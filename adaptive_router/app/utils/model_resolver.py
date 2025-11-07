"""Model resolution utilities for the Adaptive Router application."""

import logging
from typing import List, Optional

from app.models import RegistryModel
from adaptive_router.models.api import Model

logger = logging.getLogger(__name__)


def _registry_model_to_model(
    registry_model: RegistryModel,
) -> Optional[Model]:
    """Convert a RegistryModel to a Model for library compatibility.

    Args:
        registry_model: The registry model to convert

    Returns:
        Model object if conversion succeeds, None if pricing is missing/invalid

    Note:
        Returns None (with warning logged) for models with:
        - Missing pricing information
        - Invalid pricing values (None, negative, or zero)
        - Unparseable pricing strings
    """
    # Extract pricing information (costs are per token, convert to per million tokens)
    prompt_cost_per_million = 0.0
    completion_cost_per_million = 0.0

    if registry_model.pricing:
        try:
            # Handle None pricing values - they should be treated as missing
            if (
                registry_model.pricing.prompt_cost is None
                or registry_model.pricing.completion_cost is None
            ):
                raise ValueError("Pricing values cannot be None")

            prompt_cost = float(registry_model.pricing.prompt_cost)
            completion_cost = float(registry_model.pricing.completion_cost)

            # Check for invalid (negative) pricing values
            # Registry uses negative values like -1000000.0 as sentinel for "no pricing"
            # Zero costs are now accepted as valid
            if prompt_cost < 0 or completion_cost < 0:
                logger.warning(
                    "Skipping model '%s' with invalid pricing: "
                    "prompt_cost=%s, completion_cost=%s (must be non-negative)",
                    registry_model.unique_id(),
                    prompt_cost,
                    completion_cost,
                )
                return None

            # Convert from per-token to per-million-tokens
            prompt_cost_per_million = prompt_cost * 1_000_000
            completion_cost_per_million = completion_cost * 1_000_000
        except (ValueError, TypeError) as e:
            logger.warning(
                "Skipping model '%s' - failed to parse pricing: "
                "prompt_cost='%s', completion_cost='%s' (%s)",
                registry_model.unique_id(),
                registry_model.pricing.prompt_cost,
                registry_model.pricing.completion_cost,
                e,
            )
            return None
    else:
        # If pricing is missing entirely
        logger.warning(
            "Skipping model '%s' - no pricing information", registry_model.unique_id()
        )
        return None

    return Model(
        provider=registry_model.provider,
        model_name=registry_model.model_name,
        cost_per_1m_input_tokens=prompt_cost_per_million,
        cost_per_1m_output_tokens=completion_cost_per_million,
    )


def resolve_models(
    model_specs: List[str],
    registry_models: List[RegistryModel],
) -> List[Model]:
    """Resolve a list of model specifications to Model objects using exact matching.

    Args:
        model_specs: List of model specifications in "provider/model_name" format
        registry_models: List of all available registry models

    Returns:
        List of resolved Model objects

    Raises:
        ValueError: If any model specification is invalid, cannot be resolved, or has missing/invalid pricing
    """
    resolved_models = []

    for spec in model_specs:
        try:
            provider, model_name = spec.split("/", 1)
        except ValueError as e:
            raise ValueError(
                f"Invalid model specification '{spec}': expected format 'provider/model_name'"
            ) from e

        # Find exact match
        candidates = [
            m
            for m in registry_models
            if (m.provider and m.provider.lower() == provider.lower())
            and (m.model_name and m.model_name.lower() == model_name.lower())
        ]

        if not candidates:
            # Show available models from the same provider if any
            provider_models = [
                m.unique_id()
                for m in registry_models
                if m.provider and m.provider.lower() == provider.lower()
            ]

            error_msg = f"Model '{spec}' not found in registry"
            if provider_models:
                suggestions = provider_models[:5]
                error_msg += f". Available {provider} models: {', '.join(suggestions)}"
                if len(provider_models) > 5:
                    error_msg += f" (and {len(provider_models) - 5} more)"

            raise ValueError(error_msg)

        if len(candidates) > 1:
            raise ValueError(
                f"Multiple models found for '{spec}': "
                f"{[m.unique_id() for m in candidates]}"
            )

        model = _registry_model_to_model(candidates[0])
        if model is None:
            raise ValueError(
                f"Model '{spec}' found in registry but has invalid/missing pricing"
            )
        resolved_models.append(model)

    return resolved_models
