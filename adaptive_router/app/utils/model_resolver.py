"""Model resolution utilities for the Adaptive Router application."""

import logging
from typing import List, Optional

from app.models import RegistryModel
from adaptive_router.models.api import Model
from adaptive_router.utils.model_parser import parse_model_spec
from app.utils.fuzzy_matching import normalize_model_id

logger = logging.getLogger(__name__)


def _registry_model_to_model(
    registry_model: RegistryModel, raise_on_error: bool = False
) -> Optional[Model]:
    """Convert a RegistryModel to a Model for library compatibility.

    Args:
        registry_model: The registry model to convert
        raise_on_error: If True, raise ValueError on missing/invalid pricing.
                        If False, log warning and return None.

    Returns:
        Model object if conversion succeeds, None if pricing is missing/invalid and raise_on_error=False

    Raises:
        ValueError: If pricing is missing/invalid and raise_on_error=True
    """
    # Extract pricing information (costs are per token, convert to per million tokens)
    prompt_cost_per_million = 0.0
    completion_cost_per_million = 0.0

    if registry_model.pricing:
        try:
            prompt_cost = float(registry_model.pricing.prompt_cost or 0)
            completion_cost = float(registry_model.pricing.completion_cost or 0)
            # Convert from per-token to per-million-tokens
            prompt_cost_per_million = prompt_cost * 1_000_000
            completion_cost_per_million = completion_cost * 1_000_000
        except (ValueError, TypeError) as e:
            error_msg = (
                f"Failed to parse pricing for model '{registry_model.unique_id()}': "
                f"prompt_cost='{registry_model.pricing.prompt_cost}', "
                f"completion_cost='{registry_model.pricing.completion_cost}'"
            )
            if raise_on_error:
                raise ValueError(error_msg) from e
            else:
                logger.warning(error_msg)
                return None
    else:
        # If pricing is missing entirely
        error_msg = f"No pricing information for model '{registry_model.unique_id()}'"
        if raise_on_error:
            raise ValueError(error_msg)
        else:
            logger.warning(error_msg)
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
    """Resolve a list of model specifications to Model objects.

    Uses exact matching first, then falls back to fuzzy matching if no exact match is found.

    Args:
        model_specs: List of model specifications in "provider:model_name" format
        registry_models: List of all available registry models

    Returns:
        List of resolved Model objects

    Raises:
        ValueError: If any model specification is invalid, cannot be resolved, or has missing/invalid pricing
    """
    resolved_models = []

    for spec in model_specs:
        try:
            provider, model_name = parse_model_spec(spec)
        except ValueError as e:
            raise ValueError(f"Invalid model specification '{spec}': {e}") from e

        # Try exact matching first
        try:
            candidates = [
                m
                for m in registry_models
                if (m.provider and provider and m.provider.lower() == provider.lower())
                and (
                    m.model_name
                    and model_name
                    and m.model_name.lower() == model_name.lower()
                )
            ]
        except Exception as e:
            raise ValueError(
                f"Failed to fetch model '{spec}' from registry: {e}"
            ) from e

        # If exact match found, use it
        if candidates:
            if len(candidates) > 1:
                # If multiple models match, this is unexpected - the combination should be unique
                raise ValueError(
                    f"Multiple models found for '{spec}': "
                    f"{[m.unique_id() for m in candidates]}"
                )
            model = _registry_model_to_model(candidates[0], raise_on_error=True)
            assert model is not None  # Should never be None when raise_on_error=True
            resolved_models.append(model)
            continue

        # No exact match - try fuzzy matching
        logger.debug(f"No exact match for '{spec}', trying fuzzy matching...")

        # Generate normalized variants of the requested spec
        spec_variants = normalize_model_id(spec)
        logger.debug(
            f"Generated {len(spec_variants)} variants for '{spec}': {spec_variants}"
        )

        # Try to find a match using normalized variants
        fuzzy_candidates = []
        for variant in spec_variants:
            try:
                variant_provider, variant_model = parse_model_spec(variant)
            except ValueError:
                continue

            matches = [
                m
                for m in registry_models
                if (
                    m.provider
                    and variant_provider
                    and m.provider.lower() == variant_provider.lower()
                )
                and (
                    m.model_name
                    and variant_model
                    and m.model_name.lower() == variant_model.lower()
                )
            ]

            if matches:
                fuzzy_candidates.extend(matches)
                # Found a match with this variant, stop searching
                break

        # If fuzzy matching found candidates, use the first one
        if fuzzy_candidates:
            matched_model = fuzzy_candidates[0]
            logger.info(
                f"Fuzzy match: '{spec}' resolved to '{matched_model.unique_id()}' "
                f"(provider: {matched_model.provider}, model: {matched_model.model_name})"
            )
            model = _registry_model_to_model(matched_model, raise_on_error=True)
            assert model is not None  # Should never be None when raise_on_error=True
            resolved_models.append(model)
            continue

        # No match found at all - provide helpful error message
        # Show available models from the same provider if any
        provider_models = [
            m.unique_id()
            for m in registry_models
            if m.provider and m.provider.lower() == provider.lower()
        ]

        error_msg = f"Model '{spec}' not found in registry"
        if provider_models:
            # Show up to 5 similar models as suggestions
            suggestions = provider_models[:5]
            error_msg += f". Available {provider} models: {', '.join(suggestions)}"
            if len(provider_models) > 5:
                error_msg += f" (and {len(provider_models) - 5} more)"

        raise ValueError(error_msg)

    return resolved_models
