"""Model resolution utilities for the Adaptive Router application."""

from typing import List

from adaptive_router.models.registry import RegistryModel
from adaptive_router.utils.model_parser import parse_model_spec


def resolve_models(
    model_specs: List[str], registry_models: List[RegistryModel]
) -> List[RegistryModel]:
    """Resolve a list of model specifications to RegistryModel objects.

    Args:
        model_specs: List of model specifications in "provider:model_name" format
        registry_models: List of all available registry models

    Returns:
        List of resolved RegistryModel objects

    Raises:
        ValueError: If any model specification is invalid or cannot be resolved
    """
    resolved_models = []

    for spec in model_specs:
        try:
            provider, model_name = parse_model_spec(spec)
        except ValueError as e:
            raise ValueError(f"Invalid model specification '{spec}': {e}") from e

        # Fetch models from registry with provider and model_name filters
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

        if not candidates:
            raise ValueError(f"Model '{spec}' not found in registry")

        if len(candidates) > 1:
            # If multiple models match, this is unexpected - the combination should be unique
            raise ValueError(
                f"Multiple models found for '{spec}': "
                f"{[m.unique_id() for m in candidates]}"
            )

        resolved_models.append(candidates[0])

    return resolved_models
