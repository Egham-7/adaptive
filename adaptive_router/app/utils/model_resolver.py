"""Model resolution utilities for the Adaptive Router application."""

import logging
from typing import List

from adaptive_router.models.registry import RegistryModel
from adaptive_router.utils.model_parser import parse_model_spec
from app.utils.fuzzy_matching import normalize_model_id

logger = logging.getLogger(__name__)


def resolve_models(
    model_specs: List[str], registry_models: List[RegistryModel]
) -> List[RegistryModel]:
    """Resolve a list of model specifications to RegistryModel objects.

    Uses exact matching first, then falls back to fuzzy matching if no exact match is found.

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
            resolved_models.append(candidates[0])
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
            resolved_models.append(matched_model)
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
