"""Fuzzy model ID matching utilities.

Provides functions for normalizing model IDs and creating fuzzy lookup variants
to handle model naming variations (dates, versions, etc.).
"""

import logging
import re
from difflib import SequenceMatcher

from app.config import FUZZY_MATCH_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# Provider aliases mapping - maps canonical provider names to known aliases
PROVIDER_ALIASES = {
    "google": ["gemini"],  # google models can be referenced as gemini
    "gemini": ["google"],  # gemini models can be referenced as google
}


def normalize_model_id(model_id: str) -> list[str]:
    """Generate normalized variants of a model ID for fuzzy matching.

    Args:
        model_id: Original model ID (e.g., "anthropic:claude-sonnet-4-5-20250929")

    Returns:
        List of normalized variants for matching, ordered by specificity:
        1. Original ID
        2. Without date suffixes (e.g., -20250929, -2024-04-09)
        3. Without version suffixes (e.g., -latest, -preview, -v1)
        4. With dots instead of hyphens in version numbers
        5. With alternative separators (: <-> /)
        6. With provider aliases (e.g., google <-> gemini)

    Examples:
        >>> normalize_model_id("anthropic:claude-sonnet-4-5-20250929")
        [
            "anthropic:claude-sonnet-4-5-20250929",
            "anthropic:claude-sonnet-4-5",
            "anthropic:claude-sonnet-4.5"
        ]

        >>> normalize_model_id("openai:gpt-4-turbo-2024-04-09")
        [
            "openai:gpt-4-turbo-2024-04-09",
            "openai:gpt-4-turbo",
            "openai:gpt-4-turbo"
        ]

        >>> normalize_model_id("google:gemini-2.5-flash-lite")
        [
            "google:gemini-2.5-flash-lite",
            "google/gemini-2.5-flash-lite",
            "gemini:gemini-2.5-flash-lite",  # Provider alias variant
            "gemini/gemini-2.5-flash-lite"   # Provider alias with / separator
        ]
    """
    # Apply transformations sequentially
    transformations = [
        lambda x: x,  # Original
        lambda x: re.sub(r"-\d{8}$", "", x),  # Remove YYYYMMDD
        lambda x: re.sub(r"-\d{4}-\d{2}-\d{2}$", "", x),  # Remove YYYY-MM-DD
        lambda x: re.sub(
            r"-(preview|alpha|beta)-\d{2}-\d{4}$", "", x
        ),  # Remove preview/alpha/beta with MM-YYYY dates
        lambda x: re.sub(
            r"-(latest|preview|alpha|beta|v\d+)$", "", x
        ),  # Remove version suffixes
        lambda x: re.sub(
            r"(\w+)-(\d+)-(\d+)", r"\1-\2.\3", x
        ),  # Convert hyphens to dots in versions (4-5 -> 4.5)
        lambda x: re.sub(
            r"(\w+)-(\d+)\.(\d+)", r"\1-\2-\3", x
        ),  # Convert dots to hyphens in versions (4.5 -> 4-5)
    ]

    # Apply all transformations and deduplicate while preserving order
    seen: set[str] = set()
    variants: list[str] = []

    for transform in transformations:
        variant = transform(model_id)
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)

    # Add cross-separator variants (: <-> /) for better matching between systems
    cross_separator_variants: list[str] = []
    for variant in variants:
        if ":" in variant:
            cross_separator_variants.append(variant.replace(":", "/"))
        elif "/" in variant:
            cross_separator_variants.append(variant.replace("/", ":"))

    for variant in cross_separator_variants:
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)

    # Add provider alias variants (e.g., google <-> gemini)
    provider_alias_variants: list[str] = []
    for variant in variants:
        # Split on both : and / separators
        if ":" in variant:
            provider, model_name = variant.split(":", 1)
            separator = ":"
        elif "/" in variant:
            provider, model_name = variant.split("/", 1)
            separator = "/"
        else:
            continue

        # Check if this provider has aliases
        if provider.lower() in PROVIDER_ALIASES:
            for alias in PROVIDER_ALIASES[provider.lower()]:
                aliased_variant = f"{alias}{separator}{model_name}"
                provider_alias_variants.append(aliased_variant)

    for variant in provider_alias_variants:
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)

    return variants


def calculate_similarity(a: str, b: str) -> float:
    """Calculate similarity score between two strings using SequenceMatcher.

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score between 0.0 and 1.0, where 1.0 is identical
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(
    target_id: str,
    available_ids: list[str],
    threshold: float = FUZZY_MATCH_SIMILARITY_THRESHOLD,
) -> tuple[str | None, float]:
    """Find the best matching model ID from available IDs.

    Args:
        target_id: Target model ID to match
        available_ids: List of available model IDs
        threshold: Minimum similarity threshold (0.0-1.0)

    Returns:
        Tuple of (best_match_id, similarity_score) or (None, 0.0) if no match
    """
    if not available_ids:
        return None, 0.0

    # Find ID with highest similarity score
    similarities = [
        (available_id, calculate_similarity(target_id, available_id))
        for available_id in available_ids
    ]
    best_match, best_score = max(similarities, key=lambda x: x[1])

    return (best_match, best_score) if best_score >= threshold else (None, 0.0)


def enhance_model_costs_with_fuzzy_keys(
    model_costs: dict[str, float],
) -> dict[str, float]:
    """Enhance model costs dictionary with normalized variant keys for fuzzy matching.

    Creates multiple lookup keys for each model, allowing fuzzy matching.
    The original model IDs are preserved, and normalized variants are added.

    Args:
        model_costs: Original model costs dict (model_id -> cost)

    Returns:
        Enhanced dict with original + normalized keys (all pointing to costs)

    Example:
        Input: {"anthropic:claude-sonnet-4-5": 5.0}
        Output: {
            "anthropic:claude-sonnet-4-5": 5.0,
            "anthropic:claude-sonnet-4.5": 5.0,
        }
    """
    # Generate all variants for all models
    variant_mappings = [
        (variant, model_id, cost)
        for model_id, cost in model_costs.items()
        for variant in normalize_model_id(model_id)
    ]

    # Build enhanced dict with variants (excluding duplicates)
    enhanced = dict(model_costs)
    new_variants = {
        variant: cost
        for variant, model_id, cost in variant_mappings
        if variant not in enhanced and variant != model_id
    }
    enhanced.update(new_variants)

    # Log new variants
    if new_variants:
        for variant, cost in new_variants.items():
            # Find original model_id for this variant
            original = next(
                model_id
                for model_id, original_cost in model_costs.items()
                if original_cost == cost and variant in normalize_model_id(model_id)
            )
            logger.debug(f"Fuzzy variant: '{variant}' -> '{original}' (cost: {cost})")

        logger.info(
            f"Added {len(new_variants)} fuzzy matching variants for model lookup"
        )

    return enhanced
