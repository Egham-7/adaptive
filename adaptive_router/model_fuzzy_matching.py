"""Fuzzy model ID matching utilities.

Provides functions for normalizing model IDs and creating fuzzy lookup variants
to handle model naming variations (dates, versions, etc.).
"""

import logging
import re
from difflib import SequenceMatcher

from app_config import FUZZY_MATCH_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


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
    """
    variants = [model_id]  # Always include original

    # Pattern 1: Remove date suffixes (YYYYMMDD or YYYY-MM-DD format)
    without_date = re.sub(r"-\d{8}$", "", model_id)  # -20250929
    without_date = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", without_date)  # -2024-04-09
    if without_date != model_id and without_date not in variants:
        variants.append(without_date)

    # Pattern 2: Remove common version/variant suffixes
    without_suffix = re.sub(r"-(latest|preview|alpha|beta|v\d+)$", "", without_date)
    if without_suffix != without_date and without_suffix not in variants:
        variants.append(without_suffix)

    # Pattern 3: Convert version numbers to use dots (e.g., 4-5 -> 4.5)
    # Match version-like patterns: word-digit-digit
    with_dots = re.sub(r"(\w+)-(\d+)-(\d+)", r"\1-\2.\3", without_suffix)
    if with_dots != without_suffix and with_dots not in variants:
        variants.append(with_dots)

    return variants


def calculate_similarity(a: str, b: str) -> float:
    """Calculate similarity score between two strings.

    Uses SequenceMatcher for fuzzy string matching.

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
    best_match = None
    best_score = 0.0

    for available_id in available_ids:
        score = calculate_similarity(target_id, available_id)
        if score > best_score:
            best_score = score
            best_match = available_id

    if best_score >= threshold:
        return best_match, best_score

    return None, 0.0


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
    enhanced: dict[str, float] = dict(model_costs)  # Start with original
    variants_added = 0

    for model_id, cost in model_costs.items():
        variants = normalize_model_id(model_id)

        for variant in variants:
            # Add variant if not already present
            if variant not in enhanced:
                enhanced[variant] = cost
                if variant != model_id:  # Don't log the original
                    logger.debug(
                        f"Fuzzy variant: '{variant}' -> '{model_id}' (cost: {cost})"
                    )
                    variants_added += 1

    if variants_added > 0:
        logger.info(f"Added {variants_added} fuzzy matching variants for model lookup")

    return enhanced
