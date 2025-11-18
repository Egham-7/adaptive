"""Utility functions for parsing model specifications from strings."""

from typing import Tuple, Optional


def parse_model_spec(model_spec: str) -> Tuple[str, str, Optional[str]]:
    """Parse a model specification string into provider, model name, and optional variant.

    Args:
        model_spec: String in format "provider/model_name" or "provider/model_name:variant"

    Returns:
        Tuple of (provider, model_name, variant) where variant is None if not specified

    Raises:
        ValueError: If the format is invalid
    """
    if not model_spec or not model_spec.strip():
        raise ValueError("Model specification cannot be empty")

    # Check for exactly one slash
    slash_count = model_spec.count("/")
    if slash_count != 1:
        raise ValueError(
            f"Invalid model specification format: '{model_spec}'. "
            "Expected format: 'provider/model_name' or 'provider/model_name:variant'"
        )

    provider, rest = model_spec.split("/", 1)
    provider = provider.strip()

    if not provider:
        raise ValueError("Provider cannot be empty")

    # Now split the rest by ':' to get model_name and optional variant
    if ":" in rest:
        model_name, variant = rest.split(":", 1)
        model_name = model_name.strip()
        variant = variant.strip()
        if not variant:
            raise ValueError("Variant cannot be empty when specified")
    else:
        model_name = rest.strip()
        variant = None

    if not model_name:
        raise ValueError("Model name cannot be empty")

    return provider, model_name, variant
