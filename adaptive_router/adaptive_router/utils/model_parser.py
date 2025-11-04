"""Utility functions for parsing model specifications from strings."""

from typing import Tuple


def parse_model_spec(model_spec: str) -> Tuple[str, str]:
    """Parse a model specification string into provider and model name.

    Args:
        model_spec: String in format "provider:model_name"

    Returns:
        Tuple of (provider, model_name)

    Raises:
        ValueError: If the format is invalid (not exactly one colon)
    """
    if not model_spec or not model_spec.strip():
        raise ValueError("Model specification cannot be empty")

    parts = model_spec.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid model specification format: '{model_spec}'. "
            "Expected format: 'provider:model_name'"
        )

    provider, model_name = parts
    provider = provider.strip()
    model_name = model_name.strip()

    if not provider:
        raise ValueError("Provider cannot be empty")
    if not model_name:
        raise ValueError("Model name cannot be empty")

    return provider, model_name
