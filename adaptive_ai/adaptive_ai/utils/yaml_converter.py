"""
Convert YAML model data to ModelCapability objects.
Simple, fast converter for production use.
"""

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import ProviderType


def yaml_to_model_capability(yaml_data: dict, provider_name: str) -> ModelCapability:
    """
    Convert YAML model data to ModelCapability object.

    Args:
        yaml_data: Dictionary from YAML model entry
        provider_name: Provider name (e.g., "ANTHROPIC", "OPENAI")

    Returns:
        ModelCapability object with enriched data
    """
    # Map provider name to ProviderType enum (case-insensitive)
    provider_mapping: dict[str, ProviderType] = {
        "ANTHROPIC": ProviderType.ANTHROPIC,
        "OPENAI": ProviderType.OPENAI,
        "GOOGLE": ProviderType.GOOGLE,
        "GROQ": ProviderType.GROQ,
        "DEEPSEEK": ProviderType.DEEPSEEK,
        "MISTRAL": ProviderType.MISTRAL,
        "X": ProviderType.GROK,  # X provider maps to GROK enum
        "GROK": ProviderType.GROK,  # GROK provider maps to GROK enum
    }

    # Normalize provider name for case-insensitive lookup
    normalized_provider = provider_name.casefold()
    provider_type = provider_mapping.get(normalized_provider.upper(), ProviderType.OPENAI)

    return ModelCapability(
        description=yaml_data.get("description"),
        provider=provider_type,
        model_name=yaml_data.get("model_name", ""),  # Keep default for required field
        cost_per_1m_input_tokens=yaml_data.get("cost_per_1m_input_tokens"),
        cost_per_1m_output_tokens=yaml_data.get("cost_per_1m_output_tokens"),
        max_context_tokens=yaml_data.get("max_context_tokens"),
        max_output_tokens=yaml_data.get("max_output_tokens"),
        supports_function_calling=yaml_data.get("supports_function_calling"),
        languages_supported=yaml_data.get("languages_supported") or [],  # List field expects empty list, not None
        model_size_params=yaml_data.get("model_size_params"),
        latency_tier=yaml_data.get("latency_tier"),
        task_type=yaml_data.get("task_type"),
        complexity=yaml_data.get("complexity"),
    )


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for consistent lookup.

    Args:
        model_name: Raw model name

    Returns:
        Normalized model name for consistent matching
    """
    # Convert to lowercase and strip whitespace
    normalized = model_name.lower().strip()

    # Handle common variations
    common_variations = {
        "gpt4": "gpt-4",
        "gpt3.5": "gpt-3.5-turbo",
        "claude3": "claude-3-sonnet",
        "claude3.5": "claude-3-5-sonnet",
    }

    return common_variations.get(normalized, normalized)
