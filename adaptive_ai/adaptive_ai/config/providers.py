"""
Provider configurations - model names only (details loaded from YAML).
"""

from adaptive_ai.models.llm_enums import ProviderType

# Simple model name lists by provider
# All model details (pricing, capabilities, etc.) are now loaded from YAML files
provider_model_names: dict[ProviderType, list[str]] = {
    ProviderType.GOOGLE: [
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    ProviderType.MISTRAL: [
        "mistral-small-latest",
    ],
    ProviderType.OPENAI: [
        "gpt-4.1-nano",
        "gpt-4.1-mini",
    ],
    ProviderType.DEEPSEEK: [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    ProviderType.GROQ: [
        "llama-3.1-70b-versatile",
    ],
    ProviderType.GROK: [
        "grok-3-mini",
        "grok-3",
    ],
    ProviderType.ANTHROPIC: [
        "claude-3-5-haiku-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ],
}

# Backward compatibility: Empty list to maintain imports until fully migrated
# TODO: Remove this after all services migrate to YAML-first approach
provider_model_capabilities: dict[ProviderType, list] = {
    provider: [] for provider in provider_model_names.keys()
}
