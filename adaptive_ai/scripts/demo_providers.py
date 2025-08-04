"""
Provider configurations and model capabilities for all supported providers.
Generated automatically by Model Capability Agent.
"""

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import ProviderType

# Model capabilities for all providers
provider_model_capabilities: dict[ProviderType, list[ModelCapability]] = {
    ProviderType.ANTHROPIC: [
        ModelCapability(
            description="AI model with low complexity",
            provider=ProviderType.ANTHROPIC,
            model_name="claude-3-5-haiku-20241022",
            cost_per_1m_input_tokens=0.8,
            cost_per_1m_output_tokens=4.0,
            max_context_tokens=200000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=['en'],
            model_size_params="Unknown",
            latency_tier="very_low",
            task_type="creative",
            complexity="low",
        ),
        ModelCapability(
            description="AI model with expert complexity",
            provider=ProviderType.ANTHROPIC,
            model_name="claude-sonnet-4-20250514",
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=15.0,
            max_context_tokens=200000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=['en'],
            model_size_params="Unknown",
            latency_tier="medium",
            task_type="creative",
            complexity="expert",
        ),
    ],
    ProviderType.OPENAI: [
        ModelCapability(
            description="AI model with high complexity",
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            cost_per_1m_input_tokens=30.0,
            cost_per_1m_output_tokens=60.0,
            max_context_tokens=8192,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=['en'],
            model_size_params="Unknown",
            latency_tier="medium",
            task_type="general",
            complexity="high",
        ),
        ModelCapability(
            description="AI model with expert complexity",
            provider=ProviderType.OPENAI,
            model_name="gpt-4o-mini",
            cost_per_1m_input_tokens=0.15,
            cost_per_1m_output_tokens=0.6,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=['en'],
            model_size_params="Unknown",
            latency_tier="medium",
            task_type="general",
            complexity="expert",
        ),
    ]
}
