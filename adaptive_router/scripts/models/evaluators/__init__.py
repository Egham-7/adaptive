"""Provider evaluator factory for UniRouter model evaluation.

This module provides a factory pattern for creating provider-specific evaluators,
following the adaptive-backend architecture style.

Example:
    Create an evaluator for Anthropic:

    >>> from evaluators import EvaluatorFactory
    >>> evaluator = EvaluatorFactory.create(
    ...     provider="anthropic",
    ...     model="claude-sonnet-4-5-20250929",
    ...     api_key="sk-ant-..."
    ... )
    >>> predictions, accuracy, error_rate = evaluator.evaluate(questions)
"""

from .anthropic_evaluator import AnthropicEvaluator
from .base import APIError, BaseEvaluator, EvaluationError, MultipleChoiceAnswer
from .deepseek_evaluator import DeepSeekEvaluator
from .gemini_evaluator import GeminiEvaluator
from .grok_evaluator import GrokEvaluator
from .groq_evaluator import GroqEvaluator

__all__ = [
    "EvaluatorFactory",
    "BaseEvaluator",
    "AnthropicEvaluator",
    "GeminiEvaluator",
    "GrokEvaluator",
    "GroqEvaluator",
    "DeepSeekEvaluator",
    "EvaluationError",
    "APIError",
    "MultipleChoiceAnswer",
]


class EvaluatorFactory:
    """Factory for creating provider-specific evaluators.

    Follows the factory pattern from adaptive-backend's StreamFactory,
    providing a clean interface for evaluator instantiation.
    """

    # Registry mapping provider names to evaluator classes
    _EVALUATORS = {
        "anthropic": AnthropicEvaluator,
        "gemini": GeminiEvaluator,
        "grok": GrokEvaluator,
        "xai": GrokEvaluator,  # Alias for X.AI
        "groq": GroqEvaluator,
        "deepseek": DeepSeekEvaluator,
    }

    @staticmethod
    def create(
        provider: str,
        model: str,
        api_key: str,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ) -> BaseEvaluator:
        """Create a provider-specific evaluator instance.

        Args:
            provider: Provider name (anthropic, gemini, grok, groq, deepseek)
            model: Model name (e.g., "claude-sonnet-4-5-20250929")
            api_key: API key for authentication
            rate_limit_delay: Delay between API calls in seconds (default: 0.5)
            max_retries: Maximum retry attempts for failed requests (default: 3)

        Returns:
            Provider-specific evaluator instance

        Raises:
            ValueError: If provider is not supported

        Examples:
            >>> # Create Anthropic evaluator
            >>> evaluator = EvaluatorFactory.create(
            ...     provider="anthropic",
            ...     model="claude-sonnet-4-5-20250929",
            ...     api_key="sk-ant-..."
            ... )

            >>> # Create Gemini evaluator
            >>> evaluator = EvaluatorFactory.create(
            ...     provider="gemini",
            ...     model="gemini-2.5-pro",
            ...     api_key="AIza..."
            ... )
        """
        provider_lower = provider.lower()

        if provider_lower not in EvaluatorFactory._EVALUATORS:
            supported = ", ".join(sorted(EvaluatorFactory._EVALUATORS.keys()))
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {supported}"
            )

        evaluator_class = EvaluatorFactory._EVALUATORS[provider_lower]

        return evaluator_class(
            model=model,
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
        )

    @staticmethod
    def get_supported_providers() -> list[str]:
        """Get list of supported provider names.

        Returns:
            List of supported provider names
        """
        return sorted(EvaluatorFactory._EVALUATORS.keys())
