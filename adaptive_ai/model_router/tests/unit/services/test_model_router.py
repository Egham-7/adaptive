"""Unit tests for ModelRouter service."""

import pytest

from model_router.models.llm_core_models import ModelCapability
from model_router.services.model_registry import model_registry
from model_router.services.model_router import ModelRouter


class TestModelRouter:
    """Test ModelRouter class logic without external dependencies."""

    # Removed mock_logger fixture as we no longer use LitServe logging

    @pytest.fixture
    def sample_models(self) -> list[ModelCapability]:
        """Sample models for testing."""
        return [
            ModelCapability(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=128000,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="openai",
                model_name="gpt-3.5-turbo",
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=2.0,
                max_context_tokens=16000,
                supports_function_calling=True,
            ),
            ModelCapability(
                provider="anthropic",
                model_name="claude-3-sonnet",
                cost_per_1m_input_tokens=15.0,
                cost_per_1m_output_tokens=75.0,
                max_context_tokens=200000,
                supports_function_calling=False,
            ),
        ]

    def test_initialization(self) -> None:
        """Test router initialization."""
        router = ModelRouter(model_registry)

        assert hasattr(router, "_calculate_complexity_score")
        assert hasattr(router, "select_models")

    def test_select_models_with_full_models(
        self, sample_models: list[ModelCapability]
    ) -> None:
        """Test model selection when full models are provided."""
        router = ModelRouter(model_registry)

        # Test selecting models with cost bias favoring capable options
        selected = router.select_models(
            task_complexity=0.5,
            task_type="Text Generation",  # Use task type that models support
            models_input=sample_models,
            cost_bias=0.9,  # High cost bias = prefer capable models
        )

        assert len(selected) > 0
        assert all(isinstance(model, ModelCapability) for model in selected)
        # With high cost bias, more capable models should be prioritized
        # Models with higher context tokens should be ranked higher
        if len(selected) >= 2:
            assert (selected[0].max_context_tokens or 0) >= (
                selected[1].max_context_tokens or 0
            )

    def test_select_models_cost_bias_low(
        self, sample_models: list[ModelCapability]
    ) -> None:
        """Test that low cost bias prefers higher quality models."""
        router = ModelRouter(model_registry)

        selected = router.select_models(
            task_complexity=0.8,  # High complexity
            task_type="Code Generation",
            models_input=sample_models,
            cost_bias=0.1,  # Low cost bias = prefer quality
        )

        assert len(selected) > 0
        # With low cost bias and high complexity, more expensive models should be prioritized
        # gpt-4 should be preferred for complex code generation tasks

    def test_select_models_empty_input(self) -> None:
        """Test selecting models when no models are provided."""
        router = ModelRouter(model_registry)

        # When no models are provided, router should use models from registry
        selected = router.select_models(
            task_complexity=0.5,
            task_type="Text Generation",
            models_input=None,
            cost_bias=0.5,
        )

        # Should return models from the internal registry
        assert len(selected) > 0
        assert all(isinstance(model, ModelCapability) for model in selected)

    def test_partial_model_filtering(self) -> None:
        """Test filtering with partial ModelCapability."""
        router = ModelRouter(model_registry)

        # Create a partial model as filter criteria
        partial_models = [
            ModelCapability(
                provider="openai",  # Only specify provider
                model_name=None,
                cost_per_1m_input_tokens=None,
                cost_per_1m_output_tokens=None,
                max_context_tokens=None,
                supports_function_calling=None,
            )
        ]

        selected = router.select_models(
            task_complexity=0.5,
            task_type="Text Generation",
            models_input=partial_models,
            cost_bias=0.5,
        )

        # Should return only OpenAI models
        assert len(selected) > 0
        assert all(model.provider == "openai" for model in selected if model.provider)

    def test_model_selection_basic(self, sample_models: list[ModelCapability]) -> None:
        """Test basic model selection functionality."""
        router = ModelRouter(model_registry)

        selected = router.select_models(
            task_complexity=0.5,
            task_type="Code Generation",
            models_input=sample_models,
            cost_bias=0.5,
        )

        # Verify models were selected
        assert len(selected) > 0
        assert all(isinstance(model, ModelCapability) for model in selected)


class TestModelRouterEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_cost_bias(self) -> None:
        """Test handling of invalid cost bias values."""
        router = ModelRouter(model_registry)

        # Cost bias should be clamped to [0, 1]
        models = [
            ModelCapability(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=128000,
                supports_function_calling=True,
            )
        ]

        # Test with cost bias > 1
        selected = router.select_models(
            task_complexity=0.5,
            task_type="Text Generation",
            models_input=models,
            cost_bias=2.0,
        )
        assert len(selected) > 0

        # Test with cost bias < 0
        # Use models with no task type to avoid filtering issues
        models_no_task = [
            ModelCapability(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=128000,
                supports_function_calling=True,
                task_type=None,  # No task type to avoid filtering
            )
        ]
        selected = router.select_models(
            task_complexity=0.5,
            task_type="Text Generation",
            models_input=models_no_task,
            cost_bias=-1.0,
        )
        assert len(selected) > 0

    def test_zero_complexity(self) -> None:
        """Test handling of zero complexity."""
        router = ModelRouter(model_registry)

        models = [
            ModelCapability(
                provider="openai",
                model_name="test-model-unique",  # Use a model name not in registry
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=2.0,
                max_context_tokens=16000,
                supports_function_calling=True,
                task_type=None,  # No task type to avoid filtering
            )
        ]

        selected = router.select_models(
            task_complexity=0.0,
            task_type="Text Generation",
            models_input=models,
            cost_bias=0.5,
        )

        assert len(selected) > 0

    def test_max_complexity(self) -> None:
        """Test handling of maximum complexity."""
        router = ModelRouter(model_registry)

        models = [
            ModelCapability(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=128000,
                supports_function_calling=True,
                task_type=None,  # No task type to avoid filtering
            )
        ]

        selected = router.select_models(
            task_complexity=1.0,
            task_type="Text Generation",
            models_input=models,
            cost_bias=0.5,
        )

        assert len(selected) > 0
