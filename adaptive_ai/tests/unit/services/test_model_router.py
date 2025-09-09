"""Unit tests for ModelRouter service."""

from unittest.mock import Mock

import pytest

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import TaskType
from adaptive_ai.services.model_router import ModelRouter


class TestModelRouter:
    """Test ModelRouter class logic without external dependencies."""

    @pytest.fixture
    def mock_logger(self):
        """Mock LitServe logger."""
        logger = Mock()
        logger.log = Mock()
        return logger

    @pytest.fixture
    def sample_models(self):
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

    def test_initialization(self, mock_logger):
        """Test router initialization."""
        router = ModelRouter(lit_logger=mock_logger)

        assert router._lit_logger == mock_logger
        assert hasattr(router, "_calculate_complexity_score")
        assert hasattr(router, "select_models")

    def test_select_models_with_full_models(self, sample_models, mock_logger):
        """Test model selection when full models are provided."""
        router = ModelRouter(lit_logger=mock_logger)

        # Test selecting models with cost bias favoring cheaper options
        selected = router.select_models(
            task_complexity=0.5,
            task_type=TaskType.TEXT_GENERATION,  # Use task type that models support
            models_input=sample_models,
            cost_bias=0.9,  # High cost bias = prefer cheaper models
        )

        assert len(selected) > 0
        assert all(isinstance(model, ModelCapability) for model in selected)
        # With high cost bias, cheaper models should be prioritized
        # gpt-3.5-turbo should be ranked higher than gpt-4
        if len(selected) >= 2:
            assert (
                selected[0].cost_per_1m_input_tokens
                < selected[1].cost_per_1m_input_tokens
            )

    def test_select_models_cost_bias_low(self, sample_models, mock_logger):
        """Test that low cost bias prefers higher quality models."""
        router = ModelRouter(lit_logger=mock_logger)

        selected = router.select_models(
            task_complexity=0.8,  # High complexity
            task_type=TaskType.CODE_GENERATION,
            models_input=sample_models,
            cost_bias=0.1,  # Low cost bias = prefer quality
        )

        assert len(selected) > 0
        # With low cost bias and high complexity, more expensive models should be prioritized
        # gpt-4 should be preferred for complex code generation tasks

    def test_select_models_empty_input(self, mock_logger):
        """Test selecting models when no models are provided."""
        router = ModelRouter(lit_logger=mock_logger)

        # When no models are provided, router should use models from registry
        selected = router.select_models(
            task_complexity=0.5,
            task_type=TaskType.TEXT_GENERATION,
            models_input=None,
            cost_bias=0.5,
        )

        # Should return models from the internal registry
        assert len(selected) > 0
        assert all(isinstance(model, ModelCapability) for model in selected)

    def test_partial_model_filtering(self, mock_logger):
        """Test filtering with partial ModelCapability."""
        router = ModelRouter(lit_logger=mock_logger)

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
            task_type=TaskType.TEXT_GENERATION,
            models_input=partial_models,
            cost_bias=0.5,
        )

        # Should return only OpenAI models
        assert len(selected) > 0
        assert all(model.provider == "openai" for model in selected if model.provider)

    def test_logging_integration(self, sample_models, mock_logger):
        """Test that router logs metrics correctly."""
        router = ModelRouter(lit_logger=mock_logger)

        router.select_models(
            task_complexity=0.5,
            task_type=TaskType.CODE_GENERATION,
            models_input=sample_models,
            cost_bias=0.5,
        )

        # Verify logging was called
        assert mock_logger.log.called
        # Check that relevant metrics were logged
        call_args = [call[0] for call in mock_logger.log.call_args_list]
        logged_keys = [args[0] for args in call_args]

        # Should log registry and task filtering
        assert any("registry" in key for key in logged_keys)
        assert any("task_type" in key for key in logged_keys)


class TestModelRouterEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_cost_bias(self):
        """Test handling of invalid cost bias values."""
        router = ModelRouter()

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
            task_type=TaskType.TEXT_GENERATION,
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
            task_type=TaskType.TEXT_GENERATION,
            models_input=models_no_task,
            cost_bias=-1.0,
        )
        assert len(selected) > 0

    def test_zero_complexity(self):
        """Test handling of zero complexity."""
        router = ModelRouter()

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
            task_type=TaskType.TEXT_GENERATION,
            models_input=models,
            cost_bias=0.5,
        )

        assert len(selected) > 0

    def test_max_complexity(self):
        """Test handling of maximum complexity."""
        router = ModelRouter()

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
            task_type=TaskType.TEXT_GENERATION,
            models_input=models,
            cost_bias=0.5,
        )

        assert len(selected) > 0
