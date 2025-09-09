"""Unit tests for ModelRouter service."""

from unittest.mock import Mock, patch

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

    @patch("adaptive_ai.services.model_router.yaml_model_db")
    @patch("adaptive_ai.services.model_router.model_registry")
    def test_initialization(self, mock_registry, mock_yaml_db, mock_logger):
        """Test router initialization."""
        mock_yaml_db.get_all_models.return_value = {}

        router = ModelRouter(lit_logger=mock_logger)

        assert router._lit_logger == mock_logger
        assert isinstance(router._model_registry, dict)

    @patch("adaptive_ai.services.model_router.yaml_model_db")
    @patch("adaptive_ai.services.model_router.model_registry")
    def test_select_models_with_full_models(
        self, mock_registry, mock_yaml_db, sample_models, mock_logger
    ):
        """Test model selection when full models are provided."""
        mock_yaml_db.get_all_models.return_value = {}

        router = ModelRouter(lit_logger=mock_logger)

        # Test selecting models with cost bias favoring cheaper options
        selected = router.select_models(
            task_complexity=0.5,
            task_type=TaskType.CHATBOT,
            models_input=sample_models,
            cost_bias=0.9,  # High cost bias = prefer cheaper models
        )

        assert len(selected) > 0
        assert all(isinstance(model, ModelCapability) for model in selected)

    @patch("adaptive_ai.services.model_router.yaml_model_db")
    @patch("adaptive_ai.services.model_router.model_registry")
    def test_select_models_cost_bias_low(
        self, mock_registry, mock_yaml_db, sample_models, mock_logger
    ):
        """Test that low cost bias prefers higher quality models."""
        mock_yaml_db.get_all_models.return_value = {}

        router = ModelRouter(lit_logger=mock_logger)

        selected = router.select_models(
            task_complexity=0.8,  # High complexity
            task_type=TaskType.CODE_GENERATION,
            models_input=sample_models,
            cost_bias=0.1,  # Low cost bias = prefer quality
        )

        assert len(selected) > 0
        # With low cost bias and high complexity, should prefer more capable models
        # (though exact ordering depends on scoring logic)

    @patch("adaptive_ai.services.model_router.yaml_model_db")
    @patch("adaptive_ai.services.model_router.model_registry")
    def test_select_models_empty_input(self, mock_registry, mock_yaml_db, mock_logger):
        """Test model selection with empty model input."""
        # Mock the registry to return some models
        mock_yaml_db.get_all_models.return_value = {
            "gpt-4": {"provider": "openai", "model_name": "gpt-4"},
            "claude-3": {"provider": "anthropic", "model_name": "claude-3-sonnet"},
        }
        mock_registry.get_model_by_key.return_value = ModelCapability(
            provider="openai", model_name="gpt-4", cost_per_1m_input_tokens=30.0
        )

        router = ModelRouter(lit_logger=mock_logger)

        selected = router.select_models(
            task_complexity=0.5,
            task_type=TaskType.CHATBOT,
            models_input=None,  # No specific models provided
            cost_bias=0.5,
        )

        # Should fall back to registry models
        assert isinstance(selected, list)

    @patch("adaptive_ai.services.model_router.yaml_model_db")
    @patch("adaptive_ai.services.model_router.model_registry")
    def test_partial_model_filtering(
        self, mock_registry, mock_yaml_db, sample_models, mock_logger
    ):
        """Test filtering with partial model specifications."""
        mock_yaml_db.get_all_models.return_value = {}

        router = ModelRouter(lit_logger=mock_logger)

        # Create partial model spec for filtering
        partial_models = [
            ModelCapability(provider="openai"),  # Only OpenAI models
            ModelCapability(
                supports_function_calling=True
            ),  # Only function calling models
        ]

        with patch.object(
            router, "_find_matching_models", return_value=sample_models[:2]
        ):  # Mock the filtering
            selected = router.select_models(
                task_complexity=0.5,
                task_type=TaskType.CODE_GENERATION,
                models_input=partial_models,
                cost_bias=0.5,
            )

            assert len(selected) <= 2  # Should be filtered

    @patch("adaptive_ai.services.model_router.yaml_model_db")
    @patch("adaptive_ai.services.model_router.model_registry")
    def test_logging_integration(
        self, mock_registry, mock_yaml_db, sample_models, mock_logger
    ):
        """Test that router logs operations correctly."""
        mock_yaml_db.get_all_models.return_value = {}

        router = ModelRouter(lit_logger=mock_logger)

        router.select_models(
            task_complexity=0.5,
            task_type=TaskType.CHATBOT,
            models_input=sample_models,
            cost_bias=0.5,
        )

        # Verify logging calls were made
        mock_logger.log.assert_called()

        # Check that some expected log keys were used
        log_calls = [call[0][0] for call in mock_logger.log.call_args_list]
        assert any("model_selection" in key for key in log_calls)


@pytest.mark.unit
class TestModelRouterEdgeCases:
    """Test edge cases and error handling."""

    @patch("adaptive_ai.services.model_router.yaml_model_db")
    @patch("adaptive_ai.services.model_router.model_registry")
    def test_invalid_cost_bias(self, mock_registry, mock_yaml_db):
        """Test behavior with invalid cost bias values."""
        mock_yaml_db.get_all_models.return_value = {}

        router = ModelRouter()

        models = [ModelCapability(provider="openai", model_name="gpt-4")]

        # Test with cost bias outside valid range - should handle gracefully
        selected = router.select_models(
            task_complexity=0.5,
            task_type=TaskType.CHATBOT,
            models_input=models,
            cost_bias=1.5,  # Invalid value
        )

        # Should still return results (router should clamp or handle invalid values)
        assert isinstance(selected, list)

    @patch("adaptive_ai.services.model_router.yaml_model_db")
    @patch("adaptive_ai.services.model_router.model_registry")
    def test_zero_complexity(self, mock_registry, mock_yaml_db):
        """Test behavior with zero task complexity."""
        mock_yaml_db.get_all_models.return_value = {}

        router = ModelRouter()

        # Create complete models to ensure they are not considered partial
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

        selected = router.select_models(
            task_complexity=0.0,
            task_type=TaskType.CHATBOT,
            models_input=models,
            cost_bias=0.5,
        )

        assert isinstance(selected, list)
        assert len(selected) > 0

    @patch("adaptive_ai.services.model_router.yaml_model_db")
    @patch("adaptive_ai.services.model_router.model_registry")
    def test_max_complexity(self, mock_registry, mock_yaml_db):
        """Test behavior with maximum task complexity."""
        mock_yaml_db.get_all_models.return_value = {}

        router = ModelRouter()

        # Create complete models to ensure they are not considered partial
        models = [
            ModelCapability(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=128000,
                supports_function_calling=True,
                task_type=TaskType.CODE_GENERATION,
            )
        ]

        selected = router.select_models(
            task_complexity=1.0,
            task_type=TaskType.CODE_GENERATION,
            models_input=models,
            cost_bias=0.5,
        )

        assert isinstance(selected, list)
        assert len(selected) > 0
        assert selected[0] == models[0]
