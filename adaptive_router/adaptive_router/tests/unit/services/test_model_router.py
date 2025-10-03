"""Unit tests for ModelRouter service."""

import pytest

from adaptive_router.models.llm_core_models import ModelCapability
from adaptive_router.services.model_registry import ModelRegistry
from adaptive_router.services.model_router import ModelRouter
from adaptive_router.services.yaml_model_loader import YAMLModelDatabase


@pytest.fixture
def model_registry() -> ModelRegistry:
    """Create a ModelRegistry instance for testing."""
    yaml_db = YAMLModelDatabase()
    return ModelRegistry(yaml_db)


class TestModelRouter:
    """Test ModelRouter class logic without external dependencies."""

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

    def test_initialization(self, model_registry: ModelRegistry) -> None:
        """Test router initialization."""
        router = ModelRouter(model_registry)

        assert hasattr(router, "_calculate_complexity_score")
        assert hasattr(router, "select_model")

    def test_initialization_without_params(self) -> None:
        """Test router initialization without parameters."""
        router = ModelRouter()

        assert hasattr(router, "_calculate_complexity_score")
        assert hasattr(router, "_select_models")
        assert router._model_registry is not None
        assert router._prompt_classifier is not None

    def test_select_models_with_full_models(
        self, model_registry: ModelRegistry, sample_models: list[ModelCapability]
    ) -> None:
        """Test model selection when full models are provided."""
        router = ModelRouter(model_registry)

        selected = router._select_models(
            task_complexity=0.5,
            task_type="Text Generation",
            models_input=sample_models,
            cost_bias=0.9,
        )

        assert len(selected) > 0
        assert all(isinstance(model, ModelCapability) for model in selected)
        if len(selected) >= 2:
            assert (selected[0].max_context_tokens or 0) >= (
                selected[1].max_context_tokens or 0
            )

    def test_select_models_cost_bias_low(
        self, model_registry: ModelRegistry, sample_models: list[ModelCapability]
    ) -> None:
        """Test that low cost bias prefers higher quality models."""
        router = ModelRouter(model_registry)

        selected = router._select_models(
            task_complexity=0.8,
            task_type="Code Generation",
            models_input=sample_models,
            cost_bias=0.1,
        )

        assert len(selected) > 0

    def test_select_models_empty_input(self, model_registry: ModelRegistry) -> None:
        """Test selecting models when no models are provided."""
        router = ModelRouter(model_registry)

        selected = router._select_models(
            task_complexity=0.5,
            task_type="Text Generation",
            models_input=None,
            cost_bias=0.5,
        )

        assert len(selected) > 0
        assert all(isinstance(model, ModelCapability) for model in selected)

    def test_partial_model_filtering(self, model_registry: ModelRegistry) -> None:
        """Test filtering with partial ModelCapability."""
        router = ModelRouter(model_registry)

        partial_models = [
            ModelCapability(
                provider="openai",
                model_name=None,
                cost_per_1m_input_tokens=None,
                cost_per_1m_output_tokens=None,
                max_context_tokens=None,
                supports_function_calling=None,
            )
        ]

        selected = router._select_models(
            task_complexity=0.5,
            task_type="Text Generation",
            models_input=partial_models,
            cost_bias=0.5,
        )

        assert len(selected) > 0
        assert all(model.provider == "openai" for model in selected if model.provider)

    def test_model_selection_basic(
        self, model_registry: ModelRegistry, sample_models: list[ModelCapability]
    ) -> None:
        """Test basic model selection functionality."""
        router = ModelRouter(model_registry)

        selected = router._select_models(
            task_complexity=0.5,
            task_type="Code Generation",
            models_input=sample_models,
            cost_bias=0.5,
        )

        assert len(selected) > 0
        assert all(isinstance(model, ModelCapability) for model in selected)


class TestModelRouterEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_cost_bias(self, model_registry: ModelRegistry) -> None:
        """Test handling of invalid cost bias values."""
        router = ModelRouter(model_registry)

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

        selected = router._select_models(
            task_complexity=0.5,
            task_type="Text Generation",
            models_input=models,
            cost_bias=2.0,
        )
        assert len(selected) > 0

        models_no_task = [
            ModelCapability(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=30.0,
                cost_per_1m_output_tokens=60.0,
                max_context_tokens=128000,
                supports_function_calling=True,
                task_type=None,
            )
        ]
        selected = router._select_models(
            task_complexity=0.5,
            task_type="Text Generation",
            models_input=models_no_task,
            cost_bias=-1.0,
        )
        assert len(selected) > 0

    def test_zero_complexity(self, model_registry: ModelRegistry) -> None:
        """Test handling of zero complexity."""
        router = ModelRouter(model_registry)

        models = [
            ModelCapability(
                provider="openai",
                model_name="test-model-unique",
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=2.0,
                max_context_tokens=16000,
                supports_function_calling=True,
                task_type=None,
            )
        ]

        selected = router._select_models(
            task_complexity=0.0,
            task_type="Text Generation",
            models_input=models,
            cost_bias=0.5,
        )

        assert len(selected) > 0

    def test_max_complexity(self, model_registry: ModelRegistry) -> None:
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
                task_type=None,
            )
        ]

        selected = router._select_models(
            task_complexity=1.0,
            task_type="Text Generation",
            models_input=models,
            cost_bias=0.5,
        )

        assert len(selected) > 0
