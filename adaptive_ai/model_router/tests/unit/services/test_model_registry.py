"""Unit tests for ModelRegistry service."""

import time

from model_router.models.llm_core_models import ModelCapability
from model_router.services.model_registry import ModelRegistry, model_registry


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_model_registry_singleton(self) -> None:
        """Test that model_registry is a singleton instance."""
        from model_router.services.model_registry import model_registry as registry2

        assert model_registry is registry2
        assert id(model_registry) == id(registry2)

    def test_get_model_capability_existing(self) -> None:
        """Test getting an existing model capability."""
        # Test with a known model
        model = model_registry.get_model_capability("openai:gpt-4")

        assert isinstance(model, ModelCapability)
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.cost_per_1m_input_tokens == 30.0
        assert model.cost_per_1m_output_tokens == 60.0
        assert model.max_context_tokens == 8192  # Updated to match actual YAML data
        assert model.supports_function_calling is True

    def test_get_model_capability_missing(self) -> None:
        """Test getting a non-existing model capability."""
        model = model_registry.get_model_capability("nonexistent:fake-model")
        assert model is None

    def test_get_models_by_name(self) -> None:
        """Test getting models by name."""
        # Test with a model that might exist in multiple providers
        models = model_registry.get_models_by_name("gpt-4")

        assert isinstance(models, list)
        assert len(models) >= 1  # At least OpenAI should have it

        for model in models:
            assert isinstance(model, ModelCapability)
            assert model.model_name == "gpt-4"

    def test_get_models_by_name_empty(self) -> None:
        """Test getting models by name with no matches."""
        models = model_registry.get_models_by_name("nonexistent-model")
        assert models == []

    def test_is_valid_model(self) -> None:
        """Test model validation."""
        # Test valid models
        assert model_registry.is_valid_model("openai:gpt-4") is True
        assert (
            model_registry.is_valid_model("anthropic:claude-3-5-sonnet-20241022")
            is True
        )

        # Test invalid models
        assert model_registry.is_valid_model("fake:model") is False
        assert model_registry.is_valid_model("") is False

    def test_is_valid_model_name(self) -> None:
        """Test model name validation."""
        # Test valid model names
        assert model_registry.is_valid_model_name("gpt-4") is True
        assert model_registry.is_valid_model_name("gpt-3.5-turbo") is True

        # Test invalid model names
        assert model_registry.is_valid_model_name("fake-model-xyz") is False
        assert model_registry.is_valid_model_name("") is False

    def test_get_providers_for_model(self) -> None:
        """Test getting providers for a model."""
        providers = model_registry.get_providers_for_model("gpt-4")

        assert isinstance(providers, set)
        assert "openai" in providers

    def test_get_all_model_names(self) -> None:
        """Test getting all model names."""
        model_names = model_registry.get_all_model_names()

        assert isinstance(model_names, list)
        assert len(model_names) > 0

        # Check some expected models are present
        assert "openai:gpt-4" in model_names
        assert "anthropic:claude-3-5-sonnet-20241022" in model_names

    def test_find_models_matching_criteria(self) -> None:
        """Test finding models by criteria."""
        # Test finding OpenAI models
        criteria = ModelCapability(provider="openai")
        models = model_registry.find_models_matching_criteria(criteria)

        assert len(models) >= 2  # At least GPT-4 and GPT-3.5
        for model in models:
            assert model.provider == "openai"

        # Test finding models by cost threshold
        criteria = ModelCapability(
            cost_per_1m_input_tokens=5.0  # Models cheaper than this
        )
        models = model_registry.find_models_matching_criteria(criteria)

        assert len(models) >= 1
        for model in models:
            assert (
                model.cost_per_1m_input_tokens is not None
                and model.cost_per_1m_input_tokens <= 5.0
            )

    def test_find_models_with_function_calling(self) -> None:
        """Test finding models that support function calling."""
        criteria = ModelCapability(supports_function_calling=True)
        models = model_registry.find_models_matching_criteria(criteria)

        assert len(models) >= 1
        for model in models:
            assert model.supports_function_calling is True

    def test_find_models_by_complexity(self) -> None:
        """Test finding models by complexity."""
        # Test finding high complexity models
        criteria = ModelCapability(complexity="high")
        models = model_registry.find_models_matching_criteria(criteria)

        assert len(models) >= 1
        for model in models:
            assert model.complexity == "high"


class TestModelRegistryPerformance:
    """Test performance characteristics of ModelRegistry."""

    def test_lookup_performance(self) -> None:
        """Test that model lookups are fast."""
        start = time.time()

        # Perform multiple lookups
        for _ in range(100):
            model_registry.get_model_capability("openai:gpt-4")
            model_registry.is_valid_model("anthropic:claude-3-haiku-20240307")
            model_registry.get_models_by_name("gpt-3.5-turbo")

        elapsed = time.time() - start
        # Should be very fast since it's in-memory
        assert elapsed < 0.1  # 100ms for 300 operations

    def test_registry_initialization(self) -> None:
        """Test that registry initializes with expected models."""
        registry = ModelRegistry()

        # Check that basic models are loaded
        model_names = registry.get_all_model_names()

        # Should have at least the basic models defined in _load_basic_models
        expected_models = [
            "openai:gpt-4",
            "openai:gpt-3.5-turbo",
            "anthropic:claude-3-5-sonnet-20241022",
            "anthropic:claude-3-haiku-20240307",
        ]

        for expected in expected_models:
            assert expected in model_names
