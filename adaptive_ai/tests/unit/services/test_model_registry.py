"""Unit tests for ModelRegistry service."""

from unittest.mock import patch

import pytest

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.services.model_registry import model_registry


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_model_registry_singleton(self):
        """Test that model_registry is a singleton instance."""
        from adaptive_ai.services.model_registry import model_registry as registry2

        assert model_registry is registry2
        assert id(model_registry) == id(registry2)

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_get_model_by_key_existing(self, mock_yaml_db):
        """Test getting an existing model by key."""
        # Mock the YAML database response
        mock_yaml_db.get_model.return_value = {
            "provider": "openai",
            "model_name": "gpt-4",
            "cost_per_1m_input_tokens": 30.0,
            "cost_per_1m_output_tokens": 60.0,
            "max_context_tokens": 128000,
            "supports_function_calling": True,
        }

        model = model_registry.get_model_by_key("gpt-4")

        assert isinstance(model, ModelCapability)
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.cost_per_1m_input_tokens == 30.0
        mock_yaml_db.get_model.assert_called_once_with("gpt-4")

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_get_model_by_key_missing(self, mock_yaml_db):
        """Test getting a non-existing model by key."""
        mock_yaml_db.get_model.return_value = None

        model = model_registry.get_model_by_key("non-existent-model")

        assert model is None
        mock_yaml_db.get_model.assert_called_once_with("non-existent-model")

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_get_models_by_provider(self, mock_yaml_db):
        """Test getting all models from a specific provider."""
        mock_yaml_db.get_all_models.return_value = {
            "gpt-4": {
                "provider": "openai",
                "model_name": "gpt-4",
                "cost_per_1m_input_tokens": 30.0,
            },
            "gpt-3.5-turbo": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "cost_per_1m_input_tokens": 1.0,
            },
            "claude-3-sonnet": {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet",
                "cost_per_1m_input_tokens": 15.0,
            },
        }

        openai_models = model_registry.get_models_by_provider("openai")

        assert len(openai_models) == 2
        assert all(isinstance(model, ModelCapability) for model in openai_models)
        assert all(model.provider == "openai" for model in openai_models)

        model_names = [model.model_name for model in openai_models]
        assert "gpt-4" in model_names
        assert "gpt-3.5-turbo" in model_names

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_get_models_by_provider_empty(self, mock_yaml_db):
        """Test getting models for provider with no models."""
        mock_yaml_db.get_all_models.return_value = {
            "claude-3-sonnet": {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet",
            }
        }

        models = model_registry.get_models_by_provider("nonexistent_provider")

        assert len(models) == 0
        assert isinstance(models, list)

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_get_all_models(self, mock_yaml_db):
        """Test getting all available models."""
        mock_yaml_db.get_all_models.return_value = {
            "gpt-4": {
                "provider": "openai",
                "model_name": "gpt-4",
                "cost_per_1m_input_tokens": 30.0,
            },
            "claude-3-sonnet": {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet",
                "cost_per_1m_input_tokens": 15.0,
            },
        }

        all_models = model_registry.get_all_models()

        assert len(all_models) == 2
        assert all(isinstance(model, ModelCapability) for model in all_models)

        providers = [model.provider for model in all_models]
        assert "openai" in providers
        assert "anthropic" in providers

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_model_exists(self, mock_yaml_db):
        """Test checking if a model exists."""
        mock_yaml_db.get_model.side_effect = lambda key: (
            {"provider": "openai", "model_name": "gpt-4"} if key == "gpt-4" else None
        )

        assert model_registry.model_exists("gpt-4") is True
        assert model_registry.model_exists("non-existent") is False

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_get_providers(self, mock_yaml_db):
        """Test getting list of all providers."""
        mock_yaml_db.get_all_models.return_value = {
            "gpt-4": {"provider": "openai", "model_name": "gpt-4"},
            "gpt-3.5": {"provider": "openai", "model_name": "gpt-3.5-turbo"},
            "claude-3": {"provider": "anthropic", "model_name": "claude-3-sonnet"},
            "gemini": {"provider": "google", "model_name": "gemini-pro"},
        }

        providers = model_registry.get_providers()

        assert len(providers) == 3
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert isinstance(providers, list)

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_model_capability_conversion(self, mock_yaml_db):
        """Test that raw model data is correctly converted to ModelCapability."""
        mock_yaml_db.get_model.return_value = {
            "provider": "openai",
            "model_name": "gpt-4",
            "cost_per_1m_input_tokens": 30.0,
            "cost_per_1m_output_tokens": 60.0,
            "max_context_tokens": 128000,
            "supports_function_calling": True,
            "description": "GPT-4 model",
        }

        model = model_registry.get_model_by_key("gpt-4")

        assert isinstance(model, ModelCapability)
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.cost_per_1m_input_tokens == 30.0
        assert model.cost_per_1m_output_tokens == 60.0
        assert model.max_context_tokens == 128000
        assert model.supports_function_calling is True
        assert model.description == "GPT-4 model"

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_filter_models_by_criteria(self, mock_yaml_db):
        """Test filtering models by specific criteria."""
        mock_yaml_db.get_all_models.return_value = {
            "gpt-4": {
                "provider": "openai",
                "model_name": "gpt-4",
                "cost_per_1m_input_tokens": 30.0,
                "supports_function_calling": True,
                "max_context_tokens": 128000,
            },
            "gpt-3.5": {
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "cost_per_1m_input_tokens": 1.0,
                "supports_function_calling": True,
                "max_context_tokens": 16000,
            },
            "claude-3": {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet",
                "cost_per_1m_input_tokens": 15.0,
                "supports_function_calling": False,
                "max_context_tokens": 200000,
            },
        }

        # Test filtering by function calling support
        function_models = model_registry.filter_models(supports_function_calling=True)
        assert len(function_models) == 2
        assert all(model.supports_function_calling for model in function_models)

        # Test filtering by provider
        openai_models = model_registry.filter_models(provider="openai")
        assert len(openai_models) == 2
        assert all(model.provider == "openai" for model in openai_models)

        # Test filtering by max context
        large_context_models = model_registry.filter_models(min_context_tokens=100000)
        large_context_names = [model.model_name for model in large_context_models]
        assert "gpt-4" in large_context_names
        assert "claude-3-sonnet" in large_context_names
        assert "gpt-3.5-turbo" not in large_context_names

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_error_handling_malformed_data(self, mock_yaml_db):
        """Test handling of malformed model data."""
        # Mock malformed data (missing required fields)
        mock_yaml_db.get_model.return_value = {
            "provider": "openai",
            # Missing model_name
            "cost_per_1m_input_tokens": "invalid_number",  # Invalid type
        }

        # Should handle gracefully and not crash
        model = model_registry.get_model_by_key("malformed-model")

        # Depending on implementation, might return None or a partial model
        # The key is that it shouldn't crash
        assert model is None or isinstance(model, ModelCapability)


@pytest.mark.unit
class TestModelRegistryPerformance:
    """Test performance-related aspects of model registry."""

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_caching_behavior(self, mock_yaml_db):
        """Test that model registry delegates to yaml_model_db appropriately."""
        from adaptive_ai.models.llm_core_models import ModelCapability

        mock_capability = ModelCapability(provider="openai", model_name="gpt-4")
        mock_yaml_db.get_model.return_value = mock_capability

        # Call multiple times
        model1 = model_registry.get_model_capability("openai:gpt-4")
        model2 = model_registry.get_model_capability("openai:gpt-4")

        # Should delegate to yaml_model_db
        assert mock_yaml_db.get_model.call_count == 2
        assert model1 == model2
        assert model1 == mock_capability

    @patch("adaptive_ai.services.model_registry.yaml_model_db")
    def test_large_model_set_performance(self, mock_yaml_db):
        """Test performance with model registry operations."""
        from adaptive_ai.models.llm_core_models import ModelCapability

        # Create mock models
        mock_models = [
            ModelCapability(provider=f"provider-{i % 3}", model_name=f"model-{i}")
            for i in range(10)
        ]
        mock_yaml_db.get_all_models.return_value = {
            f"provider-{i % 3}:model-{i}": mock_models[i] for i in range(10)
        }

        # Test find_models_matching_criteria performance
        criteria = ModelCapability(provider="provider-0")
        matching_models = model_registry.find_models_matching_criteria(criteria)

        # Should delegate to yaml_model_db and filter results
        mock_yaml_db.get_all_models.assert_called_once()
        assert isinstance(matching_models, list)

        # Test get_providers_for_model
        mock_yaml_db.get_models_by_name.return_value = mock_models[:3]
        providers = model_registry.get_providers_for_model("test-model")

        assert isinstance(providers, set)
