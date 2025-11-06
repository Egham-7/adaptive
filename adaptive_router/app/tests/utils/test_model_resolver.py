"""Tests for model_resolver utilities."""

import pytest

from app.models import RegistryModel
from app.utils.model_resolver import _registry_model_to_model, resolve_models


class TestRegistryModelToModel:
    """Test _registry_model_to_model function."""

    def test_converts_valid_pricing(self):
        """Test conversion of model with valid pricing."""
        registry_model = RegistryModel.model_validate(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "pricing": {"prompt_cost": "0.03", "completion_cost": "0.06"},
            }
        )

        model = _registry_model_to_model(registry_model, raise_on_error=True)

        assert model is not None
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.cost_per_1m_input_tokens == 30000.0  # 0.03 * 1_000_000
        assert model.cost_per_1m_output_tokens == 60000.0  # 0.06 * 1_000_000

    def test_raises_error_when_pricing_missing_explicit(self):
        """Test that error is raised when pricing is missing for explicit models."""
        registry_model = RegistryModel.model_validate(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "pricing": None,
            }
        )

        with pytest.raises(ValueError, match="No pricing information"):
            _registry_model_to_model(registry_model, raise_on_error=True)

    def test_returns_none_when_pricing_missing_implicit(self, caplog):
        """Test that None is returned when pricing is missing for implicit models."""
        registry_model = RegistryModel.model_validate(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "pricing": None,
            }
        )

        model = _registry_model_to_model(registry_model, raise_on_error=False)

        assert model is None
        # Check that warning was logged
        assert "No pricing information" in caplog.text

    def test_raises_error_when_pricing_parsing_fails_explicit(self):
        """Test that error is raised when pricing parsing fails for explicit models."""
        registry_model = RegistryModel.model_validate(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "pricing": {
                    "prompt_cost": "invalid",
                    "completion_cost": "also_invalid",
                },
            }
        )

        with pytest.raises(ValueError, match="Failed to parse pricing"):
            _registry_model_to_model(registry_model, raise_on_error=True)

    def test_returns_none_when_pricing_parsing_fails_implicit(self, caplog):
        """Test that None is returned when pricing parsing fails for implicit models."""
        registry_model = RegistryModel.model_validate(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "pricing": {
                    "prompt_cost": "invalid",
                    "completion_cost": "also_invalid",
                },
            }
        )

        model = _registry_model_to_model(registry_model, raise_on_error=False)

        assert model is None
        # Check that warning was logged
        assert "Failed to parse pricing" in caplog.text

    def test_handles_none_pricing_values(self):
        """Test that None pricing values are treated as 0."""
        registry_model = RegistryModel.model_validate(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "pricing": {"prompt_cost": None, "completion_cost": None},
            }
        )

        model = _registry_model_to_model(registry_model, raise_on_error=True)

        assert model is not None
        # Should convert None to 0, then to 0 * 1_000_000 = 0
        assert model.cost_per_1m_input_tokens == 0.0
        assert model.cost_per_1m_output_tokens == 0.0


class TestResolveModels:
    """Test resolve_models function."""

    def test_resolves_valid_models(self):
        """Test that resolve_models works with valid models."""
        models = [
            RegistryModel.model_validate(
                {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "pricing": {"prompt_cost": "0.03", "completion_cost": "0.06"},
                }
            )
        ]

        result = resolve_models(["openai:gpt-4"], models)
        assert len(result) == 1
        assert result[0].cost_per_1m_input_tokens == 30000.0

    def test_raises_error_for_missing_pricing(self):
        """Test that resolve_models raises error for models with missing pricing."""
        models = [
            RegistryModel.model_validate(
                {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "pricing": None,
                }
            )
        ]

        with pytest.raises(ValueError, match="No pricing information"):
            resolve_models(["openai:gpt-4"], models)
