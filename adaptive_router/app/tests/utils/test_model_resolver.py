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
                "author": "openai",
                "model_name": "gpt-4",
                "pricing": {"prompt_cost": "0.03", "completion_cost": "0.06"},
            }
        )

        model = _registry_model_to_model(registry_model)

        assert model is not None
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.cost_per_1m_input_tokens == 30000.0  # 0.03 * 1_000_000
        assert model.cost_per_1m_output_tokens == 60000.0  # 0.06 * 1_000_000

    def test_raises_error_when_pricing_missing_explicit(self, caplog):
        """Test that None is returned when pricing is missing."""
        registry_model = RegistryModel.model_validate(
            {
                "author": "openai",
                "model_name": "gpt-4",
                "pricing": None,
            }
        )

        model = _registry_model_to_model(registry_model)

        assert model is None
        assert "no pricing information" in caplog.text.lower()

    def test_returns_none_when_pricing_missing_implicit(self, caplog):
        """Test that None is returned when pricing is missing for implicit models."""
        registry_model = RegistryModel.model_validate(
            {
                "author": "openai",
                "model_name": "gpt-4",
                "pricing": None,
            }
        )

        model = _registry_model_to_model(registry_model)

        assert model is None
        # Check that warning was logged
        assert "no pricing information" in caplog.text.lower()

    def test_raises_error_when_pricing_parsing_fails_explicit(self, caplog):
        """Test that None is returned when pricing parsing fails."""
        registry_model = RegistryModel.model_validate(
            {
                "author": "openai",
                "model_name": "gpt-4",
                "pricing": {
                    "prompt_cost": "invalid",
                    "completion_cost": "also_invalid",
                },
            }
        )

        model = _registry_model_to_model(registry_model)

        assert model is None
        assert "failed to parse pricing" in caplog.text.lower()

    def test_returns_none_when_pricing_parsing_fails_implicit(self, caplog):
        """Test that None is returned when pricing parsing fails for implicit models."""
        registry_model = RegistryModel.model_validate(
            {
                "author": "openai",
                "model_name": "gpt-4",
                "pricing": {
                    "prompt_cost": "invalid",
                    "completion_cost": "also_invalid",
                },
            }
        )

        model = _registry_model_to_model(registry_model)

        assert model is None
        # Check that warning was logged
        assert "failed to parse pricing" in caplog.text.lower()

    def test_handles_none_pricing_values(self):
        """Test that None pricing values are rejected (returns None)."""
        registry_model = RegistryModel.model_validate(
            {
                "author": "openai",
                "model_name": "gpt-4",
                "pricing": {"prompt_cost": None, "completion_cost": None},
            }
        )

        model = _registry_model_to_model(registry_model)

        # None/zero pricing should be rejected
        assert model is None


class TestResolveModels:
    """Test resolve_models function."""

    def test_resolves_valid_models(self):
        """Test that resolve_models works with valid models."""
        models = [
            RegistryModel.model_validate(
                {
                    "author": "openai",
                    "model_name": "gpt-4",
                    "pricing": {"prompt_cost": "0.03", "completion_cost": "0.06"},
                }
            )
        ]

        result = resolve_models(["openai/gpt-4"], models)
        assert len(result) == 1
        assert result[0].cost_per_1m_input_tokens == 30000.0

    def test_raises_error_for_missing_pricing(self):
        """Test that resolve_models raises error for models with missing pricing."""
        models = [
            RegistryModel.model_validate(
                {
                    "author": "openai",
                    "model_name": "gpt-4",
                    "pricing": None,
                }
            )
        ]

        with pytest.raises(ValueError, match="invalid/missing pricing"):
            resolve_models(["openai/gpt-4"], models)

    def test_resolves_models_with_variants(self):
        """Test that resolve_models works with model variants (e.g., :free suffix)."""
        models = [
            RegistryModel.model_validate(
                {
                    "author": "google",
                    "model_name": "gemini-2.0-flash-exp:free",
                    "pricing": {
                        "prompt_cost": "0.000001",
                        "completion_cost": "0.000001",
                    },  # Very small but positive pricing
                }
            ),
            RegistryModel.model_validate(
                {
                    "author": "google",
                    "model_name": "gemini-2.0-flash-001",
                    "pricing": {
                        "prompt_cost": "0.000015",
                        "completion_cost": "0.00012",
                    },
                }
            ),
        ]

        # Test exact match with variant
        result = resolve_models(["google/gemini-2.0-flash-exp:free"], models)
        assert len(result) == 1
        assert result[0].provider == "google"
        assert result[0].model_name == "gemini-2.0-flash-exp:free"

        # Test exact match without variant
        result = resolve_models(["google/gemini-2.0-flash-001"], models)
        assert len(result) == 1
        assert result[0].provider == "google"
        assert result[0].model_name == "gemini-2.0-flash-001"
