"""Tests for model_resolver utilities."""

import pytest

from app.models import RegistryModel
from app.utils.model_resolver import _registry_model_to_model, resolve_models


class TestRegistryModelToModel:
    """Test _registry_model_to_model function."""

    def test_converts_valid_pricing(self):
        """Test conversion of model with valid pricing."""
        registry_model = RegistryModel.model_validate({
            "provider": "openai",
            "model_name": "gpt-4",
            "pricing": {"prompt_cost": "0.03", "completion_cost": "0.06"},
        })

        model = _registry_model_to_model(registry_model, default_cost=1.0)

        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.cost_per_1m_input_tokens == 30000.0  # 0.03 * 1_000_000
        assert model.cost_per_1m_output_tokens == 60000.0  # 0.06 * 1_000_000

    def test_uses_default_cost_when_pricing_missing(self, caplog):
        """Test that default cost is used when pricing is missing."""
        registry_model = RegistryModel.model_validate({
            "provider": "openai",
            "model_name": "gpt-4",
            "pricing": None,
        })

        model = _registry_model_to_model(registry_model, default_cost=2.5)

        assert model.cost_per_1m_input_tokens == 2.5
        assert model.cost_per_1m_output_tokens == 2.5

        # Check that warning was logged
        assert "No pricing information" in caplog.text
        assert "$2.5 per 1M tokens" in caplog.text

    def test_uses_default_cost_when_pricing_parsing_fails(self, caplog):
        """Test that default cost is used when pricing parsing fails."""
        registry_model = RegistryModel.model_validate({
            "provider": "openai",
            "model_name": "gpt-4",
            "pricing": {"prompt_cost": "invalid", "completion_cost": "also_invalid"},
        })

        model = _registry_model_to_model(registry_model, default_cost=1.5)

        assert model.cost_per_1m_input_tokens == 1.5
        assert model.cost_per_1m_output_tokens == 1.5

        # Check that warning was logged
        assert "Failed to parse pricing" in caplog.text
        assert "$1.5 per 1M tokens" in caplog.text

    def test_handles_none_pricing_values(self):
        """Test that None pricing values are treated as 0."""
        registry_model = RegistryModel.model_validate({
            "provider": "openai",
            "model_name": "gpt-4",
            "pricing": {"prompt_cost": None, "completion_cost": None},
        })

        model = _registry_model_to_model(registry_model, default_cost=1.0)

        # Should convert None to 0, then to 0 * 1_000_000 = 0
        assert model.cost_per_1m_input_tokens == 0.0
        assert model.cost_per_1m_output_tokens == 0.0


class TestResolveModels:
    """Test resolve_models function."""

    def test_accepts_default_cost_parameter(self):
        """Test that resolve_models accepts default_cost parameter."""
        models = [
            RegistryModel.model_validate({
                "provider": "openai",
                "model_name": "gpt-4",
                "pricing": {"prompt_cost": "0.03", "completion_cost": "0.06"},
            })
        ]

        # Should work with default parameter
        result = resolve_models(["openai:gpt-4"], models)
        assert len(result) == 1

        # Should work with explicit parameter
        result = resolve_models(["openai:gpt-4"], models, default_cost=2.0)
        assert len(result) == 1
        assert result[0].cost_per_1m_input_tokens == 30000.0