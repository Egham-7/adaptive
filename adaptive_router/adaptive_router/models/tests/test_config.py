"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from adaptive_router.models.config import (
    ModelConfig,
    YAMLRoutingConfig,
    YAMLModelsConfig,
)


class TestModelConfig:
    """Test ModelConfig model."""

    def test_valid_model_config(self) -> None:
        """Test creating valid ModelConfig."""
        config = ModelConfig(
            id="openai:gpt-4",
            name="GPT-4",
            provider="openai",
            cost_per_1m_tokens=30.0,
            description="Advanced language model",
        )
        assert config.id == "openai:gpt-4"
        assert config.provider == "openai"
        assert config.cost_per_1m_tokens == 30.0

    def test_model_config_missing_fields(self) -> None:
        """Test ModelConfig requires all fields."""
        with pytest.raises(ValidationError):
            ModelConfig(
                id="openai:gpt-4",
                name="GPT-4",
                # Missing provider
                cost_per_1m_tokens=30.0,
            )

    def test_model_config_zero_cost(self) -> None:
        """Test ModelConfig allows zero cost."""
        config = ModelConfig(
            id="local:llama",
            name="Llama",
            provider="local",
            cost_per_1m_tokens=0.0,
            description="Free local model",
        )
        assert config.cost_per_1m_tokens == 0.0

    def test_model_config_negative_cost_allowed(self) -> None:
        """Test ModelConfig allows negative cost (no validation constraint)."""
        # Note: Pydantic doesn't validate cost is positive by default
        config = ModelConfig(
            id="test:model",
            name="Test",
            provider="test",
            cost_per_1m_tokens=-1.0,
            description="Test model",
        )
        assert config.cost_per_1m_tokens == -1.0


class TestYAMLRoutingConfig:
    """Test YAMLRoutingConfig model."""

    def test_default_values(self) -> None:
        """Test YAMLRoutingConfig has correct defaults."""
        config = YAMLRoutingConfig()
        assert config.lambda_min == 0.0
        assert config.lambda_max == 1.0
        assert config.default_cost_preference == 0.5

    def test_custom_values(self) -> None:
        """Test YAMLRoutingConfig with custom values."""
        config = YAMLRoutingConfig(
            lambda_min=0.1,
            lambda_max=0.9,
            default_cost_preference=0.3,
        )
        assert config.lambda_min == 0.1
        assert config.lambda_max == 0.9
        assert config.default_cost_preference == 0.3

    def test_lambda_min_boundary_validation(self) -> None:
        """Test lambda_min must be between 0 and 1."""
        # Valid boundaries
        YAMLRoutingConfig(lambda_min=0.0)
        YAMLRoutingConfig(lambda_min=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            YAMLRoutingConfig(lambda_min=-0.1)

        with pytest.raises(ValidationError):
            YAMLRoutingConfig(lambda_min=1.1)

    def test_lambda_max_boundary_validation(self) -> None:
        """Test lambda_max must be between 0 and 1."""
        # Valid boundaries
        YAMLRoutingConfig(lambda_max=0.0)
        YAMLRoutingConfig(lambda_max=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            YAMLRoutingConfig(lambda_max=-0.1)

        with pytest.raises(ValidationError):
            YAMLRoutingConfig(lambda_max=1.1)

    def test_cost_preference_boundary_validation(self) -> None:
        """Test default_cost_preference must be between 0 and 1."""
        # Valid boundaries
        YAMLRoutingConfig(default_cost_preference=0.0)
        YAMLRoutingConfig(default_cost_preference=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            YAMLRoutingConfig(default_cost_preference=-0.1)

        with pytest.raises(ValidationError):
            YAMLRoutingConfig(default_cost_preference=1.1)


class TestYAMLModelsConfig:
    """Test YAMLModelsConfig model."""

    def test_valid_models_config(self) -> None:
        """Test creating valid YAMLModelsConfig."""
        config = YAMLModelsConfig(
            gpt5_models=[
                ModelConfig(
                    id="openai:gpt-4",
                    name="GPT-4",
                    provider="openai",
                    cost_per_1m_tokens=30.0,
                    description="GPT-4",
                ),
            ],
            routing=YAMLRoutingConfig(),
        )
        assert len(config.gpt5_models) == 1
        assert config.routing.default_cost_preference == 0.5

    def test_empty_models_list(self) -> None:
        """Test YAMLModelsConfig allows empty models list."""
        config = YAMLModelsConfig(
            gpt5_models=[],
            routing=YAMLRoutingConfig(),
        )
        assert len(config.gpt5_models) == 0

    def test_default_routing_config(self) -> None:
        """Test YAMLModelsConfig uses default routing if not provided."""
        config = YAMLModelsConfig(gpt5_models=[])
        assert config.routing.lambda_min == 0.0
        assert config.routing.lambda_max == 1.0

    def test_multiple_models(self) -> None:
        """Test YAMLModelsConfig with multiple models."""
        config = YAMLModelsConfig(
            gpt5_models=[
                ModelConfig(
                    id="openai:gpt-4",
                    name="GPT-4",
                    provider="openai",
                    cost_per_1m_tokens=30.0,
                    description="GPT-4",
                ),
                ModelConfig(
                    id="anthropic:claude-3-sonnet",
                    name="Claude 3 Sonnet",
                    provider="anthropic",
                    cost_per_1m_tokens=15.0,
                    description="Claude 3 Sonnet",
                ),
            ],
        )
        assert len(config.gpt5_models) == 2
        assert config.gpt5_models[0].provider == "openai"
        assert config.gpt5_models[1].provider == "anthropic"
