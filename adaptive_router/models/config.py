"""Configuration models for models and routing parameters.

This module contains configuration models for model metadata, YAML configuration
parsing, and routing algorithm parameters.
"""

from typing import List

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single model.

    Attributes:
        id: Unique model identifier (format: "provider:model_name")
        name: Human-readable model name
        provider: Model provider (e.g., "openai", "anthropic")
        cost_per_1m_input_tokens: Cost per 1M input tokens
        cost_per_1m_output_tokens: Cost per 1M output tokens
        description: Model description
    """

    id: str
    name: str
    provider: str
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float
    description: str

    @property
    def cost_per_1m_tokens(self) -> float:
        """Average cost per million tokens (for backward compatibility)."""
        return (self.cost_per_1m_input_tokens + self.cost_per_1m_output_tokens) / 2.0


class YAMLRoutingConfig(BaseModel):
    """Routing configuration parameters from YAML.

    Attributes:
        lambda_min: Minimum lambda parameter (0.0 = pure quality)
        lambda_max: Maximum lambda parameter (1.0 = balanced cost-quality)
        default_cost_preference: Default cost preference (0.0=cheap, 1.0=quality)
    """

    lambda_min: float = Field(default=0.0, ge=0.0, le=1.0)
    lambda_max: float = Field(default=1.0, ge=0.0, le=1.0)
    default_cost_preference: float = Field(default=0.5, ge=0.0, le=1.0)


class YAMLModelsConfig(BaseModel):
    """Complete YAML configuration structure for models and routing.

    This model validates the structure of the unirouter_models.yaml file.

    Attributes:
        gpt5_models: List of available models for routing
        routing: Routing algorithm configuration
    """

    gpt5_models: List[ModelConfig] = Field(..., description="Available models")
    routing: YAMLRoutingConfig = Field(
        default_factory=YAMLRoutingConfig, description="Routing configuration"
    )
