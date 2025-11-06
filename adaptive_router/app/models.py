"""Type definitions for the adaptive router application.

This module contains all Pydantic models and type definitions used by the
application layer. No business logic should be in this file - only type definitions.

These types match the Go structs in adaptive-model-registry/internal/models/model.go
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


# Exception classes
class RegistryError(Exception):
    """Base error type for registry client failures."""


class RegistryConnectionError(RegistryError):
    """Raised when the registry cannot be reached."""


class RegistryResponseError(RegistryError):
    """Raised when the registry returns an unexpected response."""


# Configuration model
class RegistryClientConfig(BaseModel):
    """Configuration for the registry client.

    Attributes:
        base_url: Base URL of the registry service
        timeout: Request timeout in seconds
        default_headers: Optional default headers for requests
    """

    model_config = ConfigDict(frozen=True)

    base_url: str
    timeout: float = 5.0
    default_headers: Dict[str, str] | None = None

    def normalized_headers(self) -> Dict[str, str]:
        """Return normalized headers dictionary."""
        return dict(self.default_headers or {})


# Registry model
class RegistryModel(BaseModel):
    """Pydantic representation of the normalized Go registry Model struct.

    Attributes:
        id: Database ID
        provider: Model provider (e.g., "openai", "anthropic")
        model_name: Model name
        display_name: Human-readable display name
        description: Model description
        context_length: Maximum context window size
        pricing: Normalized pricing information (ModelPricing entity)
        architecture: Normalized architecture details (ModelArchitecture entity)
        top_provider: Top provider information (ModelTopProvider entity)
        supported_parameters: List of ModelSupportedParameter entities
        default_parameters: ModelDefaultParameters entity with JSONB parameters
        endpoints: List of ModelEndpoint entities with nested pricing
        created_at: Creation timestamp
        last_updated: Last update timestamp
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: int | None = None
    provider: str
    model_name: str
    display_name: str | None = Field(default=None, alias="display_name")
    description: str | None = None
    context_length: int | None = Field(default=None, alias="context_length")
    pricing: PricingModel | None = None
    architecture: ArchitectureModel | None = None
    top_provider: TopProviderModel | None = Field(default=None, alias="top_provider")
    supported_parameters: list[str] | None = Field(
        default=None, alias="supported_parameters"
    )
    default_parameters: Dict[str, Any] | None = Field(
        default=None, alias="default_parameters"
    )
    endpoints: list[EndpointModel] | None = None
    created_at: datetime | None = Field(default=None, alias="created_at")
    last_updated: datetime | None = Field(default=None, alias="last_updated")

    def unique_id(self) -> str:
        """Construct the router-compatible unique identifier.

        Returns:
            Unique identifier in format "provider:model"

        Raises:
            RegistryError: If provider or model_name is missing
        """
        provider = (self.provider or "").strip().lower()
        if not provider:
            raise RegistryError("registry model missing provider field")

        if not self.model_name:
            raise RegistryError(f"registry model '{provider}' missing model_name")

        model_name = self.model_name.strip().lower()

        return f"{provider}:{model_name}"

    def average_price(self) -> float | None:
        """Calculate average price from available pricing fields.

        Pricing format from normalized registry (ModelPricing entity):
        {
            "prompt_cost": "0.000015",
            "completion_cost": "0.00012",
            "request_cost": "0",
            "image_cost": "0",
            ...
        }

        Returns:
            Average of prompt and completion costs, or None if pricing unavailable
        """
        if not self.pricing:
            return None

        try:
            # Updated field names for normalized schema
            prompt_cost = float(self.pricing.prompt_cost or 0)
            completion_cost = float(self.pricing.completion_cost or 0)

            if prompt_cost == 0 and completion_cost == 0:
                return None

            return (prompt_cost + completion_cost) / 2.0

        except (ValueError, TypeError):
            return None


# ============================================================================
# Adaptive Registry Types (matching Go structs)
# ============================================================================


class PricingModel(BaseModel):
    """Pricing structure for model usage (matches Go Pricing struct)."""

    prompt_cost: str | None = None  # Cost per token for input
    completion_cost: str | None = None  # Cost per token for output
    request_cost: str | None = None  # Cost per request (optional)
    image_cost: str | None = None  # Cost per image (optional)
    image_output: str | None = None  # Cost per output image (optional)
    web_search: str | None = None  # Cost for web search (optional)
    internal_reasoning: str | None = None  # Cost for reasoning (optional)
    discount: float = 0.0  # Discount percentage


class ArchitectureModel(BaseModel):
    """Model architecture and capabilities (matches Go Architecture struct)."""

    modality: str | None = None  # e.g., "text+image->text"
    input_modalities: list[str] | None = None  # e.g., ["text", "image"]
    output_modalities: list[str] | None = None  # e.g., ["text"]
    tokenizer: str | None = None  # e.g., "GPT", "Llama3", "Nova"
    instruct_type: str | None = None  # e.g., "chatml"


class TopProviderModel(BaseModel):
    """Top provider configuration (matches Go TopProvider struct)."""

    context_length: int | None = None
    max_completion_tokens: int | None = None
    is_moderated: bool = False


class EndpointModel(BaseModel):
    """Provider endpoint configuration (matches Go Endpoint struct)."""

    name: str
    model_name: str
    context_length: int
    pricing: PricingModel
    provider_name: str
    tag: str
    quantization: str | None = None
    max_completion_tokens: int | None = None
    max_prompt_tokens: int | None = None
    supported_parameters: list[str] | None = None
    uptime_last_30m: float | None = None
    supports_implicit_caching: bool = False


# ============================================================================
# API Response Types
# ============================================================================


class ModelSelectionAPIResponse(BaseModel):
    """Enriched model selection response with full registry data.

    Attributes:
        selected_model: Full RegistryModel object for the selected model
        alternatives: List of RegistryModel objects for alternative models
    """

    selected_model: RegistryModel
    alternatives: list[RegistryModel]
