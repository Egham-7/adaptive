"""Type definitions for the adaptive router application.

This module contains all Pydantic models and type definitions used by the
application layer. No business logic should be in this file - only type definitions.

These types match the Go structs in adaptive-model-registry/internal/models/model.go
"""

from pydantic import BaseModel
from adaptive_router.models.registry import RegistryModel


# ============================================================================
# Adaptive Registry Types (matching Go structs)
# ============================================================================


class PricingModel(BaseModel):
    """Pricing structure for model usage (matches Go Pricing struct)."""

    prompt: str | None = None  # Cost per token for input
    completion: str | None = None  # Cost per token for output
    request: str | None = None  # Cost per request (optional)
    image: str | None = None  # Cost per image (optional)
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
