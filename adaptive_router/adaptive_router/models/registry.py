"""Registry models and configuration for the Adaptive model registry service.

This module contains all Pydantic models and configuration types for interacting
with the Adaptive model registry API.
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
    pricing: Dict[str, Any] | None = None
    architecture: Dict[str, Any] | None = None
    top_provider: Dict[str, Any] | None = Field(default=None, alias="top_provider")
    supported_parameters: Any | None = Field(default=None, alias="supported_parameters")
    default_parameters: Dict[str, Any] | None = Field(
        default=None, alias="default_parameters"
    )
    endpoints: Any | None = None
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
            prompt_cost = float(self.pricing.get("prompt_cost", 0))
            completion_cost = float(self.pricing.get("completion_cost", 0))

            if prompt_cost == 0 and completion_cost == 0:
                return None

            return (prompt_cost + completion_cost) / 2.0

        except (ValueError, TypeError):
            return None
