"""API response models for the adaptive router service."""

from pydantic import BaseModel
from adaptive_router.models.registry import RegistryModel


class ModelSelectionAPIResponse(BaseModel):
    """Enriched model selection response with full registry data.

    Attributes:
        selected_model: Full RegistryModel object for the selected model
        alternatives: List of RegistryModel objects for alternative models
    """

    selected_model: RegistryModel
    alternatives: list[RegistryModel]
