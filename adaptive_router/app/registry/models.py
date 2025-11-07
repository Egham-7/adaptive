"""Model registry backed by the Adaptive registry service.

This module keeps a lightweight in-memory cache of the Pydantic
``RegistryModel`` objects returned by the Adaptive model registry API. The
legacy YAML-backed implementation has been removed in favour of sourcing
canonical data from the registry service.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, List

from app.models import RegistryError, RegistryModel
from app.registry.client import AsyncRegistryClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Simple cache facade around :class:`AsyncRegistryClient`."""

    def __init__(
        self, client: AsyncRegistryClient, models: list[RegistryModel]
    ) -> None:
        self._client = client
        self._models_by_id: dict[str, RegistryModel] = {}
        self._models_by_name: dict[str, list[RegistryModel]] = defaultdict(list)
        self._populate_cache(models)

    def _populate_cache(self, registry_models: Iterable[RegistryModel]) -> None:
        """Populate internal cache from registry models.

        Args:
            registry_models: Iterable of registry models to cache
        """
        models_by_id: dict[str, RegistryModel] = {}
        models_by_name: dict[str, list[RegistryModel]] = defaultdict(list)

        for registry_model in registry_models:
            try:
                unique_id = registry_model.unique_id()
            except (RegistryError, ValueError, AttributeError) as exc:
                logger.warning(
                    "Skipping registry model with invalid identifier",
                    extra={
                        "model": registry_model.model_dump(exclude_none=True),
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                continue

            normalised_id = _normalise(unique_id)
            models_by_id[normalised_id] = registry_model

            model_name = (registry_model.model_name or "").strip()
            if model_name:
                models_by_name[_normalise(model_name)].append(registry_model)

        self._models_by_id = models_by_id
        self._models_by_name = dict(models_by_name)

        logger.info("Cached %d models from registry", len(models_by_id))

    # ------------------------------------------------------------------
    # Public accessors
    def list_models(self) -> List[RegistryModel]:
        """Return all cached models as a list."""
        return list(self._models_by_id.values())

    def get(self, unique_id: str) -> RegistryModel | None:
        """Retrieve a model by its ``provider:model`` identifier."""
        if not unique_id:
            return None
        return self._models_by_id.get(_normalise(unique_id))

    def get_by_name(self, model_name: str) -> list[RegistryModel]:
        """Return all models matching the raw ``model_name`` regardless of provider."""
        if not model_name:
            return []
        return list(self._models_by_name.get(_normalise(model_name), []))

    def providers_for_model(self, model_name: str) -> set[str]:
        """Return providers that expose the given ``model_name``."""
        models = self.get_by_name(model_name)
        return {_normalise(model.provider) for model in models if model.provider}


def _normalise(value: str | None) -> str:
    """Normalize string value to lowercase with whitespace stripped.

    Args:
        value: String value to normalize, or None

    Returns:
        Normalized lowercase string, or empty string if None
    """
    if value is None:
        return ""
    return value.strip().lower()
