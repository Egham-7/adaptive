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
from app.registry.client import RegistryClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Simple cache facade around :class:`RegistryClient`."""

    def __init__(self, client: RegistryClient, *, auto_refresh: bool = True) -> None:
        self._client = client
        self._models_by_id: dict[str, RegistryModel] = {}
        self._models_by_name: dict[str, list[RegistryModel]] = {}

        if auto_refresh:
            self.refresh()

    # ------------------------------------------------------------------
    # Loading & caching helpers
    def refresh(self) -> None:
        """Fetch the latest models from the registry service and cache them."""

        try:
            registry_models = self._client.list_models()
        except RegistryError as exc:
            logger.error(
                "Failed to load models from registry",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            raise

        self._populate_cache(registry_models)

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

        return {
            _normalise(model.provider)
            for model in self.get_by_name(model_name)
            if model.provider
        }

    def is_valid_model(self, unique_id: str) -> bool:
        return self.get(unique_id) is not None

    def is_valid_model_name(self, model_name: str) -> bool:
        return bool(self.get_by_name(model_name))

    def get_all_model_ids(self) -> list[str]:
        return list(self._models_by_id.keys())

    def get_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for model in self._models_by_id.values():
            provider = _normalise(model.provider)
            if not provider:
                continue
            stats[provider] = stats.get(provider, 0) + 1
        return stats

    # ------------------------------------------------------------------
    # Filtering helpers
    def filter(
        self,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        min_context: int | None = None,
        requires_function_calling: bool | None = None,
    ) -> list[RegistryModel]:
        """Filter cached models by common registry attributes.

        This is a very small helper intended for internal usage. Consumers that
        require more advanced querying should call the registry service
        directly.
        """

        filtered: list[RegistryModel] = []

        for model in self._models_by_id.values():
            if provider and _normalise(model.provider) != _normalise(provider):
                continue

            if model_name and _normalise(model.model_name) != _normalise(model_name):
                continue

            if min_context is not None and (
                model.context_length is None or model.context_length < min_context
            ):
                continue

            if requires_function_calling is True and not _supports_function_calling(
                model
            ):
                continue
            if requires_function_calling is False and _supports_function_calling(model):
                continue

            filtered.append(model)

        return filtered


def _supports_function_calling(model: RegistryModel) -> bool:
    """Best-effort detection based on ``supported_parameters``."""

    params = model.supported_parameters
    if params is None:
        return False

    values: Iterable[str]
    if isinstance(params, dict):
        values = [str(item) for pair in params.items() for item in pair]
    elif isinstance(params, (list, tuple, set)):
        # Handle list of SupportedParameterModel objects
        values = []
        for item in params:
            # If it's a SupportedParameterModel, extract parameter_name
            if hasattr(item, "parameter_name"):
                values.append(item.parameter_name)
            else:
                # Fallback to string conversion for backward compatibility
                values.append(str(item))
    else:
        values = [str(params)]

    supported_flags = {"tools", "functions", "function_calling", "tool_choice"}
    return any(_normalise(value) in supported_flags for value in values)


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
