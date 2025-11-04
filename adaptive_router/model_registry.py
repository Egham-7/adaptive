"""Model registry integration and cost loading."""

import logging

import httpx

from adaptive_router.models.registry import (
    RegistryClientConfig,
    RegistryConnectionError,
    RegistryError,
    RegistryResponseError,
)
from adaptive_router.registry import RegistryClient
from app_config import AppSettings, DEFAULT_MODEL_COST
from model_fuzzy_matching import enhance_model_costs_with_fuzzy_keys

logger = logging.getLogger(__name__)


def build_registry_client(
    settings: AppSettings,
) -> tuple[RegistryClient, RegistryClientConfig]:
    """Build RegistryClient with configuration and HTTP client.

    Args:
        settings: Application settings containing registry configuration

    Returns:
        Tuple of (RegistryClient, RegistryClientConfig)
    """
    config = RegistryClientConfig(
        base_url=settings.model_registry_base_url,
        timeout=settings.model_registry_timeout,
    )

    # Create httpx.Client with timeout configuration
    http_client = httpx.Client(timeout=settings.model_registry_timeout)

    return RegistryClient(config, http_client), config


def load_model_costs_from_registry(settings: AppSettings) -> dict[str, float]:
    """Load model costs from the Adaptive model registry.

    Args:
        settings: Application settings containing registry configuration

    Returns:
        Dictionary mapping model IDs to their average costs (enhanced with fuzzy variants)

    Raises:
        ValueError: If registry health check fails, no models found, or fetch fails
    """
    client, client_config = build_registry_client(settings)
    logger.info("Loading model costs from registry at %s", client_config.base_url)

    try:
        client.health_check()
    except (RegistryConnectionError, RegistryResponseError) as err:
        raise ValueError(f"Model registry health check failed: {err}") from err

    try:
        models = client.list_models()
    except RegistryError as err:
        raise ValueError(f"Failed to fetch models from registry: {err}") from err

    if not models:
        raise ValueError("Model registry returned no models")

    provider_stats: dict[str, int] = {}
    model_costs: dict[str, float] = {}

    for model in models:
        provider = (model.provider or "").strip().lower()
        if provider:
            provider_stats[provider] = provider_stats.get(provider, 0) + 1

        try:
            model_id = model.unique_id()
        except RegistryError as err:
            logger.warning("Skipping registry model without identifier: %s", err)
            continue

        avg_cost = model.average_price()
        if avg_cost is None:
            logger.warning(
                "Model %s missing pricing data, defaulting cost to %.1f",
                model_id,
                DEFAULT_MODEL_COST,
            )
            avg_cost = DEFAULT_MODEL_COST

        model_costs[model_id] = avg_cost

    logger.info(
        "Loaded costs for %d models across %d providers",
        len(model_costs),
        len(provider_stats),
    )

    # Enhance model_costs with normalized variant keys for fuzzy matching
    enhanced_costs = enhance_model_costs_with_fuzzy_keys(model_costs)

    logger.info(
        "Enhanced model costs with fuzzy matching: %d original models -> %d lookup keys",
        len(model_costs),
        len(enhanced_costs),
    )

    return enhanced_costs
