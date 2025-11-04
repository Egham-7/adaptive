"""Model router factory and initialization."""

import logging

from adaptive_router.core.router import ModelRouter
from adaptive_router.models.storage import MinIOSettings
from app_config import AppSettings
from model_registry import load_model_costs_from_registry

logger = logging.getLogger(__name__)


def create_model_router(settings: AppSettings) -> ModelRouter:
    """Create ModelRouter with MinIO and ModelRegistry integration.

    Args:
        settings: Application settings containing MinIO and registry configuration

    Returns:
        Configured ModelRouter instance

    Raises:
        ValueError: If model costs cannot be loaded from registry
    """
    logger.info("Creating ModelRouter...")

    model_costs = load_model_costs_from_registry(settings)

    minio_settings = MinIOSettings(
        endpoint_url=settings.minio_private_endpoint,
        root_user=settings.minio_root_user,
        root_password=settings.minio_root_password,
        bucket_name=settings.s3_bucket_name,
        region=settings.s3_region,
        profile_key=settings.s3_profile_key,
        connect_timeout=settings.s3_connect_timeout,
        read_timeout=settings.s3_read_timeout,
    )

    router = ModelRouter.from_minio(
        settings=minio_settings,
        model_costs=model_costs,
    )

    logger.info("ModelRouter created successfully")

    return router
