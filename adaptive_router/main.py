"""FastAPI application for adaptive_router service.

Provides HTTP API endpoints for intelligent model selection using cluster-based routing.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, AsyncIterator
import httpx

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from adaptive_router.core.router import ModelRouter
from adaptive_router.models.api import Model, ModelSelectionRequest

from app.models import (
    ModelSelectionAPIRequest,
)
from app.models import (
    RegistryConnectionError,
    RegistryError,
    RegistryResponseError,
    RegistryClientConfig,
)
from app.registry import ModelRegistry
from app.config import AppSettings
from app.health import HealthCheckResponse, HealthStatus, ServiceHealth
from app.models import ModelSelectionAPIResponse

from app.registry.client import AsyncRegistryClient
from app.utils import (
    resolve_models,
)
from app.utils.model_resolver import _registry_model_to_model


env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def extract_model_ids_from_profile(profile) -> list[str]:
    """Extract model IDs from a RouterProfile.

    Args:
        profile: RouterProfile object with llm_profiles

    Returns:
        List of model IDs (e.g., ["openai/gpt-4", "anthropic/claude-3"])
    """
    return list(profile.llm_profiles.keys())


async def load_models_for_profile_async(
    settings: AppSettings, model_ids: list[str]
) -> list[Model]:
    """Load specific models from registry asynchronously with exact matching.

    Args:
        settings: Application settings containing registry configuration
        model_ids: List of model IDs to fetch (format: "provider/model_name")

    Returns:
        List of Model objects with cost information

    Raises:
        ValueError: If registry health check fails or fetch fails
    """
    config = RegistryClientConfig(
        base_url=settings.model_registry_base_url,
        timeout=settings.model_registry_timeout,
    )
    async with httpx.AsyncClient(
        timeout=settings.model_registry_timeout
    ) as http_client:
        client = AsyncRegistryClient(config, http_client)
        logger.info(
            "Loading %d models from registry at %s", len(model_ids), config.base_url
        )

        try:
            await client.health_check()
        except (RegistryConnectionError, RegistryResponseError) as err:
            raise ValueError(f"Model registry health check failed: {err}") from err

        # Load only the specific models requested
        router_models = []
        for model_id in model_ids:
            try:
                author, model_name = model_id.split("/", 1)
                registry_model = await client.get_by_author_and_name(author, model_name)
                if registry_model is None:
                    logger.warning("Model %s not found in registry", model_id)
                    continue

                model = _registry_model_to_model(registry_model)
                if model is not None:
                    router_models.append(model)
                else:
                    logger.warning("Model %s has invalid/missing pricing", model_id)

            except (ValueError, RegistryError) as err:
                logger.warning("Failed to load model %s: %s", model_id, err)
                continue

        logger.info(
            "Loaded %d/%d models from registry",
            len(router_models),
            len(model_ids),
        )
        return router_models


# Inlined from model_router_factory.py
async def create_model_router(settings: AppSettings) -> ModelRouter:
    """Create ModelRouter with MinIO and ModelRegistry integration.

    Args:
        settings: Application settings containing MinIO and registry configuration

    Returns:
        Configured ModelRouter instance

    Raises:
        ValueError: If model costs cannot be loaded from registry
    """
    logger.info("Creating ModelRouter...")

    from adaptive_router.models.storage import MinIOSettings

    minio_settings = MinIOSettings(
        endpoint_url=settings.minio_private_endpoint,
        root_user=settings.minio_root_user,
        root_password=settings.minio_root_password,
        bucket_name=settings.s3_bucket_name,
        region=settings.s3_region,
        profile_key=settings.s3_profile_key,
        connect_timeout=int(settings.s3_connect_timeout),
        read_timeout=int(settings.s3_read_timeout),
    )

    # Load profile first to determine which models we need
    from adaptive_router.loaders.minio import MinIOProfileLoader

    loader = MinIOProfileLoader.from_settings(minio_settings)
    profile = loader.load_profile()

    # Extract model IDs from the profile
    model_ids = extract_model_ids_from_profile(profile)
    if not model_ids:
        raise ValueError("Profile contains no model IDs")

    # Load only the models referenced in the profile
    await load_models_for_profile_async(settings, model_ids)

    # Create router using the from_profile method
    router = ModelRouter.from_profile(profile=profile)

    logger.info("ModelRouter created successfully")

    return router


class AppState:
    """Application state container."""

    def __init__(self) -> None:
        """Initialize application state."""
        self.settings: AppSettings | None = None
        self.router: ModelRouter | None = None
        self.registry: ModelRegistry | None = None
        self.registry_client: AsyncRegistryClient | None = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application with dependency injection."""
    app_state = AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Lifespan event handler for startup/shutdown."""
        try:
            # Initialize settings
            app_state.settings = AppSettings()

            logger.info(
                "Starting Adaptive Router on http://%s:%d",
                app_state.settings.host,
                app_state.settings.port,
            )
            logger.info(
                "API Docs: http://%s:%d/docs",
                app_state.settings.host,
                app_state.settings.port,
            )

            # Load profile first to determine which models we need
            logger.info("Loading router profile...")
            from adaptive_router.models.storage import MinIOSettings

            minio_settings = MinIOSettings(
                endpoint_url=app_state.settings.minio_private_endpoint,
                root_user=app_state.settings.minio_root_user,
                root_password=app_state.settings.minio_root_password,
                bucket_name=app_state.settings.s3_bucket_name,
                region=app_state.settings.s3_region,
                profile_key=app_state.settings.s3_profile_key,
                connect_timeout=int(app_state.settings.s3_connect_timeout),
                read_timeout=int(app_state.settings.s3_read_timeout),
            )
            from adaptive_router.loaders.minio import MinIOProfileLoader

            loader = MinIOProfileLoader.from_settings(minio_settings)
            profile = loader.load_profile()
            profile_model_ids = extract_model_ids_from_profile(profile)
            logger.info("Loaded profile with %d model IDs", len(profile_model_ids))

            # Initialize ModelRouter with profile models
            logger.info("Initializing ModelRouter...")
            app_state.router = await create_model_router(app_state.settings)
            logger.info("ModelRouter initialized successfully")

            # Initialize ModelResolverService with only profile models
            logger.info("Initializing ModelResolverService...")
            config = RegistryClientConfig(
                base_url=app_state.settings.model_registry_base_url,
                timeout=app_state.settings.model_registry_timeout,
            )
            registry_client = AsyncRegistryClient(
                config,
                httpx.AsyncClient(timeout=app_state.settings.model_registry_timeout),
            )
            app_state.registry_client = registry_client
            models = await registry_client.list_models()
            app_state.registry = ModelRegistry(registry_client, models)
            logger.info("ModelResolverService initialized successfully")

            logger.info("FastAPI application started successfully")
        except Exception as e:
            logger.error("Failed to initialize router: %s", e, exc_info=True)
            raise

        yield

        # Cleanup on shutdown
        logger.info("Shutting down Adaptive Router...")
        if app_state.registry_client is not None:
            await app_state.registry_client._client.aclose()

    app = FastAPI(
        title="Adaptive Router",
        description="Intelligent LLM model selection API with cluster-based routing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    def configure_cors() -> None:
        """Configure CORS middleware."""
        settings = app_state.settings or AppSettings()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.origins_list,
            allow_credentials=settings.origins_list != ["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    configure_cors()

    # Dependencies
    @lru_cache()
    def get_settings() -> AppSettings:
        """Get application settings dependency."""
        if app_state.settings is None:
            app_state.settings = AppSettings()
        return app_state.settings

    async def get_registry_client() -> AsyncRegistryClient:
        """Get AsyncRegistryClient dependency."""
        if app_state.registry_client is None:
            settings = get_settings()
            config = RegistryClientConfig(
                base_url=settings.model_registry_base_url,
                timeout=settings.model_registry_timeout,
            )
            http_client = httpx.AsyncClient(timeout=settings.model_registry_timeout)
            app_state.registry_client = AsyncRegistryClient(config, http_client)
        return app_state.registry_client

    def get_registry() -> ModelRegistry:
        """Get ModelRegistry dependency."""
        if app_state.registry is None:
            raise RuntimeError("Registry not initialized")
        return app_state.registry

    async def get_router() -> ModelRouter:
        """Get ModelRouter dependency."""
        if app_state.router is None:
            settings = get_settings()
            logger.info("Lazy-initializing ModelRouter...")
            app_state.router = await create_model_router(settings)
            logger.info("ModelRouter initialized successfully")
        return app_state.router

    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        status_code=status.HTTP_200_OK,
        responses={
            200: {"description": "All services healthy"},
            503: {"description": "One or more services unhealthy"},
        },
    )
    async def health_check(
        settings: Annotated[AppSettings, Depends(get_settings)],
        registry_client: Annotated[AsyncRegistryClient, Depends(get_registry_client)],
    ) -> HealthCheckResponse:
        """Comprehensive health check including registry connectivity.

        Returns health status of:
        - Model registry connectivity
        - Router initialization status
        """
        overall_status = HealthStatus.HEALTHY

        # Check registry health
        registry_start = time.perf_counter()
        try:
            await registry_client.health_check()
            registry_health = ServiceHealth(
                status=HealthStatus.HEALTHY,
                message="Registry is accessible",
                response_time_ms=round(
                    (time.perf_counter() - registry_start) * 1000, 2
                ),
            )
        except (RegistryConnectionError, RegistryResponseError) as e:
            overall_status = HealthStatus.UNHEALTHY
            registry_health = ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Registry connection failed: {e}",
                response_time_ms=round(
                    (time.perf_counter() - registry_start) * 1000, 2
                ),
            )
        except Exception as e:
            overall_status = HealthStatus.DEGRADED
            registry_health = ServiceHealth(
                status=HealthStatus.DEGRADED,
                message=f"Registry check error: {e}",
                response_time_ms=round(
                    (time.perf_counter() - registry_start) * 1000, 2
                ),
            )

        # Check router status
        if app_state.router is not None:
            router_health = ServiceHealth(
                status=HealthStatus.HEALTHY,
                message="Router initialized",
            )
        else:
            overall_status = (
                HealthStatus.DEGRADED
                if overall_status == HealthStatus.HEALTHY
                else overall_status
            )
            router_health = ServiceHealth(
                status=HealthStatus.DEGRADED,
                message="Router not yet initialized (will initialize on first request)",
            )

        return HealthCheckResponse(
            status=overall_status,
            registry=registry_health,
            router=router_health,
        )

    # Model selection endpoint
    @app.post(
        "/select-model",
        response_model=ModelSelectionAPIResponse,
        status_code=status.HTTP_200_OK,
        responses={
            400: {"description": "Invalid request"},
            500: {"description": "Internal server error"},
        },
    )
    async def select_model(
        request: ModelSelectionAPIRequest,
        http_request: Request,
        router: Annotated[ModelRouter, Depends(get_router)],
        registry: Annotated[ModelRegistry, Depends(get_registry)],
        registry_client: Annotated[AsyncRegistryClient, Depends(get_registry_client)],
        settings: Annotated[AppSettings, Depends(get_settings)],
    ) -> ModelSelectionAPIResponse:
        """Select optimal model based on prompt analysis.

        Args:
            request: Model selection request with prompt and preferences
            http_request: FastAPI request object for logging
            router: Injected ModelRouter dependency

        Returns:
            ModelSelectionResponse with selected model and alternatives

        Raises:
            HTTPException: 400 for validation errors, 500 for server errors
        """
        start_time = time.perf_counter()

        try:
            # Resolve model specifications to RegistryModel objects
            resolved_models = None
            all_models = registry.list_models()
            if request.models:
                try:
                    resolved_models = resolve_models(request.models, all_models)
                except ValueError as e:
                    logger.error("Model resolution failed: %s", e, exc_info=True)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model resolution failed: {e}",
                    ) from e

            # Create internal request
            internal_request = ModelSelectionRequest(
                prompt=request.prompt,
                user_id=request.user_id,
                models=resolved_models,
                cost_bias=request.cost_bias,
            )

            logger.info(
                "Processing model selection request",
                extra={
                    "prompt_length": len(internal_request.prompt),
                    "cost_bias": internal_request.cost_bias,
                    "models_count": (
                        len(internal_request.models) if internal_request.models else 0
                    ),
                    "client_ip": (
                        http_request.client.host if http_request.client else "unknown"
                    ),
                },
            )

            response = router.select_model(internal_request)
            elapsed = time.perf_counter() - start_time

            logger.info(
                "Model selection completed",
                extra={
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "selected_model_id": response.model_id,
                    "alternatives_count": len(response.alternatives),
                },
            )

            # Return simplified response with model IDs only
            selected_model_id = response.model_id
            alternative_model_ids = [alt.model_id for alt in response.alternatives]

            return ModelSelectionAPIResponse(
                selected_model=selected_model_id,
                alternatives=alternative_model_ids,
            )

        except ValueError as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Validation error: %s",
                e,
                extra={"elapsed_ms": round(elapsed * 1000, 2)},
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request: validation failed",
            ) from e

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Model selection failed: %s",
                e,
                extra={"elapsed_ms": round(elapsed * 1000, 2)},
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            ) from e

    return app


app = create_app()
