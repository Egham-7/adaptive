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

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from adaptive_router.core.router import ModelRouter
from adaptive_router.models.api import Model, ModelSelectionRequest
from adaptive_router.models.storage import RouterProfile

from app.models import ModelSelectionAPIRequest
from app.config import AppSettings
from app.health import HealthCheckResponse, HealthStatus, ServiceHealth
from app.models import ModelSelectionAPIResponse

from app.utils import resolve_models


env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def extract_model_ids_from_profile(profile: RouterProfile) -> list[str]:
    """Extract model IDs from a RouterProfile."""
    return list(profile.llm_profiles.keys())


async def create_model_router(settings: AppSettings) -> tuple[ModelRouter, RouterProfile]:
    """Create ModelRouter and return the loaded profile."""
    logger.info("Creating ModelRouter...")

    from adaptive_router.models.storage import MinIOSettings
    from adaptive_router.loaders.minio import MinIOProfileLoader

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

    loader = MinIOProfileLoader.from_settings(minio_settings)
    profile = loader.load_profile()

    router = ModelRouter.from_profile(profile=profile)

    logger.info("ModelRouter created successfully")

    return router, profile


class AppState:
    """Application state container."""

    def __init__(self) -> None:
        """Initialize application state."""
        self.settings: AppSettings | None = None
        self.router: ModelRouter | None = None
        self.available_models: list[Model] | None = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application with dependency injection."""
    app_state = AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Lifespan event handler for startup/shutdown."""
        try:
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

            logger.info("Loading router profile and initializing services...")
            router, profile = await create_model_router(app_state.settings)
            app_state.router = router
            app_state.available_models = list(profile.models)
            logger.info(
                "Loaded profile with %d model IDs",
                len(extract_model_ids_from_profile(profile)),
            )
            logger.info("ModelRouter initialized successfully")

            logger.info("FastAPI application started successfully")
        except Exception as e:
            logger.error("Failed to initialize router: %s", e, exc_info=True)
            raise

        yield

        logger.info("Shutting down Adaptive Router...")

    app = FastAPI(
        title="Adaptive Router",
        description="Intelligent LLM model selection API with cluster-based routing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

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

    @lru_cache()
    def get_settings() -> AppSettings:
        """Get application settings dependency."""
        if app_state.settings is None:
            app_state.settings = AppSettings()
        return app_state.settings

    async def get_router() -> ModelRouter:
        """Get ModelRouter dependency."""
        if app_state.router is None:
            settings = get_settings()
            logger.info("Lazy-initializing ModelRouter...")
            router, profile = await create_model_router(settings)
            app_state.router = router
            app_state.available_models = list(profile.models)
            logger.info("ModelRouter initialized successfully")
        return app_state.router

    async def get_available_models() -> list[Model]:
        """Return the list of models from the loaded router profile."""
        if app_state.available_models is None:
            await get_router()
        if app_state.available_models is None:
            raise RuntimeError("Router profile not loaded yet")
        return app_state.available_models

    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        status_code=status.HTTP_200_OK,
        responses={
            200: {"description": "All services healthy"},
            503: {"description": "One or more services unhealthy"},
        },
    )
    async def health_check() -> HealthCheckResponse:
        """Comprehensive health check for router and profile models."""
        overall_status = HealthStatus.HEALTHY

        if app_state.available_models:
            registry_health = ServiceHealth(
                status=HealthStatus.HEALTHY,
                message=f"{len(app_state.available_models)} models loaded from profile",
            )
        else:
            overall_status = HealthStatus.UNHEALTHY
            registry_health = ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                message="Router profile not loaded",
            )

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
        available_models: Annotated[list[Model], Depends(get_available_models)],
    ) -> ModelSelectionAPIResponse:
        """Select optimal model based on prompt analysis."""
        start_time = time.perf_counter()

        try:
            resolved_models = None
            if request.models:
                try:
                    resolved_models = resolve_models(request.models, available_models)
                except ValueError as e:
                    logger.error("Model resolution failed: %s", e, exc_info=True)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model resolution failed: {e}",
                    ) from e

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
