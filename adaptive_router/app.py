"""FastAPI application for adaptive_router service.

Provides HTTP API endpoints for intelligent model selection using cluster-based routing.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated, AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from adaptive_router.core.router import ModelRouter
from adaptive_router.models.api import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models.registry import (
    RegistryClientConfig,
    RegistryConnectionError,
    RegistryError,
    RegistryResponseError,
)
from adaptive_router.models.storage import MinIOSettings
from adaptive_router.registry import RegistryClient
from pydantic import BaseModel

env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


# Constants
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_REGISTRY_TIMEOUT = 5.0
DEFAULT_MODEL_COST = 1.0


class HealthStatus(str, Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceHealth(BaseModel):
    """Health status of an individual service."""

    status: HealthStatus
    message: str | None = None
    response_time_ms: float | None = None


class HealthCheckResponse(BaseModel):
    """Complete health check response."""

    status: HealthStatus
    registry: ServiceHealth
    router: ServiceHealth
    timestamp: float = Field(default_factory=time.time)


class AppSettings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server settings
    host: str = Field(default=DEFAULT_HOST, description="Server host")
    port: int = Field(default=DEFAULT_PORT, description="Server port")

    # Model Registry settings
    model_registry_base_url: str = Field(
        default="http://localhost:3000",
        description="Base URL for model registry",
    )
    model_registry_timeout: float = Field(
        default=DEFAULT_REGISTRY_TIMEOUT,
        description="Timeout for registry requests in seconds",
    )

    # MinIO/S3 settings
    minio_private_endpoint: str = Field(
        default="http://localhost:9000",
        description="Private MinIO endpoint URL",
    )
    minio_root_user: str = Field(default="minioadmin", description="MinIO root user")
    minio_root_password: str = Field(
        default="minioadmin",
        description="MinIO root password",
    )
    s3_bucket_name: str = Field(
        default="adaptive-router-profiles",
        description="S3 bucket name",
    )
    s3_region: str = Field(default="us-east-1", description="S3 region")
    s3_profile_key: str = Field(
        default="global/profile.json",
        description="S3 profile key path",
    )
    s3_connect_timeout: str = Field(default="5", description="S3 connect timeout")
    s3_read_timeout: str = Field(default="30", description="S3 read timeout")

    # CORS settings
    allowed_origins: str = Field(
        default="",
        description="Comma-separated list of allowed origins",
    )

    @property
    def origins_list(self) -> list[str]:
        """Parse allowed origins into a list."""
        if not self.allowed_origins:
            return ["*"]
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


def _build_registry_client(
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
        Dictionary mapping model IDs to their average costs

    Raises:
        ValueError: If registry health check fails, no models found, or fetch fails
    """
    client, client_config = _build_registry_client(settings)
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

    return model_costs


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


class AppState:
    """Application state container."""

    def __init__(self) -> None:
        """Initialize application state."""
        self.settings: AppSettings | None = None
        self.router: ModelRouter | None = None


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

            # Initialize ModelRouter
            logger.info("Initializing ModelRouter...")
            app_state.router = create_model_router(app_state.settings)
            logger.info("ModelRouter initialized successfully")
            logger.info("FastAPI application started successfully")
        except Exception as e:
            logger.error("Failed to initialize router: %s", e, exc_info=True)
            raise

        yield

        # Cleanup on shutdown
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
    def get_settings() -> AppSettings:
        """Get application settings dependency."""
        if app_state.settings is None:
            app_state.settings = AppSettings()
        return app_state.settings

    def get_router() -> ModelRouter:
        """Get ModelRouter dependency."""
        if app_state.router is None:
            settings = get_settings()
            logger.info("Lazy-initializing ModelRouter...")
            app_state.router = create_model_router(settings)
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
            client, _ = _build_registry_client(settings)
            client.health_check()
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
        response_model=ModelSelectionResponse,
        status_code=status.HTTP_200_OK,
        responses={
            400: {"description": "Invalid request"},
            500: {"description": "Internal server error"},
        },
    )
    async def select_model(
        request: ModelSelectionRequest,
        http_request: Request,
        router: Annotated[ModelRouter, Depends(get_router)],
    ) -> ModelSelectionResponse:
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
            logger.info(
                "Processing model selection request",
                extra={
                    "prompt_length": len(request.prompt),
                    "cost_bias": request.cost_bias,
                    "models_count": len(request.models) if request.models else 0,
                    "client_ip": (
                        http_request.client.host if http_request.client else "unknown"
                    ),
                },
            )

            response = router.select_model(request)
            elapsed = time.perf_counter() - start_time

            logger.info(
                "Model selection completed",
                extra={
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "selected_provider": response.provider,
                    "selected_model": response.model,
                    "alternatives_count": len(response.alternatives),
                },
            )

            return response

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
