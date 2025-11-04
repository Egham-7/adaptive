"""FastAPI application for adaptive_router service.

Provides HTTP API endpoints for intelligent model selection using cluster-based routing.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from adaptive_router.core.router import ModelRouter
from adaptive_router.models.api import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models.registry import (
    RegistryConnectionError,
    RegistryResponseError,
)
from app_config import AppSettings
from app_health import HealthCheckResponse, HealthStatus, ServiceHealth
from model_registry import build_registry_client
from model_router_factory import create_model_router

env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


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
            client, _ = build_registry_client(settings)
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
