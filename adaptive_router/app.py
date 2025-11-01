"""FastAPI application for adaptive_router service.

Provides HTTP API endpoints for intelligent model selection using cluster-based routing.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from adaptive_router.models.api import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.core.router import ModelRouter
from adaptive_router.models.storage import MinIOSettings
from adaptive_router.models.registry import (
    RegistryClientConfig,
    RegistryConnectionError,
    RegistryError,
    RegistryResponseError,
)
from adaptive_router.registry import RegistryClient

env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid float value for %s=%s, falling back to %s", key, raw, default
        )
        return default


def _registry_headers_from_env() -> Dict[str, str]:
    headers: Dict[str, str] = {}

    api_key = os.getenv("MODEL_REGISTRY_API_KEY")
    if api_key:
        header_name = os.getenv("MODEL_REGISTRY_AUTH_HEADER", "Authorization")
        if header_name.lower() == "authorization" and not api_key.lower().startswith(
            "bearer "
        ):
            headers[header_name] = f"Bearer {api_key}"
        else:
            headers[header_name] = api_key

    extra_headers = os.getenv("MODEL_REGISTRY_HEADERS")
    if extra_headers:
        pairs = [pair.strip() for pair in extra_headers.split(",") if pair.strip()]
        for pair in pairs:
            if ":" not in pair:
                logger.warning(
                    "Ignoring invalid header format in MODEL_REGISTRY_HEADERS: %s", pair
                )
                continue
            name, value = pair.split(":", 1)
            headers[name.strip()] = value.strip()

    return headers


def _build_registry_client() -> tuple[RegistryClient, RegistryClientConfig]:
    """Build RegistryClient with configuration and HTTP client.

    Returns:
        Tuple of (RegistryClient, RegistryClientConfig)
    """
    base_url = os.getenv("MODEL_REGISTRY_BASE_URL", "http://localhost:3000")
    timeout = _env_float("MODEL_REGISTRY_TIMEOUT", 5.0)
    headers = _registry_headers_from_env()

    config = RegistryClientConfig(
        base_url=base_url,
        timeout=timeout,
        default_headers=headers or None,
    )

    # Create httpx.Client with timeout configuration
    http_client = httpx.Client(timeout=timeout)

    return RegistryClient(config, http_client), config


def load_model_costs_from_registry() -> Dict[str, float]:
    """Load model costs from the Adaptive model registry."""
    client, client_config = _build_registry_client()
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

    provider_stats: Dict[str, int] = {}
    model_costs: Dict[str, float] = {}

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
                "Model %s missing pricing data, defaulting cost to 1.0", model_id
            )
            avg_cost = 1.0

        model_costs[model_id] = avg_cost

    logger.info(
        "Loaded costs for %d models across %d providers",
        len(model_costs),
        len(provider_stats),
    )

    return model_costs


def create_model_router() -> ModelRouter:
    """Create ModelRouter with MinIO and ModelRegistry integration."""
    logger.info("Creating ModelRouter...")

    model_costs = load_model_costs_from_registry()

    endpoint_url: str = os.getenv(
        "MINIO_PRIVATE_ENDPOINT", "http://localhost:9000"
    ) or os.getenv("MINIO_ENDPOINT", "http://localhost:9000")

    minio_settings = MinIOSettings(
        endpoint_url=endpoint_url,
        root_user=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        root_password=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        bucket_name=os.getenv("S3_BUCKET_NAME", "adaptive-router-profiles"),
        region=os.getenv("S3_REGION", "us-east-1"),
        profile_key=os.getenv("S3_PROFILE_KEY", "global/profile.json"),
        connect_timeout=os.getenv("S3_CONNECT_TIMEOUT") or "5",
        read_timeout=os.getenv("S3_READ_TIMEOUT") or "30",
    )

    router = ModelRouter.from_minio(
        settings=minio_settings,
        model_costs=model_costs,
    )

    logger.info("ModelRouter created successfully")

    return router


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    router_instance = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Lifespan event handler for startup/shutdown."""
        nonlocal router_instance
        try:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", "8000")
            logger.info(f"Starting Adaptive Router on http://{host}:{port}")
            logger.info(f"API Docs: http://{host}:{port}/docs")
            logger.info("Initializing ModelRouter...")
            router_instance = create_model_router()
            logger.info("ModelRouter initialized successfully")
            logger.info("FastAPI application started successfully")
        except Exception as e:
            logger.error(f"Failed to initialize router: {e}")
            raise
        yield

    app = FastAPI(
        title="Adaptive Router",
        description="Intelligent LLM model selection API with cluster-based routing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
    allowed_origins = (
        [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
        if allowed_origins_str
        else ["*"]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allowed_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_router() -> ModelRouter:
        """Get or create ModelRouter instance."""
        nonlocal router_instance
        if router_instance is None:
            logger.info("Initializing ModelRouter...")
            router_instance = create_model_router()
            logger.info("ModelRouter initialized successfully")
        return router_instance

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/select-model", response_model=ModelSelectionResponse)
    async def select_model(
        request: ModelSelectionRequest,
        http_request: Request,
    ) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis."""
        start_time = time.perf_counter()

        try:
            router = get_router()

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
                f"Validation error: {e}",
                extra={"elapsed_ms": round(elapsed * 1000, 2)},
                exc_info=True,
            )
            raise HTTPException(
                status_code=400, detail="Invalid request: validation failed"
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                f"Model selection failed: {e}",
                extra={"elapsed_ms": round(elapsed * 1000, 2)},
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error",
            )

    return app


app = create_app()
