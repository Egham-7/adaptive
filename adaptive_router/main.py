"""FastAPI application for adaptive_router service.

Provides HTTP API endpoints for intelligent model selection using cluster-based routing.
"""

import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator
import httpx

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from adaptive_router.core.router import ModelRouter
from adaptive_router.models.api import Model
from adaptive_router.models.api import (
    ModelSelectionAPIRequest,
)
from app.models import (
    RegistryConnectionError,
    RegistryError,
    RegistryResponseError,
    RegistryModel,
    RegistryClientConfig,
)
from app.config import AppSettings
from app.health import HealthCheckResponse, HealthStatus, ServiceHealth
from app.models import ModelSelectionAPIResponse
from app.registry import RegistryClient, ModelRegistry
from app.utils import (
    resolve_models,
)


env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


# Inlined from model_registry.py
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


def load_models_from_registry(settings: AppSettings) -> list[Model]:
    """Load typed Model objects from the Adaptive model registry.

    Args:
        settings: Application settings containing registry configuration

    Returns:
        List of Model objects with cost information

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
    router_models: list[Model] = []

    for reg_model in models:
        provider = (reg_model.provider or "").strip().lower()
        if provider:
            provider_stats[provider] = provider_stats.get(provider, 0) + 1

        try:
            reg_model.unique_id()
        except RegistryError as err:
            logger.warning("Skipping registry model without identifier: %s", err)
            continue

        # Extract pricing information (costs are per token, convert to per million tokens)
        prompt_cost_per_million = 0.0
        completion_cost_per_million = 0.0

        if reg_model.pricing:
            try:
                prompt_cost = float(reg_model.pricing.get("prompt_cost", 0))
                completion_cost = float(reg_model.pricing.get("completion_cost", 0))
                # Convert from per-token to per-million-tokens
                prompt_cost_per_million = prompt_cost * 1_000_000
                completion_cost_per_million = completion_cost * 1_000_000
            except (ValueError, TypeError):
                # If pricing parsing fails, use default values
                pass

        # If we couldn't extract pricing, use defaults
        if prompt_cost_per_million == 0 and completion_cost_per_million == 0:
            prompt_cost_per_million = 1.0  # DEFAULT_MODEL_COST
            completion_cost_per_million = 1.0  # DEFAULT_MODEL_COST

        # Skip models with invalid (negative or zero) pricing
        if prompt_cost_per_million <= 0 or completion_cost_per_million <= 0:
            logger.warning(
                "Skipping model %s:%s with invalid pricing (prompt: %.6f, completion: %.6f)",
                reg_model.provider,
                reg_model.model_name,
                prompt_cost_per_million,
                completion_cost_per_million,
            )
            continue

        # Create typed Model object
        router_model = Model(
            provider=reg_model.provider,
            model_name=reg_model.model_name,
            cost_per_1m_input_tokens=prompt_cost_per_million,
            cost_per_1m_output_tokens=completion_cost_per_million,
        )
        router_models.append(router_model)

    logger.info(
        "Loaded %d models across %d providers from registry",
        len(router_models),
        len(provider_stats),
    )

    return router_models


# Inlined from model_router_factory.py
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

    models = load_models_from_registry(settings)

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

    router = ModelRouter.from_minio(
        settings=minio_settings,
        models=models,
    )

    logger.info("ModelRouter created successfully")

    return router


# Inlined from model_fuzzy_matching.py
def normalize_model_id(model_id: str) -> list[str]:
    """Generate normalized variants of a model ID for fuzzy matching.

    Args:
        model_id: Original model ID (e.g., "anthropic:claude-sonnet-4-5-20250929")

    Returns:
        List of normalized variants for matching, ordered by specificity:
        1. Original ID
        2. Without date suffixes (e.g., -20250929, -2024-04-09)
        3. Without version suffixes (e.g., -latest, -preview, -v1)
        4. With dots instead of hyphens in version numbers

    Examples:
        >>> normalize_model_id("anthropic:claude-sonnet-4-5-20250929")
        [
            "anthropic:claude-sonnet-4-5-20250929",
            "anthropic:claude-sonnet-4-5",
            "anthropic:claude-sonnet-4.5"
        ]

        >>> normalize_model_id("openai:gpt-4-turbo-2024-04-09")
        [
            "openai:gpt-4-turbo-2024-04-09",
            "openai:gpt-4-turbo",
            "openai:gpt-4-turbo"
        ]
    """
    # Apply transformations sequentially
    transformations = [
        lambda x: x,  # Original
        lambda x: re.sub(r"-\d{8}$", "", x),  # Remove YYYYMMDD
        lambda x: re.sub(r"-\d{4}-\d{2}-\d{2}$", "", x),  # Remove YYYY-MM-DD
        lambda x: re.sub(
            r"-(latest|preview|alpha|beta|v\d+)$", "", x
        ),  # Remove version suffixes
        lambda x: re.sub(
            r"(\w+)-(\d+)-(\d+)", r"\1-\2.\3", x
        ),  # Convert hyphens to dots in versions (4-5 -> 4.5)
        lambda x: re.sub(
            r"(\w+)-(\d+)\.(\d+)", r"\1-\2-\3", x
        ),  # Convert dots to hyphens in versions (4.5 -> 4-5)
    ]

    # Apply all transformations and deduplicate while preserving order
    seen: set[str] = set()
    variants: list[str] = []

    for transform in transformations:
        variant = transform(model_id)
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)

    # Add cross-separator variants (: <-> /) for better matching between systems
    cross_separator_variants: list[str] = []
    for variant in variants:
        if ":" in variant:
            cross_separator_variants.append(variant.replace(":", "/"))
        elif "/" in variant:
            cross_separator_variants.append(variant.replace("/", ":"))

    for variant in cross_separator_variants:
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)

    return variants


def calculate_similarity(a: str, b: str) -> float:
    """Calculate similarity score between two strings using SequenceMatcher.

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score between 0.0 and 1.0, where 1.0 is identical
    """
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(
    target_id: str,
    available_ids: list[str],
    threshold: float = 0.8,
) -> tuple[str | None, float]:
    """Find the best matching model ID from available IDs.

    Args:
        target_id: Target model ID to match
        available_ids: List of available model IDs
        threshold: Minimum similarity threshold (0.0-1.0)

    Returns:
        Tuple of (best_match_id, similarity_score) or (None, 0.0) if no match
    """
    if not available_ids:
        return None, 0.0

    # Find ID with highest similarity score
    similarities = [
        (available_id, calculate_similarity(target_id, available_id))
        for available_id in available_ids
    ]
    best_match, best_score = max(similarities, key=lambda x: x[1])

    return (best_match, best_score) if best_score >= threshold else (None, 0.0)


class AppState:
    """Application state container."""

    def __init__(self) -> None:
        """Initialize application state."""
        self.settings: AppSettings | None = None
        self.router: ModelRouter | None = None
        self.registry: ModelRegistry | None = None


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

            # Initialize ModelResolverService
            logger.info("Initializing ModelResolverService...")
            registry_client, _ = build_registry_client(app_state.settings)
            app_state.registry = ModelRegistry(registry_client)
            logger.info("ModelResolverService initialized successfully")

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
            if app_state.registry is None:
                raise HTTPException(
                    detail="Model registry not initialized",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            # Resolve model specifications to RegistryModel objects
            resolved_models = None
            all_models = app_state.registry.list_models()
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
            internal_request = request.to_internal_request(resolved_models)

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

            # Transform library response to API response with full RegistryModel data
            selected_model_id = response.model_id
            selected_registry_model = app_state.registry.get(selected_model_id)

            if selected_registry_model is None:
                logger.warning(
                    f"Selected model {selected_model_id} not found in registry, using minimal data"
                )
                # Fallback: create minimal RegistryModel by parsing model_id
                parts = selected_model_id.split(":", 1)
                selected_registry_model = RegistryModel(
                    provider=parts[0] if len(parts) == 2 else "unknown",
                    model_name=parts[1] if len(parts) == 2 else selected_model_id,
                )

            # Lookup alternatives in registry
            alternative_registry_models = []
            for alt in response.alternatives:
                alt_model_id = alt.model_id
                alt_registry_model = app_state.registry.get(alt_model_id)

                if alt_registry_model is None:
                    logger.warning(
                        f"Alternative model {alt_model_id} not found in registry, skipping"
                    )
                    continue

                alternative_registry_models.append(alt_registry_model)

            return ModelSelectionAPIResponse(
                selected_model=selected_registry_model,
                alternatives=alternative_registry_models,
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
