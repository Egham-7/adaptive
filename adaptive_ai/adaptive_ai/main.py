"""FastAPI application for intelligent model selection and routing.

This module provides a FastAPI-based API that handles model selection requests with
ML-driven classification and intelligent routing via Modal service integration.

Key Components:
- FastAPI application with async endpoints
- Model classification and complexity analysis via Modal
- Intelligent model selection with cost optimization
- Health check endpoints

Architecture Flow:
1. Receive ModelSelectionRequest with prompt and preferences
2. Classify prompt using Modal API (task type, complexity, etc.)
3. Route to optimal model based on classification and cost bias
4. Return ModelSelectionResponse with selected model and alternatives
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import signal
import sys
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from hypercorn.asyncio import serve
from hypercorn.config import Config

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_ai.models.llm_enums import TaskType
from adaptive_ai.services.model_registry import model_registry
from adaptive_ai.services.model_router import ModelRouter
from adaptive_ai.services.prompt_classifier import get_prompt_classifier

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set specific loggers
    logger = logging.getLogger(__name__)
    logger.info(
        "Adaptive AI service starting with log level: %s", settings.logging.level
    )

    # Suppress noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# Global components initialized on startup
prompt_classifier = None
model_router = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown events."""
    global prompt_classifier, model_router

    # Startup
    logger.info("Initializing Adaptive AI services...")

    # Initialize services
    prompt_classifier = get_prompt_classifier()
    model_router = ModelRouter(model_registry)

    logger.info("Adaptive AI services initialized successfully")

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down Adaptive AI services...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Adaptive AI - Model Selection Service",
        description="Intelligent model selection and routing with ML-driven classification",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint."""
        try:
            # Check Modal service health
            modal_health = (
                prompt_classifier.health_check()
                if prompt_classifier
                else {"status": "not_initialized"}
            )

            return {
                "status": "healthy",
                "service": "adaptive_ai",
                "modal_service": modal_health,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(
                status_code=503, detail=f"Service unhealthy: {e!s}"
            ) from e

    @app.post("/predict", response_model=ModelSelectionResponse)
    async def predict(request: ModelSelectionRequest) -> ModelSelectionResponse:
        """Process model selection request with intelligent routing.

        This endpoint handles the core functionality of analyzing prompts and
        selecting optimal models based on ML classification and cost optimization.
        """
        try:
            logger.info("Processing model selection request")

            # Classify prompt using Modal API
            classification_result = await classify_prompt_async(request)

            # Process the request with classification results
            response = await process_request_async(request, classification_result)

            logger.info("Model selection request completed successfully")
            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {e!s}"
            ) from e

    return app


async def classify_prompt_async(request: ModelSelectionRequest) -> ClassificationResult:
    """Classify single prompt for task type asynchronously."""
    start_time = time.perf_counter()

    # Extract prompt from request - it's a required field
    prompt = request.prompt

    try:
        # Use the async Modal classification method directly
        if prompt_classifier is None:
            raise RuntimeError("Prompt classifier not initialized")
        results = await prompt_classifier.classify_prompts_async([prompt])

        elapsed = time.perf_counter() - start_time
        logger.info(f"Modal classification completed in {elapsed:.3f}s")

        # Validate results and return first classification or fallback
        if results and len(results) > 0 and results[0]:
            return results[0]
        else:
            logger.warning(
                "Modal returned empty or invalid results, using default classification"
            )
            return get_default_classification()

    except Exception as e:
        # Log the error and use default classification as fallback
        elapsed = time.perf_counter() - start_time
        logger.warning(
            f"Modal classification failed after {elapsed:.3f}s, using default: {e}"
        )
        return get_default_classification()


def get_default_classification() -> ClassificationResult:
    """Get default classification when Modal fails."""
    return ClassificationResult(
        # Required fields
        task_type_1=["Other"],
        prompt_complexity_score=[0.5],
        domain=["General"],
        # Optional fields
        task_type_2=["Unknown"],
        task_type_prob=[0.5],
        creativity_scope=[0.5],
        reasoning=[0.5],
        contextual_knowledge=[0.5],
        domain_knowledge=[0.5],
        number_of_few_shots=[0],
        no_label_reason=[0.5],
        constraint_ct=[0.5],
    )


def convert_to_task_type(task_type_str: str | None) -> TaskType:
    """Convert string task type to TaskType enum."""
    if not task_type_str:
        return TaskType.OTHER

    # Try to find matching TaskType enum value
    for task_type in TaskType:
        if task_type.value == task_type_str:
            return task_type

    # Fallback to OTHER if no match found
    return TaskType.OTHER


async def process_request_async(
    request: ModelSelectionRequest,
    classification: ClassificationResult,
) -> ModelSelectionResponse:
    """Process a single model selection request."""
    # Extract parameters
    task_type_str = (
        classification.task_type_1[0] if classification.task_type_1 else None
    )
    task_type = convert_to_task_type(task_type_str)
    models_input = request.models
    cost_bias = request.cost_bias

    # Select models
    start_time = time.perf_counter()
    task_complexity = (
        classification.prompt_complexity_score[0]
        if classification.prompt_complexity_score
        else 0.5
    )

    # Run model selection
    if model_router is None:
        raise RuntimeError("Model router not initialized")

    # Debug logging
    logger.info(
        f"Model selection inputs: task_complexity={task_complexity}, task_type={task_type}, models_input={len(models_input) if models_input else 0}, cost_bias={cost_bias}"
    )

    selected_models = model_router.select_models(
        task_complexity,
        task_type,
        models_input,
        cost_bias or 0.5,  # Handle None case
    )

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Model selection completed in {elapsed:.3f}s, selected {len(selected_models)} models"
    )

    if not selected_models:
        logger.error(
            f"No models selected with inputs: task_complexity={task_complexity}, task_type={task_type}, models_input={models_input}, cost_bias={cost_bias}"
        )
        raise ValueError("No eligible models found")

    # Build response
    best_model = selected_models[0]

    # Ensure we have valid provider and model names
    if not best_model.provider:
        raise ValueError("Selected model missing provider")
    if not best_model.model_name:
        raise ValueError("Selected model missing model_name")

    alternatives = [
        Alternative(provider=alt.provider, model=alt.model_name)
        for alt in selected_models[1:]
        if alt.provider
        and alt.model_name  # Skip alternatives with missing provider/model
    ]

    return ModelSelectionResponse(
        provider=best_model.provider,
        model=best_model.model_name,
        alternatives=alternatives,
    )


def create_hypercorn_config() -> Config:
    """Create Hypercorn configuration from settings."""
    settings = get_settings()
    config = Config()

    # Basic server configuration
    config.bind = [f"{settings.server.host}:{settings.server.port}"]
    config.accesslog = "-" if settings.fastapi.access_log else None
    config.loglevel = settings.fastapi.log_level

    # Additional Hypercorn-specific configurations
    config.keep_alive_timeout = 5
    config.graceful_timeout = 30

    return config


async def serve_with_graceful_shutdown() -> None:
    """Serve the application with graceful shutdown handling."""
    logger = logging.getLogger(__name__)

    # Create shutdown event
    shutdown_event = asyncio.Event()

    def _signal_handler(signum: int, frame: Any) -> None:
        logger.info("Received signal %d, initiating graceful shutdown...", signum)
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Create configuration
    config = create_hypercorn_config()

    logger.info("Starting Hypercorn server on %s", ", ".join(config.bind))

    try:
        # Serve the application
        await serve(app, config, shutdown_trigger=shutdown_event.wait)  # type: ignore[arg-type]
    except Exception as e:
        logger.exception("Server error: %s", e)
        raise


def main() -> None:
    """Run the FastAPI application with proper logging setup."""
    # Setup logging first
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting Adaptive AI Service with Hypercorn")

    try:
        settings = get_settings()

        logger.info(
            "Starting server on %s:%d",
            settings.server.host,
            settings.server.port,
        )

        # Note: workers parameter is not used in programmatic API
        # For multi-process deployment, use a process manager like gunicorn
        if settings.fastapi.workers > 1:
            logger.warning(
                "Workers setting (%d) ignored in programmatic mode. "
                "Use a process manager for multi-process deployment.",
                settings.fastapi.workers,
            )

        # Run the server
        asyncio.run(serve_with_graceful_shutdown())

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.exception("Failed to start Adaptive AI Service: %s", e)
        sys.exit(1)


# Create app instance for hypercorn to discover
app = create_app()

if __name__ == "__main__":
    main()
