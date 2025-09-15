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

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
import logging
import sys
import time
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pythonjsonlogger.jsonlogger import JsonFormatter

from model_router.core.config import get_settings
from model_router.models.llm_core_models import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from model_router.services.model_registry import model_registry
from model_router.services.model_router import ModelRouter
from model_router.services.prompt_classifier import (
    PromptClassifier,
    get_prompt_classifier,
)

logger = logging.getLogger(__name__)

# Context variable for request correlation
request_id_context: ContextVar[str] = ContextVar("request_id", default="")


def setup_logging() -> None:
    """Configure structured JSON logging for the application."""
    settings = get_settings()

    # Create JSON formatter with request correlation
    class RequestCorrelationFormatter(jsonlogger.JsonFormatter):
        def add_fields(
            self,
            log_record: dict[str, Any],
            record: logging.LogRecord,
            message_dict: dict[str, Any],
        ) -> None:
            super().add_fields(log_record, record, message_dict)

            # Add request correlation ID if available
            request_id = request_id_context.get("")
            if request_id:
                log_record["request_id"] = request_id

            # Add standard fields
            log_record["timestamp"] = record.created
            log_record["service"] = "model_router"

    # Configure JSON formatter
    json_formatter = RequestCorrelationFormatter(  # type: ignore[no-untyped-call]
        "%(timestamp)s %(name)s %(levelname)s %(message)s"
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.logging.level.upper(), logging.INFO))

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add JSON handler
    json_handler = logging.StreamHandler(sys.stdout)
    json_handler.setFormatter(json_formatter)
    root_logger.addHandler(json_handler)

    # Set specific loggers
    logger = logging.getLogger(__name__)
    logger.info(
        "Adaptive AI service starting with structured JSON logging",
        extra={"log_level": settings.logging.level},
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("hypercorn.access").setLevel(logging.WARNING)


# Global components initialized on startup
prompt_classifier: "PromptClassifier | None" = None
model_router: ModelRouter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    global prompt_classifier, model_router

    # Startup
    setup_logging()
    logger.info("Initializing Adaptive AI services...")

    # Initialize services
    prompt_classifier = get_prompt_classifier()
    model_router = ModelRouter(model_registry)

    logger.info("Adaptive AI services initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Adaptive AI services...")

    # Close the prompt classifier's AsyncClient to prevent socket leaks
    if prompt_classifier:
        try:
            await prompt_classifier.aclose()
            logger.info("Prompt classifier AsyncClient closed successfully")
        except Exception as e:
            logger.error(
                "Failed to close prompt classifier AsyncClient",
                extra={"error": str(e), "error_type": type(e).__name__},
            )


# Create FastAPI app instance
app = FastAPI(
    title="Adaptive AI - Model Selection Service",
    description="Intelligent model selection and routing with ML-driven classification",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware with secure configuration
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allowed_origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)


@app.middleware("http")
async def request_correlation_middleware(request: Request, call_next: Any) -> Any:
    """Middleware to handle request correlation via X-Request-ID header."""
    # Get or generate request ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())

    # Set request ID in context
    token = request_id_context.set(request_id)

    try:
        # Process the request
        response = await call_next(request)

        # Add request ID to response headers for client correlation
        response.headers["X-Request-ID"] = request_id

        return response
    finally:
        # Clean up context
        request_id_context.reset(token)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    try:
        # Check Modal service health using async method
        modal_health = (
            await prompt_classifier.health_check_async()
            if prompt_classifier
            else {"status": "not_initialized"}
        )

        return {
            "status": "healthy",
            "service": "model_router",
            "modal_service": modal_health,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(
            "Health check failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e!s}") from e


@app.post("/predict", response_model=ModelSelectionResponse)
async def predict(request: ModelSelectionRequest) -> ModelSelectionResponse:
    """Process model selection request with intelligent routing.

    This endpoint handles the core functionality of analyzing prompts and
    selecting optimal models based on ML classification and cost optimization.
    """
    try:
        logger.info(
            "Processing model selection request",
            extra={
                "prompt_length": len(request.prompt),
                "cost_bias": request.cost_bias,
                "models_count": len(request.models) if request.models else 0,
            },
        )

        # Classify prompt using Modal API
        start_time = time.perf_counter()
        if prompt_classifier is None:
            raise RuntimeError("Prompt classifier not initialized")

        classification = await prompt_classifier.classify_prompt_async(request.prompt)
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Modal classification completed",
            extra={
                "classification_time_ms": round(elapsed * 1000, 2),
                "task_type": classification.task_type_1,
                "complexity_score": classification.prompt_complexity_score,
            },
        )

        # Extract task type directly from classification
        task_type = (
            classification.task_type_1 if classification.task_type_1 else "Other"
        )

        # Extract task complexity
        task_complexity = (
            classification.prompt_complexity_score
            if classification.prompt_complexity_score is not None
            else 0.5
        )

        # Select models
        if model_router is None:
            raise RuntimeError("Model router not initialized")

        selected_models = model_router.select_models(
            task_complexity,
            task_type,
            request.models,
            request.cost_bias if request.cost_bias is not None else 0.5,
        )

        if not selected_models:
            raise ValueError("No eligible models found")

        # Build response
        best_model = selected_models[0]
        if not best_model.provider or not best_model.model_name:
            raise ValueError("Selected model missing provider or model_name")

        alternatives = [
            Alternative(provider=alt.provider, model=alt.model_name)
            for alt in selected_models[1:]
            if alt.provider and alt.model_name
        ]

        logger.info(
            "Model selection request completed successfully",
            extra={
                "selected_provider": best_model.provider,
                "selected_model": best_model.model_name,
                "alternatives_count": len(alternatives),
                "task_type": task_type,
                "task_complexity": task_complexity,
            },
        )
        return ModelSelectionResponse(
            provider=best_model.provider,
            model=best_model.model_name,
            alternatives=alternatives,
        )

    except ValueError as e:
        logger.error(
            "Validation error in model selection",
            extra={"error": str(e), "error_type": "ValidationError"},
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Model selection failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e!s}"
        ) from e
