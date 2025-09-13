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
import logging
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adaptive_ai.services.prompt_classifier import PromptClassifier

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.llm_core_models import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
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
            "service": "adaptive_ai",
            "modal_service": modal_health,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e!s}") from e


@app.post("/predict", response_model=ModelSelectionResponse)
async def predict(request: ModelSelectionRequest) -> ModelSelectionResponse:
    """Process model selection request with intelligent routing.

    This endpoint handles the core functionality of analyzing prompts and
    selecting optimal models based on ML classification and cost optimization.
    """
    try:
        logger.info("Processing model selection request")

        # Classify prompt using Modal API
        start_time = time.perf_counter()
        if prompt_classifier is None:
            raise RuntimeError("Prompt classifier not initialized")

        classification = await prompt_classifier.classify_prompt_async(request.prompt)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Modal classification completed in {elapsed:.3f}s")

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

        logger.info("Model selection request completed successfully")
        return ModelSelectionResponse(
            provider=best_model.provider,
            model=best_model.model_name,
            alternatives=alternatives,
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e!s}"
        ) from e
