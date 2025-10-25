"""FastAPI application for adaptive_router service.

Provides HTTP API endpoints for intelligent model selection using cluster-based routing.
"""

import logging
import time
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from adaptive_router.models.llm_core_models import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.model_router import ModelRouter

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Adaptive Router",
        description="Intelligent LLM model selection API with cluster-based routing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize model router (singleton pattern)
    router_instance = None

    def get_router() -> ModelRouter:
        """Get or create ModelRouter instance.

        Returns:
            ModelRouter instance
        """
        nonlocal router_instance
        if router_instance is None:
            logger.info("Initializing ModelRouter...")
            router_instance = ModelRouter()
            logger.info("ModelRouter initialized successfully")
        return router_instance

    @app.on_event("startup")
    async def startup_event():
        """Initialize router on startup."""
        try:
            get_router()
            logger.info("FastAPI application started successfully")
        except Exception as e:
            logger.error(f"Failed to initialize router: {e}")
            raise

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint.

        Returns:
            Dictionary with health status

        Example:
            ```
            GET /health
            {
                "status": "healthy"
            }
            ```
        """
        return {"status": "healthy"}

    @app.post("/select_model", response_model=ModelSelectionResponse)
    async def select_model(
        request: ModelSelectionRequest,
        http_request: Request,
    ) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis.

        Uses cluster-based intelligent routing to select the best LLM model
        based on prompt characteristics, cost preferences, and model capabilities.

        Args:
            request: Model selection request with prompt and preferences
            http_request: FastAPI Request object for logging

        Returns:
            ModelSelectionResponse with selected model and alternatives

        Raises:
            HTTPException: If validation fails or routing error occurs

        Example:
            ```
            POST /select_model
            {
                "prompt": "Write a Python function to calculate factorial",
                "cost_bias": 0.5,
                "models": [
                    {
                        "provider": "openai",
                        "model_name": "gpt-4"
                    }
                ]
            }
            ```
        """
        start_time = time.perf_counter()

        try:
            router = get_router()

            logger.info(
                "Processing model selection request",
                extra={
                    "prompt_length": len(request.prompt),
                    "cost_bias": request.cost_bias,
                    "models_count": len(request.models) if request.models else 0,
                    "client_ip": http_request.client.host if http_request.client else "unknown",
                },
            )

            # Perform model selection
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
            )
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                f"Model selection failed: {e}",
                extra={"elapsed_ms": round(elapsed * 1000, 2)},
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error during model selection: {str(e)}",
            )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unexpected errors.

        Args:
            request: FastAPI Request object
            exc: Exception that was raised

        Returns:
            JSON response with error details
        """
        logger.error(
            f"Unhandled exception: {exc}",
            extra={
                "path": request.url.path,
                "method": request.method,
            },
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


# Create application instance
app = create_app()
