"""FastAPI application for adaptive_router service.

Provides HTTP API endpoints for intelligent model selection using cluster-based routing.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from adaptive_router.models.llm_core_models import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.model_router import ModelRouter

# Load environment variables
env_file = Path(".env")
if env_file.exists():
    load_dotenv(env_file)
elif (parent_env := Path("../.env")).exists():
    load_dotenv(parent_env)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

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

    # Configure CORS
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

    # Initialize model router (singleton pattern)
    router_instance = None

    def get_router() -> ModelRouter:
        """Get or create ModelRouter instance."""
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
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", "8000")
            logger.info(f"Starting Adaptive Router on http://{host}:{port}")
            logger.info(f"API Docs: http://{host}:{port}/docs")
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
        """
        return {"status": "healthy"}

    @app.post("/select-model", response_model=ModelSelectionResponse)
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

    return app


# Create application instance
app = create_app()
