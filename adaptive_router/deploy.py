"""Modal deployment for Model Router Service.

This module deploys the model router as a single Modal function with:
- GPU acceleration for ML inference (NVIDIA T4)
- Auto-scaling with configurable min/max containers
- Direct function call interface
- JWT authentication for security
"""

import logging
import time

import modal
from fastapi import HTTPException, Request

from adaptive_router.models.llm_core_models import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.model_router import ModelRouter

# Modal deployment constants
APP_NAME = "adaptive_router"
GPU_TYPE = "T4"
TIMEOUT = 600
SCALEDOWN_WINDOW = 300
MIN_CONTAINERS = 0
MAX_CONTAINERS = 1

logger = logging.getLogger(__name__)


def get_modal_image() -> modal.Image:
    """Create optimized Modal image with minimal dependencies.

    Only includes packages actually imported by the model router:
    - torch, transformers, huggingface-hub, numpy: For PromptClassifier ML inference
    - pydantic, pydantic-settings: For data models and configuration
    - pyyaml: For YAMLModelDatabase model metadata loading
    - pyjwt: For JWT authentication (jwt module)
    - fastapi: For Modal endpoint
    """
    return (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install(
            [
                # Core ML dependencies (required for PromptClassifier)
                "torch",
                "transformers",
                "huggingface-hub",
                "numpy",
                # Data models and configuration
                "pydantic",
                "pydantic-settings",
                # YAML model database
                "pyyaml",
                # JWT authentication
                "pyjwt",
                # FastAPI endpoint
                "fastapi",
            ]
        )
        .env({"HF_HOME": "/models"})
        .add_local_file("config.py", "/root/config.py")
        .add_local_python_source("adaptive_router")
    )


app = modal.App(APP_NAME, image=get_modal_image())

# Volume for caching HuggingFace models
model_cache_volume = modal.Volume.from_name("adaptive-router-volume")
MODEL_DIR = "/models"


@app.function(
    gpu=GPU_TYPE,
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    secrets=[modal.Secret.from_name("jwt")],
    volumes={MODEL_DIR: model_cache_volume},
)
@modal.fastapi_endpoint(method="POST")
def select_model(
    request: ModelSelectionRequest, http_request: Request
) -> ModelSelectionResponse:
    """Select optimal model based on prompt analysis.

    Requires JWT authentication via Authorization header.

    Args:
        request: Model selection request with prompt and preferences
        http_request: FastAPI Request object for JWT validation

    Returns:
        ModelSelectionResponse: Selected model with provider and alternatives

    Raises:
        HTTPException: If JWT authentication fails (401)
        ValueError: If validation fails or no eligible models found
        RuntimeError: If inference or routing errors occur
    """
    import jwt_auth

    # Verify JWT token
    try:
        token_payload = jwt_auth.verify_jwt_token(http_request)
        logger.info(
            "JWT authentication successful",
            extra={"user": token_payload.get("sub", "unknown")},
        )
    except HTTPException as e:
        logger.error(
            "JWT authentication failed",
            extra={"status_code": e.status_code, "detail": e.detail},
        )
        raise

    logger.info(
        "Processing model selection request",
        extra={
            "prompt_length": len(request.prompt),
            "cost_bias": request.cost_bias,
            "models_count": len(request.models) if request.models else 0,
            "user": token_payload.get("sub", "unknown"),
        },
    )

    model_router_instance = ModelRouter()
    start_time = time.perf_counter()
    response = model_router_instance.select_model(request)
    elapsed = time.perf_counter() - start_time

    logger.info(
        "Model selection completed successfully",
        extra={
            "elapsed_ms": round(elapsed * 1000, 2),
            "selected_provider": response.provider,
            "selected_model": response.model,
            "alternatives_count": len(response.alternatives),
        },
    )

    return response


if __name__ == "__main__":
    import os

    print("🚀 Adaptive Router - Modal Deployment")
    print("=" * 60)
    print("✨ Features:")
    print(f"  • 🧠 ML inference with GPU acceleration ({GPU_TYPE})")
    print("  • ⚡ Auto-scaling serverless deployment")
    print("  • 🎯 Intelligent model selection")
    print("  • 🔒 JWT authentication for security")
    print("")
    print("🚀 Deploy: modal deploy deploy.py")
    print("")
    print("📞 Usage:")
    print("  from modal import Function")
    print(f'  select_model = Function.lookup("{APP_NAME}", "select_model")')
    print("  result = select_model.remote(request)")
    print("")
    print("🔒 Authentication:")
    print("  Set JWT_SECRET environment variable before deployment")
    print("  Requests must include 'Authorization: Bearer <jwt_token>' header")
    print("")
    print("⚙️  Configuration:")
    print(f"  • App name: {APP_NAME}")
    print(f"  • GPU: {GPU_TYPE}")
    print(f"  • Timeout: {TIMEOUT}s")
    print(f"  • Scale down: {SCALEDOWN_WINDOW}s")
    print(f"  • Min containers: {MIN_CONTAINERS}")
    print(f"  • Max containers: {MAX_CONTAINERS}")
    print(
        f"  • JWT Secret: {'✅ Modal Secrets' if os.getenv('jwt_auth') else '❌ Not configured'}"
    )
