"""Modal deployment for Model Router Service.

This module deploys the model router as a single Modal function with:
- GPU acceleration for ML inference (NVIDIA T4)
- Auto-scaling with configurable min/max containers
- Direct function call interface
- JWT authentication for security
"""

import importlib.util
import logging
import os
import time

import modal
from fastapi import HTTPException, Request

from config import get_settings
from adaptive_router.models.llm_core_models import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.model_router import ModelRouter

# Import local JWT module
spec = importlib.util.spec_from_file_location(
    "jwt_auth", os.path.join(os.path.dirname(__file__), "jwt_auth.py")
)
if spec and spec.loader:
    jwt_auth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jwt_auth)
else:
    raise ImportError("Could not load JWT authentication module")

logger = logging.getLogger(__name__)

settings = get_settings()
model_router_instance = ModelRouter()

# Create Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        [
            # Core ML dependencies
            "torch",
            "transformers",
            "huggingface-hub",
            "numpy",
            "accelerate",
            # API dependencies
            "pydantic",
            "pydantic-settings",
            # LLM integration
            "langchain",
            "langchain-core",
            "openai",
            "pyyaml",
            "cachetools",
            # JWT authentication
            "python-jose[cryptography]",
            "pyjwt",
        ]
    )
    .add_local_file("config.py", "/root/config.py")
    .add_local_python_source("adaptive_router")
)

app = modal.App(settings.modal_deployment.app_name, image=image)


@app.function(
    gpu=settings.modal_deployment.gpu_type,
    timeout=settings.modal_deployment.timeout,
    scaledown_window=settings.modal_deployment.scaledown_window,
    min_containers=settings.modal_deployment.min_containers,
    max_containers=settings.modal_deployment.max_containers,
    secrets=[modal.Secret.from_name("jwt-auth")],
)
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
    print("🚀 Adaptive Router - Modal Deployment")
    print("=" * 60)
    print("✨ Features:")
    print(
        f"  • 🧠 ML inference with GPU acceleration ({settings.modal_deployment.gpu_type})"
    )
    print("  • ⚡ Auto-scaling serverless deployment")
    print("  • 🎯 Intelligent model selection")
    print("  • 🔒 JWT authentication for security")
    print("")
    print("🚀 Deploy: modal deploy deploy.py")
    print("")
    print("📞 Usage:")
    print("  from modal import Function")
    print(
        '  select_model = Function.lookup("prompt-task-complexity-classifier", "select_model")'
    )
    print("  result = select_model.remote(request)")
    print("")
    print("🔒 Authentication:")
    print("  Set JWT_SECRET environment variable before deployment")
    print("  Requests must include 'Authorization: Bearer <jwt_token>' header")
    print("")
    print("⚙️  Configuration:")
    print(f"  • App name: {settings.modal_deployment.app_name}")
    print(f"  • GPU: {settings.modal_deployment.gpu_type}")
    print(f"  • Timeout: {settings.modal_deployment.timeout}s")
    print(f"  • Scale down: {settings.modal_deployment.scaledown_window}s")
    print(f"  • Min containers: {settings.modal_deployment.min_containers}")
    print(f"  • Max containers: {settings.modal_deployment.max_containers}")
    print(
        f"  • JWT Secret: {'✅ Modal Secrets' if os.getenv('jwt_auth') else '❌ Not configured'}"
    )
