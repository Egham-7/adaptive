import logging
import traceback
import modal
from typing import List
from fastapi import Request, HTTPException, status
from prompt_task_complexity_classifier import (
    get_prompt_classifier,
    ClassificationResult,
    ClassifyRequest,
    ClassifyBatchRequest,
    verify_jwt_token,
    get_config,
)


# Load configuration
config = get_config()

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        [
            "torch",
            "transformers",
            "huggingface-hub",
            "numpy",
            "accelerate",
            "fastapi[standard]",
            "pydantic",
            "PyJWT",
            "python-jose[cryptography]",
            "pyyaml",
            "tiktoken",
            "pydantic_settings",
        ]
    )
    .add_local_python_source("prompt_task_complexity_classifier")
)

app = modal.App(config.deployment.app_name, image=image)


@app.function(
    gpu=config.deployment.gpu_type,
    timeout=config.deployment.ml_timeout,
    scaledown_window=config.deployment.scaledown_window,
    secrets=[modal.Secret.from_name(config.deployment.modal_secret_name)],
    min_containers=config.deployment.min_containers,
    max_containers=config.deployment.max_containers,
)
@modal.fastapi_endpoint(method="POST")
def classify(
    classify_request: ClassifyRequest, request: Request
) -> ClassificationResult:
    """Classify a single prompt for task complexity and type.

    Args:
        classify_request: Request containing the prompt to classify
        request: FastAPI Request object for JWT verification

    Returns:
        ClassificationResult: Detailed classification results

    Raises:
        HTTPException: If authentication fails or inference errors occur
    """
    try:
        # Verify JWT token
        verify_jwt_token(request)

        # Get cached classifier and run inference
        classifier = get_prompt_classifier()
        result = classifier.classify_prompt(classify_request.prompt)

        return ClassificationResult(**result)

    except HTTPException:
        # Re-raise authentication/authorization errors
        raise
    except Exception as e:
        # Log and convert inference errors to HTTP 500
        logging.error(f"Classification failed: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}",
        )


@app.function(
    gpu=config.deployment.gpu_type,
    timeout=config.deployment.ml_timeout,
    scaledown_window=config.deployment.scaledown_window,
    min_containers=config.deployment.min_containers,
    max_containers=config.deployment.max_containers,
    secrets=[modal.Secret.from_name(config.deployment.modal_secret_name)],
)
@modal.fastapi_endpoint(method="POST")
def classify_batch(
    batch_request: ClassifyBatchRequest, request: Request
) -> List[ClassificationResult]:
    """Classify multiple prompts in batch for task complexity and type.

    Args:
        batch_request: Request containing list of prompts to classify
        request: FastAPI Request object for JWT verification

    Returns:
        List[ClassificationResult]: Classification results for each prompt

    Raises:
        HTTPException: If authentication fails or inference errors occur
    """
    try:
        # Verify JWT token
        verify_jwt_token(request)

        # Get cached classifier and run batch inference
        classifier = get_prompt_classifier()
        results = [
            classifier.classify_prompt(prompt) for prompt in batch_request.prompts
        ]

        return [ClassificationResult(**result) for result in results]

    except HTTPException:
        # Re-raise authentication/authorization errors
        raise
    except Exception as e:
        # Log and convert inference errors to HTTP 500
        logging.error(f"Batch classification failed: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch classification failed: {str(e)}",
        )


if __name__ == "__main__":
    print(f"🚀 {config.deployment.app_name} - Modal Deployment")
    print("=" * 60)
    print("✨ Features:")
    print(f"  • 🧠 ML inference with GPU acceleration ({config.deployment.gpu_type})")
    print(
        f"  • 🔐 JWT authentication with Modal secret '{config.deployment.modal_secret_name}'"
    )
    print("  • ⚡ Individual function endpoints with auto-scaling")
    print("  • 📦 Optimized container lifecycle")
    print("")
    print("🚀 Deploy: modal deploy deploy.py")
    print("🌐 Endpoints:")
    print("  POST /classify - Single prompt classification")
    print("  POST /classify_batch - Batch classification")
    print("  GET /health - Health check (no auth)")
    print("")
    print("⚙️  Configuration:")
    print(f"  • App name: {config.deployment.app_name}")
    print(f"  • Model: {config.deployment.model_name}")
    print(f"  • GPU: {config.deployment.gpu_type}")
    print(f"  • ML timeout: {config.deployment.ml_timeout}s")
    print(f"  • Web timeout: {config.deployment.web_timeout}s")
    print(f"  • Scale down: {config.deployment.scaledown_window}s")
    print(f"  • Min containers: {config.deployment.min_containers}")
    print(f"  • Max containers: {config.deployment.max_containers}")
