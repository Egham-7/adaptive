from __future__ import annotations

import logging
import traceback
import modal
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request, FastAPI

# Create image with all dependencies - using latest versions
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
        ]
    )
    .add_local_python_source("prompt_task_complexity_classifier")
)

app = modal.App("prompt-task-complexity-classifier", image=image)


@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("jwt")],
    scaledown_window=300,
    timeout=600,
    max_containers=1,
    min_containers=0,
)
class PromptClassifier:
    """Prompt task complexity classifier with ML inference and FastAPI endpoints."""

    @modal.enter()
    def load_model(self) -> None:
        """Load the model on container startup."""
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from prompt_task_complexity_classifier.task_complexity_model import CustomModel

        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Get config
        try:
            from prompt_task_complexity_classifier.config import get_config

            config = get_config()
            model_name = config.deployment.model_name
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            print("🔄 Using default NVIDIA classifier model")
            model_name = "nvidia/prompt-task-and-complexity-classifier"

        print(f"🚀 Loading model: {model_name}")
        print(
            f"🎮 GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
        )

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model config and create custom model
        model_config = AutoConfig.from_pretrained(model_name)
        self.model = CustomModel(
            target_sizes=getattr(model_config, "target_sizes", {}),
            task_type_map=getattr(model_config, "task_type_map", {}),
            weights_map=getattr(model_config, "weights_map", {}),
            divisor_map=getattr(model_config, "divisor_map", {}),
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("✅ Model loaded on GPU")
        else:
            print("⚠️ Model loaded on CPU")

        self.model.eval()
        print("✅ Model ready!")

    def _classify_prompt(self, prompt: str) -> Dict[str, Any]:
        """Internal method to classify a single prompt."""
        # Tokenize
        encoded = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to GPU if available
        if self.torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Run inference
        with self.torch.no_grad():
            try:
                result = self.model(encoded)
                # Extract single result from batch
                return {
                    key: values[0] if isinstance(values, list) else values
                    for key, values in result.items()
                }
            except Exception as e:
                # Log the full error with context and stacktrace
                self.logger.error(
                    f"Model inference failed for prompt (length: {len(prompt)}): {str(e)}"
                )
                self.logger.error(f"Full stacktrace: {traceback.format_exc()}")

                # Re-raise the exception to propagate to FastAPI layer for HTTP 5xx
                raise RuntimeError(f"Model inference failed: {str(e)}") from e

    @modal.asgi_app()
    def create_fastapi_app(self) -> "FastAPI":
        """Create FastAPI app with all endpoints."""
        from fastapi import FastAPI, HTTPException, status
        import jwt
        import os

        def _verify_jwt_token(request: Request) -> Dict[str, Any]:
            """Verify JWT token from Authorization header."""
            # Extract Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Authorization header missing",
                )

            # Check Bearer token format
            if not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authorization header format",
                )

            token = auth_header.split(" ", 1)[1]

            # Get JWT secret from environment
            jwt_secret = os.environ.get("jwt_auth")
            if not jwt_secret:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="JWT secret not configured",
                )

            try:
                # Verify and decode JWT token
                payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
                return payload  # type: ignore[no-any-return]
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
                )
            except jwt.InvalidTokenError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
                )

        # Create FastAPI app
        web_app = FastAPI(
            title="Prompt Task Complexity Classifier",
            description="ML-powered prompt classification API",
            version="1.0.0",
        )

        @web_app.post("/classify")
        async def classify_endpoint(
            classify_request: Dict[str, Any], request: Request
        ) -> Dict[str, Any]:
            """Classify a single prompt."""
            # Import and cast inside function to avoid Modal import issues
            from prompt_task_complexity_classifier.models import (
                ClassifyRequest,
                ClassificationResult,
            )

            _verify_jwt_token(request)
            # Cast dict to proper model for validation
            validated_request = ClassifyRequest(**classify_request)
            result = self._classify_prompt(validated_request.prompt)
            # Return as dict to avoid response model issues
            return ClassificationResult(**result).model_dump()

        @web_app.post("/classify_batch")
        async def classify_batch_endpoint(
            batch_request: Dict[str, Any], request: Request
        ) -> List[Dict[str, Any]]:
            """Classify multiple prompts in batch."""
            # Import and cast inside function to avoid Modal import issues
            from prompt_task_complexity_classifier.models import (
                ClassifyBatchRequest,
                ClassificationResult,
            )

            _verify_jwt_token(request)
            # Cast dict to proper model for validation
            validated_request = ClassifyBatchRequest(**batch_request)
            results = [
                self._classify_prompt(prompt) for prompt in validated_request.prompts
            ]
            # Return as list of dicts to avoid response model issues
            return [ClassificationResult(**result).model_dump() for result in results]

        @web_app.get("/health")
        async def health_endpoint() -> Dict[str, str]:
            """Health check endpoint - no auth required."""
            return {"status": "healthy", "service": "prompt-task-complexity-classifier"}

        return web_app


if __name__ == "__main__":
    print("🚀 Prompt Task Complexity Classifier - Fixed Modal Deployment")
    print("=" * 60)
    print("✨ Features:")
    print("  • 🧠 ML inference with GPU acceleration")
    print("  • 🌐 FastAPI endpoints via ASGI")
    print("  • 📦 Single-file deployment")
    print("")
    print("🚀 Deploy: modal deploy deploy.py")
    print("🌐 Endpoints:")
    print("  POST /classify - Single prompt classification")
    print("  POST /classify_batch - Batch classification")
    print("  GET /health - Health check")
    print("")
