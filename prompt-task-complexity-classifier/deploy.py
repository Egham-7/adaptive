"""Simple Modal deployment for prompt task complexity classifier.

Just one function that does ML inference with FastAPI endpoint - following Modal best practices.
"""

from __future__ import annotations

import logging
import traceback
import modal
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_task_complexity_classifier.models import (
        ClassificationResult,
        ClassifyRequest,
        ClassifyBatchRequest,
    )
    from fastapi import Request

# Create image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "torch>=2.2.0,<2.5.0",
            "transformers>=4.52.4,<5",
            "huggingface-hub>=0.32.0,<0.35",
            "numpy>=1.24.0,<3.0",
            "accelerate>=1.8.1,<2",
            "fastapi[standard]>=0.110.0",
            "pydantic>=2.11.5,<3",
            "PyJWT>=2.8.0",
            "python-jose[cryptography]>=3.3.0",
            "pyyaml>=6.0.2",
            "tiktoken >=0.11.0",
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

    def _verify_jwt_token(self, request: "Request") -> Dict[str, Any]:
        """Verify JWT token from Authorization header."""
        import jwt
        import os
        from fastapi import HTTPException, status

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
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

    @modal.fastapi_endpoint(method="POST", docs=True)
    def classify(
        self, classify_request: "ClassifyRequest", request: "Request"
    ) -> "ClassificationResult":
        """FastAPI endpoint for single prompt classification."""
        from prompt_task_complexity_classifier.models import ClassificationResult

        # Verify JWT token
        self._verify_jwt_token(request)

        result = self._classify_prompt(classify_request.prompt)
        return ClassificationResult(**result)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def classify_batch(
        self, batch_request: "ClassifyBatchRequest", request: "Request"
    ) -> List["ClassificationResult"]:
        """FastAPI endpoint for batch prompt classification."""
        from prompt_task_complexity_classifier.models import ClassificationResult

        # Verify JWT token
        self._verify_jwt_token(request)

        results = [self._classify_prompt(prompt) for prompt in batch_request.prompts]
        return [ClassificationResult(**result) for result in results]

    @modal.fastapi_endpoint(method="GET", docs=True)
    def health(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "prompt-task-complexity-classifier"}


if __name__ == "__main__":
    print("🚀 Prompt Task Complexity Classifier - Simple Modal Deployment")
    print("=" * 60)
    print("✨ Features:")
    print("  • 🧠 ML inference with GPU acceleration")
    print("  • 🌐 FastAPI endpoints")
    print("  • 📦 Single-file deployment")
    print("")
    print("🚀 Deploy: modal deploy deploy.py")
    print("🌐 Endpoints:")
    print("  POST /classify - Single prompt classification")
    print("  POST /classify_batch - Batch classification")
    print("  GET /health - Health check")
    print("")
