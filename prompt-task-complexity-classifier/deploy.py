"""Clean Modal deployment for prompt task complexity classifier.

Simple, production-ready deployment with:
- Clean package structure with prompt_task_complexity_classifier/
- Dual image strategy (ML + Web)
- Straightforward API endpoints
- JWT authentication
"""

import os
from typing import Dict, List, Any, TYPE_CHECKING
import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"
GPU_TYPE = "T4"
APP_NAME = "prompt-task-complexity-classifier"

# ============================================================================
# MODAL IMAGES
# ============================================================================

# ML image for NVIDIA model inference
ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "torch>=2.2.0,<2.5.0",
            "transformers>=4.52.4,<5",
            "huggingface-hub>=0.32.0,<0.35",
            "numpy>=1.24.0,<3.0",
            "accelerate>=1.8.1,<2",
        ]
    )
    .add_local_dir(
        "prompt_task_complexity_classifier",
        remote_path="/root/prompt_task_complexity_classifier",
    )
)

# Web image for FastAPI endpoints
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "fastapi>=0.110.0",
            "pydantic>=2.5.0",
            "PyJWT>=2.8.0",
            "numpy>=1.24.0,<3.0",
            "python-jose[cryptography]>=3.3.0",
        ]
    )
    .add_local_dir(
        "prompt_task_complexity_classifier",
        remote_path="/root/prompt_task_complexity_classifier",
    )
)

# ============================================================================
# MODAL APPLICATION
# ============================================================================

app = modal.App(APP_NAME)

# ============================================================================
# NVIDIA MODEL CLASS
# ============================================================================


@app.cls(
    image=ml_image,
    gpu=GPU_TYPE,
    secrets=[modal.Secret.from_name("jwt")],
    scaledown_window=300,
    timeout=600,
    max_containers=1,
    min_containers=0,
)
class PromptTaskComplexityClassifier:
    """Prompt task complexity classifier with GPU acceleration."""

    @modal.enter()
    def load_model(self) -> None:
        """Load the prompt task complexity classifier model."""
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from prompt_task_complexity_classifier.task_complexity_model import (
            get_model_classes,
        )

        _, _, CustomModelClass = get_model_classes()

        print(f"🚀 Loading prompt task complexity classifier: {MODEL_NAME}")
        print(
            f"🎮 GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
        )

        self.torch = torch
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = CustomModelClass(
            target_sizes=self.config.target_sizes,
            task_type_map=self.config.task_type_map,
            weights_map=self.config.weights_map,
            divisor_map=self.config.divisor_map,
        ).from_pretrained(MODEL_NAME)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  Model loaded on CPU")

        self.model.eval()
        print("✅ Prompt task complexity classifier ready!")

    @modal.method()
    def classify(self, prompts: List[str]) -> Dict[str, List[Any]]:
        """Classify a batch of prompts."""
        if not prompts:
            raise ValueError("No prompts provided")

        print(f"🔍 Classifying {len(prompts)} prompts...")

        # Tokenize
        encoded_texts = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to GPU
        if self.torch.cuda.is_available():
            encoded_texts = {k: v.cuda() for k, v in encoded_texts.items()}

        # Run inference
        with self.torch.no_grad():
            results: Dict[str, List[Any]] = self.model(encoded_texts)

        print(f"✅ Classification complete for {len(prompts)} prompts")
        return results

    @modal.method()
    def classify_single(self, prompt: str) -> Dict[str, Any]:
        """Classify a single prompt and return individual result."""
        # Use existing classify method with single prompt
        batch_result = self.classify([prompt])

        # Extract single result from batch (index 0)
        single_result = {
            "task_type_1": batch_result["task_type_1"][0],
            "task_type_2": batch_result["task_type_2"][0],
            "task_type_prob": batch_result["task_type_prob"][0],
            "creativity_scope": batch_result["creativity_scope"][0],
            "reasoning": batch_result["reasoning"][0],
            "contextual_knowledge": batch_result["contextual_knowledge"][0],
            "prompt_complexity_score": batch_result["prompt_complexity_score"][0],
            "domain_knowledge": batch_result["domain_knowledge"][0],
            "number_of_few_shots": batch_result["number_of_few_shots"][0],
            "no_label_reason": batch_result["no_label_reason"][0],
            "constraint_ct": batch_result["constraint_ct"][0],
        }
        return single_result

    @modal.method()
    def classify_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Classify a batch of prompts and return list of individual results."""
        # Use existing classify method
        batch_result = self.classify(prompts)

        # Convert batch result to list of individual results
        individual_results = []
        for i in range(len(prompts)):
            individual_result = {
                "task_type_1": batch_result["task_type_1"][i],
                "task_type_2": batch_result["task_type_2"][i],
                "task_type_prob": batch_result["task_type_prob"][i],
                "creativity_scope": batch_result["creativity_scope"][i],
                "reasoning": batch_result["reasoning"][i],
                "contextual_knowledge": batch_result["contextual_knowledge"][i],
                "prompt_complexity_score": batch_result["prompt_complexity_score"][i],
                "domain_knowledge": batch_result["domain_knowledge"][i],
                "number_of_few_shots": batch_result["number_of_few_shots"][i],
                "no_label_reason": batch_result["no_label_reason"][i],
                "constraint_ct": batch_result["constraint_ct"][i],
            }
            individual_results.append(individual_result)

        return individual_results


# ============================================================================
# FASTAPI WEB INTERFACE
# ============================================================================


@app.function(
    image=web_image,
    secrets=[modal.Secret.from_name("jwt")],
    scaledown_window=60,
    timeout=300,
    max_containers=1,
    cpu=2,
)
@modal.asgi_app()
def serve() -> "FastAPI":
    """Serve the FastAPI application."""
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
    from typing import Annotated
    import jwt

    app = FastAPI(
        title="Prompt Task Complexity Classifier API",
        version="1.0.0",
        description="GPU-accelerated prompt task complexity classification service",
    )
    security = HTTPBearer()

    # Pydantic models
    class ClassificationResult(BaseModel):
        """Classification result for a single prompt."""

        task_type_1: str = Field(description="Primary task type")
        task_type_2: str = Field(description="Secondary task type")
        task_type_prob: float = Field(
            description="Confidence score for primary task type"
        )
        creativity_scope: float = Field(description="Creativity level required (0-1)")
        reasoning: float = Field(description="Reasoning complexity required (0-1)")
        contextual_knowledge: float = Field(
            description="Context knowledge requirement (0-1)"
        )
        prompt_complexity_score: float = Field(
            description="Overall prompt complexity (0-1)"
        )
        domain_knowledge: float = Field(
            description="Domain-specific knowledge requirement (0-1)"
        )
        number_of_few_shots: float = Field(description="Few-shot learning requirement")
        no_label_reason: float = Field(
            description="Confidence in classification accuracy (0-1)"
        )
        constraint_ct: float = Field(description="Constraint complexity detected (0-1)")

    class ClassifyRequest(BaseModel):
        """Request model for batch prompt classification."""

        prompts: Annotated[List[str], Field(min_length=1, max_length=100)] = Field(
            description="List of prompts to classify"
        )

    class SingleClassifyRequest(BaseModel):
        """Request model for single prompt classification."""

        prompt: str = Field(
            description="Single prompt to classify", min_length=1, max_length=10000
        )

    def verify_jwt_token(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> str:
        """Verify JWT token from Authorization header."""
        try:
            token = credentials.credentials
            secret = os.environ.get("jwt_auth")
            if not secret:
                raise Exception("JWT secret not configured")

            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                options={"require_exp": True, "require_sub": True},
            )

            user = payload.get("sub") or payload.get("user")
            if not user:
                raise Exception("Invalid token: missing subject")
            return str(user)

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed"
            )

    @app.post("/classify", response_model=List[ClassificationResult])
    def classify_prompts(
        request: ClassifyRequest, user: str = Depends(verify_jwt_token)
    ) -> List[ClassificationResult]:
        """Classify batch of prompts."""
        print(
            f"Classification request from user: {user}, prompts: {len(request.prompts)}"
        )

        try:
            classifier = PromptTaskComplexityClassifier()
            results = classifier.classify_batch.remote(request.prompts)
            print(f"Classification completed for {len(request.prompts)} prompts")
            return [ClassificationResult(**result) for result in results]

        except Exception as e:
            print(f"Classification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Classification failed: {str(e)}",
            )

    @app.post("/classify/single", response_model=ClassificationResult)
    def classify_single_prompt(
        request: SingleClassifyRequest, user: str = Depends(verify_jwt_token)
    ) -> ClassificationResult:
        """Classify a single prompt."""
        print(f"Single classification request from user: {user}")

        try:
            classifier = PromptTaskComplexityClassifier()
            result = classifier.classify_single.remote(request.prompt)
            print("Single prompt classification completed")
            return ClassificationResult(**result)

        except Exception as e:
            print(f"Single prompt classification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Single prompt classification failed: {str(e)}",
            )

    @app.get("/health")
    def health_check() -> Dict[str, str]:
        """Basic health check endpoint."""
        return {"status": "healthy", "service": "prompt-task-complexity-classifier"}

    # TODO: Implement /health/detailed endpoint
    # Should return: service info, version, uptime, model status, dependencies

    # TODO: Implement /benchmark endpoint
    # Should return: latency measurements, throughput tests, timestamp results

    # TODO: Add JWT token caching with expiry/refresh logic
    # Current implementation verifies tokens but doesn't cache them

    # TODO: Add retry/backoff logic around classifier.classify.remote() calls
    # Current implementation has no retry on Modal API failures

    # TODO: Add structured logging with correlation IDs
    # Current implementation only uses basic print statements

    return app


# ============================================================================
# DEPLOYMENT INFO
# ============================================================================

if __name__ == "__main__":
    print("🚀 Prompt Task Complexity Classifier - Clean Modal Deployment")
    print("=" * 60)
    print("✨ Features:")
    print("  • 🧠 Complete prompt task complexity model implementation")
    print("  • 🎮 GPU acceleration (T4)")
    print("  • 🔐 JWT authentication")
    print("  • 📦 Clean package structure")
    print("  • ⚡ Simple, synchronous API endpoints")
    print("")
    print("🚀 Deploy: modal deploy deploy.py")
    print("🌐 Endpoints:")
    print("  POST /classify - Batch classification")
    print("  POST /classify/single - Single prompt classification")
    print("  GET /health - Health check")
    print("")
    print(f"🧠 Model: {MODEL_NAME}")
    print(f"🎮 GPU: {GPU_TYPE}")
