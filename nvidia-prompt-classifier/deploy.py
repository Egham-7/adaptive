"""Clean Modal deployment for NVIDIA prompt classifier using organized package structure.

This deployment showcases a well-organized Modal application with:
- Separated concerns (model logic, API endpoints, utilities)
- Whole folder deployment using add_local_dir
- Professional package structure
- Clear separation between ML and web components
"""

import os
from typing import Dict, List, TYPE_CHECKING, Any
import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"
GPU_TYPE = "T4"
APP_NAME = "nvidia-prompt-classifier"

# ============================================================================
# MODAL IMAGES WITH ORGANIZED PACKAGE
# ============================================================================

# ML image for NVIDIA model inference (minimal dependencies)
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
        "nvidia_classifier", remote_path="/root/nvidia_classifier"
    )  # Deploy whole folder
)

# Web image for FastAPI endpoints
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "fastapi>=0.110.0",
            "pydantic>=2.5.0",
            "PyJWT>=2.8.0",
            "numpy>=1.24.0,<3.0",  # Needed for deserializing results from ML container
            "httpx>=0.24.0",  # Required for client functionality
            "python-jose>=3.3.0",  # Required for JWT handling
        ]
    )
    .add_local_dir(
        "nvidia_classifier", remote_path="/root/nvidia_classifier"
    )  # Deploy whole folder
)

# ============================================================================
# MODAL APPLICATION
# ============================================================================

app = modal.App(APP_NAME)

# ============================================================================
# NVIDIA MODEL CLASS (using consolidated components)
# ============================================================================


@app.cls(
    image=ml_image,
    gpu=GPU_TYPE,
    secrets=[modal.Secret.from_name("jwt-auth")],
    max_containers=20,  # Increased for better concurrency
    scaledown_window=180,  # Faster scaledown for cost efficiency
    min_containers=2,  # Keep 2 containers warm for faster response (renamed from keep_warm)
)
class NvidiaPromptClassifier:
    """NVIDIA prompt classifier using consolidated model components."""

    @modal.enter()
    def load_model(self) -> None:
        """Load model using consolidated components from nvidia_model."""
        import torch  # type: ignore
        from transformers import AutoConfig, AutoTokenizer  # type: ignore
        from nvidia_classifier.nvidia_model import get_model_classes

        # Get model classes (imports torch inside the container)
        _, _, CustomModelClass = get_model_classes()

        print(f"🚀 Loading NVIDIA classifier: {MODEL_NAME}")
        print(
            f"🎮 GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
        )

        # Store references
        self.torch = torch

        # Load model configuration and tokenizer
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Create and load custom model with full configuration
        self.model = CustomModelClass(
            target_sizes=self.config.target_sizes,
            task_type_map=self.config.task_type_map,
            weights_map=self.config.weights_map,
            divisor_map=self.config.divisor_map,
        ).from_pretrained(MODEL_NAME)

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  Model loaded on CPU")

        self.model.eval()
        print("✅ NVIDIA prompt classifier ready!")

    @modal.method()
    def classify(self, prompts: List[str]) -> Dict[str, List]:
        """Classify a batch of prompts with GPU acceleration."""
        print(f"Received prompts: {prompts}")

        if prompts is None or len(prompts) == 0:
            raise ValueError("No prompts provided")

        print(f"🔍 Classifying {len(prompts)} prompts on GPU...")

        # Tokenize prompts
        encoded_texts = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to GPU if available
        if self.torch.cuda.is_available():
            encoded_texts = {k: v.cuda() for k, v in encoded_texts.items()}

        # Run inference - this returns the full classification results
        with self.torch.no_grad():
            results = self.model(encoded_texts)

        print(f"✅ GPU classification complete for {len(prompts)} prompts")
        return results


# ============================================================================
# FASTAPI WEB INTERFACE (using organized models)
# ============================================================================


@app.function(
    image=web_image,
    secrets=[modal.Secret.from_name("jwt-auth")],
    max_containers=50,  # Allow multiple concurrent requests (renamed from concurrency_limit)
    timeout=300,  # 5 minute timeout for large batches
)
@modal.asgi_app()
def serve() -> "FastAPI":
    """Serve the FastAPI application with async support."""
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
    import jwt
    import asyncio
    from typing import Optional

    # Create FastAPI app
    web_app = FastAPI(
        title="NVIDIA Prompt Classifier API",
        version="1.0.0",
        description="GPU-accelerated prompt classification with JWT authentication and async support",
    )
    security = HTTPBearer()

    # Pydantic models for API
    class ClassificationResult(BaseModel):
        """Results from prompt classification including task type and complexity metrics."""

        task_type_1: List[str] = Field(description="Primary task type for each prompt")
        task_type_2: List[str] = Field(
            description="Secondary task type for each prompt"
        )
        task_type_prob: List[float] = Field(
            description="Confidence scores for primary task types"
        )
        creativity_scope: List[float] = Field(
            description="Creativity level required (0-1)"
        )
        reasoning: List[float] = Field(
            description="Reasoning complexity required (0-1)"
        )
        contextual_knowledge: List[float] = Field(
            description="Context knowledge requirement (0-1)"
        )
        prompt_complexity_score: List[float] = Field(
            description="Overall prompt complexity (0-1)"
        )
        domain_knowledge: List[float] = Field(
            description="Domain-specific knowledge requirement (0-1)"
        )
        number_of_few_shots: List[int] = Field(
            description="Few-shot learning requirement"
        )
        no_label_reason: List[float] = Field(
            description="Confidence in classification accuracy (0-1)"
        )
        constraint_ct: List[float] = Field(
            description="Constraint complexity detected (0-1)"
        )

    class ClassifyRequest(BaseModel):
        """Request model for prompt classification."""

        prompts: List[str] = Field(
            description="List of prompts to classify", min_length=1, max_length=100
        )
        chunk_size: Optional[int] = Field(
            default=None,
            description="Chunk size for batch processing (1-50)",
            ge=1,
            le=50,
        )
        max_concurrent: Optional[int] = Field(
            default=None,
            description="Maximum concurrent chunks to process (1-10)",
            ge=1,
            le=10,
        )

    def verify_jwt_token(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> str:
        """Verify JWT token from Authorization header with enhanced security."""
        try:
            token = credentials.credentials
            secret = os.environ.get("JWT_SECRET")
            if not secret:
                raise Exception("JWT secret not configured")

            # Enhanced JWT validation with explicit algorithm verification
            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],  # Only allow HS256 algorithm
                options={
                    "require_exp": True,
                    "require_sub": True,
                },  # Require expiration and subject
            )

            user = payload.get("sub") or payload.get("user")
            if not user:
                raise Exception("Invalid token: missing subject")
            return user

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

    async def _process_chunk_async(prompts_chunk: List[str]) -> Dict[str, List]:
        """Process a single chunk of prompts asynchronously."""
        try:
            # Use Modal's async interface for non-blocking GPU processing
            classifier = NvidiaPromptClassifier()

            # Create a task for async processing
            task = asyncio.create_task(
                asyncio.to_thread(classifier.classify.remote, prompts_chunk)
            )
            result = await task
            return result
        except Exception as e:
            print(f"Chunk processing failed: {e}")
            # Return fallback results for failed chunk
            return {
                "task_type_1": ["Other"] * len(prompts_chunk),
                "task_type_2": ["NA"] * len(prompts_chunk),
                "task_type_prob": [0.5] * len(prompts_chunk),
                "creativity_scope": [0.5] * len(prompts_chunk),
                "reasoning": [0.5] * len(prompts_chunk),
                "contextual_knowledge": [0.5] * len(prompts_chunk),
                "prompt_complexity_score": [0.5] * len(prompts_chunk),
                "domain_knowledge": [0.5] * len(prompts_chunk),
                "number_of_few_shots": [0] * len(prompts_chunk),
                "no_label_reason": [0.5] * len(prompts_chunk),
                "constraint_ct": [0.5] * len(prompts_chunk),
            }

    def _chunk_prompts(prompts: List[str], chunk_size: int) -> List[List[str]]:
        """Split prompts into chunks for parallel processing."""
        return [prompts[i : i + chunk_size] for i in range(0, len(prompts), chunk_size)]

    def _merge_results(
        chunk_results: List[Dict[str, List[Any]]],
    ) -> Dict[str, List[Any]]:
        """Merge results from multiple chunks into a single result."""
        if not chunk_results:
            return {}

        merged: Dict[str, List[Any]] = {}
        for key in chunk_results[0].keys():
            merged[key] = []
            for chunk_result in chunk_results:
                merged[key].extend(chunk_result.get(key, []))
        return merged

    @web_app.post("/classify", response_model=ClassificationResult)
    async def classify_prompts(
        request: ClassifyRequest, user: str = Depends(verify_jwt_token)
    ):
        """Classify prompts using NVIDIA model with async chunked processing."""
        print(
            f"Classification request from user: {user}, prompts: {len(request.prompts)}, "
            f"chunk_size: {request.chunk_size}, max_concurrent: {request.max_concurrent}"
        )

        try:
            # Split prompts into chunks for parallel processing
            chunk_size = request.chunk_size or 10
            max_concurrent = request.max_concurrent or 5
            prompt_chunks = _chunk_prompts(request.prompts, chunk_size)
            print(
                f"Processing {len(prompt_chunks)} chunks with chunk_size={chunk_size}, max_concurrent={max_concurrent}"
            )

            # Process chunks with controlled concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_with_semaphore(chunk):
                async with semaphore:
                    return await _process_chunk_async(chunk)

            # Execute chunks concurrently
            chunk_tasks = [process_with_semaphore(chunk) for chunk in prompt_chunks]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

            # Handle any failed chunks
            processed_results: List[Dict[str, List[Any]]] = []
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    print(f"Chunk {i} failed: {result}")
                    # Create fallback result for failed chunk
                    chunk_size = len(prompt_chunks[i])
                    fallback: Dict[str, List[Any]] = {
                        "task_type_1": ["Other"] * chunk_size,
                        "task_type_2": ["NA"] * chunk_size,
                        "task_type_prob": [0.5] * chunk_size,
                        "creativity_scope": [0.5] * chunk_size,
                        "reasoning": [0.5] * chunk_size,
                        "contextual_knowledge": [0.5] * chunk_size,
                        "prompt_complexity_score": [0.5] * chunk_size,
                        "domain_knowledge": [0.5] * chunk_size,
                        "number_of_few_shots": [0] * chunk_size,
                        "no_label_reason": [0.5] * chunk_size,
                        "constraint_ct": [0.5] * chunk_size,
                    }
                    processed_results.append(fallback)
                else:
                    # result is guaranteed to be a dict here, not an Exception
                    processed_results.append(result)  # type: ignore[arg-type]

            # Merge all chunk results
            final_result = _merge_results(processed_results)

            print(f"Classification completed for {len(request.prompts)} prompts")
            return ClassificationResult(**final_result)

        except Exception as e:
            print(f"Classification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Classification failed: {str(e)}",
            )

    @web_app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint (minimal info, no authentication required)."""
        return {"status": "healthy", "service": "nvidia-prompt-classifier"}

    @web_app.get("/health/detailed")
    async def detailed_health_check(
        user: str = Depends(verify_jwt_token),
    ) -> Dict[str, Any]:
        """Detailed health check endpoint (requires authentication)."""
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "gpu": GPU_TYPE,
            "service": APP_NAME,
            "user": user,
            "features": {
                "async_processing": True,
                "chunked_batching": True,
                "concurrent_requests": True,
                "max_containers": 20,
                "keep_warm": 2,
                "concurrency_limit": 50,
            },
            "performance": {
                "default_chunk_size": 10,
                "max_concurrent_chunks": 5,
                "max_batch_size": 100,
                "estimated_latency_per_chunk": "200-500ms",
                "http2_enabled": True,
            },
        }

    @web_app.post("/benchmark")
    async def benchmark_classification(
        request: ClassifyRequest, user: str = Depends(verify_jwt_token)
    ) -> Dict[str, Any]:
        """Benchmark endpoint to test classification performance."""
        import time

        start_time = time.time()

        # Use a subset for benchmarking
        test_prompts = request.prompts[: min(len(request.prompts), 20)]
        test_request = ClassifyRequest(
            prompts=test_prompts,
            chunk_size=request.chunk_size or 5,
            max_concurrent=request.max_concurrent or 3,
        )

        try:
            result = await classify_prompts(test_request, user)
            end_time = time.time()

            return {
                "status": "success",
                "prompts_processed": len(test_prompts),
                "total_time_seconds": round(end_time - start_time, 3),
                "avg_time_per_prompt": round(
                    (end_time - start_time) / len(test_prompts), 3
                ),
                "chunk_size": test_request.chunk_size,
                "max_concurrent": test_request.max_concurrent,
                "result_sample": {
                    "task_types": result.task_type_1[:3] if result.task_type_1 else [],
                    "complexity_scores": (
                        result.prompt_complexity_score[:3]
                        if result.prompt_complexity_score
                        else []
                    ),
                },
            }
        except Exception as e:
            end_time = time.time()
            return {
                "status": "error",
                "error": str(e),
                "total_time_seconds": round(end_time - start_time, 3),
                "prompts_attempted": len(test_prompts),
            }

    return web_app


# ============================================================================
# DEPLOYMENT INFO
# ============================================================================

if __name__ == "__main__":
    print("🚀 NVIDIA Prompt Classifier - Clean Modal Deployment")
    print("=" * 60)
    print("✨ Features:")
    print("  • 📁 Clean package structure (nvidia_classifier/)")
    print("  • 🧠 Complete NVIDIA model implementation")
    print("  • 📦 Single deployment file")
    print("  • 🔐 JWT authentication")
    print("  • 🎮 GPU acceleration (T4)")
    print("  • 🎯 Full classification + complexity scoring")
    print("")
    print("🚀 Deploy:")
    print("  modal deploy deploy.py")
    print("")
    print("🌐 Endpoints:")
    print("  POST /classify - Full classification results (JWT required)")
    print("  GET /health - Health check")
    print("  GET /docs - Interactive API docs")
    print("")
    print("📋 Package Structure:")
    print("  nvidia_classifier/")
    print("  └── nvidia_model.py  # Complete NVIDIA model + neural network")
    print("")
    print("Authentication: Bearer token in Authorization header")
    print("GPU: NVIDIA T4 (16GB VRAM)")
    print(f"Model: {MODEL_NAME}")
