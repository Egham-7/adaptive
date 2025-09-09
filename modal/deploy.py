"""Clean Modal deployment for NVIDIA prompt classifier using organized package structure.

This deployment showcases a well-organized Modal application with:
- Separated concerns (model logic, API endpoints, utilities)
- Whole folder deployment using add_local_dir
- Professional package structure
- Clear separation between ML and web components
"""

import os
from typing import Dict, List
import modal

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"
GPU_TYPE = "T4"
APP_NAME = "nvidia-prompt-classifier"

# ============================================================================
# MODAL IMAGES WITH ORGANIZED PACKAGE
# ============================================================================

# ML image for NVIDIA model inference
ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.2.0,<2.5.0",
        "transformers>=4.52.4,<5",
        "huggingface-hub>=0.32.0,<0.35", 
        "numpy>=1.24.0,<3.0",
        "accelerate>=1.8.1,<2",
    ])
    .add_local_dir("nvidia_classifier", remote_path="/root/nvidia_classifier")  # Deploy whole folder
)

# Web image for FastAPI endpoints
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi>=0.110.0",
        "pydantic>=2.5.0", 
        "PyJWT>=2.8.0",
        "numpy>=1.24.0,<3.0",  # Needed for deserializing results from ML container
    ])
    .add_local_dir("nvidia_classifier", remote_path="/root/nvidia_classifier")  # Deploy whole folder
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
    max_containers=10,
    scaledown_window=300,
)
class NvidiaPromptClassifier:
    """NVIDIA prompt classifier using consolidated model components."""
    
    @modal.enter()
    def load_model(self):
        """Load model using consolidated components from nvidia_model."""
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from nvidia_classifier.nvidia_model import get_model_classes
        
        # Get model classes (imports torch inside the container)
        _, _, CustomModelClass = get_model_classes()
        
        print(f"🚀 Loading NVIDIA classifier: {MODEL_NAME}")
        print(f"🎮 GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        
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
)
@modal.asgi_app()
def serve():
    """Serve the FastAPI application."""
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
    import jwt
    
    # Create FastAPI app
    web_app = FastAPI(
        title="NVIDIA Prompt Classifier API", 
        version="1.0.0",
        description="GPU-accelerated prompt classification with JWT authentication"
    )
    security = HTTPBearer()

    # Pydantic models for API
    class ClassificationResult(BaseModel):
        """Results from prompt classification including task type and complexity metrics."""
        
        task_type_1: List[str] = Field(description="Primary task type for each prompt")
        task_type_2: List[str] = Field(description="Secondary task type for each prompt") 
        task_type_prob: List[float] = Field(description="Confidence scores for primary task types")
        creativity_scope: List[float] = Field(description="Creativity level required (0-1)")
        reasoning: List[float] = Field(description="Reasoning complexity required (0-1)")
        contextual_knowledge: List[float] = Field(description="Context knowledge requirement (0-1)")
        prompt_complexity_score: List[float] = Field(description="Overall prompt complexity (0-1)")
        domain_knowledge: List[float] = Field(description="Domain-specific knowledge requirement (0-1)")
        number_of_few_shots: List[int] = Field(description="Few-shot learning requirement")
        no_label_reason: List[float] = Field(description="Confidence in classification accuracy (0-1)")
        constraint_ct: List[float] = Field(description="Constraint complexity detected (0-1)")

    class ClassifyRequest(BaseModel):
        """Request model for prompt classification."""
        prompts: List[str] = Field(description="List of prompts to classify", min_length=1)

    def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
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
                options={"require_exp": True, "require_sub": True}  # Require expiration and subject
            )
            
            user = payload.get("sub") or payload.get("user")
            if not user:
                raise Exception("Invalid token: missing subject")
            return user
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )

    @web_app.post("/classify", response_model=ClassificationResult)
    async def classify_prompts(
        request: ClassifyRequest, 
        user: str = Depends(verify_jwt_token)
    ):
        """Classify prompts using NVIDIA model (requires JWT authentication)."""
        print(f"Classification request from user: {user}, prompts: {len(request.prompts)}")
        
        # Run classification using Modal method
        result = NvidiaPromptClassifier().classify.remote(request.prompts)
        
        print(f"Classification completed for {len(request.prompts)} prompts")
        return ClassificationResult(**result)

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint (minimal info, no authentication required)."""
        return {
            "status": "healthy",
            "service": "nvidia-prompt-classifier"
        }
        
    @web_app.get("/health/detailed")
    async def detailed_health_check(user: str = Depends(verify_jwt_token)):
        """Detailed health check endpoint (requires authentication)."""
        return {
            "status": "healthy", 
            "model": MODEL_NAME,
            "gpu": GPU_TYPE,
            "service": APP_NAME,
            "user": user
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
    print("  ├── nvidia_model.py  # Complete NVIDIA model + neural network")
    print("  └── utils/           # Constants and configuration")
    print("")
    print("Authentication: Bearer token in Authorization header")
    print("GPU: NVIDIA T4 (16GB VRAM)")
    print(f"Model: {MODEL_NAME}")