"""Modal deployment for NVIDIA prompt classifier with GPU acceleration and JWT authentication.

This module deploys the NVIDIA prompt-task-and-complexity-classifier model to Modal
with T4 GPU acceleration, batch processing, and JWT authentication for secure
communication with the adaptive_ai service.
"""

import os
from typing import Dict, List

import modal

# Modal configuration
MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"

# Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        # ML dependencies
        "torch>=2.2.0,<2.5.0",
        "transformers>=4.52.4,<5",
        "huggingface-hub>=0.32.0,<0.35", 
        "numpy>=1.24.0,<3.0",
        "accelerate>=1.8.1,<2",
        # Web framework and auth
        "fastapi>=0.110.0",
        "pydantic>=2.5.0",
        "PyJWT>=2.8.0",
    ])
)

app = modal.App("nvidia-prompt-classifier")


@app.cls(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("jwt-auth")],
    concurrency_limit=10,
    container_idle_timeout=300,
)
class NvidiaPromptClassifier:
    """Modal class for NVIDIA prompt classifier with GPU acceleration."""
    
    @modal.enter()
    def load_model(self):
        """Load the NVIDIA prompt classifier model on container startup."""
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from huggingface_hub import PyTorchModelHubMixin
        import numpy as np
        
        # Model architecture classes
        class MeanPooling(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask

        class MulticlassHead(nn.Module):
            def __init__(self, input_size: int, num_classes: int) -> None:
                super().__init__()
                self.fc = nn.Linear(input_size, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        class CustomModel(nn.Module, PyTorchModelHubMixin):
            def __init__(self, target_sizes: Dict[str, int]) -> None:
                super().__init__()
                self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base", use_safetensors=True)
                self.target_sizes = list(target_sizes.values())
                
                self.heads = nn.ModuleList([
                    MulticlassHead(self.backbone.config.hidden_size, sz) for sz in self.target_sizes
                ])
                self.pool = MeanPooling()

            def forward(self, batch: Dict[str, torch.Tensor]) -> List[List[List[float]]]:
                """Forward pass returning raw logits only."""
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                mean_pooled_representation = self.pool(last_hidden_state, attention_mask)
                logits = [head(mean_pooled_representation) for head in self.heads]
                
                # Convert to regular Python lists for JSON serialization
                logits_list = []
                for logit_tensor in logits:
                    logit_list = logit_tensor.detach().cpu().tolist()
                    logits_list.append(logit_list)
                
                return logits_list

        # Store references
        self.torch = torch
        self.CustomModel = CustomModel
        
        print("Loading NVIDIA prompt classifier...")
        print(f"Model: {MODEL_NAME}")
        print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        
        # Load model configuration and tokenizer
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Create and load custom model
        self.model = CustomModel(
            target_sizes=self.config.target_sizes
        ).from_pretrained(MODEL_NAME)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            print("Model loaded on CPU")
            
        self.model.eval()
        print("NVIDIA prompt classifier loaded successfully!")

    @modal.method()
    def classify_raw(self, prompts: List[str]) -> List[List[List[float]]]:
        """Classify a batch of prompts and return raw logits."""
        if not prompts:
            raise ValueError("No prompts provided")
            
        print(f"Getting raw logits for {len(prompts)} prompts...")
        
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

        # Run inference
        with self.torch.no_grad():
            raw_logits = self.model(encoded_texts)

        print(f"Raw logits generated for {len(prompts)} prompts")
        return raw_logits


# Serve FastAPI app
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("jwt-auth")],
)
@modal.asgi_app()
def serve():
    """Serve the FastAPI application."""
    import jwt
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
    
    # Create FastAPI app
    web_app = FastAPI(
        title="NVIDIA Prompt Classifier API", 
        version="1.0.0",
        description="GPU-accelerated prompt classification with JWT authentication"
    )
    security = HTTPBearer()

    # Pydantic models
    class ClassifyRequest(BaseModel):
        """Request model for prompt classification."""
        prompts: List[str] = Field(description="List of prompts to classify", min_items=1)

    class RawLogitsResponse(BaseModel):
        """Response model for raw logits."""
        logits: List[List[List[float]]] = Field(description="Raw model logits for classification")

    def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
        """Verify JWT token from Authorization header."""
        token = credentials.credentials
        secret = os.environ.get("JWT_SECRET")
        
        if not secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT secret not configured"
            )
        
        try:
            payload = jwt.decode(token, secret, algorithms=["HS256"])
            user = payload.get("sub") or payload.get("user", "unknown")
            return user
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint (no authentication required)."""
        return {
            "status": "healthy", 
            "model": MODEL_NAME,
            "gpu": "T4",
            "service": "nvidia-prompt-classifier"
        }

    @web_app.post("/classify_raw", response_model=RawLogitsResponse)
    async def classify_prompts_raw(
        request: ClassifyRequest, 
        user: str = Depends(verify_jwt_token)
    ):
        """Get raw logits from NVIDIA model (requires JWT authentication)."""
        print(f"Raw logits request from user: {user}, prompts: {len(request.prompts)}")
        
        # Run classification to get raw logits
        classifier = NvidiaPromptClassifier()
        raw_logits = classifier.classify_raw.remote(request.prompts)
        
        print(f"Raw logits generated for {len(request.prompts)} prompts")
        return RawLogitsResponse(logits=raw_logits)

    return web_app


if __name__ == "__main__":
    print("NVIDIA Prompt Classifier Modal Deployment")
    print("=========================================")
    print("Use 'modal serve app.py' for development")
    print("Use 'modal deploy app.py' for production")
    print("")
    print("Endpoints:")
    print("  POST /classify_raw - Get raw model logits (JWT required)")
    print("  GET /health - Health check")
    print("")
    print("Authentication: Bearer token in Authorization header")
    print("GPU: NVIDIA T4")
    print(f"Model: {MODEL_NAME}")