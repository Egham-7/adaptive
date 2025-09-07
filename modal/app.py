"""Modal deployment for NVIDIA prompt classifier with GPU acceleration and JWT authentication.

This module deploys the NVIDIA prompt-task-and-complexity-classifier model to Modal
with T4 GPU acceleration, batch processing, and JWT authentication for secure
communication with the adaptive_ai service.

Architecture:
- GPU-accelerated NVIDIA DeBERTa-v3-based classifier
- JWT authentication between adaptive_ai and Modal
- FastAPI endpoints with Bearer token validation
- Auto-scaling serverless inference

Usage:
    modal deploy app.py
    
API Endpoints:
    POST /classify - Classify prompts (requires JWT Bearer token)
    GET /health - Health check (no authentication)
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import modal

# Import FastAPI and other web dependencies inside Modal container functions
# ML imports moved inside Modal container functions to avoid local import issues

# ML imports moved inside Modal container functions to avoid local import issues

# Modal configuration
MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"

# Modal image with dependencies for NVIDIA classifier
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        # ML dependencies for NVIDIA classifier
        "torch>=2.2.0,<2.5.0",
        "transformers>=4.52.4,<5",
        "huggingface-hub>=0.32.0,<0.35", 
        "numpy>=1.24.0,<3.0",
        "accelerate>=1.8.1,<2",
        # Web framework and auth
        "fastapi>=0.100.0",
        "python-jose[cryptography]>=3.3.0,<4.0",
        "PyJWT>=2.8.0,<3.0",  # Alternative JWT library
        "pydantic>=2.11.5,<3",
    ])
)

app = modal.App("nvidia-prompt-classifier")


# JWT utility functions  
@app.function(image=image, secrets=[modal.Secret.from_name("jwt-auth")])
def verify_jwt_token_modal(token: str) -> str:
    """Verify JWT token using Modal secret."""
    try:
        # Try python-jose first
        from jose import JWTError, jwt
        jwt_lib = "jose"
    except ImportError:
        # Fallback to PyJWT
        import jwt
        from jwt import InvalidTokenError as JWTError
        jwt_lib = "PyJWT"
    
    try:
        # Get JWT secret from Modal environment (set via secrets)
        jwt_secret = os.environ.get("JWT_SECRET")
        if not jwt_secret:
            raise Exception("JWT secret not configured")
        
        # Decode and verify token
        if jwt_lib == "jose":
            payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        else:
            payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        
        # Extract user/service identifier
        user = payload.get("sub") or payload.get("user")
        if not user:
            raise Exception("Invalid token: missing subject")
            
        return user
        
    except JWTError as e:
        raise Exception(f"Invalid token: {str(e)}")
    except Exception as e:
        raise Exception(f"JWT verification failed: {str(e)}")


# Modal class for the NVIDIA prompt classifier
@app.cls(
    image=image,
    gpu=modal.gpu.T4(),  # T4 GPU sufficient for DeBERTa-v3-base
    container_idle_timeout=300,  # 5 minute idle timeout for cost optimization
    secrets=[modal.Secret.from_name("jwt-auth")],  # JWT secret for authentication
    allow_concurrent_inputs=10,  # Allow concurrent requests for better throughput
)
class NvidiaPromptClassifier:
    """Modal class for NVIDIA prompt classifier with GPU acceleration."""
    
    @modal.enter()
    def load_model(self):
        """Load the NVIDIA prompt classifier model on container startup."""
        # Import ML libraries inside Modal container
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from huggingface_hub import PyTorchModelHubMixin
        import numpy as np
        
        # Define neural network components inside container
        class MeanPooling(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                )
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                return mean_embeddings

        class MulticlassHead(nn.Module):
            def __init__(self, input_size: int, num_classes: int) -> None:
                super().__init__()
                self.fc = nn.Linear(input_size, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        class CustomModel(nn.Module, PyTorchModelHubMixin):
            def __init__(self, target_sizes: Dict[str, int], task_type_map: Dict[str, str], 
                         weights_map: Dict[str, List[float]], divisor_map: Dict[str, float]) -> None:
                super().__init__()
                self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base", use_safetensors=True)
                self.target_sizes = list(target_sizes.values())
                self.task_type_map = task_type_map
                self.weights_map = weights_map
                self.divisor_map = divisor_map
                
                self.heads = nn.ModuleList([
                    MulticlassHead(self.backbone.config.hidden_size, sz) for sz in self.target_sizes
                ])
                self.pool = MeanPooling()

            def compute_results(self, preds: torch.Tensor, target: str, decimal: int = 4):
                if target == "task_type":
                    top2_indices = torch.topk(preds, k=2, dim=1).indices
                    softmax_probs = torch.softmax(preds, dim=1)
                    top2_probs = softmax_probs.gather(1, top2_indices)
                    top2 = top2_indices.detach().cpu().tolist()
                    top2_prob = top2_probs.detach().cpu().tolist()
                    
                    top2_strings = [[self.task_type_map[str(idx)] for idx in sample] for sample in top2]
                    top2_prob_rounded = [[round(value, 3) for value in sublist] for sublist in top2_prob]
                    
                    for i, sublist in enumerate(top2_prob_rounded):
                        if sublist[1] < 0.1:
                            top2_strings[i][1] = "NA"
                            
                    task_type_1 = [sublist[0] for sublist in top2_strings]
                    task_type_2 = [sublist[1] for sublist in top2_strings]
                    task_type_prob = [sublist[0] for sublist in top2_prob_rounded]
                    
                    return (task_type_1, task_type_2, task_type_prob)
                else:
                    preds = torch.softmax(preds, dim=1)
                    weights = np.array(self.weights_map[target])
                    weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
                    scores = weighted_sum / self.divisor_map[target]
                    scores = [round(value, decimal) for value in scores]
                    
                    if target == "number_of_few_shots":
                        int_scores = [max(0, round(x)) for x in scores]
                        return int_scores
                    return scores

            def _extract_classification_results(self, logits: List[torch.Tensor]) -> Dict[str, List]:
                result = {}
                task_type_logits = logits[0]
                task_type_results = self.compute_results(task_type_logits, target="task_type")
                if isinstance(task_type_results, tuple):
                    result["task_type_1"] = task_type_results[0]
                    result["task_type_2"] = task_type_results[1]
                    result["task_type_prob"] = task_type_results[2]

                classifications = [
                    ("creativity_scope", logits[1]), ("reasoning", logits[2]),
                    ("contextual_knowledge", logits[3]), ("number_of_few_shots", logits[4]),
                    ("domain_knowledge", logits[5]), ("no_label_reason", logits[6]),
                    ("constraint_ct", logits[7]),
                ]

                for target, target_logits in classifications:
                    target_results = self.compute_results(target_logits, target=target)
                    if isinstance(target_results, list):
                        result[target] = target_results

                return result

            def _calculate_complexity_scores(self, results: Dict[str, List], task_types: List[str]) -> List[float]:
                task_type_weights = {
                    "Open QA": [0.2, 0.3, 0.15, 0.2, 0.15], "Closed QA": [0.1, 0.35, 0.2, 0.25, 0.1],
                    "Summarization": [0.2, 0.25, 0.25, 0.1, 0.2], "Text Generation": [0.4, 0.2, 0.15, 0.1, 0.15],
                    "Code Generation": [0.1, 0.3, 0.2, 0.3, 0.1], "Chatbot": [0.25, 0.25, 0.15, 0.1, 0.25],
                    "Classification": [0.1, 0.35, 0.25, 0.2, 0.1], "Rewrite": [0.2, 0.2, 0.3, 0.1, 0.2],
                    "Brainstorming": [0.5, 0.2, 0.1, 0.1, 0.1], "Extraction": [0.05, 0.3, 0.3, 0.15, 0.2],
                    "Other": [0.25, 0.25, 0.2, 0.15, 0.15],
                }

                complexity_scores = []
                for i, task_type in enumerate(task_types):
                    weights = task_type_weights.get(task_type, [0.3, 0.3, 0.2, 0.1, 0.1])
                    score = round(
                        weights[0] * results.get("creativity_scope", [])[i] +
                        weights[1] * results.get("reasoning", [])[i] +
                        weights[2] * results.get("constraint_ct", [])[i] +
                        weights[3] * results.get("domain_knowledge", [])[i] +
                        weights[4] * results.get("contextual_knowledge", [])[i], 5
                    )
                    complexity_scores.append(score)
                return complexity_scores

            def process_logits(self, logits: List[torch.Tensor]) -> Dict[str, List]:
                batch_results = self._extract_classification_results(logits)
                if "task_type_1" in batch_results:
                    task_types = batch_results["task_type_1"]
                    complexity_scores = self._calculate_complexity_scores(batch_results, task_types)
                    batch_results["prompt_complexity_score"] = complexity_scores
                return batch_results

            def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List]:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                mean_pooled_representation = self.pool(last_hidden_state, attention_mask)
                logits = [head(mean_pooled_representation) for head in self.heads]
                return self.process_logits(logits)

        # Store classes for use in methods
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
            target_sizes=self.config.target_sizes,
            task_type_map=self.config.task_type_map,
            weights_map=self.config.weights_map,
            divisor_map=self.config.divisor_map,
        ).from_pretrained(MODEL_NAME)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            print("Model loaded on CPU")
            
        self.model.eval()  # Set to evaluation mode
        print("NVIDIA prompt classifier loaded successfully!")

    @modal.method()
    def classify(self, prompts: List[str]) -> Dict[str, List]:
        """Classify a batch of prompts with GPU acceleration."""
        print(f"Received prompts: {prompts}")
        
        if prompts is None or len(prompts) == 0:
            raise ValueError("No prompts provided")
            
        print(f"Classifying {len(prompts)} prompts on GPU...")
        
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
            results = self.model(encoded_texts)

        print(f"GPU classification complete for {len(prompts)} prompts")
        return results


# Serve FastAPI app
@app.function(image=image, secrets=[modal.Secret.from_name("jwt-auth")])
@modal.asgi_app()
def serve():
    """Serve the FastAPI application."""
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from jose import JWTError, jwt
    from pydantic import BaseModel, Field
    
    # Create FastAPI app inside the function
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
        prompts: List[str] = Field(description="List of prompts to classify", min_items=1)

    def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
        """Verify JWT token from Authorization header using Modal function."""
        try:
            # Use Modal function to verify token with secret access
            user = verify_jwt_token_modal.remote(credentials.credentials)
            return user
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
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
        """Health check endpoint (no authentication required)."""
        return {
            "status": "healthy", 
            "model": MODEL_NAME,
            "gpu": "T4",
            "service": "nvidia-prompt-classifier"
        }

    return web_app


if __name__ == "__main__":
    print("NVIDIA Prompt Classifier Modal Deployment")
    print("=========================================")
    print("Use 'modal serve app.py' for development")
    print("Use 'modal deploy app.py' for production")
    print("")
    print("Endpoints:")
    print("  POST /classify - Classify prompts (JWT required)")
    print("  GET /health - Health check")
    print("")
    print("Authentication: Bearer token in Authorization header")
    print("GPU: NVIDIA T4 (16GB VRAM)")
    print(f"Model: {MODEL_NAME}")