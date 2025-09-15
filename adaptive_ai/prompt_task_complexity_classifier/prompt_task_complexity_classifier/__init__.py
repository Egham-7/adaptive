"""Prompt Task Complexity Classifier Package.

Clean package containing the prompt task complexity classifier model implementation
for Modal deployment.

Modules:
    - task_complexity_model: Complete ML model implementation with GPU acceleration
    - config: Configuration management
    - models: Pydantic models for API requests and responses
    - prompt_classifier: Main classifier service for inference
"""

__version__ = "1.0.0"

# Import all main components for easy access
from .task_complexity_model import CustomModel, MeanPooling, MulticlassHead
from .prompt_classifier import PromptClassifier, get_prompt_classifier
from .config import (
    ClassifierConfig,
    ServiceConfig,
    UserTestConfig,
    DeploymentConfig,
    get_config,
)
from .models import ClassificationResult, ClassifyRequest, ClassifyBatchRequest
from .utils import verify_jwt_token

# Export available components
__all__ = [
    # Model components
    "CustomModel",
    "MeanPooling",
    "MulticlassHead",
    # Main classifier
    "PromptClassifier",
    "get_prompt_classifier",
    # Configuration
    "ClassifierConfig",
    "ServiceConfig",
    "UserTestConfig",
    "DeploymentConfig",
    "get_config",
    # Pydantic models
    "ClassificationResult",
    "ClassifyRequest",
    "ClassifyBatchRequest",
    # Utilities
    "verify_jwt_token",
]
