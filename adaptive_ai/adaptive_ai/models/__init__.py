# __init__.py
"""
Models module for Adaptive AI.

This module contains data models, request/response schemas,
and type definitions used throughout the application.
"""

# Import all enums that define standard categories
# Import data models specific to prompt classification results
from .llm_classification_models import ClassificationResult

# Import core data models for model capabilities, selection logic, and general requests/results
from .llm_core_models import (
    ModelCapability,
    ModelSelectionConfig,
    ModelSelectionRequest,
    TaskModelEntry,
    TaskModelMapping,
)
from .llm_enums import (
    DifficultyLevel,
    ProtocolType,
    ProviderType,
    TaskType,
)

# Import data models related to the orchestrator's response structure
from .llm_orchestration_models import (
    Alternative,
    MinionInfo,
    OpenAIParameters,
    OrchestratorResponse,
    StandardLLMInfo,
)

# Define what is exposed when a user imports * from this package or references the package
__all__ = [
    "Alternative",
    # Classification Models
    "ClassificationResult",
    "DifficultyLevel",
    "MinionInfo",
    # Core Models
    "ModelCapability",
    "ModelSelectionConfig",
    "ModelSelectionRequest",
    # Orchestration Models
    "OpenAIParameters",
    "OrchestratorResponse",
    "ProtocolType",
    # Enums
    "ProviderType",
    "StandardLLMInfo",
    "TaskModelEntry",  # NEW: Export TaskModelEntry
    "TaskModelMapping",
    "TaskType",
]
