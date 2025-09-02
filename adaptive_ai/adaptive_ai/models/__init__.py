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
    ModelEntry,
    ModelRouterConfig,
    ModelSelectionConfig,
    ModelSelectionRequest,
    TaskModelMapping,
)
from .llm_enums import (
    ProtocolType,
    ProviderType,
    TaskType,
)

# Note: llm_orchestration_models removed - orchestration models no longer needed

# Define what is exposed when a user imports * from this package or references the package
__all__ = [
    # Classification Models
    "ClassificationResult",
    # Core Models
    "ModelCapability",
    "ModelEntry",
    "ModelRouterConfig",
    "ModelSelectionConfig",
    "ModelSelectionRequest",
    # Enums
    "ProtocolType",
    "ProviderType",
    "TaskModelMapping",
    "TaskType",
]
