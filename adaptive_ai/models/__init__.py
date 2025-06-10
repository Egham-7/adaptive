"""
Model definitions and configurations for the adaptive AI system.
"""

# Export all types
from .types import (
    ProviderType,
    TaskType,
    DifficultyLevel,
    TaskTypeParametersType,
    ModelCapability,
    ModelInfo,
    PromptScores,
    DifficultyThresholds,
    TaskDifficultyConfig,
    TaskModelMapping,
    ModelParameters,
    ModelSelectionError,
)

from .requests import ModelSelectionResponse


__all__ = [
    # Types
    "ProviderType",
    "TaskType",
    "DifficultyLevel",
    "TaskTypeParametersType",
    "ModelCapability",
    "ModelInfo",
    "PromptScores",
    "DifficultyThresholds",
    "TaskDifficultyConfig",
    "TaskModelMapping",
    "ModelParameters",
    "ModelSelectionError",
    "ModelSelectionResponse",
]
