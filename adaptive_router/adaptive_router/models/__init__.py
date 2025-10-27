# __init__.py
"""
Models module for Adaptive AI.
"""

from .llm_core_models import (
    Alternative,
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from .routing_models import (
    ClusterMetadata,
    CodeQuestion,
    EvaluationResult,
    HealthResponse,
    MCQAnswer,
    ModelConfig,
    ModelFeatures,
    QuestionRoutingRequest,
    QuestionRoutingResponse,
    RoutingDecision,
)

__all__ = [
    "Alternative",
    "ModelCapability",
    "ModelSelectionRequest",
    "ModelSelectionResponse",
    "ClusterMetadata",
    "CodeQuestion",
    "EvaluationResult",
    "HealthResponse",
    "MCQAnswer",
    "ModelConfig",
    "ModelFeatures",
    "QuestionRoutingRequest",
    "QuestionRoutingResponse",
    "RoutingDecision",
]
