"""UniRouter - Cluster-based intelligent LLM routing.

This module implements the UniRouter algorithm for cost-aware model selection
using semantic clustering and per-cluster error rates.

Key Components:
    - UniRouter: Main routing class that selects models based on cluster assignment
    - ClusterEngine: K-means clustering engine for semantic question grouping
    - FeatureExtractor: Hybrid TF-IDF + embedding feature extraction
    - ModelConfig: Configuration for available models
    - RoutingDecision: Result of routing with selected model and alternatives

Example:
    >>> from adaptive_router.services.unirouter import UniRouter, ClusterEngine
    >>> cluster_engine = ClusterEngine(n_clusters=20)
    >>> cluster_engine.fit(training_questions)
    >>> router = UniRouter(cluster_engine, model_features, models)
    >>> decision = router.route("Write a Python sorting function", cost_preference=0.5)
    >>> print(f"Selected: {decision.selected_model_id}")

References:
    UniRouter algorithm: https://arxiv.org/abs/2502.08773
"""

from adaptive_router.services.unirouter.cluster_engine import ClusterEngine
from adaptive_router.services.unirouter.feature_extractor import FeatureExtractor
from adaptive_router.services.unirouter.router import UniRouter
from adaptive_router.services.unirouter.schemas import (
    CodeQuestion,
    ModelConfig,
    RoutingDecision,
)

__all__ = [
    "ClusterEngine",
    "FeatureExtractor",
    "UniRouter",
    "CodeQuestion",
    "ModelConfig",
    "RoutingDecision",
]
