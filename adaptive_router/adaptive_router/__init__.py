"""Adaptive Router - Intelligent LLM model selection library.

This package provides intelligent model routing using cluster-based selection
with per-cluster error rates, cost optimization, and model capability matching.

Basic Usage:
    >>> from adaptive_router import ModelRouter, ModelSelectionRequest, MinIOSettings
    >>>
    >>> settings = MinIOSettings(
    ...     endpoint_url="https://minio.example.com",
    ...     root_user="admin",
    ...     root_password="password",
    ...     bucket_name="profiles"
    ... )
    >>> router = ModelRouter.from_minio(settings, model_costs)
    >>> request = ModelSelectionRequest(prompt="Write a Python function", cost_bias=0.5)
    >>> response = router.select_model(request)
    >>> print(f"Selected: {response.provider}/{response.model}")

Advanced Usage:
    >>> from adaptive_router import (
    ...     ClusterEngine,
    ...     FeatureExtractor,
    ...     LocalFileProfileLoader,
    ...     RouterProfile,
    ... )
    >>>
    >>> # Custom profile loading
    >>> loader = LocalFileProfileLoader(profile_path="custom_profile.json")
    >>> profile = loader.load_profile()
    >>> router = ModelRouter.from_profile(profile, model_costs)
"""

# ============================================================================
# TIER 1: Essential API (90% of users)
# ============================================================================

# Core routing service
from adaptive_router.core.router import ModelRouter

# Request/Response models
from adaptive_router.models.api import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)

# Storage configuration (needed for initialization)
from adaptive_router.models.storage import MinIOSettings

# ============================================================================
# TIER 2: Configuration & Integration
# ============================================================================

# Model types for routing
from adaptive_router.models.api import Model

# Profile loaders (for custom profile loading)
from adaptive_router.loaders import (
    LocalFileProfileLoader,
    MinIOProfileLoader,
    ProfileLoader,
)

# Storage types (profile structure)
from adaptive_router.models.storage import (
    ClusterCentersData,
    ProfileMetadata,
    RouterProfile,
    ScalerParameters,
    ScalerParametersData,
    TFIDFVocabularyData,
)

# Configuration types (YAML and routing config)
from adaptive_router.models.config import (
    ModelConfig,
    YAMLModelsConfig,
    YAMLRoutingConfig,
)

# ============================================================================
# TIER 3: Advanced API (Scripts, testing, custom implementations)
# ============================================================================

# Core ML components
from adaptive_router.core import (
    ClusterEngine,
    FeatureExtractor,
)

# Routing internals and public types
from adaptive_router.models.routing import (
    ModelFeatureVector,
    ModelFeatures,
    ModelInfo,
    ModelPricing,
    RoutingDecision,
)

# Evaluation and testing
from adaptive_router.models.evaluation import (
    CodeQuestion,
    EvaluationResult,
    MCQAnswer,
)

# Health check
from adaptive_router.models.health import HealthResponse

# ============================================================================
# Package metadata
# ============================================================================

__version__ = "0.1.0"

__all__ = [
    # ========================================================================
    # Tier 1: Essential API
    # ========================================================================
    "ModelRouter",
    "ModelSelectionRequest",
    "ModelSelectionResponse",
    "Alternative",
    "MinIOSettings",
    # ========================================================================
    # Tier 2: Configuration & Integration
    # ========================================================================
    # Model types
    "Model",
    # Loaders
    "ProfileLoader",
    "LocalFileProfileLoader",
    "MinIOProfileLoader",
    # Storage types
    "RouterProfile",
    "ProfileMetadata",
    "ClusterCentersData",
    "ScalerParameters",
    "ScalerParametersData",
    "TFIDFVocabularyData",
    # Configuration
    "ModelConfig",
    "YAMLModelsConfig",
    "YAMLRoutingConfig",
    # ========================================================================
    # Tier 2.5: Public Routing Types (clean API)
    # ========================================================================
    "ModelInfo",
    "ModelPricing",
    # ========================================================================
    # Tier 3: Advanced API
    # ========================================================================
    # Core components
    "ClusterEngine",
    "FeatureExtractor",
    # Routing internals
    "ModelFeatureVector",
    "ModelFeatures",
    "RoutingDecision",
    # Evaluation
    "CodeQuestion",
    "MCQAnswer",
    "EvaluationResult",
    # Health
    "HealthResponse",
]
