"""Intelligent model routing using cluster-based selection.

This module provides the ModelRouter class for selecting optimal LLM models
based on cluster-specific error rates, cost optimization, and model capabilities.

All routing logic is consolidated here: MinIO loading, cluster-based routing,
and response conversion.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from adaptive_router.core.storage_config import MinIOSettings
from adaptive_router.models.llm_core_models import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models import ModelConfig, RoutingDecision
from adaptive_router.services.cluster_engine import ClusterEngine
from adaptive_router.services.storage_profile_loader import StorageProfileLoader

logger = logging.getLogger(__name__)

# Cost estimation constants
_ESTIMATED_TOKEN_COUNT = 2000  # Assumed average token count for cost estimation
_TOKENS_PER_MILLION = 1_000_000  # Tokens per million for pricing calculations
_EPSILON = 1e-10  # Small value for floating point comparisons


class ModelRouter:
    """Intelligent model routing using cluster-based selection.

    Selects optimal LLM models based on cluster-specific error rates,
    cost optimization, and model capability matching.

    Loads cluster profiles from MinIO S3 storage and performs intelligent
    routing using the UniRouter algorithm with per-cluster error rates.
    """

    def __init__(
        self,
        config_file: Path | None = None,
        router_service: Any = None,  # For backwards compatibility with tests
    ) -> None:
        """Initialize ModelRouter.

        Args:
            config_file: Path to Router models YAML config. If None, uses default.
            router_service: Deprecated. Kept for backwards compatibility with tests.
                           If provided, uses its internal router instead of loading.
        """
        # Backwards compatibility: if router_service provided (from tests), use it
        if router_service is not None:
            self._router = router_service.router
            logger.info("ModelRouter initialized with provided router_service (test mode)")
            return

        # Auto-detect config file path
        if config_file is None:
            service_dir = Path(__file__).parent.parent  # adaptive_router/
            config_file = service_dir / "config" / "unirouter_models.yaml"

        self.config_file = Path(config_file)

        # Load Router from MinIO
        self._router = self._load_router()
        logger.info(
            f"ModelRouter initialized with {len(self._router.models)} models, "
            f"K={self._router.cluster_engine.n_clusters} clusters"
        )

    def _load_router(self) -> _Router:
        """Load Router from MinIO S3 storage (Railway deployment).

        This method loads all profile data from the MinIO bucket:
        - Cluster centers (K-means centroids)
        - Model error rates per cluster
        - TF-IDF vocabulary
        - Scaler parameters

        Returns:
            Initialized _Router instance

        Raises:
            FileNotFoundError: If profile not found in MinIO
            ValueError: If MinIO configuration is invalid or profile data is corrupted
        """
        logger.info("Loading Router profile from MinIO storage...")

        # Load MinIO settings (Railway native)
        try:
            # Pydantic Settings loads from environment variables automatically
            minio_settings = MinIOSettings()  # type: ignore[call-arg]
            storage_loader = StorageProfileLoader.from_minio_settings(minio_settings)
            logger.info(
                f"MinIO storage configured: bucket={minio_settings.bucket_name}, "
                f"endpoint={minio_settings.endpoint_url}"
            )
        except Exception as e:
            raise ValueError(
                f"MinIO configuration error. Required environment variables:\n"
                f"  - S3_BUCKET_NAME (required)\n"
                f"  - MINIO_PUBLIC_ENDPOINT (required, e.g., https://minio.railway.app)\n"
                f"  - MINIO_ROOT_USER (required)\n"
                f"  - MINIO_ROOT_PASSWORD (required)\n"
                f"\n"
                f"Error: {e}"
            )

        # Load profile from storage
        profile_data = storage_loader.load_global_profile()

        # Extract components from profile
        cluster_centers_data = profile_data["cluster_centers"]
        model_profiles = profile_data["llm_profiles"]
        tfidf_data = profile_data["tfidf_vocabulary"]
        scaler_data = profile_data["scaler_parameters"]
        metadata = profile_data["metadata"]

        n_clusters = metadata["n_clusters"]
        logger.info(f"Loaded profile from MinIO: {n_clusters} clusters")

        # Build cluster engine from MinIO data
        cluster_engine = self._build_cluster_engine_from_data(
            cluster_centers_data=cluster_centers_data,
            tfidf_data=tfidf_data,
            scaler_data=scaler_data,
            metadata=metadata,
        )

        logger.info(
            f"Loaded cluster engine from storage: {n_clusters} clusters, "
            f"silhouette score: {metadata.get('silhouette_score', 'N/A')}"
        )

        # Load models config
        with open(self.config_file) as f:
            models_config = yaml.safe_load(f)

        # Parse models
        models = [ModelConfig(**m) for m in models_config["gpt5_models"]]

        # Prepare model_features dict combining error rates and cost
        model_features = {}
        for model in models:
            model_id = model.id
            model_name = model.name

            # Try to find profile by ID first, then by name
            profile = model_profiles.get(model_id) or model_profiles.get(model_name)

            if profile:
                model_features[model_id] = {
                    "error_rates": profile,
                    "cost_per_1m_tokens": model.cost_per_1m_tokens,
                }
                logger.debug(f"Loaded profile for {model_id}")
            else:
                logger.warning(
                    f"Model {model_id} ({model_name}) not in profiles, skipping"
                )

        if not model_features:
            raise ValueError("No valid model features found in llm_profiles")

        # Get routing config
        routing_config = models_config.get("routing", {})

        # Initialize Router
        router = _Router(
            cluster_engine=cluster_engine,
            model_features=model_features,
            models=models,
            lambda_min=routing_config.get("lambda_min", 0.0),
            lambda_max=routing_config.get("lambda_max", 1.0),
            default_cost_preference=routing_config.get("default_cost_preference", 0.5),
        )

        logger.info(
            f"Router initialized from storage: {len(models)} models, "
            f"lambda range [{router.lambda_min}, {router.lambda_max}]"
        )

        return router

    def _build_cluster_engine_from_data(
        self,
        cluster_centers_data: dict,
        tfidf_data: dict,
        scaler_data: dict,
        metadata: dict,
    ) -> ClusterEngine:
        """Build ClusterEngine from storage profile data.

        Args:
            cluster_centers_data: Cluster centers and assignments
            tfidf_data: TF-IDF vocabulary and IDF scores
            scaler_data: StandardScaler parameters
            metadata: Metadata dictionary with config info

        Returns:
            Reconstructed ClusterEngine
        """
        import platform

        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import StandardScaler

        from adaptive_router.services.feature_extractor import FeatureExtractor

        # Determine device (CPU for macOS, CUDA if available otherwise)
        device = (
            "cpu"
            if platform.system() == "Darwin"
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Loading embedding model on device: {device}")

        # Create fresh SentenceTransformer model
        embedding_model_name = metadata["embedding_model"]
        embedding_model = SentenceTransformer(
            embedding_model_name, device=device, trust_remote_code=True
        )

        # Create FeatureExtractor with fresh model
        feature_extractor = FeatureExtractor(
            embedding_model=embedding_model_name,
            tfidf_max_features=metadata.get("tfidf_max_features", 5000),
            tfidf_ngram_range=tuple(metadata.get("tfidf_ngram_range", [1, 2])),
        )

        # Replace the embedding model (already loaded above)
        feature_extractor.embedding_model = embedding_model

        # Restore TF-IDF vocabulary
        feature_extractor.tfidf_vectorizer.vocabulary_ = tfidf_data["vocabulary"]
        feature_extractor.tfidf_vectorizer.idf_ = np.array(tfidf_data["idf"])

        # Restore scaler parameters
        logger.info("Restoring scaler parameters from MinIO data...")

        # Restore embedding scaler
        feature_extractor.embedding_scaler = StandardScaler()
        feature_extractor.embedding_scaler.mean_ = np.array(
            scaler_data["embedding_scaler"]["mean"]
        )
        feature_extractor.embedding_scaler.scale_ = np.array(
            scaler_data["embedding_scaler"]["scale"]
        )
        feature_extractor.embedding_scaler.n_features_in_ = len(
            feature_extractor.embedding_scaler.mean_
        )

        # Restore TF-IDF scaler
        feature_extractor.tfidf_scaler = StandardScaler()
        feature_extractor.tfidf_scaler.mean_ = np.array(
            scaler_data["tfidf_scaler"]["mean"]
        )
        feature_extractor.tfidf_scaler.scale_ = np.array(
            scaler_data["tfidf_scaler"]["scale"]
        )
        feature_extractor.tfidf_scaler.n_features_in_ = len(
            feature_extractor.tfidf_scaler.mean_
        )
        logger.info("✅ Scaler parameters restored")

        feature_extractor.is_fitted = True

        # Create ClusterEngine
        cluster_engine = ClusterEngine(
            n_clusters=cluster_centers_data["n_clusters"],
            embedding_model=embedding_model_name,
            tfidf_max_features=metadata.get("tfidf_max_features", 5000),
            tfidf_ngram_range=tuple(metadata.get("tfidf_ngram_range", [1, 2])),
        )
        cluster_engine.feature_extractor = feature_extractor

        # Restore K-means cluster centers
        cluster_engine.kmeans.cluster_centers_ = np.array(
            cluster_centers_data["cluster_centers"]
        )
        # Set required K-means attributes
        cluster_engine.kmeans._n_threads = 1
        cluster_engine.kmeans.n_iter_ = 0  # Already fitted

        cluster_engine.is_fitted = True
        cluster_engine.silhouette = metadata.get("silhouette_score", 0.0)

        logger.info(
            f"Built cluster engine from storage data: {cluster_centers_data['n_clusters']} clusters, "
            f"{cluster_centers_data['feature_dim']} features"
        )

        return cluster_engine

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis.

        This is the main public API method. Uses cluster-based routing to select
        the best model based on prompt characteristics, cost preferences, and
        historical per-cluster error rates.

        Args:
            request: ModelSelectionRequest with prompt and optional models/cost_bias

        Returns:
            ModelSelectionResponse with selected provider, model, and alternatives

        Raises:
            ValueError: If requested models are not supported or validation fails
            RuntimeError: If routing fails
        """
        # Extract and validate allowed models if provided
        allowed_model_ids: List[str] | None = None
        if request.models:
            supported = self.get_supported_models()

            # Build list of requested model IDs
            # Note: For filtering, we only need provider:model_name, even if other fields are missing
            requested = []
            for m in request.models:
                if m.provider and m.model_name:
                    # Construct ID in the same format as router uses
                    model_id = f"{m.provider.lower()}:{m.model_name.lower()}"
                    requested.append(model_id)

            # Validate all requested models are supported
            unsupported = [m for m in requested if m not in supported]
            if unsupported:
                raise ValueError(
                    f"Models not supported by Router: {unsupported}. "
                    f"Supported models: {supported}"
                )

            # Pass the validated model IDs to router
            allowed_model_ids = requested

        # Map cost_bias (0.0=cheap, 1.0=quality) to cost_preference
        cost_preference = request.cost_bias if request.cost_bias is not None else None

        # Route the question using internal Router
        decision = self._router.route(
            question_text=request.prompt,
            cost_preference=cost_preference,
            allowed_models=allowed_model_ids,
        )

        # Parse model ID to extract provider and model name
        # Format: "openai:gpt-5-mini" -> provider="openai", model="gpt-5-mini"
        if ":" not in decision.selected_model_id:
            raise ValueError(
                f"Invalid model ID format: {decision.selected_model_id}. "
                f"Expected format: 'provider:model_name' (e.g., 'openai:gpt-4')"
            )

        selected_model_parts = decision.selected_model_id.split(":", 1)
        if len(selected_model_parts) != 2 or not all(selected_model_parts):
            raise ValueError(
                f"Invalid model ID format: {decision.selected_model_id}. "
                f"Both provider and model_name must be non-empty. "
                f"Expected format: 'provider:model_name'"
            )

        provider, model_name = selected_model_parts

        # Convert alternatives to Alternative objects
        alternatives_list = []
        for alt in decision.alternatives[:3]:  # Limit to top 3 alternatives
            alt_model_id = alt["model_id"]
            alt_parts = alt_model_id.split(":", 1)
            if len(alt_parts) == 2:
                alternatives_list.append(
                    Alternative(provider=alt_parts[0], model=alt_parts[1])
                )

        # Convert RoutingDecision to ModelSelectionResponse
        response = ModelSelectionResponse(
            provider=provider,
            model=model_name,
            alternatives=alternatives_list,
        )

        logger.info(
            f"Selected model: {provider}/{model_name} "
            f"(cluster {decision.cluster_id}, "
            f"accuracy {decision.predicted_accuracy:.2%}, "
            f"score {decision.routing_score:.3f})"
        )

        return response

    def get_supported_models(self) -> List[str]:
        """Get list of models this router supports.

        Returns:
            List of model IDs in format "provider:model_name"
        """
        return list(self._router.models.keys())

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about loaded clusters.

        Returns:
            Dictionary with cluster statistics including n_clusters,
            embedding_model, supported_models, and lambda parameters
        """
        return {
            "n_clusters": self._router.cluster_engine.n_clusters,
            "embedding_model": self._router.cluster_engine.feature_extractor.embedding_model_name,
            "supported_models": self.get_supported_models(),
            "lambda_min": self._router.lambda_min,
            "lambda_max": self._router.lambda_max,
            "default_cost_preference": self._router.default_cost_preference,
        }


class _Router:
    """Internal routing engine using cluster-based error rates.

    This is a private implementation class. Use ModelRouter instead.
    """

    def __init__(
        self,
        cluster_engine: ClusterEngine,
        model_features: Dict[str, Any],
        models: List[ModelConfig],
        lambda_min: float = 0.0,
        lambda_max: float = 1.0,
        default_cost_preference: float = 0.5,
    ) -> None:
        """Initialize internal Router.

        Args:
            cluster_engine: Fitted ClusterEngine for question assignment
            model_features: Model feature vectors (error rates + cost)
            models: List of available models
            lambda_min: Minimum lambda parameter
            lambda_max: Maximum lambda parameter
            default_cost_preference: Default cost-quality trade-off (0.0-1.0)
        """
        self.cluster_engine = cluster_engine
        self.model_features = model_features
        self.models = {m.id: m for m in models}

        # Lambda parameter range [0.0, 1.0]
        # 0.0 = no cost penalty (pure quality)
        # 1.0 = equal weight to error rate and cost (both normalized)
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.default_cost_preference = default_cost_preference

        # Pre-compute normalized costs
        all_costs = [f["cost_per_1m_tokens"] for f in model_features.values()]
        self.min_cost = min(all_costs)
        self.max_cost = max(all_costs)

        logger.info(f"Internal Router initialized with {len(models)} models")

    def route(
        self,
        question_text: str,
        cost_preference: float | None = None,
        allowed_models: List[str] | None = None,
    ) -> RoutingDecision:
        """Route a question to the optimal model.

        Args:
            question_text: Question to route
            cost_preference: 0.0=cheap, 1.0=quality (default from config)
            allowed_models: Optional list of model IDs to restrict routing to.
                           If None, considers all available models.

        Returns:
            RoutingDecision with selected model and reasoning
        """
        start_time = time.time()

        # Use default if not specified
        if cost_preference is None:
            cost_preference = self.default_cost_preference

        # 1. Assign to cluster
        cluster_id, distance = self.cluster_engine.assign_question(question_text)

        # 2. Calculate lambda parameter
        lambda_param = self._calculate_lambda(cost_preference)

        # 3. Determine which models to consider
        if allowed_models is not None:
            # Filter to only allowed models
            allowed_set = set(allowed_models)
            models_to_score = {
                model_id: features
                for model_id, features in self.model_features.items()
                if model_id in allowed_set
            }

            if not models_to_score:
                raise ValueError(
                    f"No valid models found in allowed list. "
                    f"Allowed: {allowed_models}, Available: {list(self.model_features.keys())}"
                )
        else:
            # Use all available models
            models_to_score = self.model_features

        # 4. Compute routing scores for each model
        model_scores = {}

        for model_id, features in models_to_score.items():
            error_rate = features["error_rates"][cluster_id]
            cost = features["cost_per_1m_tokens"]

            normalized_cost = self._normalize_cost(cost)
            score = error_rate + lambda_param * normalized_cost

            model_scores[model_id] = {
                "score": score,
                "error_rate": error_rate,
                "accuracy": 1.0 - error_rate,
                "cost": cost,
                "normalized_cost": normalized_cost,
            }

        # 5. Select best model (lowest score)
        best_model_id = min(model_scores, key=lambda k: model_scores[k]["score"])
        best_scores = model_scores[best_model_id]

        # 6. Prepare decision
        model = self.models[best_model_id]

        routing_time = (time.time() - start_time) * 1000

        # Generate reasoning
        reasoning = self._generate_reasoning(
            cluster_id=cluster_id,
            cost_preference=cost_preference,
            lambda_param=lambda_param,
            selected_scores=best_scores,
        )

        # Prepare alternatives
        alternatives = [
            {
                "model_id": mid,
                "model_name": self.models[mid].name,
                "score": scores["score"],
                "accuracy": scores["accuracy"],
                "cost": scores["cost"],
            }
            for mid, scores in sorted(
                model_scores.items(), key=lambda x: x[1]["score"]
            )
            if mid != best_model_id
        ]

        return RoutingDecision(
            selected_model_id=best_model_id,
            selected_model_name=model.name,
            routing_score=best_scores["score"],
            predicted_accuracy=best_scores["accuracy"],
            estimated_cost=best_scores["cost"]
            * _ESTIMATED_TOKEN_COUNT
            / _TOKENS_PER_MILLION,
            cluster_id=cluster_id,
            cluster_confidence=1.0 / (1.0 + distance),  # Convert distance to confidence
            lambda_param=lambda_param,
            reasoning=reasoning,
            alternatives=alternatives,
            routing_time_ms=routing_time,
        )

    def _calculate_lambda(self, cost_preference: float) -> float:
        """Calculate lambda parameter.

        Args:
            cost_preference: 0.0=cheap, 1.0=quality

        Returns:
            Lambda parameter (higher = more cost penalty)
        """
        # Invert: high quality preference = low lambda (cost matters less)
        lambda_param = self.lambda_max - cost_preference * (
            self.lambda_max - self.lambda_min
        )

        return lambda_param

    def _normalize_cost(self, cost: float) -> float:
        """Normalize cost to [0, 1] range.

        Args:
            cost: Model cost per 1M tokens

        Returns:
            Normalized cost
        """
        cost_range = self.max_cost - self.min_cost
        if cost_range < _EPSILON:
            return 0.0

        return float((cost - self.min_cost) / cost_range)

    def _generate_reasoning(
        self,
        cluster_id: int,
        cost_preference: float,
        lambda_param: float,
        selected_scores: Dict[str, Any],
    ) -> str:
        """Generate human-readable reasoning for routing decision.

        Args:
            cluster_id: Assigned cluster
            cost_preference: User's cost preference
            lambda_param: Calculated lambda
            selected_scores: Scores for selected model

        Returns:
            Reasoning string
        """
        parts = []

        # Cluster info
        parts.append(f"Question assigned to cluster {cluster_id}")

        # Preference info
        if cost_preference < 0.3:
            parts.append(f"Cost-optimized routing (λ={lambda_param:.2f})")
        elif cost_preference < 0.7:
            parts.append(f"Balanced cost-accuracy routing (λ={lambda_param:.2f})")
        else:
            parts.append(f"Quality-optimized routing (λ={lambda_param:.2f})")

        # Performance info
        accuracy = selected_scores["accuracy"]
        if accuracy >= 0.95:
            parts.append(f"Excellent predicted accuracy ({accuracy:.0%})")
        elif accuracy >= 0.75:
            parts.append(f"Strong predicted accuracy ({accuracy:.0%})")
        else:
            parts.append(f"Best available option ({accuracy:.0%} predicted)")

        return "; ".join(parts)
