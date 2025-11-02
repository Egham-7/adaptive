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
from typing import Any, Dict, List

import numpy as np

from adaptive_router.loaders.local import LocalFileProfileLoader
from adaptive_router.loaders.minio import MinIOProfileLoader
from adaptive_router.models.api import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models.routing import ModelFeatureVector
from adaptive_router.models.storage import RouterProfile, MinIOSettings
from adaptive_router.models.registry import RegistryModel
from adaptive_router.core.cluster_engine import ClusterEngine

logger = logging.getLogger(__name__)

_EPSILON = 1e-10


class ModelRouter:
    """Intelligent model routing using cluster-based selection.

    Selects optimal LLM models based on cluster-specific error rates,
    cost optimization, and model capability matching.

    Loads cluster profiles from MinIO S3 storage and performs intelligent
    routing using the UniRouter algorithm with per-cluster error rates.
    """

    def __init__(
        self,
        profile: RouterProfile,
        model_costs: Dict[str, float],
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5,
        allow_trust_remote_code: bool = False,
    ) -> None:
        """Initialize ModelRouter with injected dependencies.

        Args:
            profile: RouterProfile with cluster data and error rates per model
            model_costs: Dict mapping model_id to cost_per_1m_tokens
            lambda_min: Minimum lambda value for cost-quality tradeoff (default: 0.0)
            lambda_max: Maximum lambda value for cost-quality tradeoff (default: 2.0)
            default_cost_preference: Default cost preference when not specified (default: 0.5)
            allow_trust_remote_code: Allow execution of remote code in embedding models.
                                   WARNING: Enabling this allows arbitrary code execution from
                                   remote sources and should only be used with trusted models.
                                   Defaults to False for security.

        Raises:
            ValueError: If model_costs don't match profile.llm_profiles
        """
        n_clusters = profile.metadata.n_clusters
        logger.info(f"Initializing ModelRouter with {n_clusters} clusters")

        if allow_trust_remote_code:
            logger.warning(
                "WARNING: allow_trust_remote_code=True enables execution of remote code "
                "from embedding models. This should only be used with trusted models."
            )

        self.cluster_engine = self._build_cluster_engine_from_data(
            profile, allow_trust_remote_code
        )

        logger.info(
            f"Loaded cluster engine: {n_clusters} clusters, "
            f"silhouette score: {profile.metadata.silhouette_score or 'N/A'}"
        )

        self.model_features: Dict[str, ModelFeatureVector] = {}

        for model_id, error_rates in profile.llm_profiles.items():
            if model_id not in model_costs:
                logger.warning(f"Model {model_id} missing cost data, skipping")
                continue

            self.model_features[model_id] = ModelFeatureVector(
                error_rates=error_rates,
                cost_per_1m_tokens=model_costs[model_id],
            )
            logger.debug(
                f"Loaded profile for {model_id} with cost {model_costs[model_id]}"
            )

        if not self.model_features:
            raise ValueError("No valid model features found in llm_profiles")

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.default_cost_preference = default_cost_preference

        all_costs = [f.cost_per_1m_tokens for f in self.model_features.values()]
        self.min_cost = min(all_costs)
        self.max_cost = max(all_costs)

        logger.info(
            f"ModelRouter initialized with {len(self.model_features)} models, "
            f"K={self.cluster_engine.n_clusters} clusters, "
            f"lambda range [{self.lambda_min}, {self.lambda_max}]"
        )

    def _build_cluster_engine_from_data(
        self,
        profile: RouterProfile,
        allow_trust_remote_code: bool,
    ) -> ClusterEngine:
        """Build ClusterEngine from storage profile data.

        Args:
            profile: Validated RouterProfile from storage

        Returns:
            Reconstructed ClusterEngine
        """
        import platform

        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import StandardScaler

        from adaptive_router.core.feature_extractor import FeatureExtractor

        # Determine device (CPU for macOS, CUDA if available otherwise)
        device = (
            "cpu"
            if platform.system() == "Darwin"
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Loading embedding model on device: {device}")

        # Create fresh SentenceTransformer model
        embedding_model_name = profile.metadata.embedding_model
        embedding_model = SentenceTransformer(
            embedding_model_name,
            device=device,
            trust_remote_code=allow_trust_remote_code,
        )

        # Create FeatureExtractor with fresh model
        feature_extractor = FeatureExtractor(
            embedding_model=embedding_model_name,
            tfidf_max_features=profile.metadata.tfidf_max_features,
            tfidf_ngram_range=tuple(profile.metadata.tfidf_ngram_range),  # type: ignore[arg-type]
            allow_trust_remote_code=allow_trust_remote_code,
        )

        # Replace the embedding model (already loaded above)
        feature_extractor.embedding_model = embedding_model

        # Restore TF-IDF vocabulary
        feature_extractor.tfidf_vectorizer.vocabulary_ = (
            profile.tfidf_vocabulary.vocabulary
        )
        feature_extractor.tfidf_vectorizer.idf_ = np.array(profile.tfidf_vocabulary.idf)

        # Restore scaler parameters
        logger.info("Restoring scaler parameters from MinIO data...")

        # Restore embedding scaler
        feature_extractor.embedding_scaler = StandardScaler()
        feature_extractor.embedding_scaler.mean_ = np.array(
            profile.scaler_parameters.embedding_scaler.mean
        )
        feature_extractor.embedding_scaler.scale_ = np.array(
            profile.scaler_parameters.embedding_scaler.scale
        )
        feature_extractor.embedding_scaler.n_features_in_ = len(
            feature_extractor.embedding_scaler.mean_
        )

        # Restore TF-IDF scaler
        feature_extractor.tfidf_scaler = StandardScaler()
        feature_extractor.tfidf_scaler.mean_ = np.array(
            profile.scaler_parameters.tfidf_scaler.mean
        )
        feature_extractor.tfidf_scaler.scale_ = np.array(
            profile.scaler_parameters.tfidf_scaler.scale
        )
        feature_extractor.tfidf_scaler.n_features_in_ = len(
            feature_extractor.tfidf_scaler.mean_
        )
        logger.info("Scaler parameters restored")

        feature_extractor.is_fitted = True

        # Create ClusterEngine
        cluster_engine = ClusterEngine(
            n_clusters=profile.cluster_centers.n_clusters,
            embedding_model=embedding_model_name,
            tfidf_max_features=profile.metadata.tfidf_max_features,
            tfidf_ngram_range=tuple(profile.metadata.tfidf_ngram_range),  # type: ignore[arg-type]
        )
        cluster_engine.feature_extractor = feature_extractor

        # Restore K-means cluster centers
        cluster_engine.kmeans.cluster_centers_ = np.array(
            profile.cluster_centers.cluster_centers
        )
        # Set required K-means attributes
        cluster_engine.kmeans._n_threads = 1
        cluster_engine.kmeans.n_iter_ = 0  # Already fitted
        cluster_engine.kmeans.n_features_in_ = (
            cluster_engine.kmeans.cluster_centers_.shape[1]
        )

        cluster_engine.is_fitted = True
        cluster_engine.silhouette = profile.metadata.silhouette_score or 0.0

        logger.info(
            f"Built cluster engine from storage data: {profile.cluster_centers.n_clusters} clusters, "
            f"{profile.cluster_centers.feature_dim} features"
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
        start_time = time.time()

        # Extract and validate allowed models if provided
        allowed_model_ids: List[str] | None = None
        if request.models:
            allowed_model_ids = self._filter_models_by_request(request.models)

        # Map cost_bias (0.0=cheap, 1.0=quality) to cost_preference
        cost_preference = (
            request.cost_bias
            if request.cost_bias is not None
            else self.default_cost_preference
        )

        # Route the question - all routing logic inline now
        # 1. Assign to cluster
        cluster_id, distance = self.cluster_engine.assign_question(request.prompt)

        # 2. Calculate lambda parameter
        lambda_param = self._calculate_lambda(cost_preference)

        # 3. Determine which models to consider
        if allowed_model_ids is not None:
            # Filter to only allowed models
            allowed_set = set(allowed_model_ids)
            models_to_score = {
                model_id: features
                for model_id, features in self.model_features.items()
                if model_id in allowed_set
            }

            if not models_to_score:
                raise ValueError(
                    f"No valid models found in allowed list. "
                    f"Allowed: {allowed_model_ids}, Available: {list(self.model_features.keys())}"
                )
        else:
            # Use all available models
            models_to_score = self.model_features

        # 4. Compute routing scores for each model
        model_scores = {}

        for model_id, features in models_to_score.items():
            error_rate = features.error_rates[cluster_id]
            cost = features.cost_per_1m_tokens

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

        routing_time = (time.time() - start_time) * 1000

        # Generate reasoning
        self._generate_reasoning(
            cluster_id=cluster_id,
            cost_preference=cost_preference,
            lambda_param=lambda_param,
            selected_scores=best_scores,
        )

        # Prepare alternatives
        alternatives_data = [
            {
                "model_id": mid,
                "score": scores["score"],
                "accuracy": scores["accuracy"],
                "cost": scores["cost"],
            }
            for mid, scores in sorted(model_scores.items(), key=lambda x: x[1]["score"])
            if mid != best_model_id
        ]

        # Parse model ID to extract provider and model name
        if ":" not in best_model_id:
            raise ValueError(
                f"Invalid model ID format: {best_model_id}. "
                f"Expected format: 'provider:model_name' (e.g., 'openai:gpt-4')"
            )

        selected_model_parts = best_model_id.split(":", 1)
        if len(selected_model_parts) != 2 or not all(selected_model_parts):
            raise ValueError(
                f"Invalid model ID format: {best_model_id}. "
                f"Both provider and model_name must be non-empty. "
                f"Expected format: 'provider:model_name'"
            )

        provider, model_name = selected_model_parts

        # Convert alternatives to Alternative objects
        alternatives_list = []
        for alt in alternatives_data[:3]:  # Limit to top 3 alternatives
            alt_model_id: str = alt["model_id"]  # type: ignore[assignment]
            alt_parts = alt_model_id.split(":", 1)
            if len(alt_parts) == 2:
                alternatives_list.append(
                    Alternative(provider=alt_parts[0], model=alt_parts[1])
                )

        # Convert to ModelSelectionResponse
        response = ModelSelectionResponse(
            provider=provider,
            model=model_name,
            alternatives=alternatives_list,
        )

        logger.info(
            f"Selected model: {provider}/{model_name} "
            f"(cluster {cluster_id}, "
            f"accuracy {best_scores['accuracy']:.2%}, "
            f"score {best_scores['score']:.3f}, "
            f"routing_time {routing_time:.2f}ms)"
        )

        return response

    @classmethod
    def from_profile(
        cls,
        profile: RouterProfile,
        model_costs: Dict[str, float],
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5,
        allow_trust_remote_code: bool = False,
    ) -> ModelRouter:
        return cls(
            profile=profile,
            model_costs=model_costs,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            default_cost_preference=default_cost_preference,
            allow_trust_remote_code=allow_trust_remote_code,
        )

    @classmethod
    def from_minio(
        cls,
        settings: MinIOSettings,
        model_costs: Dict[str, float],
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5,
        allow_trust_remote_code: bool = False,
    ) -> ModelRouter:
        loader = MinIOProfileLoader.from_settings(settings)
        profile = loader.load_profile()
        return cls.from_profile(
            profile=profile,
            model_costs=model_costs,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            default_cost_preference=default_cost_preference,
            allow_trust_remote_code=allow_trust_remote_code,
        )

    @classmethod
    def from_local_file(
        cls,
        profile_path: str | Path,
        model_costs: Dict[str, float],
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5,
        allow_trust_remote_code: bool = False,
    ) -> ModelRouter:
        loader = LocalFileProfileLoader(profile_path=Path(profile_path))
        profile = loader.load_profile()
        return cls.from_profile(
            profile=profile,
            model_costs=model_costs,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            default_cost_preference=default_cost_preference,
            allow_trust_remote_code=allow_trust_remote_code,
        )

    def _filter_models_by_request(
        self, models: List[RegistryModel]
    ) -> List[str] | None:
        """Filter supported models based on request model specifications.

        This method provides a scalable filtering mechanism that can handle:
        - Full model specifications (provider + model_name)
        - Provider-only filters (just provider specified)
        - Future filters (e.g., cost thresholds, capabilities, etc.)

        Args:
            models: List of RegistryModel objects with filter criteria

        Returns:
            List of allowed model IDs in "provider:model_name" format,
            or None if no filtering should be applied

        Raises:
            ValueError: If requested models are not supported or no models match filters
        """
        supported = self.get_supported_models()

        # Separate full model specs from partial filters
        explicit_models = []
        provider_filters = []

        for m in models:
            if m.provider and m.model_name:
                # Full model specification - exact match required
                model_id = f"{m.provider.lower()}:{m.model_name.lower()}"
                explicit_models.append(model_id)
            elif m.provider:
                # Provider-only filter - match all models from this provider
                provider_filters.append(m.provider.lower())
            # Future: Add more filter types here (e.g., cost_threshold, supports_function_calling, etc.)

        # Apply filters in order of specificity
        allowed_model_ids = []

        # 1. Explicit model specifications take highest priority
        if explicit_models:
            # Validate all requested models are supported
            unsupported = [m for m in explicit_models if m not in supported]
            if unsupported:
                raise ValueError(
                    f"Models not supported by Router: {unsupported}. "
                    f"Supported models: {supported}"
                )
            allowed_model_ids.extend(explicit_models)

        # 2. Apply provider filters
        if provider_filters:
            provider_filtered = [
                model_id
                for model_id in supported
                if any(
                    model_id.startswith(f"{provider}:") for provider in provider_filters
                )
            ]
            if not provider_filtered:
                raise ValueError(
                    f"No supported models found for providers: {provider_filters}. "
                    f"Supported models: {supported}"
                )
            allowed_model_ids.extend(provider_filtered)

        # 3. Future filters can be added here (e.g., capability filters, cost filters)
        # Example:
        # if cost_threshold_filters:
        #     cost_filtered = [
        #         model_id for model_id in supported
        #         if self.model_features[model_id].cost_per_1m_tokens <= threshold
        #     ]
        #     allowed_model_ids.extend(cost_filtered)

        # Remove duplicates while preserving order
        if allowed_model_ids:
            seen = set()
            unique_models = []
            for model_id in allowed_model_ids:
                if model_id not in seen:
                    seen.add(model_id)
                    unique_models.append(model_id)
            return unique_models

        # No filters specified - use all models
        return None

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

    def get_supported_models(self) -> List[str]:
        """Get list of models this router supports.

        Returns:
            List of model IDs in format "provider:model_name"
        """
        return list(self.model_features.keys())

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about loaded clusters.

        Returns:
            Dictionary with cluster statistics including n_clusters,
            embedding_model, supported_models, and lambda parameters
        """
        return {
            "n_clusters": self.cluster_engine.n_clusters,
            "embedding_model": self.cluster_engine.feature_extractor.embedding_model_name,
            "supported_models": self.get_supported_models(),
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "default_cost_preference": self.default_cost_preference,
        }
