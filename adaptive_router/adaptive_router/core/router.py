"""Intelligent model routing using cluster-based selection.

This module provides the ModelRouter class for selecting optimal LLM models
based on cluster-specific error rates, cost optimization, and model capabilities.

All routing logic is consolidated here: MinIO loading, cluster-based routing,
and response conversion.
"""

from __future__ import annotations

import logging
import time
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from adaptive_router.loaders.local import LocalFileProfileLoader
from adaptive_router.loaders.minio import MinIOProfileLoader
from adaptive_router.models.api import (
    Alternative,
    Model,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models.routing import (
    ModelFeatureVector,
    ModelScore,
    AlternativeScore,
)
from adaptive_router.models.storage import RouterProfile, MinIOSettings
from adaptive_router.core.cluster_engine import ClusterEngine
from adaptive_router.core.feature_extractor import FeatureExtractor
from adaptive_router.core.exceptions import (
    ModelNotFoundError,
    InvalidModelFormatError,
)


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
        models: List[Model],
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5,
        allow_trust_remote_code: bool = False,
    ) -> None:
        """Initialize ModelRouter with injected dependencies.

        Args:
            profile: RouterProfile with cluster data and error rates per model
            models: List of Model objects with cost information
            lambda_min: Minimum lambda value for cost-quality tradeoff (default: 0.0)
            lambda_max: Maximum lambda value for cost-quality tradeoff (default: 2.0)
            default_cost_preference: Default cost preference when not specified (default: 0.5)
            allow_trust_remote_code: Allow execution of remote code in embedding models.
                                   WARNING: Enabling this allows arbitrary code execution from
                                   remote sources and should only be used with trusted models.
                                   Defaults to False for security.

        Raises:
            ModelNotFoundError: If models don't match profile.llm_profiles
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

        # Create mapping from model_id to cost
        model_costs: Dict[str, float] = {}
        for model in models:
            model_id = model.unique_id()
            model_costs[model_id] = model.cost_per_1m_tokens

        self.model_features: Dict[str, ModelFeatureVector] = {}

        for model_id, error_rates in profile.llm_profiles.items():
            if model_id not in model_costs:
                logger.warning(f"Model {model_id} missing cost data, skipping")
                continue

            # Extract input and output costs from the model
            matching_models = [m for m in models if m.unique_id() == model_id]
            if not matching_models:
                logger.warning(f"Model {model_id} not found in models list, skipping")
                continue

            model = matching_models[0]
            self.model_features[model_id] = ModelFeatureVector(
                error_rates=error_rates,
                cost_per_1m_input_tokens=model.cost_per_1m_input_tokens,
                cost_per_1m_output_tokens=model.cost_per_1m_output_tokens,
            )
            logger.debug(
                f"Loaded profile for {model_id} with cost {model_costs[model_id]}"
            )

        if not self.model_features:
            raise ModelNotFoundError("No valid model features found in llm_profiles")

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.default_cost_preference = default_cost_preference

        all_costs = [f.cost_per_1m_tokens for f in self.model_features.values()]
        self.min_cost = min(all_costs)
        self.max_cost = max(all_costs)

    @cached_property
    def cost_range(self) -> float:
        """Cache cost range calculation for performance."""
        return self.max_cost - self.min_cost

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
            allow_trust_remote_code: Whether to allow remote code execution

        Returns:
            Reconstructed ClusterEngine
        """
        # Load and configure embedding model
        embedding_model = self._load_embedding_model(
            profile.metadata.embedding_model, allow_trust_remote_code
        )

        # Restore feature extractor with scalers and vocabulary
        feature_extractor = self._restore_feature_extractor(
            profile, embedding_model, allow_trust_remote_code
        )

        # Restore cluster engine with K-means parameters
        cluster_engine = self._restore_cluster_engine(profile, feature_extractor)

        return cluster_engine

    def _load_embedding_model(
        self,
        embedding_model_name: str,
        allow_trust_remote_code: bool,
    ) -> SentenceTransformer:
        """Load and configure the embedding model.

        Args:
            embedding_model_name: Name of the SentenceTransformer model
            allow_trust_remote_code: Whether to allow remote code execution

        Returns:
            Configured SentenceTransformer model
        """
        import platform
        import torch

        # Determine device (CPU for macOS, CUDA if available otherwise)
        device = (
            "cpu"
            if platform.system() == "Darwin"
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Loading embedding model on device: {device}")

        # Suppress the clean_up_tokenization_spaces warning during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*clean_up_tokenization_spaces.*",
                category=FutureWarning,
            )
            embedding_model = SentenceTransformer(
                embedding_model_name,
                device=device,
                trust_remote_code=allow_trust_remote_code,
            )
        # Explicitly set clean_up_tokenization_spaces to False for future compatibility
        embedding_model.tokenizer.clean_up_tokenization_spaces = False

        return embedding_model

    def _restore_feature_extractor(
        self,
        profile: RouterProfile,
        embedding_model: SentenceTransformer,
        allow_trust_remote_code: bool,
    ) -> FeatureExtractor:
        """Restore FeatureExtractor from profile data.

        Args:
            profile: RouterProfile with stored parameters
            embedding_model: Loaded SentenceTransformer model
            allow_trust_remote_code: Trust remote code setting

        Returns:
            Configured FeatureExtractor
        """
        from sklearn.preprocessing import StandardScaler
        from adaptive_router.core.feature_extractor import FeatureExtractor

        # Create FeatureExtractor with fresh model
        feature_extractor = FeatureExtractor(
            embedding_model=profile.metadata.embedding_model,
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

        return feature_extractor

    def _restore_cluster_engine(
        self,
        profile: RouterProfile,
        feature_extractor: FeatureExtractor,
    ) -> ClusterEngine:
        """Restore ClusterEngine from profile data.

        Args:
            profile: RouterProfile with cluster data
            feature_extractor: Configured FeatureExtractor

        Returns:
            Configured ClusterEngine
        """
        # Create ClusterEngine
        cluster_engine = ClusterEngine(
            n_clusters=profile.cluster_centers.n_clusters,
            embedding_model=profile.metadata.embedding_model,
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

        cluster_engine.silhouette = profile.metadata.silhouette_score or 0.0

        logger.info(
            f"Built cluster engine from storage data: {profile.cluster_centers.n_clusters} clusters, "
            f"{profile.cluster_centers.feature_dim} features"
        )

        return cluster_engine

    def select_model(
        self,
        request: ModelSelectionRequest,
        *,
        cost_bias: float | None = None,
    ) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis.

        This is the main public API method. Uses cluster-based routing to select
        the best model based on prompt characteristics, cost preferences, and
        historical per-cluster error rates.

        Args:
            request: ModelSelectionRequest with prompt and optional model constraints
            cost_bias: Override default cost preference (0.0=cheap, 1.0=quality).
                      If not provided, uses the request's cost_bias or default_cost_preference.

        Returns:
            ModelSelectionResponse with selected provider, model, and alternatives

        Raises:
            ModelNotFoundError: If requested models are not supported
            InvalidModelFormatError: If model ID format is invalid

        Examples:
            >>> router.select_model(ModelSelectionRequest(prompt="How do I sort a list?"))
            >>> router.select_model(request, cost_bias=0.8)  # Quality-focused
        """
        start_time = time.time()

        # Extract and validate allowed models if provided
        allowed_model_ids: Optional[List[str]] = None
        if request.models:
            allowed_model_ids = self._filter_models_by_request(request.models)

        # Map cost_bias (0.0=cheap, 1.0=quality) to cost_preference
        # Priority: explicit parameter > request field > default
        cost_preference = (
            cost_bias
            if cost_bias is not None
            else (
                request.cost_bias
                if request.cost_bias is not None
                else self.default_cost_preference
            )
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
                raise ModelNotFoundError(
                    f"No valid models found in allowed list. "
                    f"Allowed: {allowed_model_ids}, Available: {list(self.model_features.keys())}"
                )
        else:
            # Use all available models
            models_to_score = self.model_features

        # 4. Compute routing scores for each model
        model_scores: Dict[str, ModelScore] = {}

        for model_id, features in models_to_score.items():
            error_rate = features.error_rates[cluster_id]
            cost = features.cost_per_1m_tokens

            normalized_cost = self._normalize_cost(cost)
            score = error_rate + lambda_param * normalized_cost

            model_scores[model_id] = ModelScore(
                score=score,
                error_rate=error_rate,
                accuracy=1.0 - error_rate,
                cost=cost,
                normalized_cost=normalized_cost,
            )

        # 5. Select best model (lowest score)
        best_model_id = min(model_scores, key=lambda k: model_scores[k].score)
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
        alternatives: List[AlternativeScore] = [
            AlternativeScore(
                model_id=mid,
                score=scores.score,
                accuracy=scores.accuracy,
                cost=scores.cost,
            )
            for mid, scores in sorted(model_scores.items(), key=lambda x: x[1].score)
            if mid != best_model_id
        ]

        # Parse model ID to extract provider and model name
        if "/" not in best_model_id:
            raise InvalidModelFormatError(
                f"Invalid model ID format: {best_model_id}. "
                f"Expected format: 'provider/model_name' (e.g., 'openai/gpt-4')"
            )

        selected_model_parts = best_model_id.split("/", 1)
        if len(selected_model_parts) != 2 or not all(selected_model_parts):
            raise InvalidModelFormatError(
                f"Invalid model ID format: {best_model_id}. "
                f"Both provider and model_name must be non-empty. "
                f"Expected format: 'provider/model_name'"
            )

        # Convert alternatives to Alternative objects
        alternatives_list = []
        for alt in alternatives[:3]:  # Limit to top 3 alternatives
            alternatives_list.append(Alternative(model_id=alt.model_id))

        # Convert to ModelSelectionResponse
        response = ModelSelectionResponse(
            model_id=best_model_id,
            alternatives=alternatives_list,
        )

        logger.info(
            f"Selected model: {best_model_id} "
            f"(cluster {cluster_id}, "
            f"accuracy {best_scores.accuracy:.2%}, "
            f"score {best_scores.score:.3f}, "
            f"routing_time {routing_time:.2f}ms)"
        )

        return response

    def route(self, prompt: str, cost_bias: float | None = None) -> str:
        """Quick routing - just give me the model ID.

        Convenience method for simple routing use cases. Returns only the selected
        model ID without the full response object.

        Args:
            prompt: Question text to route
            cost_bias: Cost preference (0.0=cheap, 1.0=quality). Uses default if None.

        Returns:
            Model ID string (e.g., "openai/gpt-4")

        Raises:
            ModelNotFoundError: If no suitable models found
            InvalidModelFormatError: If model ID format is invalid

        Examples:
            >>> router.route("How do I sort a list?")
            'openai/gpt-3.5-turbo'
            >>> router.route("Complex analysis needed", cost_bias=0.9)
            'openai/gpt-4'
        """
        request = ModelSelectionRequest(prompt=prompt, cost_bias=cost_bias)
        response = self.select_model(request, cost_bias=cost_bias)
        return response.model_id

    @classmethod
    def from_profile(
        cls,
        profile: RouterProfile,
        models: List[Model],
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5,
        allow_trust_remote_code: bool = False,
    ) -> ModelRouter:
        return cls(
            profile=profile,
            models=models,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            default_cost_preference=default_cost_preference,
            allow_trust_remote_code=allow_trust_remote_code,
        )

    @classmethod
    def from_minio(
        cls,
        settings: MinIOSettings,
        models: List[Model],
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5,
        allow_trust_remote_code: bool = False,
    ) -> ModelRouter:
        loader = MinIOProfileLoader.from_settings(settings)
        profile = loader.load_profile()
        return cls.from_profile(
            profile=profile,
            models=models,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            default_cost_preference=default_cost_preference,
            allow_trust_remote_code=allow_trust_remote_code,
        )

    @classmethod
    def from_local_file(
        cls,
        profile_path: str | Path,
        models: List[Model],
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5,
        allow_trust_remote_code: bool = False,
    ) -> ModelRouter:
        loader = LocalFileProfileLoader(profile_path=Path(profile_path))
        profile = loader.load_profile()
        return cls.from_profile(
            profile=profile,
            models=models,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            default_cost_preference=default_cost_preference,
            allow_trust_remote_code=allow_trust_remote_code,
        )

    def _filter_models_by_request(self, models: List[Model]) -> Optional[List[str]]:
        """Filter supported models based on request model specifications.

        This method provides a scalable filtering mechanism that can handle:
        - Full model specifications (provider + model_name)
        - Provider-only filters (just provider specified)
        - Future filters (e.g., cost thresholds, capabilities, etc.)

        Args:
            models: List of Model objects with filter criteria

        Returns:
            List of allowed model IDs in "provider/model_name" format,
            or None if no filtering should be applied

        Raises:
            ModelNotFoundError: If requested models are not supported or no models match filters
        """
        supported = self.get_supported_models()

        # Separate full model specs from partial filters
        explicit_models = []
        provider_filters = []

        for m in models:
            if m.provider and m.model_name:
                # Full model specification - exact match required
                model_id = f"{m.provider.lower()}/{m.model_name.lower()}"
                explicit_models.append(model_id)
            elif m.provider:
                # Provider-only filter - match all models from this provider
                provider_filters.append(m.provider.lower())
            # Future: Add more filter types here (e.g., cost_threshold, supports_function_calling, etc.)

        # Apply filters in order of specificity
        allowed_model_ids = []

        # 1. Explicit model specifications take highest priority
        if explicit_models:
            # Filter to only supported models instead of raising error
            unsupported = [m for m in explicit_models if m not in supported]
            if unsupported:
                logger.warning(
                    f"Dropping unsupported models from request: {unsupported}. "
                    f"Supported models: {supported}"
                )
            # Only add models that are actually supported
            supported_explicit = [m for m in explicit_models if m in supported]
            allowed_model_ids.extend(supported_explicit)

        # 2. Apply provider filters
        if provider_filters:
            provider_filtered = [
                model_id
                for model_id in supported
                if any(
                    model_id.startswith(f"{provider}/") for provider in provider_filters
                )
            ]
            if not provider_filtered:
                raise ModelNotFoundError(
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

            # Ensure we have at least one model after filtering
            if not unique_models:
                raise ModelNotFoundError(
                    f"No supported models remain after filtering. "
                    f"Requested models were filtered out. "
                    f"Supported models: {supported}"
                )
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
        if self.cost_range < _EPSILON:
            return 0.0

        return float((cost - self.min_cost) / self.cost_range)

    def _generate_reasoning(
        self,
        cluster_id: int,
        cost_preference: float,
        lambda_param: float,
        selected_scores: ModelScore,
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
        accuracy = selected_scores.accuracy
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
            List of model IDs in format "provider/model_name"
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
