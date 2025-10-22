"""Router service wrapper for adaptive_router integration.

This service bridges the cluster-based routing with adaptive_router's
ModelSelectionRequest/Response API.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from adaptive_router.core.storage_config import MinIOSettings
from adaptive_router.models.llm_core_models import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.storage_profile_loader import StorageProfileLoader
from adaptive_router.services.cluster_engine import ClusterEngine
from adaptive_router.services.router import Router
from adaptive_router.services.routing_schemas import (
    ModelConfig,
)

logger = logging.getLogger(__name__)


class RouterService:
    """Service layer for Router integration with adaptive_router.

    This service:
    1. Loads Router from MinIO S3 bucket (no local files used)
    2. Provides model selection compatible with adaptive_router API
    3. Optionally supports Modal GPU for feature extraction (future enhancement)
    """

    def __init__(
        self,
        config_file: Path | None = None,
        use_modal_gpu: bool = False,
    ):
        """Initialize Router service.

        All profile data is loaded from MinIO S3 bucket (Railway deployment).
        No local data files are used in production.

        Args:
            config_file: Path to Router models YAML config
            use_modal_gpu: If True, attempt to use Modal GPU for feature extraction
        """
        # Auto-detect config file path
        if config_file is None:
            service_dir = Path(__file__).parent.parent  # adaptive_router/
            config_file = service_dir / "config" / "unirouter_models.yaml"

        self.config_file = Path(config_file)
        self.use_modal_gpu = use_modal_gpu
        self.modal_feature_extractor = None

        # Try to connect to Modal if requested
        if use_modal_gpu:
            try:
                from modal import Function

                self.modal_feature_extractor = Function.lookup(
                    "unirouter-feature-extractor", "extract_features"
                )
                logger.info("Using Modal GPU for feature extraction")
            except Exception as e:
                logger.warning(f"Modal GPU unavailable, falling back to local: {e}")
                self.use_modal_gpu = False

        # Load Router
        self.router = self._load_router()
        logger.info(
            f"RouterService initialized with {len(self.router.models)} models, "
            f"K={self.router.cluster_engine.n_clusters} clusters"
        )

    def _load_router(self) -> Router:
        """Load Router from MinIO S3 storage (Railway deployment).

        This method loads all profile data from the MinIO bucket:
        - Cluster centers (K-means centroids)
        - Model error rates per cluster
        - TF-IDF vocabulary
        - Scaler parameters

        Local data files in adaptive_router/data/ are NOT used.

        Returns:
            Initialized Router instance

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
        router = Router(
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

        from adaptive_router.services.feature_extractor import (
            FeatureExtractor,
        )

        # Determine device (CPU for macOS, CUDA if available otherwise)
        device = (
            "cpu"
            if platform.system() == "Darwin"
            else "cuda" if torch.cuda.is_available() else "cpu"
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
        """Select optimal model using Router.

        Args:
            request: Model selection request with prompt and cost bias

        Returns:
            Model selection response with routing decision

        Raises:
            ValueError: If requested models are not supported
        """
        # Validate models if provided
        if request.models:
            supported = self.get_supported_models()
            requested = [m.unique_id for m in request.models if not m.is_partial]
            unsupported = [m for m in requested if m not in supported]

            if unsupported:
                raise ValueError(
                    f"Models not supported by Router: {unsupported}. "
                    f"Supported models: {supported}"
                )

        # Map cost_bias (0.0=cheap, 1.0=quality) to cost_preference
        # Note: request.cost_bias might be None
        cost_preference = request.cost_bias if request.cost_bias is not None else None

        # Route the question using Router
        decision = self.router.route(
            question_text=request.prompt,
            cost_preference=cost_preference,
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
        from adaptive_router.models.llm_core_models import Alternative

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
        """Get list of models Router supports.

        Returns:
            List of model IDs
        """
        return list(self.router.models.keys())

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about loaded clusters.

        Returns:
            Dictionary with cluster statistics
        """
        return {
            "n_clusters": self.router.cluster_engine.n_clusters,
            "embedding_model": self.router.cluster_engine.feature_extractor.embedding_model_name,
            "supported_models": self.get_supported_models(),
            "lambda_min": self.router.lambda_min,
            "lambda_max": self.router.lambda_max,
            "default_cost_preference": self.router.default_cost_preference,
        }

    def _extract_features_remote(self, question_text: str) -> np.ndarray:
        """Extract features via Modal GPU service (future enhancement).

        Args:
            question_text: Question to extract features for

        Returns:
            5384D feature vector

        Raises:
            RuntimeError: If Modal call fails
        """
        if not self.modal_feature_extractor:
            raise RuntimeError("Modal feature extractor not available")

        try:
            result = self.modal_feature_extractor.remote([question_text])
            features = np.array(result["features"][0])
            logger.debug(
                f"Modal feature extraction: {result['inference_time_ms']:.1f}ms"
            )
            return features
        except Exception as e:
            logger.error(f"Modal feature extraction failed: {e}")
            raise RuntimeError(f"Modal feature extraction failed: {e}") from e
