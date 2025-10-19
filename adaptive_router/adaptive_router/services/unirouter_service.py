"""UniRouter service wrapper for adaptive_router integration.

This service bridges UniRouter's cluster-based routing with adaptive_router's
ModelSelectionRequest/Response API.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from adaptive_router.models.llm_core_models import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.unirouter.cluster_engine import ClusterEngine
from adaptive_router.services.unirouter.router import UniRouter
from adaptive_router.services.unirouter.schemas import (
    ModelConfig,
)

logger = logging.getLogger(__name__)


class ModulePathUnpickler(pickle.Unpickler):
    """Custom unpickler that handles module path changes from unirouter.* to adaptive_router.services.unirouter.*"""

    def find_class(self, module, name):
        """Override find_class to redirect old unirouter paths."""
        if module.startswith("unirouter."):
            # Map old paths to new paths
            old_to_new = {
                "unirouter.clustering.cluster_engine": "adaptive_router.services.unirouter.cluster_engine",
                "unirouter.clustering.feature_extractor": "adaptive_router.services.unirouter.feature_extractor",
                "unirouter.models.schemas": "adaptive_router.services.unirouter.schemas",
            }

            new_module = old_to_new.get(
                module,
                module.replace("unirouter.", "adaptive_router.services.unirouter."),
            )
            logger.debug(f"Remapping pickle module: {module} -> {new_module}")
            return super().find_class(new_module, name)

        return super().find_class(module, name)


class UniRouterService:
    """Service layer for UniRouter integration with adaptive_router.

    This service:
    1. Loads UniRouter from saved artifacts (cluster engine, model features, config)
    2. Provides model selection compatible with adaptive_router API
    3. Optionally supports Modal GPU for feature extraction (future enhancement)
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        config_file: Path | None = None,
        use_modal_gpu: bool = False,
    ):
        """Initialize UniRouter service.

        Args:
            data_dir: Path to UniRouter data directory with clusters/
            config_file: Path to UniRouter models YAML config
            use_modal_gpu: If True, attempt to use Modal GPU for feature extraction
        """
        # Auto-detect paths relative to this file
        if data_dir is None:
            service_dir = Path(__file__).parent.parent  # adaptive_router/
            data_dir = service_dir / "data" / "unirouter"
        if config_file is None:
            service_dir = Path(__file__).parent.parent  # adaptive_router/
            config_file = service_dir / "config" / "unirouter_models.yaml"

        self.data_dir = Path(data_dir)
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

        # Load UniRouter
        self.router = self._load_router()
        logger.info(
            f"UniRouterService initialized with {len(self.router.models)} models, "
            f"K={self.router.cluster_engine.n_clusters} clusters"
        )

    def _load_router(self) -> UniRouter:
        """Load UniRouter from saved artifacts (prefers JSON over pickle).

        Returns:
            Initialized UniRouter instance

        Raises:
            FileNotFoundError: If required data files are missing
            ValueError: If data files are corrupted or incompatible
        """
        cluster_dir = self.data_dir / "clusters"
        cluster_pkl = cluster_dir / "cluster_engine.pkl"
        cluster_centers_json = cluster_dir / "cluster_centers.json"
        tfidf_vocab_json = cluster_dir / "tfidf_vocabulary.json"
        metadata_file = cluster_dir / "metadata.json"
        profiles_file = cluster_dir / "llm_profiles.json"

        # Verify required files exist
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Required file not found: {metadata_file}\n"
                f"Run training first or check data directory: {self.data_dir}"
            )
        if not profiles_file.exists():
            raise FileNotFoundError(
                f"Required file not found: {profiles_file}\n"
                f"Run training first or check data directory: {self.data_dir}"
            )

        # Load metadata
        with open(metadata_file) as f:
            metadata = json.load(f)

        n_clusters = metadata["n_clusters"]
        logger.info(f"Loading UniRouter with K={n_clusters} clusters")

        # Try loading from JSON first (preferred), fall back to pickle
        if cluster_centers_json.exists() and tfidf_vocab_json.exists():
            logger.info("Loading cluster engine from JSON files (preferred method)...")
            cluster_engine = self._load_cluster_engine_from_json(cluster_dir, metadata)
        elif cluster_pkl.exists():
            logger.warning(
                "Loading from pickle (deprecated). "
                "Run training to generate JSON files for better Git performance."
            )
            # Load cluster engine from pickle using custom unpickler
            with open(cluster_pkl, "rb") as f:
                cluster_engine = ModulePathUnpickler(f).load()

            if not isinstance(cluster_engine, ClusterEngine):
                raise ValueError(
                    f"Loaded cluster engine has wrong type: {type(cluster_engine)}"
                )

            # WORKAROUND for macOS: Reload the embedding model fresh to avoid version conflicts
            try:
                import platform

                if platform.system() == "Darwin":
                    logger.info("Reloading embedding model for macOS compatibility...")
                    from sentence_transformers import SentenceTransformer

                    device = "cpu"
                    model_name = cluster_engine.feature_extractor.embedding_model_name
                    cluster_engine.feature_extractor.embedding_model = (
                        SentenceTransformer(
                            model_name, device=device, trust_remote_code=True
                        )
                    )
                    logger.info(f"Reloaded embedding model on {device}")
            except Exception as e:
                logger.warning(
                    f"Could not reload embedding model, continuing anyway: {e}"
                )
        else:
            raise FileNotFoundError(
                f"No cluster data found in {cluster_dir}\n"
                f"Expected either:\n"
                f"  - JSON files: cluster_centers.json + tfidf_vocabulary.json (preferred)\n"
                f"  - Pickle file: cluster_engine.pkl (deprecated)\n"
                f"Run training to generate cluster data."
            )

        logger.info(
            f"Loaded cluster engine: {n_clusters} clusters, "
            f"silhouette score: {metadata.get('silhouette_score', 'N/A')}"
        )

        # Load model features (per-cluster error rates)
        with open(profiles_file) as f:
            model_profiles = json.load(f)

        # Load models config
        with open(self.config_file) as f:
            models_config = yaml.safe_load(f)

        # Parse models
        models = [ModelConfig(**m) for m in models_config["gpt5_models"]]

        # Prepare model_features dict combining error rates and cost
        # Note: profiles may use model names without provider prefix
        model_features = {}
        for model in models:
            model_id = model.id  # e.g., "openai:gpt-5-codex"
            model_name = model.name  # e.g., "gpt-5-codex"

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
            raise ValueError("No valid model features found in llm_profiles.json")

        # Get routing config
        routing_config = models_config.get("routing", {})

        # Initialize UniRouter
        router = UniRouter(
            cluster_engine=cluster_engine,
            model_features=model_features,
            models=models,
            lambda_min=routing_config.get("lambda_min", 0.0),
            lambda_max=routing_config.get("lambda_max", 1.0),
            default_cost_preference=routing_config.get("default_cost_preference", 0.5),
        )

        logger.info(
            f"UniRouter initialized: {len(models)} models, "
            f"lambda range [{router.lambda_min}, {router.lambda_max}]"
        )

        return router

    def _load_cluster_engine_from_json(
        self, cluster_dir: Path, metadata: dict
    ) -> ClusterEngine:
        """Load ClusterEngine from JSON files (preferred method).

        Args:
            cluster_dir: Directory containing JSON files
            metadata: Metadata dictionary with config info

        Returns:
            Reconstructed ClusterEngine

        Raises:
            FileNotFoundError: If required JSON files are missing
        """
        import platform

        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import StandardScaler

        from adaptive_router.services.unirouter.feature_extractor import (
            FeatureExtractor,
        )

        cluster_centers_file = cluster_dir / "cluster_centers.json"
        tfidf_vocab_file = cluster_dir / "tfidf_vocabulary.json"

        # Load cluster centers
        with open(cluster_centers_file) as f:
            cluster_data = json.load(f)

        # Load TF-IDF vocabulary
        with open(tfidf_vocab_file) as f:
            tfidf_data = json.load(f)

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

        # Restore TF-IDF vocabulary from JSON
        feature_extractor.tfidf_vectorizer.vocabulary_ = tfidf_data["vocabulary"]
        feature_extractor.tfidf_vectorizer.idf_ = np.array(tfidf_data["idf"])

        # Restore scaler parameters from JSON (if available)
        scaler_params_file = cluster_dir / "scaler_parameters.json"
        if scaler_params_file.exists():
            logger.info("Loading scaler parameters from JSON...")
            with open(scaler_params_file) as f:
                scaler_data = json.load(f)

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
            logger.info("✅ Scaler parameters restored from JSON")
        else:
            # Fallback: create unfitted scalers (for backward compatibility)
            logger.warning(
                "Scaler parameters file not found. Routing may fail. "
                "Run migration script to generate scaler_parameters.json"
            )
            feature_extractor.embedding_scaler = StandardScaler()
            feature_extractor.tfidf_scaler = StandardScaler()

        feature_extractor.is_fitted = True

        # Create ClusterEngine
        cluster_engine = ClusterEngine(
            n_clusters=cluster_data["n_clusters"],
            embedding_model=embedding_model_name,
            tfidf_max_features=metadata.get("tfidf_max_features", 5000),
            tfidf_ngram_range=tuple(metadata.get("tfidf_ngram_range", [1, 2])),
        )
        cluster_engine.feature_extractor = feature_extractor

        # Restore K-means cluster centers
        cluster_engine.kmeans.cluster_centers_ = np.array(
            cluster_data["cluster_centers"]
        )
        # Set required K-means attributes
        cluster_engine.kmeans._n_threads = 1
        cluster_engine.kmeans.n_iter_ = 0  # Already fitted

        cluster_engine.is_fitted = True
        cluster_engine.silhouette = metadata.get("silhouette_score", 0.0)

        logger.info(
            f"Loaded cluster engine from JSON: {cluster_data['n_clusters']} clusters, "
            f"{cluster_data['feature_dim']} features"
        )

        return cluster_engine

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResponse:
        """Select optimal model using UniRouter.

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
                    f"Models not supported by UniRouter: {unsupported}. "
                    f"Supported models: {supported}"
                )

        # Map cost_bias (0.0=cheap, 1.0=quality) to cost_preference
        # Note: request.cost_bias might be None
        cost_preference = request.cost_bias if request.cost_bias is not None else None

        # Route the question using UniRouter
        decision = self.router.route(
            question_text=request.prompt,
            cost_preference=cost_preference,
        )

        # Parse model ID to extract provider and model name
        # Format: "openai:gpt-5-mini" -> provider="openai", model="gpt-5-mini"
        selected_model_parts = decision.selected_model_id.split(":", 1)
        if len(selected_model_parts) != 2:
            raise ValueError(
                f"Invalid model ID format: {decision.selected_model_id}. "
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
        """Get list of models UniRouter supports.

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
