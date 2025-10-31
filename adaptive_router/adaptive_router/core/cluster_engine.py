"""Unified clustering engine for semantic clustering of coding questions."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from adaptive_router.models import CodeQuestion
from adaptive_router.core.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class ClusterEngine:
    """Unified engine for clustering questions using K-means on hybrid features."""

    def __init__(
        self,
        n_clusters: int = 20,
        max_iter: int = 300,
        random_state: int = 42,
        n_init: int = 10,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tfidf_max_features: int = 5000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:
        """Initialize clustering engine.

        Args:
            n_clusters: Number of clusters (K)
            max_iter: Maximum iterations for K-means
            random_state: Random seed for reproducibility
            n_init: Number of K-means runs with different centroid seeds
            embedding_model: HuggingFace model for semantic embeddings
            tfidf_max_features: Maximum TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF
        """
        logger.info(f"Initializing ClusterEngine with K={n_clusters}")

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            embedding_model=embedding_model,
            tfidf_max_features=tfidf_max_features,
            tfidf_ngram_range=tfidf_ngram_range,
        )

        # K-means clusterer with spherical k-means (normalize features)
        # This uses cosine similarity instead of Euclidean distance
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            n_init=n_init,
            verbose=0,
            algorithm="lloyd",  # Use Lloyd algorithm for better convergence
        )

        # Flag to normalize features (spherical k-means)
        self.use_spherical = True

        self.n_clusters = n_clusters
        self.is_fitted = False
        self.cluster_assignments: np.ndarray = np.array([])
        self.silhouette: float = 0.0
        self.questions: List[CodeQuestion] = []

    def fit(self, questions: List[CodeQuestion]) -> "ClusterEngine":
        """Fit clustering model on questions.

        Args:
            questions: List of CodeQuestion objects

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting clustering model on {len(questions)} questions...")
        self.questions = questions

        # Extract hybrid features
        features = self.feature_extractor.fit_transform(questions)

        # Normalize features for spherical k-means (cosine similarity)
        if self.use_spherical:
            from sklearn.preprocessing import normalize

            features = normalize(features, norm="l2")
            logger.info(
                "Applied L2 normalization for spherical K-means (cosine similarity)"
            )

        # Perform K-means clustering
        logger.info("Running K-means clustering...")
        self.kmeans.fit(features)
        self.cluster_assignments = self.kmeans.labels_

        # Compute silhouette score
        logger.info("Computing silhouette score...")
        unique_labels = np.unique(self.cluster_assignments)

        if len(unique_labels) > 1:
            self.silhouette = float(
                silhouette_score(features, self.cluster_assignments)
            )
        else:
            self.silhouette = float("nan")
            logger.warning(
                "Silhouette score is undefined: all points assigned to a single cluster"
            )

        self.is_fitted = True

        logger.info(f"Clustering complete! Silhouette score: {self.silhouette:.3f}")
        self._log_cluster_distribution()

        return self

    def predict(self, questions: List[CodeQuestion]) -> np.ndarray:
        """Predict cluster assignments for new questions.

        Args:
            questions: List of CodeQuestion objects

        Returns:
            Numpy array of cluster IDs

        Raises:
            ValueError: If predict is called before fit
        """
        if not self.is_fitted:
            raise ValueError("Must call fit before predict")

        # Extract features
        features = self.feature_extractor.transform(questions)

        # Normalize if using spherical k-means
        if self.use_spherical:
            from sklearn.preprocessing import normalize

            features = normalize(features, norm="l2")

        # Predict clusters
        return self.kmeans.predict(features)

    def assign_clusters(self, questions: List[CodeQuestion]) -> np.ndarray:
        """Assign questions to clusters (alias for predict).

        This method name matches the paper's terminology: "assign each validation
        set prompt to a cluster".

        Args:
            questions: List of CodeQuestion objects

        Returns:
            Numpy array of cluster IDs

        Raises:
            ValueError: If called before fit
        """
        return self.predict(questions)

    def assign_question(self, question_text: str) -> Tuple[int, float]:
        """Assign a single question to the nearest cluster.

        Args:
            question_text: Raw question text

        Returns:
            Tuple of (cluster_id, distance_to_centroid)

        Raises:
            ValueError: If called before fit
        """
        if not self.is_fitted:
            raise ValueError("Must call fit before assign_question")

        # Extract features directly from text (no wrapper needed)
        features = self.feature_extractor.transform([question_text])

        # Normalize if using spherical k-means (consistency with fit/predict)
        if self.use_spherical:
            from sklearn.preprocessing import normalize

            features = normalize(features, norm="l2")

        # Predict cluster and compute distance
        cluster_id = int(self.kmeans.predict(features)[0])
        distances = self.kmeans.transform(features)[0]
        distance = float(distances[cluster_id])

        return cluster_id, distance

    def save(self, output_dir: Path, save_pickle: bool = False) -> Dict[str, str]:
        """Save clustering artifacts as lightweight JSON (+ optional pickle).

        Saves lightweight JSON files for Git tracking and optionally full pickle for local use.

        Args:
            output_dir: Directory to save artifacts
            save_pickle: Whether to save pickle file (for local use, gitignored)

        Returns:
            Dictionary with file paths
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving cluster engine to {output_dir}")

        # 1. Save cluster centers as JSON (Git-friendly)
        cluster_centers_file = output_dir / "cluster_centers.json"
        cluster_data = {
            "cluster_centers": self.kmeans.cluster_centers_.tolist(),
            "cluster_assignments": {
                q.question_id: int(cluster_id)
                for q, cluster_id in zip(self.questions, self.cluster_assignments)
            },
            "n_clusters": self.n_clusters,
            "feature_dim": self.kmeans.cluster_centers_.shape[1],
            "feature_info": self.feature_extractor.get_feature_info(),
        }
        with open(cluster_centers_file, "w") as f:
            json.dump(cluster_data, f, indent=2)

        cluster_centers_size = cluster_centers_file.stat().st_size / 1024
        logger.info(f"✅ Saved cluster_centers.json ({cluster_centers_size:.1f} KB)")

        # 2. Save TF-IDF vocabulary as JSON (Git-friendly)
        tfidf_vocab_file = output_dir / "tfidf_vocabulary.json"

        # Convert vocabulary to native Python types (numpy int64 -> int)
        vocabulary_native = {
            str(k): int(v)
            for k, v in self.feature_extractor.tfidf_vectorizer.vocabulary_.items()
        }

        tfidf_data = {
            "vocabulary": vocabulary_native,
            "idf": self.feature_extractor.tfidf_vectorizer.idf_.tolist(),
            "max_features": int(self.feature_extractor.tfidf_vectorizer.max_features),
            "ngram_range": list(self.feature_extractor.tfidf_vectorizer.ngram_range),
        }
        with open(tfidf_vocab_file, "w") as f:
            json.dump(tfidf_data, f, indent=2)

        tfidf_vocab_size = tfidf_vocab_file.stat().st_size / 1024
        logger.info(f"✅ Saved tfidf_vocabulary.json ({tfidf_vocab_size:.1f} KB)")

        # 3. Save scaler parameters as JSON (Git-friendly)
        logger.info("Saving scaler parameters...")
        scaler_params_file = output_dir / "scaler_parameters.json"
        scaler_data = {
            "embedding_scaler": {
                "mean": self.feature_extractor.embedding_scaler.mean_.tolist(),
                "scale": self.feature_extractor.embedding_scaler.scale_.tolist(),
            },
            "tfidf_scaler": {
                "mean": self.feature_extractor.tfidf_scaler.mean_.tolist(),
                "scale": self.feature_extractor.tfidf_scaler.scale_.tolist(),
            },
        }
        with open(scaler_params_file, "w") as f:
            json.dump(scaler_data, f, indent=2)

        scaler_params_size = scaler_params_file.stat().st_size / 1024
        logger.info(f"✅ Saved scaler_parameters.json ({scaler_params_size:.1f} KB)")

        # 4. Save metadata with enhanced config info
        cluster_info = self.get_cluster_info()
        metadata = {
            "n_clusters": cluster_info["n_clusters"],
            "n_train_questions": cluster_info["n_questions"],
            "silhouette_score": cluster_info["silhouette_score"],
            "embedding_model": self.feature_extractor.embedding_model_name,
            "embedding_dim": self.feature_extractor.embedding_dim,
            "tfidf_max_features": self.feature_extractor.tfidf_vectorizer.max_features,
            "tfidf_ngram_range": list(
                self.feature_extractor.tfidf_vectorizer.ngram_range
            ),
            "total_features": self.feature_extractor.embedding_dim
            + self.feature_extractor.tfidf_vectorizer.max_features,
            "cluster_sizes": cluster_info["cluster_sizes"],
        }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("✅ Saved metadata.json")

        # Prepare return dictionary
        result = {
            "cluster_file": str(cluster_centers_file),
            "tfidf_vocab_file": str(tfidf_vocab_file),
            "scaler_params_file": str(scaler_params_file),
            "metadata_file": str(metadata_file),
        }

        # 5. Optionally save pickle for local use (will be gitignored)
        if save_pickle:
            pickle_file = output_dir / "cluster_engine.pkl"
            with open(pickle_file, "wb") as f:
                pickle.dump(self, f)

            pickle_size = pickle_file.stat().st_size / 1024 / 1024
            logger.info(
                f"✅ Saved cluster_engine.pkl ({pickle_size:.1f} MB) [local only, gitignored]"
            )
            result["pickle_file"] = str(pickle_file)

            total_json_size = (
                cluster_centers_size
                + tfidf_vocab_size
                + scaler_params_size
                + (metadata_file.stat().st_size / 1024)
            )
            logger.info(f"\n📊 Total JSON size: {total_json_size:.1f} KB (Git-tracked)")
            logger.info(f"📊 Pickle size: {pickle_size:.1f} MB (local only)")
            logger.info(
                f"📊 Size reduction for Git: {(1 - total_json_size / 1024 / pickle_size) * 100:.1f}%"
            )

        return result

    def _log_cluster_distribution(self) -> None:
        """Log distribution of questions across clusters."""
        unique, counts = np.unique(self.cluster_assignments, return_counts=True)

        logger.info("Cluster distribution:")
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(self.cluster_assignments)) * 100
            logger.info(
                f"  Cluster {cluster_id}: {count} questions ({percentage:.1f}%)"
            )

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about clustering results.

        Returns:
            Dictionary with clustering statistics
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}

        unique, counts = np.unique(self.cluster_assignments, return_counts=True)

        return {
            "n_clusters": self.n_clusters,
            "n_questions": len(self.questions),
            "silhouette_score": self.silhouette,
            "cluster_sizes": {
                int(cluster_id): int(count) for cluster_id, count in zip(unique, counts)
            },
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "avg_cluster_size": float(counts.mean()),
        }
