"""Unified clustering engine for semantic clustering of coding questions."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted

from adaptive_router.models import CodeQuestion
from adaptive_router.core.feature_extractor import FeatureExtractor
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class ClusterEngine(BaseEstimator):
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
        # Store parameters for saving/loading
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_init = n_init
        self.embedding_model = embedding_model
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range

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

        self.cluster_assignments: np.ndarray = np.array([])
        self.silhouette: float = 0.0

    def fit(self, questions: List[CodeQuestion]) -> "ClusterEngine":
        """Fit clustering model on questions.

        Args:
            questions: List of CodeQuestion objects

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting clustering model on {len(questions)} questions...")

        # Extract hybrid features
        features = self.feature_extractor.fit_transform(questions)

        # Normalize features for spherical k-means (cosine similarity)
        features = normalize(features, norm="l2")

        # Perform K-means clustering
        self.kmeans.fit(features)
        self.cluster_assignments = self.kmeans.labels_

        # Compute silhouette score
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

        logger.info(f"Clustering complete! Silhouette score: {self.silhouette:.3f}")

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
        check_is_fitted(self, ["kmeans"])

        # Extract features
        features = self.feature_extractor.transform(questions)

        # Normalize features for spherical k-means (cosine similarity)
        from sklearn.preprocessing import normalize

        features = normalize(features, norm="l2")

        # Predict clusters
        return self.kmeans.predict(features)

    def assign_question(self, question_text: str) -> Tuple[int, float]:
        """Assign a single question to the nearest cluster.

        Args:
            question_text: Raw question text

        Returns:
            Tuple of (cluster_id, distance_to_centroid)

        Raises:
            ValueError: If called before fit
        """
        check_is_fitted(self, ["kmeans"])

        # Extract features directly from text (no wrapper needed)
        features = self.feature_extractor.transform([question_text])

        # Normalize features for spherical k-means (cosine similarity)
        features = normalize(features, norm="l2")

        # Predict cluster and compute distance
        cluster_id = int(self.kmeans.predict(features)[0])
        distances = self.kmeans.transform(features)[0]
        distance = float(distances[cluster_id])

        return cluster_id, distance

    def save(self, filepath: Path) -> Dict[str, Path]:
        """Save the clustering engine to JSON files.

        Args:
            filepath: Directory path to save the engine files

        Returns:
            Dictionary with paths to saved files

        Raises:
            Exception: If called before fit
        """
        if not hasattr(self, "kmeans") or not hasattr(self.kmeans, "cluster_centers_"):
            raise Exception("Cannot save unfitted")

        filepath.mkdir(parents=True, exist_ok=True)

        # Save cluster centers
        cluster_centers_file = filepath / "cluster_centers.json"
        cluster_data = {
            "cluster_centers": self.kmeans.cluster_centers_.tolist(),
            "cluster_assignments": {
                q_id: int(cluster_id)
                for q_id, cluster_id in zip(
                    [f"q{i}" for i in range(len(self.cluster_assignments))],
                    self.cluster_assignments,
                )
            },
            "n_clusters": self.n_clusters,
            "feature_info": {"total_dim": self.kmeans.cluster_centers_.shape[1]},
        }
        with open(cluster_centers_file, "w") as f:
            json.dump(cluster_data, f, indent=2)

        # Save metadata
        metadata_file = filepath / "metadata.json"
        metadata = {
            "n_clusters": self.n_clusters,
            "embedding_model": self.embedding_model,
            "tfidf_max_features": self.tfidf_max_features,
            "tfidf_ngram_range": list(self.tfidf_ngram_range),
            "silhouette_score": float(self.silhouette),
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved cluster engine to {filepath}")
        return {"cluster_file": cluster_centers_file, "metadata_file": metadata_file}

    @classmethod
    def load(cls, filepath: Path) -> "ClusterEngine":
        """Load a clustering engine from JSON files.

        Args:
            filepath: Directory path containing the engine files

        Returns:
            Loaded ClusterEngine instance
        """
        cluster_file = filepath / "cluster_centers.json"
        metadata_file = filepath / "metadata.json"

        with open(cluster_file) as f:
            cluster_data = json.load(f)

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Reconstruct the engine
        engine = cls(
            n_clusters=metadata["n_clusters"],
            embedding_model=metadata["embedding_model"],
            tfidf_max_features=metadata["tfidf_max_features"],
            tfidf_ngram_range=tuple(metadata["tfidf_ngram_range"]),
        )

        # Set fitted state
        from sklearn.cluster import KMeans
        import numpy as np

        engine.kmeans = KMeans(n_clusters=metadata["n_clusters"])
        engine.kmeans.cluster_centers_ = np.array(cluster_data["cluster_centers"])
        engine.cluster_assignments = np.array(
            list(cluster_data["cluster_assignments"].values())
        )
        engine.silhouette = metadata["silhouette_score"]

        logger.info(f"Loaded cluster engine from {filepath}")
        return engine

    @property
    def cluster_stats(self) -> Dict[str, Any]:
        """Get information about clustering results.

        Returns:
            Dictionary with clustering statistics
        """
        check_is_fitted(self, ["kmeans"])

        unique, counts = np.unique(self.cluster_assignments, return_counts=True)

        return {
            "n_clusters": self.n_clusters,
            "n_questions": len(self.cluster_assignments),
            "silhouette_score": self.silhouette,
            "cluster_sizes": {
                int(cluster_id): int(count) for cluster_id, count in zip(unique, counts)
            },
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "avg_cluster_size": float(counts.mean()),
        }
