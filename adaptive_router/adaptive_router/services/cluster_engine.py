"""Unified clustering engine for semantic clustering of coding questions."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from adaptive_router.models import ClusterMetadata, CodeQuestion
from adaptive_router.services.feature_extractor import FeatureExtractor

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
        embedding_weight: float = 0.7,
        tfidf_weight: float = 0.3,
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
            embedding_weight: Weight for embedding features
            tfidf_weight: Weight for TF-IDF features
        """
        logger.info(f"Initializing ClusterEngine with K={n_clusters}")

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            embedding_model=embedding_model,
            tfidf_max_features=tfidf_max_features,
            tfidf_ngram_range=tfidf_ngram_range,
            embedding_weight=embedding_weight,
            tfidf_weight=tfidf_weight,
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
        self.silhouette = float(silhouette_score(features, self.cluster_assignments))

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
            question_text: Question text

        Returns:
            Tuple of (cluster_id, distance_to_centroid)

        Raises:
            ValueError: If called before fit
        """
        if not self.is_fitted:
            raise ValueError("Must call fit before assign_question")

        # Create temporary CodeQuestion
        temp_question = CodeQuestion(
            question_id="temp",
            question=question_text,
            choices=["A", "B", "C", "D"],
            answer="A",
        )

        # Extract features
        features = self.feature_extractor.transform([temp_question])

        # Predict cluster and compute distance
        cluster_id = int(self.kmeans.predict(features)[0])
        distances = self.kmeans.transform(features)[0]
        distance = float(distances[cluster_id])

        return cluster_id, distance

    def save(self, output_dir: Path) -> Dict[str, str]:
        """Save clustering artifacts as lightweight JSON.

        Args:
            output_dir: Directory to save artifacts

        Returns:
            Dictionary with file paths
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving clustering artifacts to {output_dir}")

        # Prepare cluster centers (lightweight)
        cluster_centers = self.kmeans.cluster_centers_.tolist()

        # Prepare cluster assignments with question IDs
        cluster_data = {
            "cluster_centers": cluster_centers,
            "cluster_assignments": {
                q.question_id: int(cluster_id)
                for q, cluster_id in zip(self.questions, self.cluster_assignments)
            },
            "n_clusters": self.n_clusters,
            "feature_info": self.feature_extractor.get_feature_info(),
        }

        # Save cluster data
        cluster_file = output_dir / "cluster_centers.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_data, f, indent=2)

        # Save metadata
        metadata = ClusterMetadata(
            n_clusters=self.n_clusters,
            n_questions=len(self.questions),
            clustering_method="K-means",
            embedding_model=self.feature_extractor.embedding_model_name,
            timestamp=datetime.now().isoformat(),
            silhouette_score=self.silhouette,
        )

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        logger.info(
            f"Saved cluster_centers.json ({cluster_file.stat().st_size / 1024:.1f} KB)"
        )
        logger.info(
            f"Saved metadata.json ({metadata_file.stat().st_size / 1024:.1f} KB)"
        )

        return {
            "cluster_file": str(cluster_file),
            "metadata_file": str(metadata_file),
        }

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
