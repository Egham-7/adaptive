"""Feature extraction for clustering: TF-IDF + Semantic Embeddings."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from adaptive_router.services.unirouter.schemas import CodeQuestion

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract hybrid features combining TF-IDF and semantic embeddings."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tfidf_max_features: int = 5000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        embedding_weight: float = 0.7,
        tfidf_weight: float = 0.3,
    ) -> None:
        """Initialize feature extractor.

        Args:
            embedding_model: HuggingFace model for semantic embeddings
            tfidf_max_features: Maximum TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF
            embedding_weight: Weight for embedding features (0.0-1.0)
            tfidf_weight: Weight for TF-IDF features (0.0-1.0)
        """
        logger.info(f"Initializing FeatureExtractor with model: {embedding_model}")

        # Store model name
        self.embedding_model_name = embedding_model

        # Semantic embedding model (force CPU for macOS compatibility)
        import platform

        device = (
            "cpu"
            if platform.system() == "Darwin"
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Loading embedding model on device: {device}")

        # Load with trust_remote_code for compatibility
        try:
            self.embedding_model = SentenceTransformer(
                embedding_model, device=device, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(
                f"Failed to load with trust_remote_code, trying without: {e}"
            )
            self.embedding_model = SentenceTransformer(embedding_model, device=device)

        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
        )

        # Scalers for normalization
        self.embedding_scaler = StandardScaler()
        self.tfidf_scaler = StandardScaler()

        # Feature weights
        self.embedding_weight = embedding_weight
        self.tfidf_weight = tfidf_weight

        self.is_fitted = False

        logger.info(
            f"Feature dimensions: {self.embedding_dim} (embeddings) + "
            f"{tfidf_max_features} (TF-IDF)"
        )

    def fit_transform(self, questions: List[CodeQuestion]) -> np.ndarray:
        """Fit on questions and transform to hybrid features.

        Args:
            questions: List of CodeQuestion objects

        Returns:
            Numpy array of shape (n_questions, n_features)
        """
        logger.info(f"Extracting features from {len(questions)} questions...")

        # Extract question texts
        texts = [q.question for q in questions]

        # 1. Generate semantic embeddings
        logger.info("Generating semantic embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )

        # 2. Generate TF-IDF features
        logger.info("Generating TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()

        # 3. Normalize features individually
        logger.info("Normalizing features...")
        embeddings_normalized = self.embedding_scaler.fit_transform(embeddings)
        tfidf_normalized = self.tfidf_scaler.fit_transform(tfidf_features)

        # 4. Concatenate features (not add - different dimensions!)
        logger.info("Combining features...")
        hybrid_features = np.concatenate(
            [embeddings_normalized, tfidf_normalized], axis=1
        )

        self.is_fitted = True

        logger.info(f"Feature extraction complete! Shape: {hybrid_features.shape}")
        return hybrid_features

    def transform(self, questions: List[CodeQuestion]) -> np.ndarray:
        """Transform questions to hybrid features (must call fit_transform first).

        Args:
            questions: List of CodeQuestion objects

        Returns:
            Numpy array of shape (n_questions, n_features)

        Raises:
            ValueError: If transform is called before fit_transform
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_transform before transform")

        logger.info(f"Transforming {len(questions)} questions...")

        # Extract question texts
        texts = [q.question for q in questions]

        # 1. Generate semantic embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True,
        )

        # 2. Generate TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()

        # 3. Normalize features
        embeddings_normalized = self.embedding_scaler.transform(embeddings)
        tfidf_normalized = self.tfidf_scaler.transform(tfidf_features)

        # 4. Concatenate features
        hybrid_features = np.concatenate(
            [embeddings_normalized, tfidf_normalized], axis=1
        )

        return hybrid_features

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about extracted features.

        Returns:
            Dictionary with feature extraction details
        """
        return {
            "embedding_dim": self.embedding_dim,
            "tfidf_features": self.tfidf_vectorizer.max_features,
            "tfidf_vocabulary_size": (
                len(self.tfidf_vectorizer.vocabulary_) if self.is_fitted else 0
            ),
            "embedding_weight": self.embedding_weight,
            "tfidf_weight": self.tfidf_weight,
            "total_features": (self.embedding_dim + self.tfidf_vectorizer.max_features),
            "is_fitted": self.is_fitted,
        }
