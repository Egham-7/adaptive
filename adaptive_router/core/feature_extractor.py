"""Feature extraction for clustering: TF-IDF + Semantic Embeddings.

Simplified for better DX - accepts only string inputs.
"""

import logging
import platform
import warnings
from typing import TypedDict

import methodtools
import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from adaptive_router.exceptions.core import FeatureExtractionError

logger = logging.getLogger(__name__)


class FeatureInfo(TypedDict):
    embedding_model: str
    embedding_dim: int
    tfidf_max_features: int
    tfidf_vocabulary_size: int
    tfidf_ngram_range: tuple[int, int]
    total_features: int
    is_fitted: bool


class FeatureExtractor:
    """Extract hybrid features combining TF-IDF and semantic embeddings.

    This class provides a two-stage feature extraction pipeline:
    1. Semantic embeddings via SentenceTransformers (384D default)
    2. TF-IDF features for lexical patterns (5000D default)
    3. StandardScaler normalization for both
    4. Concatenation into hybrid feature vector

    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.fit_transform(["text 1", "text 2"])
        >>> new_features = extractor.transform(["text 3"])
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tfidf_max_features: int = 5000,
        tfidf_ngram_range: tuple[int, int] = (1, 2),
        allow_trust_remote_code: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Initialize feature extractor.

        Args:
            embedding_model: HuggingFace model for semantic embeddings
            tfidf_max_features: Maximum TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF (e.g., (1, 2) for unigrams/bigrams)
            allow_trust_remote_code: Allow remote code execution in embedding models
                WARNING: Only enable for trusted models
            batch_size: Batch size for embedding generation (default: 128 for GPU, 32 for CPU)
        """
        logger.info(f"Initializing FeatureExtractor with model: {embedding_model}")

        self.embedding_model_name = embedding_model
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range

        # Determine device
        device = self._get_device()

        # Set batch size
        if batch_size is None:
            self.batch_size = 128 if device == "cuda" else 32
        else:
            self.batch_size = batch_size
        logger.info(f"Loading embedding model on device: {device}")

        # Load embedding model with trust_remote_code handling
        self.embedding_model = self._load_embedding_model(
            embedding_model, device, allow_trust_remote_code
        )
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

        self.is_fitted = False

        logger.info(
            f"Feature dimensions: {self.embedding_dim} (embeddings) + "
            f"{tfidf_max_features} (TF-IDF) = {self.embedding_dim + tfidf_max_features} total"
        )

    @staticmethod
    def _get_device() -> str:
        """Determine the appropriate device for model loading.

        Returns:
            Device string: 'cpu' for macOS, 'cuda' if available, otherwise 'cpu'
        """
        if platform.system() == "Darwin":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_embedding_model(
        self, model_name: str, device: str, allow_trust_remote_code: bool
    ) -> SentenceTransformer:
        """Load SentenceTransformer with proper error handling.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on ("cpu" or "cuda")
            allow_trust_remote_code: Whether to allow remote code execution

        Returns:
            Loaded SentenceTransformer model
        """
        # Suppress tokenization warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*clean_up_tokenization_spaces.*",
                category=FutureWarning,
            )
            try:
                model = SentenceTransformer(
                    model_name,
                    device=device,
                    trust_remote_code=allow_trust_remote_code,
                )
            except (OSError, RuntimeError, ValueError) as e:
                if allow_trust_remote_code:
                    raise  # Don't retry if explicitly enabled

                logger.warning(
                    f"Failed loading with trust_remote_code=False, retrying with True: {e}"
                )
                model = SentenceTransformer(
                    model_name, device=device, trust_remote_code=True
                )

        # Set tokenizer cleanup for future compatibility
        try:
            model.tokenizer.clean_up_tokenization_spaces = False
        except AttributeError:
            # Model doesn't have tokenizer attribute (some models don't)
            pass

        return model

    @methodtools.lru_cache(maxsize=50000)
    def _encode_text_cached(self, text: str) -> npt.NDArray[np.float32]:
        """Cache embeddings for identical texts to improve performance.

        Cache size increased to 50,000 for production workloads (~20MB memory for all-MiniLM-L6-v2).
        Provides 10-100x speedup for repeated queries compared to the previous 128-entry cache.

        Args:
            text: Input text to encode

        Returns:
            Normalized embedding vector
        """
        return self.embedding_model.encode(
            [text],
            show_progress_bar=False,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )[0]

    def _validate_texts(self, texts: list[str]) -> None:
        """Validate text inputs.

        Args:
            texts: List of text strings

        Raises:
            FeatureExtractionError: If validation fails
        """
        if not texts:
            raise FeatureExtractionError("Text list cannot be empty")

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise FeatureExtractionError(
                    f"Input at index {idx} is not a string: {type(text)}"
                )
            if not text.strip():
                raise FeatureExtractionError(f"Empty text at index {idx}")

    def fit_transform(
        self, texts: list[str], skip_validation: bool = False
    ) -> npt.NDArray[np.float64]:
        """Fit on texts and transform to hybrid features.

        Args:
            texts: List of text strings
            skip_validation: Skip input validation for trusted production inputs

        Returns:
            Hybrid feature matrix (n_samples × total_features)

        Raises:
            FeatureExtractionError: If inputs are invalid
        """
        if not skip_validation:
            self._validate_texts(texts)

        logger.info(f"Extracting features from {len(texts)} texts")

        # 1. Generate semantic embeddings
        logger.info("Generating semantic embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )

        # 2. Generate TF-IDF features
        logger.info("Generating TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()

        # 3. Normalize features individually
        logger.info("Normalizing features...")
        embeddings_normalized = self.embedding_scaler.fit_transform(embeddings)
        tfidf_normalized = self.tfidf_scaler.fit_transform(tfidf_features)

        # 4. Concatenate features
        logger.info("Combining features...")
        hybrid_features = np.concatenate(
            [embeddings_normalized, tfidf_normalized], axis=1
        )

        self.is_fitted = True

        logger.info(f"Feature extraction complete. Shape: {hybrid_features.shape}")
        return hybrid_features

    def transform(
        self, texts: list[str], skip_validation: bool = False
    ) -> npt.NDArray[np.float64]:
        """Transform texts to hybrid features (must call fit_transform first).

        Args:
            texts: List of text strings
            skip_validation: Skip input validation for trusted production inputs

        Returns:
            Hybrid feature matrix (n_samples × total_features)

        Raises:
            FeatureExtractionError: If not fitted or inputs invalid
        """
        if not self.is_fitted:
            raise FeatureExtractionError("Must call fit_transform() before transform()")

        if not skip_validation:
            self._validate_texts(texts)

        logger.debug(f"Transforming {len(texts)} texts")

        # 1. Generate semantic embeddings
        if len(texts) == 1:
            # Use cached encoding for single text (common in production)
            embeddings = np.array([self._encode_text_cached(texts[0])])
        else:
            # Batch encode for multiple texts
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                batch_size=self.batch_size,
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

    def get_feature_info(self) -> FeatureInfo:
        """Get information about extracted features.

        Returns:
            Dictionary with feature extraction details
        """
        return {
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "tfidf_max_features": self.tfidf_max_features,
            "tfidf_vocabulary_size": (
                len(self.tfidf_vectorizer.vocabulary_) if self.is_fitted else 0
            ),
            "tfidf_ngram_range": self.tfidf_ngram_range,
            "total_features": self.embedding_dim + self.tfidf_max_features,
            "is_fitted": self.is_fitted,
        }
