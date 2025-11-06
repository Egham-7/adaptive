"""Tests for FeatureExtractor."""

import numpy as np
import pytest

from adaptive_router.core.feature_extractor import FeatureExtractor
from adaptive_router.models import CodeQuestion


@pytest.fixture
def sample_questions() -> list[CodeQuestion]:
    """Create sample code questions for testing."""
    return [
        CodeQuestion(question="How do I sort a list in Python?", language="python"),
        CodeQuestion(question="What is a lambda function?", language="python"),
        CodeQuestion(
            question="How to reverse a string in JavaScript?", language="javascript"
        ),
        CodeQuestion(question="Explain async/await in Python", language="python"),
        CodeQuestion(question="How to use map in JavaScript?", language="javascript"),
    ]


@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    """Create a FeatureExtractor with test configuration."""
    return FeatureExtractor(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        tfidf_max_features=100,
        tfidf_ngram_range=(1, 2),
    )


class TestFeatureExtractorInitialization:
    """Test FeatureExtractor initialization."""

    def test_default_initialization(self) -> None:
        """Test FeatureExtractor initializes with default parameters."""
        extractor = FeatureExtractor()
        assert (
            extractor.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert not extractor.is_fitted

    def test_custom_parameters(self) -> None:
        """Test FeatureExtractor with custom parameters."""
        extractor = FeatureExtractor(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            tfidf_max_features=1000,
            tfidf_ngram_range=(1, 3),
        )
        assert extractor.tfidf_vectorizer.max_features == 1000
        assert extractor.tfidf_vectorizer.ngram_range == (1, 3)


class TestFeatureExtractorFit:
    """Test FeatureExtractor fitting."""

    def test_fit_updates_state(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test that fit updates extractor state correctly."""
        assert not feature_extractor.is_fitted

        feature_extractor.fit(sample_questions)

        assert feature_extractor.is_fitted

    def test_fit_returns_self(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test that fit returns self for method chaining."""
        result = feature_extractor.fit(sample_questions)
        assert result is feature_extractor

    def test_fit_with_empty_questions(
        self, feature_extractor: FeatureExtractor
    ) -> None:
        """Test fit handles empty questions list."""
        with pytest.raises((ValueError, AttributeError)):
            feature_extractor.fit([])


class TestFeatureExtractorTransform:
    """Test FeatureExtractor transformation."""

    def test_transform_single_question(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test transforming a single question."""
        feature_extractor.fit(sample_questions)

        new_question = CodeQuestion(
            question="How to use list comprehension in Python?", language="python"
        )
        features = feature_extractor.transform(new_question)

        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        assert len(features) > 0

    def test_transform_batch(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test batch transformation of multiple questions."""
        feature_extractor.fit(sample_questions)

        new_questions = [
            CodeQuestion(question="Python list sorting", language="python"),
            CodeQuestion(question="JavaScript array methods", language="javascript"),
        ]

        features = feature_extractor.transform_batch(new_questions)

        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        assert features.shape[0] == len(new_questions)

    def test_transform_before_fit_raises_error(
        self, feature_extractor: FeatureExtractor
    ) -> None:
        """Test transform raises error when not fitted."""
        question = CodeQuestion(question="Test question", language="python")

        with pytest.raises((ValueError, AttributeError)):
            feature_extractor.transform(question)


class TestFeatureExtractorFitTransform:
    """Test FeatureExtractor fit_transform."""

    def test_fit_transform(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test fit_transform combines fit and transform."""
        features = feature_extractor.fit_transform(sample_questions)

        assert feature_extractor.is_fitted
        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        assert features.shape[0] == len(sample_questions)

    def test_fit_transform_equals_fit_then_transform(
        self, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test fit_transform produces same result as fit then transform."""
        extractor1 = FeatureExtractor(tfidf_max_features=100)
        extractor2 = FeatureExtractor(tfidf_max_features=100)

        features1 = extractor1.fit_transform(sample_questions)

        extractor2.fit(sample_questions)
        features2 = extractor2.transform_batch(sample_questions)

        np.testing.assert_array_almost_equal(features1, features2)


class TestFeatureExtractorPersistence:
    """Test FeatureExtractor save/load functionality."""

    def test_save_and_load(
        self,
        feature_extractor: FeatureExtractor,
        sample_questions: list[CodeQuestion],
        tmp_path,
    ) -> None:
        """Test saving and loading feature extractor."""
        feature_extractor.fit(sample_questions)

        # Save to temp directory
        save_path = tmp_path / "test_feature_extractor"
        feature_extractor.save(save_path)

        # Load from saved path
        loaded_extractor = FeatureExtractor.load(save_path)

        assert loaded_extractor.is_fitted
        assert (
            loaded_extractor.embedding_model_name
            == feature_extractor.embedding_model_name
        )

        # Test that loaded extractor produces same features
        test_question = sample_questions[0]
        original_features = feature_extractor.transform(test_question)
        loaded_features = loaded_extractor.transform(test_question)

        np.testing.assert_array_almost_equal(original_features, loaded_features)


class TestFeatureExtractorEdgeCases:
    """Test FeatureExtractor edge cases."""

    def test_transform_with_special_characters(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test transformation handles special characters."""
        feature_extractor.fit(sample_questions)

        special_question = CodeQuestion(
            question="How to use @decorator and #comment in Python?",
            language="python",
        )
        features = feature_extractor.transform(special_question)

        assert isinstance(features, np.ndarray)
        assert not np.isnan(features).any()

    def test_transform_with_empty_question(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test transformation handles empty question text."""
        feature_extractor.fit(sample_questions)

        empty_question = CodeQuestion(question="", language="python")
        features = feature_extractor.transform(empty_question)

        assert isinstance(features, np.ndarray)
        # Features should still be generated (likely zeros or defaults)
        assert len(features) > 0
