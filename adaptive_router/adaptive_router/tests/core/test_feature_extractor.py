"""Tests for FeatureExtractor."""

import numpy as np
import pytest

from adaptive_router.core.feature_extractor import FeatureExtractor
from adaptive_router.models import CodeQuestion


@pytest.fixture
def sample_questions() -> list[CodeQuestion]:
    """Create sample code questions for testing."""
    return [
        CodeQuestion(
            question_id="q1",
            question="How do I sort a list in Python?",
            choices=["A) sorted()", "B) sort()", "C) order()", "D) arrange()"],
            answer="A",
        ),
        CodeQuestion(
            question_id="q2",
            question="What is a lambda function?",
            choices=[
                "A) Anonymous function",
                "B) Named function",
                "C) Class method",
                "D) Built-in",
            ],
            answer="A",
        ),
        CodeQuestion(
            question_id="q3",
            question="How to reverse a string in JavaScript?",
            choices=[
                "A) reverse()",
                "B) split().reverse().join()",
                "C) backwards()",
                "D) flip()",
            ],
            answer="B",
        ),
        CodeQuestion(
            question_id="q4",
            question="Explain async/await in Python",
            choices=[
                "A) Async/await syntax",
                "B) Threading",
                "C) Multiprocessing",
                "D) Synchronous",
            ],
            answer="A",
        ),
        CodeQuestion(
            question_id="q5",
            question="How to use map in JavaScript?",
            choices=["A) map()", "B) forEach()", "C) filter()", "D) reduce()"],
            answer="A",
        ),
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

        feature_extractor.fit_transform(sample_questions)

        assert feature_extractor.is_fitted

    def test_fit_returns_self(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test that fit returns self for method chaining."""
        result = feature_extractor.fit_transform(sample_questions)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(sample_questions)

    def test_fit_with_empty_questions(
        self, feature_extractor: FeatureExtractor
    ) -> None:
        """Test fit handles empty questions list."""
        with pytest.raises((ValueError, AttributeError)):
            feature_extractor.fit_transform([])


class TestFeatureExtractorTransform:
    """Test FeatureExtractor transformation."""

    def test_transform_single_question(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test transforming a single question."""
        feature_extractor.fit_transform(sample_questions)

        new_question = CodeQuestion(
            question_id="test_q_new",
            question="How to use list comprehension in Python?",
            choices=[
                "A) [x for x in list]",
                "B) list.comp()",
                "C) for x in list: x",
                "D) map()",
            ],
            answer="A",
        )
        features = feature_extractor.transform([new_question])

        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        assert features.shape[0] == 1
        assert features.shape[1] > 0

    def test_transform_before_fit_raises_error(
        self, feature_extractor: FeatureExtractor
    ) -> None:
        """Test transform raises error when not fitted."""
        question = CodeQuestion(
            question_id="test_q_single",
            question="Test question",
            choices=["A) Answer A", "B) Answer B", "C) Answer C", "D) Answer D"],
            answer="A",
        )

        with pytest.raises((ValueError, AttributeError)):
            feature_extractor.transform([question])


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

        extractor2.fit_transform(sample_questions)
        features2 = extractor2.transform(sample_questions)

        np.testing.assert_array_almost_equal(features1, features2)


class TestFeatureExtractorEdgeCases:
    """Test FeatureExtractor edge cases."""

    def test_transform_with_special_characters(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test transformation handles special characters."""
        feature_extractor.fit_transform(sample_questions)

        special_question = CodeQuestion(
            question="How to use @decorator and #comment in Python?",
            question_id="test_q_special",
            choices=["A) Special answer", "B) Other", "C) Another", "D) Last"],
            answer="A",
        )
        features = feature_extractor.transform([special_question])

        assert isinstance(features, np.ndarray)
        assert not np.isnan(features).any()

    def test_transform_with_empty_question(
        self, feature_extractor: FeatureExtractor, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test transformation handles empty question text."""
        feature_extractor.fit_transform(sample_questions)

        empty_question = CodeQuestion(
            question_id="test_q_empty",
            question="",
            choices=["A) Empty", "B) Blank", "C) Nothing", "D) Void"],
            answer="A",
        )
        features = feature_extractor.transform([empty_question])

        assert isinstance(features, np.ndarray)
        # Features should still be generated (likely zeros or defaults)
        assert len(features) > 0
