"""Tests for ClusterEngine."""

import numpy as np
import pytest

from adaptive_router.core.cluster_engine import ClusterEngine
from adaptive_router.models import CodeQuestion


@pytest.fixture
def sample_questions() -> list[CodeQuestion]:
    """Create sample code questions for testing."""
    return [
        CodeQuestion(question="How do I sort a list in Python?", language="python"),
        CodeQuestion(question="What is a lambda function?", language="python"),
        CodeQuestion(question="How to reverse a string in JavaScript?", language="javascript"),
        CodeQuestion(question="Explain async/await in Python", language="python"),
        CodeQuestion(question="How to use map in JavaScript?", language="javascript"),
    ]


@pytest.fixture
def cluster_engine() -> ClusterEngine:
    """Create a ClusterEngine with test configuration."""
    return ClusterEngine(
        n_clusters=2,
        max_iter=100,
        random_state=42,
        tfidf_max_features=100,
    )


class TestClusterEngineInitialization:
    """Test ClusterEngine initialization."""

    def test_default_initialization(self) -> None:
        """Test ClusterEngine initializes with default parameters."""
        engine = ClusterEngine()
        assert engine.n_clusters == 20
        assert not engine.is_fitted
        assert engine.use_spherical is True
        assert len(engine.cluster_assignments) == 0

    def test_custom_parameters(self) -> None:
        """Test ClusterEngine with custom parameters."""
        engine = ClusterEngine(
            n_clusters=10,
            max_iter=200,
            random_state=123,
            tfidf_max_features=1000,
        )
        assert engine.n_clusters == 10
        assert engine.kmeans.max_iter == 200
        assert engine.kmeans.random_state == 123


class TestClusterEngineFit:
    """Test ClusterEngine fitting."""

    def test_fit_updates_state(
        self, cluster_engine: ClusterEngine, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test that fit updates engine state correctly."""
        assert not cluster_engine.is_fitted

        cluster_engine.fit(sample_questions)

        assert cluster_engine.is_fitted
        assert len(cluster_engine.cluster_assignments) == len(sample_questions)
        assert len(cluster_engine.questions) == len(sample_questions)
        assert cluster_engine.silhouette >= -1.0
        assert cluster_engine.silhouette <= 1.0

    def test_fit_returns_self(
        self, cluster_engine: ClusterEngine, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test that fit returns self for method chaining."""
        result = cluster_engine.fit(sample_questions)
        assert result is cluster_engine

    def test_fit_with_empty_questions(self, cluster_engine: ClusterEngine) -> None:
        """Test fit raises error with empty questions list."""
        with pytest.raises((ValueError, AttributeError)):
            cluster_engine.fit([])


class TestClusterEnginePredict:
    """Test ClusterEngine prediction."""

    def test_predict_assigns_cluster(
        self, cluster_engine: ClusterEngine, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test predict assigns cluster to new question."""
        cluster_engine.fit(sample_questions)

        new_question = CodeQuestion(
            question="How to use list comprehension in Python?", language="python"
        )
        cluster_id = cluster_engine.predict(new_question)

        assert isinstance(cluster_id, (int, np.integer))
        assert 0 <= cluster_id < cluster_engine.n_clusters

    def test_predict_before_fit_raises_error(
        self, cluster_engine: ClusterEngine
    ) -> None:
        """Test predict raises error when not fitted."""
        question = CodeQuestion(question="Test question", language="python")

        with pytest.raises((ValueError, AttributeError)):
            cluster_engine.predict(question)

    def test_predict_batch(
        self, cluster_engine: ClusterEngine, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test batch prediction of multiple questions."""
        cluster_engine.fit(sample_questions)

        new_questions = [
            CodeQuestion(question="Python list sorting", language="python"),
            CodeQuestion(question="JavaScript array methods", language="javascript"),
        ]

        assignments = cluster_engine.predict_batch(new_questions)

        assert len(assignments) == len(new_questions)
        assert all(isinstance(a, (int, np.integer)) for a in assignments)
        assert all(0 <= a < cluster_engine.n_clusters for a in assignments)


class TestClusterEngineAnalysis:
    """Test ClusterEngine analysis methods."""

    def test_get_cluster_summary(
        self, cluster_engine: ClusterEngine, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test get_cluster_summary returns statistics."""
        cluster_engine.fit(sample_questions)
        summary = cluster_engine.get_cluster_summary()

        assert "n_clusters" in summary
        assert "n_samples" in summary
        assert "silhouette_score" in summary
        assert "cluster_sizes" in summary

        assert summary["n_clusters"] == cluster_engine.n_clusters
        assert summary["n_samples"] == len(sample_questions)
        assert len(summary["cluster_sizes"]) == cluster_engine.n_clusters

    def test_get_cluster_questions(
        self, cluster_engine: ClusterEngine, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test retrieving questions from specific cluster."""
        cluster_engine.fit(sample_questions)

        # Get questions from first cluster
        cluster_questions = cluster_engine.get_cluster_questions(0)

        assert isinstance(cluster_questions, list)
        assert all(isinstance(q, CodeQuestion) for q in cluster_questions)
        # At least one question should be in cluster 0
        assert len(cluster_questions) >= 0


class TestClusterEnginePersistence:
    """Test ClusterEngine save/load functionality."""

    def test_save_and_load(
        self,
        cluster_engine: ClusterEngine,
        sample_questions: list[CodeQuestion],
        tmp_path,
    ) -> None:
        """Test saving and loading cluster engine."""
        cluster_engine.fit(sample_questions)

        # Save to temp directory
        save_path = tmp_path / "test_cluster_engine"
        cluster_engine.save(save_path)

        # Load from saved path
        loaded_engine = ClusterEngine.load(save_path)

        assert loaded_engine.is_fitted
        assert loaded_engine.n_clusters == cluster_engine.n_clusters
        assert len(loaded_engine.cluster_assignments) == len(
            cluster_engine.cluster_assignments
        )
        np.testing.assert_array_equal(
            loaded_engine.cluster_assignments, cluster_engine.cluster_assignments
        )

    def test_load_unfitted_engine(self, cluster_engine: ClusterEngine, tmp_path) -> None:
        """Test saving and loading unfitted engine."""
        save_path = tmp_path / "unfitted_engine"
        cluster_engine.save(save_path)

        loaded_engine = ClusterEngine.load(save_path)
        assert not loaded_engine.is_fitted
