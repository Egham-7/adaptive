"""Tests for ClusterEngine."""

import numpy as np
import pytest

from adaptive_router.core.cluster_engine import ClusterEngine
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
        cluster_engine.fit(sample_questions)

        assert hasattr(cluster_engine.kmeans, "cluster_centers_")
        assert len(cluster_engine.cluster_assignments) == len(sample_questions)

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
        cluster_id, _ = cluster_engine.assign_question(new_question.question)

        assert isinstance(cluster_id, (int, np.integer))
        assert 0 <= cluster_id < cluster_engine.n_clusters

    def test_predict_before_fit_raises_error(
        self, cluster_engine: ClusterEngine
    ) -> None:
        """Test predict raises error when not fitted."""
        question = CodeQuestion(
            question_id="test_q_single",
            question="Test question",
            choices=["A) Answer A", "B) Answer B", "C) Answer C", "D) Answer D"],
            answer="A",
        )

        with pytest.raises(Exception, match="Must call fit_transform before transform"):
            cluster_engine.assign_question(question.question)

    def test_predict_batch(
        self, cluster_engine: ClusterEngine, sample_questions: list[CodeQuestion]
    ) -> None:
        """Test batch prediction of multiple questions."""
        cluster_engine.fit(sample_questions)

        new_questions = [
            CodeQuestion(
                question_id="test_q_batch1",
                question="Python list sorting",
                choices=["A) sorted()", "B) sort()", "C) order()", "D) arrange()"],
                answer="A",
            ),
            CodeQuestion(
                question_id="test_q_batch2",
                question="JavaScript array methods",
                choices=[
                    "A) array.map()",
                    "B) array.forEach()",
                    "C) array.filter()",
                    "D) array.reduce()",
                ],
                answer="A",
            ),
        ]

        assignments = cluster_engine.predict(new_questions)

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
        summary = cluster_engine.cluster_stats

        assert "n_clusters" in summary
        assert "n_questions" in summary
        assert "silhouette_score" in summary
        assert "cluster_sizes" in summary

        assert summary["n_clusters"] == cluster_engine.n_clusters
        assert summary["n_questions"] == len(sample_questions)
        assert len(summary["cluster_sizes"]) == cluster_engine.n_clusters


class TestClusterEnginePersistence:
    """Test ClusterEngine save functionality."""

    def test_save_fitted_engine(
        self,
        cluster_engine: ClusterEngine,
        sample_questions: list[CodeQuestion],
        tmp_path,
    ) -> None:
        """Test saving fitted cluster engine."""
        cluster_engine.fit(sample_questions)

        # Save to temp directory
        save_path = tmp_path / "test_cluster_engine"
        cluster_engine.save(save_path)

        assert save_path.exists()

    def test_save_unfitted_engine_raises_error(
        self, cluster_engine: ClusterEngine, tmp_path
    ) -> None:
        """Test saving unfitted engine raises error."""
        save_path = tmp_path / "unfitted_engine"

        with pytest.raises(Exception, match="Cannot save unfitted"):
            cluster_engine.save(save_path)
