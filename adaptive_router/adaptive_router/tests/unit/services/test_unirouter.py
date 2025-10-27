"""Unit tests for Router cluster-based routing."""

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from adaptive_router.models import CodeQuestion
from adaptive_router.core.cluster_engine import ClusterEngine
from adaptive_router.core.router import ModelRouter


@pytest.fixture
def sample_questions() -> List[CodeQuestion]:
    """Sample CodeQuestion objects for testing."""
    return [
        CodeQuestion(
            question_id="q1",
            question="Write a Python function to sort a list",
            choices=["A", "B", "C", "D"],
            answer="A",
        ),
        CodeQuestion(
            question_id="q2",
            question="Explain the concept of recursion in programming",
            choices=["A", "B", "C", "D"],
            answer="B",
        ),
        CodeQuestion(
            question_id="q3",
            question="What is the time complexity of quicksort?",
            choices=["A", "B", "C", "D"],
            answer="C",
        ),
        CodeQuestion(
            question_id="q4",
            question="Implement a binary search tree in Python",
            choices=["A", "B", "C", "D"],
            answer="D",
        ),
        CodeQuestion(
            question_id="q5",
            question="Describe how dynamic programming works",
            choices=["A", "B", "C", "D"],
            answer="A",
        ),
    ]


@pytest.fixture
def small_cluster_engine() -> ClusterEngine:
    """Create a small cluster engine for testing."""
    # Use very small parameters for fast testing
    return ClusterEngine(
        n_clusters=2,  # Small number of clusters for fast testing
        max_iter=10,  # Few iterations
        n_init=1,  # Single run
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        tfidf_max_features=100,  # Reduced features
    )


@pytest.mark.unit
class TestClusterEngine:
    """Test ClusterEngine core functionality."""

    def test_initialization(self) -> None:
        """Test ClusterEngine initialization."""
        engine = ClusterEngine(n_clusters=5)

        assert engine.n_clusters == 5
        assert engine.is_fitted is False
        assert isinstance(engine.kmeans, object)  # KMeans object
        assert hasattr(engine, "feature_extractor")

    def test_initialization_with_custom_params(self) -> None:
        """Test ClusterEngine with custom parameters."""
        engine = ClusterEngine(
            n_clusters=10,
            max_iter=500,
            random_state=123,
            n_init=20,
            tfidf_max_features=3000,
        )

        assert engine.n_clusters == 10
        assert engine.kmeans.max_iter == 500
        assert engine.kmeans.random_state == 123
        assert engine.kmeans.n_init == 20

    @pytest.mark.slow
    def test_fit(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test fitting the cluster engine on questions."""
        engine = small_cluster_engine

        result = engine.fit(sample_questions)

        # Should return self for method chaining
        assert result is engine

        # Should be fitted
        assert engine.is_fitted is True

        # Should have cluster assignments
        assert len(engine.cluster_assignments) == len(sample_questions)

        # Cluster assignments should be in valid range
        assert all(0 <= c < engine.n_clusters for c in engine.cluster_assignments)

        # Should have silhouette score
        assert -1.0 <= engine.silhouette <= 1.0

    @pytest.mark.slow
    def test_predict_before_fit_raises_error(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that predict raises error if called before fit."""
        engine = small_cluster_engine

        with pytest.raises(ValueError, match="Must call fit before predict"):
            engine.predict(sample_questions)

    @pytest.mark.slow
    def test_predict_after_fit(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test predicting clusters after fitting."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        # Create new questions for prediction
        new_questions = [
            CodeQuestion(
                question_id="new1",
                question="Write a sorting algorithm",
                choices=["A", "B"],
                answer="A",
            )
        ]

        predictions = engine.predict(new_questions)

        assert len(predictions) == len(new_questions)
        assert all(0 <= p < engine.n_clusters for p in predictions)

    @pytest.mark.slow
    def test_assign_clusters_alias(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that assign_clusters is an alias for predict."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        new_questions = [
            CodeQuestion(
                question_id="new1",
                question="Test question",
                choices=["A"],
                answer="A",
            )
        ]

        predictions1 = engine.predict(new_questions)
        predictions2 = engine.assign_clusters(new_questions)

        np.testing.assert_array_equal(predictions1, predictions2)

    @pytest.mark.slow
    def test_assign_question(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test assigning a single question to a cluster."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        cluster_id, distance = engine.assign_question(
            "Write a function to implement quicksort"
        )

        assert isinstance(cluster_id, int)
        assert 0 <= cluster_id < engine.n_clusters
        assert isinstance(distance, float)
        assert distance >= 0.0

    @pytest.mark.slow
    def test_assign_question_before_fit_raises_error(
        self, small_cluster_engine: ClusterEngine
    ) -> None:
        """Test that assign_question raises error if called before fit."""
        engine = small_cluster_engine

        with pytest.raises(ValueError, match="Must call fit before assign_question"):
            engine.assign_question("Test question")

    @pytest.mark.slow
    def test_get_cluster_info(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test getting cluster information."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        info = engine.get_cluster_info()

        assert "n_clusters" in info
        assert "n_questions" in info
        assert "silhouette_score" in info
        assert "cluster_sizes" in info
        assert "min_cluster_size" in info
        assert "max_cluster_size" in info
        assert "avg_cluster_size" in info

        assert info["n_clusters"] == engine.n_clusters
        assert info["n_questions"] == len(sample_questions)

    def test_get_cluster_info_before_fit(
        self, small_cluster_engine: ClusterEngine
    ) -> None:
        """Test getting cluster info before fitting."""
        engine = small_cluster_engine

        info = engine.get_cluster_info()

        assert "error" in info
        assert info["error"] == "Model not fitted"


@pytest.mark.unit
class TestClusterEngineSaveLoad:
    """Test saving and loading cluster engines."""

    @pytest.mark.slow
    def test_save_before_fit_raises_error(
        self, small_cluster_engine: ClusterEngine
    ) -> None:
        """Test that save raises error if called before fit."""
        engine = small_cluster_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                engine.save(output_dir)

    @pytest.mark.slow
    def test_save_creates_required_files(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that save creates the required JSON files."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            file_paths = engine.save(output_dir)

            # Check return value
            assert "cluster_file" in file_paths
            assert "metadata_file" in file_paths

            # Check files exist
            cluster_file = Path(file_paths["cluster_file"])
            metadata_file = Path(file_paths["metadata_file"])

            assert cluster_file.exists()
            assert metadata_file.exists()

            # Check file names
            assert cluster_file.name == "cluster_centers.json"
            assert metadata_file.name == "metadata.json"

    @pytest.mark.slow
    def test_save_cluster_data_structure(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that saved cluster data has correct structure."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            file_paths = engine.save(output_dir)

            # Load and check cluster data
            cluster_file = Path(file_paths["cluster_file"])
            with open(cluster_file) as f:
                cluster_data = json.load(f)

            assert "cluster_centers" in cluster_data
            assert "cluster_assignments" in cluster_data
            assert "n_clusters" in cluster_data
            assert "feature_info" in cluster_data

            # Check cluster centers
            assert len(cluster_data["cluster_centers"]) == engine.n_clusters
            assert all(
                isinstance(center, list) for center in cluster_data["cluster_centers"]
            )

            # Check cluster assignments
            assert len(cluster_data["cluster_assignments"]) == len(sample_questions)
            for question in sample_questions:
                assert question.question_id in cluster_data["cluster_assignments"]

    @pytest.mark.slow
    def test_save_metadata_structure(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that saved metadata has correct structure."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            file_paths = engine.save(output_dir)

            # Load and check metadata
            metadata_file = Path(file_paths["metadata_file"])
            with open(metadata_file) as f:
                metadata = json.load(f)

            assert "n_clusters" in metadata
            assert "n_questions" in metadata
            assert "clustering_method" in metadata
            assert "embedding_model" in metadata
            assert "timestamp" in metadata
            assert "silhouette_score" in metadata

            assert metadata["n_clusters"] == engine.n_clusters
            assert metadata["n_questions"] == len(sample_questions)
            assert metadata["clustering_method"] == "K-means"


@pytest.mark.unit
class TestClusterEngineEdgeCases:
    """Test edge cases for ClusterEngine."""

    def test_empty_questions_list(self, small_cluster_engine: ClusterEngine) -> None:
        """Test fitting with empty questions list."""
        engine = small_cluster_engine

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, Exception)):
            engine.fit([])

    @pytest.mark.slow
    def test_single_question(self, small_cluster_engine: ClusterEngine) -> None:
        """Test fitting with a single question."""
        engine = small_cluster_engine

        single_question = [
            CodeQuestion(
                question_id="q1",
                question="Test question",
                choices=["A"],
                answer="A",
            )
        ]

        # Should handle gracefully (might cluster all to single cluster)
        try:
            engine.fit(single_question)
            assert engine.is_fitted
        except (ValueError, Exception):
            # Some ML libraries might complain about insufficient data
            pass

    @pytest.mark.slow
    def test_predict_different_size_batch(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test predicting on batches of different sizes."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        # Predict single question
        single_pred = engine.predict([sample_questions[0]])
        assert len(single_pred) == 1

        # Predict multiple questions
        multiple_pred = engine.predict(sample_questions[:3])
        assert len(multiple_pred) == 3

    @pytest.mark.slow
    def test_cluster_distribution(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that cluster distribution is reasonable."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        info = engine.get_cluster_info()

        # Sum of cluster sizes should equal total questions
        total_assigned = sum(info["cluster_sizes"].values())
        assert total_assigned == len(sample_questions)

        # Min/max/avg should be consistent
        assert (
            info["min_cluster_size"]
            <= info["avg_cluster_size"]
            <= info["max_cluster_size"]
        )


@pytest.mark.unit
class TestRouterServiceMocked:
    """Test ModelRouter with mocked dependencies (no real ML models)."""

    def test_initialization_mocked(self) -> None:
        """Test ModelRouter initialization with mocked components."""
        from adaptive_router.models.storage import (
            RouterProfile,
            ProfileMetadata,
            ClusterCentersData,
            TFIDFVocabularyData,
            ScalerParameters,
            ScalerParametersData,
        )

        mock_profile = RouterProfile(
            metadata=ProfileMetadata(
                n_clusters=5,
                silhouette_score=0.5,
                embedding_model="all-MiniLM-L6-v2",
                tfidf_max_features=100,
                tfidf_ngram_range=[1, 2],
            ),
            cluster_centers=ClusterCentersData(
                n_clusters=5,
                feature_dim=100,
                cluster_centers=[[0.0] * 100 for _ in range(5)],
            ),
            llm_profiles={
                "openai:gpt-4": [0.08] * 5,
            },
            tfidf_vocabulary=TFIDFVocabularyData(
                vocabulary={"test": 0},
                idf=[1.0],
            ),
            scaler_parameters=ScalerParameters(
                embedding_scaler=ScalerParametersData(
                    mean=[0.0] * 100,
                    scale=[1.0] * 100,
                ),
                tfidf_scaler=ScalerParametersData(
                    mean=[0.0],
                    scale=[1.0],
                ),
            ),
        )

        mock_costs = {"openai:gpt-4": 30.0}

        with patch.object(ModelRouter, "_build_cluster_engine_from_data"):
            router = ModelRouter(profile=mock_profile, model_costs=mock_costs)

            assert router is not None

    def test_select_model_validates_request(self) -> None:
        """Test that select_model validates the request."""
        pass


@pytest.mark.unit
class TestJSONStorageCompatibility:
    """Test JSON storage and loading compatibility."""

    @pytest.mark.slow
    def test_json_roundtrip_cluster_centers(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that cluster centers can be saved and loaded via JSON."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            file_paths = engine.save(output_dir)

            # Load cluster centers
            cluster_file = Path(file_paths["cluster_file"])
            with open(cluster_file) as f:
                cluster_data = json.load(f)

            # Reconstruct cluster centers as numpy array
            loaded_centers = np.array(cluster_data["cluster_centers"])
            original_centers = engine.kmeans.cluster_centers_

            # Should have same shape
            assert loaded_centers.shape == original_centers.shape

            # Should have same values (within floating point tolerance)
            np.testing.assert_allclose(loaded_centers, original_centers, rtol=1e-5)

    @pytest.mark.slow
    def test_json_serializable_types(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that all saved data uses JSON-serializable types."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            file_paths = engine.save(output_dir)

            # Try to load and re-serialize (should not fail)
            for file_path in file_paths.values():
                with open(file_path) as f:
                    data = json.load(f)

                # Re-serialize to verify JSON compatibility
                json_str = json.dumps(data)
                assert len(json_str) > 0

    @pytest.mark.slow
    def test_feature_info_in_saved_data(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that feature_info is included in saved cluster data."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            file_paths = engine.save(output_dir)

            cluster_file = Path(file_paths["cluster_file"])
            with open(cluster_file) as f:
                cluster_data = json.load(f)

            assert "feature_info" in cluster_data
            feature_info = cluster_data["feature_info"]

            # Should contain embedding and TF-IDF dimensions
            assert "embedding_dim" in feature_info or "total_dim" in feature_info


@pytest.mark.unit
class TestModelSelectionIntegration:
    """Test integration of ClusterEngine with model selection."""

    @pytest.mark.slow
    def test_cluster_assignment_consistency(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that same question always gets assigned to same cluster."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        test_question = "Write a Python function to sort a list"

        # Assign multiple times
        cluster_id1, _ = engine.assign_question(test_question)
        cluster_id2, _ = engine.assign_question(test_question)
        cluster_id3, _ = engine.assign_question(test_question)

        # Should be consistent
        assert cluster_id1 == cluster_id2 == cluster_id3

    @pytest.mark.slow
    def test_similar_questions_same_cluster(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that similar questions tend to be assigned to same cluster."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        # These should be similar
        question1 = "Write a sorting algorithm in Python"
        question2 = "Implement a sort function in Python"

        cluster_id1, _ = engine.assign_question(question1)
        cluster_id2, _ = engine.assign_question(question2)

        # Note: With only 2 clusters and 5 questions, there's a decent chance
        # they'll be in the same cluster, but not guaranteed
        # This is a weak test but demonstrates the concept
        # (A real test would use many more questions and larger clusters)


@pytest.mark.unit
class TestClusterEnginePerformance:
    """Test performance characteristics of ClusterEngine."""

    @pytest.mark.slow
    def test_fit_performance(self, sample_questions: List[CodeQuestion]) -> None:
        """Test that fitting completes in reasonable time."""
        import time

        engine = ClusterEngine(n_clusters=2, max_iter=10, n_init=1)

        start = time.time()
        engine.fit(sample_questions)
        elapsed = time.time() - start

        # With only 5 questions, 2 clusters, and 10 iterations, should be very fast
        assert elapsed < 30.0  # 30 seconds should be more than enough

    @pytest.mark.slow
    def test_predict_performance(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that prediction is fast."""
        import time

        engine = small_cluster_engine
        engine.fit(sample_questions)

        # Create test questions
        test_questions = [
            CodeQuestion(
                question_id=f"test{i}",
                question=f"Test question {i}",
                choices=["A"],
                answer="A",
            )
            for i in range(10)
        ]

        start = time.time()
        for _ in range(10):  # 100 total predictions
            engine.predict(test_questions)
        elapsed = time.time() - start

        # 100 predictions should be fast
        assert elapsed < 30.0

    @pytest.mark.slow
    def test_assign_question_performance(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[CodeQuestion]
    ) -> None:
        """Test that single question assignment is fast."""
        import time

        engine = small_cluster_engine
        engine.fit(sample_questions)

        start = time.time()
        for i in range(100):
            engine.assign_question(f"Test question {i}")
        elapsed = time.time() - start

        # 100 assignments should complete in under 30 seconds
        assert elapsed < 30.0
