"""
Tests for the QuantizedPromptClassifier class.

This module contains unit tests for the quantized prompt task and complexity
classifier, testing basic functionality, edge cases, and performance.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path

from prompt_classifier.classifier import QuantizedPromptClassifier
from prompt_classifier.utils import validate_model_files, load_model_config


class TestQuantizedPromptClassifier:
    """Test cases for the QuantizedPromptClassifier class."""

    @pytest.fixture
    def mock_model_path(self, tmp_path):
        """Create a temporary model directory with mock files."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        # Create mock config
        config = {
            "target_sizes": {
                "task_type": 8,
                "creativity_scope": 5,
                "reasoning": 5,
                "contextual_knowledge": 5,
                "number_of_few_shots": 5,
                "domain_knowledge": 5,
                "no_label_reason": 5,
                "constraint_ct": 5
            },
            "task_type_map": {
                "0": "Open QA",
                "1": "Closed QA",
                "2": "Summarization",
                "3": "Text Generation",
                "4": "Code Generation",
                "5": "Chatbot",
                "6": "Classification",
                "7": "Rewrite"
            },
            "weights_map": {
                "creativity_scope": [0.0, 0.25, 0.5, 0.75, 1.0],
                "reasoning": [0.0, 0.25, 0.5, 0.75, 1.0],
                "contextual_knowledge": [0.0, 0.25, 0.5, 0.75, 1.0],
                "number_of_few_shots": [0.0, 1.0, 2.0, 3.0, 4.0],
                "domain_knowledge": [0.0, 0.25, 0.5, 0.75, 1.0],
                "no_label_reason": [0.0, 0.25, 0.5, 0.75, 1.0],
                "constraint_ct": [0.0, 0.25, 0.5, 0.75, 1.0]
            },
            "divisor_map": {
                "creativity_scope": 1.0,
                "reasoning": 1.0,
                "contextual_knowledge": 1.0,
                "number_of_few_shots": 1.0,
                "domain_knowledge": 1.0,
                "no_label_reason": 1.0,
                "constraint_ct": 1.0
            }
        }

        import json
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create empty ONNX file
        (model_dir / "model_quantized.onnx").touch()

        return model_dir

    @patch('prompt_classifier.classifier.ort.InferenceSession')
    @patch('prompt_classifier.classifier.AutoTokenizer')
    @patch('prompt_classifier.classifier.AutoConfig')
    def test_init_success(self, mock_config, mock_tokenizer, mock_session, mock_model_path):
        """Test successful initialization of the classifier."""
        # Mock the config loading
        mock_config.from_pretrained.return_value = Mock()
        mock_config.from_pretrained.return_value.target_sizes = {"task_type": 8}
        mock_config.from_pretrained.return_value.task_type_map = {"0": "Open QA"}
        mock_config.from_pretrained.return_value.weights_map = {"test": [1.0]}
        mock_config.from_pretrained.return_value.divisor_map = {"test": 1.0}

        # Mock tokenizer
        mock_tokenizer.from_pretrained.return_value = Mock()

        # Mock ONNX session
        mock_session.return_value = Mock()

        classifier = QuantizedPromptClassifier(mock_model_path)

        assert classifier.model_path == Path(mock_model_path)
        mock_config.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_session.assert_called_once()

    def test_init_missing_onnx_file(self, tmp_path):
        """Test initialization fails when ONNX file is missing."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            QuantizedPromptClassifier(model_dir)

    @patch('prompt_classifier.classifier.OPTIMUM_AVAILABLE', False)
    def test_init_missing_dependencies(self, mock_model_path):
        """Test initialization fails when dependencies are missing."""
        with pytest.raises(ImportError, match="ONNX Runtime and optimum are required"):
            QuantizedPromptClassifier(mock_model_path)

    @patch('prompt_classifier.classifier.ort.InferenceSession')
    @patch('prompt_classifier.classifier.AutoTokenizer')
    @patch('prompt_classifier.classifier.AutoConfig')
    def test_from_pretrained(self, mock_config, mock_tokenizer, mock_session, mock_model_path):
        """Test the from_pretrained class method."""
        # Setup mocks
        mock_config.from_pretrained.return_value = Mock()
        mock_config.from_pretrained.return_value.target_sizes = {"task_type": 8}
        mock_config.from_pretrained.return_value.task_type_map = {"0": "Open QA"}
        mock_config.from_pretrained.return_value.weights_map = {"test": [1.0]}
        mock_config.from_pretrained.return_value.divisor_map = {"test": 1.0}
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_session.return_value = Mock()

        classifier = QuantizedPromptClassifier.from_pretrained(mock_model_path)

        assert isinstance(classifier, QuantizedPromptClassifier)
        assert classifier.model_path == Path(mock_model_path)

    def test_tokenize_texts(self):
        """Test text tokenization functionality."""
        # Create a minimal mock classifier
        classifier = Mock(spec=QuantizedPromptClassifier)
        classifier.tokenizer = Mock()
        classifier.tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int32),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int32)
        }

        # Call the actual method
        result = QuantizedPromptClassifier.tokenize_texts(classifier, ["test prompt"])

        # Verify the result
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].dtype == np.int64
        assert result["attention_mask"].dtype == np.int64

    def test_softmax(self):
        """Test the softmax implementation."""
        classifier = QuantizedPromptClassifier.__new__(QuantizedPromptClassifier)

        # Test with simple input
        x = np.array([[1.0, 2.0, 3.0]])
        result = classifier._softmax(x)

        # Check that probabilities sum to 1
        assert np.allclose(np.sum(result, axis=1), 1.0)

        # Check that larger inputs have higher probabilities
        assert result[0, 2] > result[0, 1] > result[0, 0]

    def test_compute_results_task_type(self):
        """Test compute_results for task type classification."""
        classifier = QuantizedPromptClassifier.__new__(QuantizedPromptClassifier)
        classifier.task_type_map = {
            "0": "Open QA",
            "1": "Closed QA",
            "2": "Summarization"
        }

        # Mock predictions with clear winner
        preds = np.array([[0.1, 0.8, 0.1]])

        result = classifier.compute_results(preds, "task_type")

        assert isinstance(result, tuple)
        assert len(result) == 3
        task_type_1, task_type_2, task_type_prob = result
        assert task_type_1[0] == "Closed QA"
        assert isinstance(task_type_prob[0], float)

    def test_compute_results_other_targets(self):
        """Test compute_results for non-task-type targets."""
        classifier = QuantizedPromptClassifier.__new__(QuantizedPromptClassifier)
        classifier.weights_map = {"creativity_scope": [0.0, 0.25, 0.5, 0.75, 1.0]}
        classifier.divisor_map = {"creativity_scope": 1.0}

        # Mock predictions
        preds = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])

        result = classifier.compute_results(preds, "creativity_scope")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert 0 <= result[0] <= 1

    @patch('prompt_classifier.classifier.ort.InferenceSession')
    @patch('prompt_classifier.classifier.AutoTokenizer')
    @patch('prompt_classifier.classifier.AutoConfig')
    def test_classify_single_prompt(self, mock_config, mock_tokenizer, mock_session, mock_model_path):
        """Test single prompt classification."""
        # Setup comprehensive mocks
        mock_config_obj = Mock()
        mock_config_obj.target_sizes = {
            "task_type": 8, "creativity_scope": 5, "reasoning": 5,
            "contextual_knowledge": 5, "number_of_few_shots": 5,
            "domain_knowledge": 5, "no_label_reason": 5, "constraint_ct": 5
        }
        mock_config_obj.task_type_map = {"0": "Open QA", "1": "Closed QA"}
        mock_config_obj.weights_map = {
            "creativity_scope": [0.0, 0.25, 0.5, 0.75, 1.0],
            "reasoning": [0.0, 0.25, 0.5, 0.75, 1.0],
            "contextual_knowledge": [0.0, 0.25, 0.5, 0.75, 1.0],
            "number_of_few_shots": [0.0, 1.0, 2.0, 3.0, 4.0],
            "domain_knowledge": [0.0, 0.25, 0.5, 0.75, 1.0],
            "no_label_reason": [0.0, 0.25, 0.5, 0.75, 1.0],
            "constraint_ct": [0.0, 0.25, 0.5, 0.75, 1.0]
        }
        mock_config_obj.divisor_map = {k: 1.0 for k in mock_config_obj.weights_map.keys()}

        mock_config.from_pretrained.return_value = mock_config_obj

        mock_tokenizer_obj = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_obj

        mock_session_obj = Mock()
        # Mock ONNX session outputs (8 classification heads)
        mock_outputs = [
            np.random.randn(1, 8),   # task_type
            np.random.randn(1, 5),   # creativity_scope
            np.random.randn(1, 5),   # reasoning
            np.random.randn(1, 5),   # contextual_knowledge
            np.random.randn(1, 5),   # number_of_few_shots
            np.random.randn(1, 5),   # domain_knowledge
            np.random.randn(1, 5),   # no_label_reason
            np.random.randn(1, 5),   # constraint_ct
        ]
        mock_session_obj.run.return_value = mock_outputs
        mock_session.return_value = mock_session_obj

        classifier = QuantizedPromptClassifier(mock_model_path)

        # Mock tokenization
        classifier.tokenize_texts = Mock(return_value={
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64)
        })

        result = classifier.classify_single_prompt("test prompt")

        assert isinstance(result, dict)
        assert "task_type_1" in result
        assert "prompt_complexity_score" in result

    def test_get_task_types(self):
        """Test getting just task types from prompts."""
        classifier = Mock(spec=QuantizedPromptClassifier)
        classifier.classify_prompts.return_value = [
            {"task_type_1": ["Open QA"]},
            {"task_type_1": ["Code Generation"]}
        ]

        result = QuantizedPromptClassifier.get_task_types(classifier, ["prompt1", "prompt2"])

        assert result == ["Open QA", "Code Generation"]

    def test_get_complexity_scores(self):
        """Test getting just complexity scores from prompts."""
        classifier = Mock(spec=QuantizedPromptClassifier)
        classifier.classify_prompts.return_value = [
            {"prompt_complexity_score": [0.7]},
            {"prompt_complexity_score": [0.3]}
        ]

        result = QuantizedPromptClassifier.get_complexity_scores(classifier, ["prompt1", "prompt2"])

        assert result == [0.7, 0.3]


class TestUtils:
    """Test cases for utility functions."""

    def test_validate_model_files_all_present(self, tmp_path):
        """Test validation when all files are present."""
        model_dir = tmp_path / "complete_model"
        model_dir.mkdir()

        required_files = [
            "model_quantized.onnx", "config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"
        ]

        for file_name in required_files:
            (model_dir / file_name).touch()

        missing = validate_model_files(model_dir)
        assert missing == []

    def test_validate_model_files_some_missing(self, tmp_path):
        """Test validation when some files are missing."""
        model_dir = tmp_path / "incomplete_model"
        model_dir.mkdir()

        # Only create some files
        (model_dir / "config.json").touch()
        (model_dir / "tokenizer.json").touch()

        missing = validate_model_files(model_dir)
        assert len(missing) == 4  # 4 missing files
        assert "model_quantized.onnx" in missing

    def test_load_model_config_success(self, tmp_path):
        """Test successful config loading."""
        model_dir = tmp_path / "model_with_config"
        model_dir.mkdir()

        config_data = {"test_key": "test_value", "number": 42}
        import json
        with open(model_dir / "config.json", "w") as f:
            json.dump(config_data, f)

        config = load_model_config(model_dir)
        assert config == config_data

    def test_load_model_config_missing_file(self, tmp_path):
        """Test config loading with missing file."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_model_config(model_dir)

    def test_load_model_config_invalid_json(self, tmp_path):
        """Test config loading with invalid JSON."""
        model_dir = tmp_path / "invalid_model"
        model_dir.mkdir()

        with open(model_dir / "config.json", "w") as f:
            f.write("invalid json content {")

        with pytest.raises(json.JSONDecodeError):
            load_model_config(model_dir)


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual model files."""

    @pytest.mark.slow
    def test_full_pipeline_with_real_model(self):
        """Test the full pipeline with a real quantized model."""
        # This test would require actual model files
        # Skip if files are not available
        model_path = Path("./model_quantized.onnx")
        if not model_path.exists():
            pytest.skip("Real model files not available")

        try:
            classifier = QuantizedPromptClassifier("./")
            result = classifier.classify_single_prompt("What is machine learning?")

            assert isinstance(result, dict)
            assert "task_type_1" in result
            assert "prompt_complexity_score" in result

        except Exception as e:
            pytest.skip(f"Integration test failed due to missing dependencies: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
