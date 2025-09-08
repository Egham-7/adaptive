"""Tests for classification models."""

import pytest

from adaptive_ai.models.llm_classification_models import ClassificationResult


def create_classification_result(
    task_type_1=None, task_type_2=None, size=1, **overrides
):
    """Helper to create ClassificationResult with all required fields."""
    if task_type_1 is None:
        task_type_1 = ["Code Generation"] * size
    if task_type_2 is None:
        task_type_2 = ["Other"] * size

    defaults = {
        "task_type_1": task_type_1,
        "task_type_2": task_type_2,
        "task_type_prob": [0.8] * size,
        "creativity_scope": [0.2] * size,
        "reasoning": [0.7] * size,
        "contextual_knowledge": [0.3] * size,
        "prompt_complexity_score": [0.65] * size,
        "domain_knowledge": [0.4] * size,
        "number_of_few_shots": [0] * size,
        "no_label_reason": [0.9] * size,
        "constraint_ct": [0.1] * size,
    }
    defaults.update(overrides)
    return ClassificationResult(**defaults)


class TestClassificationResult:
    """Test ClassificationResult model."""

    def test_minimal_classification_result(self):
        """Test creating ClassificationResult with minimal fields."""
        result = ClassificationResult(
            task_type_1=["Code Generation"],
            task_type_2=["Other"],
            task_type_prob=[0.8],
            creativity_scope=[0.2],
            reasoning=[0.7],
            contextual_knowledge=[0.3],
            prompt_complexity_score=[0.65],
            domain_knowledge=[0.4],
            number_of_few_shots=[0],
            no_label_reason=[0.9],
            constraint_ct=[0.1],
        )

        assert result.task_type_1 == ["Code Generation"]
        assert result.prompt_complexity_score == [0.65]
        assert result.domain_knowledge == [0.4]

    def test_full_classification_result(self):
        """Test creating ClassificationResult with all fields."""
        result = ClassificationResult(
            task_type_1=["Code Generation", "Open QA"],
            task_type_2=["Summarization", "Classification"],
            task_type_prob=[0.89, 0.76],
            creativity_scope=[0.2, 0.8],
            reasoning=[0.7, 0.4],
            contextual_knowledge=[0.3, 0.6],
            prompt_complexity_score=[0.75, 0.65],
            domain_knowledge=[0.1, 0.9],
            number_of_few_shots=[0, 3],
            no_label_reason=[0.9, 0.85],
            constraint_ct=[0.2, 0.5],
        )

        assert result.task_type_1 == ["Code Generation", "Open QA"]
        assert result.prompt_complexity_score == [0.75, 0.65]
        assert result.domain_knowledge == [0.1, 0.9]

    def test_task_type_validation(self):
        """Test task type list validation."""
        # Valid task types
        result = ClassificationResult(task_type_1=["code", "chat", "analysis"])
        assert len(result.task_type_1) == 3

        # Empty list should be allowed
        result = ClassificationResult(task_type_1=[])
        assert result.task_type_1 == []

        # Single item list
        result = ClassificationResult(task_type_1=["code"])
        assert result.task_type_1 == ["code"]

    def test_complexity_score_validation(self):
        """Test complexity score validation."""
        # Valid complexity scores
        result = ClassificationResult(prompt_complexity_score=[0.0])
        assert result.prompt_complexity_score == [0.0]

        result = ClassificationResult(prompt_complexity_score=[0.5])
        assert result.prompt_complexity_score == [0.5]

        result = ClassificationResult(prompt_complexity_score=[1.0])
        assert result.prompt_complexity_score == [1.0]

        # Multiple scores (if supported)
        result = ClassificationResult(prompt_complexity_score=[0.3, 0.7, 0.9])
        assert len(result.prompt_complexity_score) == 3

    def test_domain_classification(self):
        """Test domain classification fields."""
        result = ClassificationResult(domain_1=["technology", "science", "business"])

        assert result.domain_1 == ["technology", "science", "business"]
        assert len(result.domain_1) == 3

    def test_complex_classification_scenario(self):
        """Test realistic classification scenario."""
        result = ClassificationResult(
            task_type_1=["code", "technical_writing"],
            prompt_complexity_score=[0.82],
            domain_1=["software_engineering", "documentation"],
        )

        # Verify all fields are set correctly
        assert "code" in result.task_type_1
        assert "technical_writing" in result.task_type_1
        assert result.prompt_complexity_score[0] == 0.82
        assert "software_engineering" in result.domain_1
        assert "documentation" in result.domain_1

    def test_serialization(self):
        """Test ClassificationResult serialization."""
        original = ClassificationResult(
            task_type_1=["code", "analysis"],
            prompt_complexity_score=[0.65],
            domain_1=["technology"],
        )

        # Serialize to dict
        data = original.model_dump()

        assert data["task_type_1"] == ["code", "analysis"]
        assert data["prompt_complexity_score"] == [0.65]
        assert data["domain_1"] == ["technology"]

        # Deserialize from dict
        restored = ClassificationResult(**data)

        assert restored.task_type_1 == original.task_type_1
        assert restored.prompt_complexity_score == original.prompt_complexity_score
        assert restored.domain_1 == original.domain_1

    def test_none_values_handling(self):
        """Test handling of None values."""
        result = ClassificationResult(
            task_type_1=None, prompt_complexity_score=None, domain_1=None
        )

        assert result.task_type_1 is None
        assert result.prompt_complexity_score is None
        assert result.domain_1 is None

    def test_empty_lists_vs_none(self):
        """Test distinction between empty lists and None."""
        result1 = ClassificationResult(task_type_1=[])
        result2 = ClassificationResult(task_type_1=None)

        assert result1.task_type_1 == []
        assert result2.task_type_1 is None
        assert result1.task_type_1 != result2.task_type_1

    def test_model_equality(self):
        """Test equality comparison between ClassificationResult instances."""
        result1 = ClassificationResult(
            task_type_1=["code"], prompt_complexity_score=[0.5]
        )

        result2 = ClassificationResult(
            task_type_1=["code"], prompt_complexity_score=[0.5]
        )

        result3 = ClassificationResult(
            task_type_1=["chat"], prompt_complexity_score=[0.5]
        )

        assert result1 == result2
        assert result1 != result3

    def test_json_serialization(self):
        """Test JSON serialization compatibility."""
        result = ClassificationResult(
            task_type_1=["code", "analysis"],
            prompt_complexity_score=[0.75],
            domain_1=["technology"],
        )

        # Test that it can be converted to JSON-compatible format
        json_data = result.model_dump()

        # All values should be JSON-serializable
        import json

        json_string = json.dumps(json_data)

        # Should be able to round-trip
        restored_data = json.loads(json_string)
        restored_result = ClassificationResult(**restored_data)

        assert restored_result == result


@pytest.mark.unit
class TestClassificationResultEdgeCases:
    """Test edge cases for ClassificationResult."""

    def test_very_large_lists(self):
        """Test with very large classification lists."""
        size = 100
        large_task_list_1 = [f"task_{i}" for i in range(size)]
        large_task_list_2 = [f"secondary_task_{i}" for i in range(size)]
        large_score_list = [i / 100.0 for i in range(size)]
        large_int_list = [i % 5 for i in range(size)]  # few_shots values 0-4

        result = ClassificationResult(
            task_type_1=large_task_list_1,
            task_type_2=large_task_list_2,
            task_type_prob=large_score_list,
            creativity_scope=large_score_list,
            reasoning=large_score_list,
            contextual_knowledge=large_score_list,
            prompt_complexity_score=large_score_list,
            domain_knowledge=large_score_list,
            number_of_few_shots=large_int_list,
            no_label_reason=large_score_list,
            constraint_ct=large_score_list,
        )

        assert len(result.task_type_1) == 100
        assert len(result.prompt_complexity_score) == 100
        assert result.task_type_1[0] == "task_0"
        assert result.prompt_complexity_score[0] == 0.0

    def test_unicode_task_types(self):
        """Test handling of unicode characters in task types."""
        result = create_classification_result(
            task_type_1=["코딩", "分析", "créatif", "🤖"],
            task_type_2=["Other", "Other", "Other", "Other"],
            size=4,
        )

        assert "코딩" in result.task_type_1
        assert "🤖" in result.task_type_1
        assert len(result.task_type_1) == 4

    def test_extreme_complexity_scores(self):
        """Test edge cases for complexity scores."""
        # Very small positive numbers
        result1 = create_classification_result(prompt_complexity_score=[0.000001])
        assert result1.prompt_complexity_score[0] == 0.000001

        # Very close to 1
        result2 = create_classification_result(prompt_complexity_score=[0.999999])
        assert result2.prompt_complexity_score[0] == 0.999999

        # Exactly 0 and 1
        result3 = create_classification_result(
            prompt_complexity_score=[0.0, 1.0], size=2
        )
        assert 0.0 in result3.prompt_complexity_score
        assert 1.0 in result3.prompt_complexity_score

    def test_mixed_data_types_in_lists(self):
        """Test that lists maintain type consistency."""
        # All strings for task types
        result = create_classification_result(
            task_type_1=["code", "analysis", "creative"], size=3
        )

        assert all(isinstance(item, str) for item in result.task_type_1)

        # All floats for complexity scores
        result = create_classification_result(
            prompt_complexity_score=[0.1, 0.5, 0.9], size=3
        )

        assert all(
            isinstance(item, int | float) for item in result.prompt_complexity_score
        )
