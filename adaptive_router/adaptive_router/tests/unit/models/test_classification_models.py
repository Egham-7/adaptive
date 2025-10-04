"""Tests for classification models."""

import json
from typing import Any

import pytest

from adaptive_router.models.llm_classification_models import ClassificationResult


def create_classification_result(
    task_type_1: str | None = None, task_type_2: str | None = None, **overrides: Any
) -> ClassificationResult:
    """Helper to create ClassificationResult with all required fields."""
    if task_type_1 is None:
        task_type_1 = "Code Generation"
    if task_type_2 is None:
        task_type_2 = "Other"

    defaults = {
        # All fields are required
        "task_type_1": task_type_1,
        "task_type_2": task_type_2,
        "task_type_prob": 0.8,
        "creativity_scope": 0.2,
        "reasoning": 0.7,
        "contextual_knowledge": 0.3,
        "prompt_complexity_score": 0.65,
        "domain_knowledge": 0.4,
        "number_of_few_shots": 0.0,
        "no_label_reason": 0.9,
        "constraint_ct": 0.1,
    }
    defaults.update(overrides)
    return ClassificationResult(**defaults)  # type: ignore[arg-type]


class TestClassificationResult:
    """Test ClassificationResult model."""

    def test_minimal_classification_result(self) -> None:
        """Test creating ClassificationResult with minimal fields."""
        result = create_classification_result(
            # Required fields
            task_type_1="Code Generation",
            prompt_complexity_score=0.65,
            # Optional fields
            task_type_2="Other",
            task_type_prob=0.8,
            creativity_scope=0.2,
            reasoning=0.7,
            contextual_knowledge=0.3,
            domain_knowledge=0.4,
            number_of_few_shots=0.0,
            no_label_reason=0.9,
            constraint_ct=0.1,
        )

        assert result.task_type_1 == "Code Generation"
        assert result.prompt_complexity_score == 0.65
        assert result.domain_knowledge == 0.4

    def test_full_classification_result(self) -> None:
        """Test creating ClassificationResult with all fields."""
        result = create_classification_result(
            # Required fields
            task_type_1="Code Generation",
            prompt_complexity_score=0.75,
            # Optional fields
            task_type_2="Summarization",
            task_type_prob=0.89,
            creativity_scope=0.2,
            reasoning=0.7,
            contextual_knowledge=0.3,
            domain_knowledge=0.1,
            number_of_few_shots=0.0,
            no_label_reason=0.9,
            constraint_ct=0.2,
        )

        assert result.task_type_1 == "Code Generation"
        assert result.prompt_complexity_score == 0.75
        assert result.domain_knowledge == 0.1

    def test_task_type_validation(self) -> None:
        """Test task type validation."""
        # Valid task types
        result = create_classification_result(
            task_type_1="code",
            prompt_complexity_score=0.5,
        )
        assert isinstance(result.task_type_1, str)
        assert result.task_type_1 == "code"

        # All fields are required now, so we use "NA" for task_type_2
        result = create_classification_result(
            task_type_1="code",
            prompt_complexity_score=0.5,
            task_type_2="NA",
        )
        assert result.task_type_2 == "NA"
        assert result.task_type_1 == "code"

        # String value
        result = create_classification_result(
            task_type_1="code",
            prompt_complexity_score=0.5,
        )
        assert result.task_type_1 == "code"

    def test_complexity_score_validation(self) -> None:
        """Test complexity score validation."""
        # Valid complexity scores
        result = create_classification_result(
            task_type_1="Test",
            prompt_complexity_score=0.0,
        )
        assert result.prompt_complexity_score == 0.0

        result = create_classification_result(
            task_type_1="Test",
            prompt_complexity_score=0.5,
        )
        assert result.prompt_complexity_score == 0.5

        result = create_classification_result(
            task_type_1="Test",
            prompt_complexity_score=1.0,
        )
        assert result.prompt_complexity_score == 1.0

        # Score should be a single float value
        result = create_classification_result(
            task_type_1="Test",
            prompt_complexity_score=0.7,
        )
        assert isinstance(result.prompt_complexity_score, int | float)
        assert result.prompt_complexity_score == 0.7

    def test_domain_classification(self) -> None:
        """Test domain classification fields (using optional fields)."""
        result = create_classification_result(
            task_type_1="Test",
            prompt_complexity_score=0.5,
            task_type_2="secondary_task",
            domain_knowledge=0.6,
        )

        assert result.task_type_2 == "secondary_task"
        assert result.domain_knowledge == 0.6

    def test_complex_classification_scenario(self) -> None:
        """Test realistic classification scenario."""
        result = create_classification_result(
            # Required fields
            task_type_1="code",
            prompt_complexity_score=0.82,
            # Optional fields
            task_type_2="technical_writing",
            domain_knowledge=0.7,
            reasoning=0.8,
            creativity_scope=0.3,
        )

        # Verify all fields are set correctly
        assert result.task_type_1 == "code"
        assert result.task_type_2 == "technical_writing"
        assert result.prompt_complexity_score == 0.82
        assert result.domain_knowledge == 0.7
        assert result.reasoning == 0.8
        assert result.creativity_scope == 0.3

    def test_serialization(self) -> None:
        """Test ClassificationResult serialization."""
        original = create_classification_result(
            # Required fields
            task_type_1="code",
            prompt_complexity_score=0.65,
            # Optional fields
            task_type_2="analysis",
            domain_knowledge=0.7,
        )

        # Serialize to dict
        data = original.model_dump()

        assert data["task_type_1"] == "code"
        assert data["prompt_complexity_score"] == 0.65
        assert data["task_type_2"] == "analysis"
        assert data["domain_knowledge"] == 0.7

        # Deserialize from dict
        restored = create_classification_result(**data)

        assert restored.task_type_1 == original.task_type_1
        assert restored.prompt_complexity_score == original.prompt_complexity_score
        assert restored.task_type_2 == original.task_type_2
        assert restored.domain_knowledge == original.domain_knowledge

    def test_none_values_handling(self) -> None:
        """Test handling of None values - all fields are now required."""
        result = create_classification_result(
            # All fields are required now
            task_type_1="Test",
            prompt_complexity_score=0.5,
            task_type_2="NA",  # Use "NA" instead of None
            task_type_prob=0.0,  # Use 0.0 instead of None
        )

        assert result.task_type_1 == "Test"
        assert result.prompt_complexity_score == 0.5
        assert result.task_type_2 == "NA"
        assert result.task_type_prob == 0.0

    def test_empty_strings_vs_none(self) -> None:
        """Test distinction between empty strings and default values."""
        result1 = create_classification_result(
            task_type_1="Test",
            prompt_complexity_score=0.5,
            task_type_2="",  # Empty string
        )
        result2 = create_classification_result(
            task_type_1="Test",
            prompt_complexity_score=0.5,
            task_type_2="NA",  # Default "NA" value
        )

        assert result1.task_type_2 == ""
        assert result2.task_type_2 == "NA"
        assert result1.task_type_2 != result2.task_type_2

    def test_model_equality(self) -> None:
        """Test equality comparison between ClassificationResult instances."""
        result1 = create_classification_result(
            task_type_1="code",
            prompt_complexity_score=0.5,
            domain_knowledge=0.6,
        )

        result2 = create_classification_result(
            task_type_1="code",
            prompt_complexity_score=0.5,
            domain_knowledge=0.6,
        )

        result3 = create_classification_result(
            task_type_1="chat",
            prompt_complexity_score=0.5,
            domain_knowledge=0.6,
        )

        assert result1 == result2
        assert result1 != result3

    def test_json_serialization(self) -> None:
        """Test JSON serialization compatibility."""
        result = create_classification_result(
            task_type_1="code",
            prompt_complexity_score=0.75,
            task_type_2="analysis",
            domain_knowledge=0.8,
        )

        # Test that it can be converted to JSON-compatible format
        json_data = result.model_dump()

        # All values should be JSON-serializable
        json_string = json.dumps(json_data)

        # Should be able to round-trip
        restored_data = json.loads(json_string)
        restored_result = create_classification_result(**restored_data)

        assert restored_result == result


@pytest.mark.unit
class TestClassificationResultEdgeCases:
    """Test edge cases for ClassificationResult."""

    def test_single_values_not_lists(self) -> None:
        """Test that ClassificationResult now uses individual values, not lists."""
        result = create_classification_result(
            # Required fields
            task_type_1="task_42",
            prompt_complexity_score=0.75,
            # Optional fields
            task_type_2="secondary_task_7",
            task_type_prob=0.85,
            creativity_scope=0.3,
            reasoning=0.9,
            contextual_knowledge=0.6,
            domain_knowledge=0.8,
            number_of_few_shots=2.0,
            no_label_reason=0.95,
            constraint_ct=0.4,
        )

        # All fields should be individual values, not lists
        assert isinstance(result.task_type_1, str)
        assert isinstance(result.prompt_complexity_score, int | float)
        assert isinstance(result.task_type_2, str)
        assert result.task_type_1 == "task_42"
        assert result.prompt_complexity_score == 0.75
        assert result.task_type_2 == "secondary_task_7"
        assert result.number_of_few_shots == 2.0

    def test_unicode_task_types(self) -> None:
        """Test handling of unicode characters in task types."""
        result1 = create_classification_result(
            task_type_1="코딩",
            task_type_2="Other",
        )

        result2 = create_classification_result(
            task_type_1="🤖",
            task_type_2="créatif",
        )

        assert result1.task_type_1 == "코딩"
        assert result2.task_type_1 == "🤖"
        assert result2.task_type_2 == "créatif"
        assert isinstance(result1.task_type_1, str)
        assert isinstance(result2.task_type_1, str)

    def test_extreme_complexity_scores(self) -> None:
        """Test edge cases for complexity scores."""
        # Very small positive numbers
        result1 = create_classification_result(prompt_complexity_score=0.000001)
        assert result1.prompt_complexity_score == 0.000001

        # Very close to 1
        result2 = create_classification_result(prompt_complexity_score=0.999999)
        assert result2.prompt_complexity_score == 0.999999

        # Exactly 0 and 1
        result3 = create_classification_result(prompt_complexity_score=0.0)
        result4 = create_classification_result(prompt_complexity_score=1.0)
        assert result3.prompt_complexity_score == 0.0
        assert result4.prompt_complexity_score == 1.0

    def test_data_type_consistency(self) -> None:
        """Test that individual values maintain type consistency."""
        # Strings for task types
        result = create_classification_result(
            task_type_1="code",
            task_type_2="analysis",
        )

        assert isinstance(result.task_type_1, str)
        assert isinstance(result.task_type_2, str)

        # Floats for complexity scores
        result = create_classification_result(
            prompt_complexity_score=0.5,
            reasoning=0.8,
            creativity_scope=0.3,
        )

        assert isinstance(result.prompt_complexity_score, int | float)
        assert isinstance(result.reasoning, int | float)
        assert isinstance(result.creativity_scope, int | float)
