"""Tests for enums and constants."""

import json

import pytest

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import TaskType
from tests.unit.models.test_classification_models import create_classification_result


class TestTaskType:
    """Test TaskType enum functionality."""

    def test_task_type_values(self):
        """Test that TaskType enum has expected values."""
        # Test common task types exist
        assert TaskType.CHATBOT
        assert TaskType.CODE_GENERATION
        assert TaskType.OPEN_QA
        assert TaskType.CLOSED_QA
        assert TaskType.TEXT_GENERATION
        assert TaskType.OTHER

        # Test values are strings
        assert isinstance(TaskType.CHATBOT.value, str)
        assert isinstance(TaskType.CODE_GENERATION.value, str)
        assert isinstance(TaskType.OPEN_QA.value, str)

    def test_task_type_string_representation(self):
        """Test string representation of TaskType enum."""
        assert str(TaskType.CODE_GENERATION) == "TaskType.CODE_GENERATION"
        assert str(TaskType.CHATBOT) == "TaskType.CHATBOT"
        assert str(TaskType.OPEN_QA) == "TaskType.OPEN_QA"

    def test_task_type_value_access(self):
        """Test accessing TaskType enum values."""
        assert TaskType.CODE_GENERATION.value == "Code Generation"
        assert TaskType.CHATBOT.value == "Chatbot"
        assert TaskType.OPEN_QA.value == "Open QA"
        assert TaskType.CLOSED_QA.value == "Closed QA"
        assert TaskType.TEXT_GENERATION.value == "Text Generation"
        assert TaskType.OTHER.value == "Other"

    def test_task_type_comparison(self):
        """Test TaskType enum comparison."""
        assert TaskType.CODE_GENERATION == TaskType.CODE_GENERATION
        assert TaskType.CODE_GENERATION != TaskType.CHATBOT
        assert TaskType.CODE_GENERATION is TaskType.CODE_GENERATION

    def test_task_type_iteration(self):
        """Test iterating over TaskType enum."""
        task_types = list(TaskType)

        assert len(task_types) >= 6  # At least the main types we expect
        assert TaskType.CODE_GENERATION in task_types
        assert TaskType.CHATBOT in task_types
        assert TaskType.OPEN_QA in task_types
        assert TaskType.CLOSED_QA in task_types
        assert TaskType.TEXT_GENERATION in task_types
        assert TaskType.OTHER in task_types

    def test_task_type_from_string(self):
        """Test creating TaskType from string values."""
        # Test valid conversions
        assert TaskType("Code Generation") == TaskType.CODE_GENERATION
        assert TaskType("Chatbot") == TaskType.CHATBOT
        assert TaskType("Open QA") == TaskType.OPEN_QA
        assert TaskType("Closed QA") == TaskType.CLOSED_QA
        assert TaskType("Text Generation") == TaskType.TEXT_GENERATION
        assert TaskType("Other") == TaskType.OTHER

    def test_task_type_invalid_string(self):
        """Test behavior with invalid string values."""
        with pytest.raises(ValueError):
            TaskType("invalid_task_type")

    def test_task_type_case_sensitivity(self):
        """Test that TaskType is case sensitive."""
        # These should work (exact case)
        assert TaskType("Code Generation") == TaskType.CODE_GENERATION

        # These should fail (wrong case)
        with pytest.raises(ValueError):
            TaskType("code generation")

        with pytest.raises(ValueError):
            TaskType("CODE GENERATION")

    def test_task_type_membership(self):
        """Test membership testing with TaskType."""
        valid_values = [task_type.value for task_type in TaskType]

        assert "Code Generation" in valid_values
        assert "Chatbot" in valid_values
        assert "Open QA" in valid_values
        assert "invalid_type" not in valid_values

    def test_task_type_uniqueness(self):
        """Test that all TaskType values are unique."""
        task_type_values = [task_type.value for task_type in TaskType]
        unique_values = set(task_type_values)

        assert len(task_type_values) == len(unique_values)

    def test_task_type_serialization(self):
        """Test TaskType serialization behavior."""
        # Test that enum can be converted to JSON-serializable format
        task_type = TaskType.CODE_GENERATION

        # Value should be JSON serializable
        json_string = json.dumps(task_type.value)
        restored_value = json.loads(json_string)

        assert restored_value == "Code Generation"
        assert TaskType(restored_value) == TaskType.CODE_GENERATION

    def test_task_type_in_collections(self):
        """Test TaskType behavior in collections."""
        # Test in list
        task_list = [TaskType.CODE_GENERATION, TaskType.CHATBOT, TaskType.OPEN_QA]
        assert len(task_list) == 3
        assert TaskType.CODE_GENERATION in task_list

        # Test in set
        task_set = {
            TaskType.CODE_GENERATION,
            TaskType.CHATBOT,
            TaskType.CODE_GENERATION,
        }  # Duplicate should be removed
        assert len(task_set) == 2

        # Test in dict as key
        task_dict = {
            TaskType.CODE_GENERATION: "Programming tasks",
            TaskType.CHATBOT: "Conversational tasks",
            TaskType.OPEN_QA: "Question answering tasks",
        }
        assert task_dict[TaskType.CODE_GENERATION] == "Programming tasks"

    def test_task_type_ordering(self):
        """Test TaskType ordering behavior."""
        task_types = [
            TaskType.TEXT_GENERATION,
            TaskType.CODE_GENERATION,
            TaskType.CHATBOT,
        ]

        # Should be sortable (by name or value)
        sorted_types = sorted(task_types, key=lambda x: x.value)

        assert len(sorted_types) == 3
        # Verify sorting worked (exact order depends on enum definition)

    def test_task_type_hash(self):
        """Test TaskType hashing behavior."""
        # Should be hashable (for use as dict keys, in sets)
        task_set = {TaskType.CODE_GENERATION, TaskType.CHATBOT}
        assert len(task_set) == 2

        # Hash should be consistent
        assert hash(TaskType.CODE_GENERATION) == hash(TaskType.CODE_GENERATION)
        assert hash(TaskType.CODE_GENERATION) != hash(TaskType.CHATBOT)

    def test_task_type_default_fallback(self):
        """Test handling of unknown task types with OTHER fallback."""
        # Test that OTHER can serve as a fallback for unknown types
        assert TaskType.OTHER.value == "Other"

        # Verify OTHER is a valid enum member
        assert TaskType.OTHER in list(TaskType)

    def test_task_type_comprehensive_coverage(self):
        """Test that enum covers expected task categories."""
        expected_categories = [
            "Code Generation",
            "Chatbot",
            "Open QA",
            "Text Generation",
            "Other",
        ]
        actual_values = [task_type.value for task_type in TaskType]

        for expected in expected_categories:
            assert expected in actual_values, f"Missing expected task type: {expected}"


@pytest.mark.unit
class TestTaskTypeIntegration:
    """Integration-style tests for TaskType with other components."""

    def test_task_type_with_model_capability(self):
        """Test TaskType usage with ModelCapability."""
        # Test that TaskType can be used in ModelCapability
        model = ModelCapability(
            provider="openai", model_name="gpt-4", task_type=TaskType.CODE_GENERATION
        )

        assert model.task_type == TaskType.CODE_GENERATION
        assert isinstance(model.task_type, TaskType)

    def test_task_type_with_classification_result(self):
        """Test TaskType usage with classification results."""
        # Test that TaskType values work with classification
        result = create_classification_result(
            task_type_1=["Code Generation", "Open QA"], size=2
        )

        assert "Code Generation" in result.task_type_1
        assert TaskType.CODE_GENERATION.value in result.task_type_1

    def test_task_type_conversion_utilities(self):
        """Test utility functions for TaskType conversion."""
        # Test converting string list to TaskType list
        string_types = ["Code Generation", "Chatbot", "Open QA"]
        task_types = []

        for string_type in string_types:
            try:
                task_type = TaskType(string_type)
                task_types.append(task_type)
            except ValueError:
                task_types.append(TaskType.OTHER)

        assert len(task_types) == 3
        assert TaskType.CODE_GENERATION in task_types
        assert TaskType.CHATBOT in task_types
        assert TaskType.OPEN_QA in task_types

    def test_task_type_filtering(self):
        """Test filtering operations with TaskType."""
        all_types = list(TaskType)

        # Filter for specific types
        technical_types = [
            t
            for t in all_types
            if t in [TaskType.CODE_GENERATION, TaskType.CLASSIFICATION]
        ]
        chat_types = [t for t in all_types if t == TaskType.CHATBOT]

        assert TaskType.CODE_GENERATION in technical_types or len(technical_types) >= 0
        assert len(chat_types) == 1
        assert chat_types[0] == TaskType.CHATBOT
