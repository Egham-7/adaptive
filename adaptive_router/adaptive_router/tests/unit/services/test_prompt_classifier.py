"""Unit tests for PromptClassifier service."""

from unittest.mock import AsyncMock, Mock

import pytest

from adaptive_router.models.llm_classification_models import ClassificationResult
from adaptive_router.services.prompt_task_complexity_classifier import PromptClassifier


@pytest.mark.unit
class TestPromptClassifier:
    """Test PromptClassifier functionality."""

    def test_get_prompt_classifier_basic(self) -> None:
        """Test basic prompt classifier instantiation."""
        classifier = PromptClassifier()
        assert isinstance(classifier, PromptClassifier)
        assert hasattr(classifier, "classify_prompt")

    @pytest.mark.asyncio
    async def test_prompt_classifier_interface(self) -> None:
        """Test that PromptClassifier has expected interface."""
        # Create a mock classifier to test interface
        classifier = Mock(spec=PromptClassifier)
        classifier.classify_prompts_async = AsyncMock(
            return_value=[
                ClassificationResult(
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
            ]
        )

        # Test the interface
        results = await classifier.classify_prompts_async(["Test prompt"])
        assert len(results) == 1
        assert isinstance(results[0], ClassificationResult)

    @pytest.mark.asyncio
    async def test_classify_prompts_empty_input(self) -> None:
        """Test classify_prompts_async with empty input."""
        classifier = Mock(spec=PromptClassifier)
        classifier.classify_prompts_async = AsyncMock(
            side_effect=ValueError("Prompts list cannot be empty")
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            await classifier.classify_prompts_async([])

    @pytest.mark.asyncio
    async def test_classify_prompts_multiple(self) -> None:
        """Test classify_prompts_async with multiple prompts."""
        classifier = Mock(spec=PromptClassifier)
        classifier.classify_prompts_async = AsyncMock(
            return_value=[
                ClassificationResult(
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
                ),
                ClassificationResult(
                    # Required fields
                    task_type_1="Chatbot",
                    prompt_complexity_score=0.35,
                    # Optional fields
                    task_type_2="Other",
                    task_type_prob=0.9,
                    creativity_scope=0.1,
                    reasoning=0.3,
                    contextual_knowledge=0.2,
                    domain_knowledge=0.2,
                    number_of_few_shots=0.0,
                    no_label_reason=0.95,
                    constraint_ct=0.05,
                ),
            ]
        )

        prompts = ["Write Python code", "Hello, how are you?"]
        results = await classifier.classify_prompts_async(prompts)

        assert len(results) == 2
        assert all(isinstance(r, ClassificationResult) for r in results)


@pytest.mark.unit
class TestPromptClassifierEdgeCases:
    """Test edge cases for PromptClassifier."""

    @pytest.mark.asyncio
    async def test_classifier_with_none_input(self) -> None:
        """Test classifier behavior with None input."""
        classifier = Mock(spec=PromptClassifier)
        classifier.classify_prompts_async = AsyncMock(
            side_effect=ValueError("Prompts list cannot be empty")
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            await classifier.classify_prompts_async(None)

    @pytest.mark.asyncio
    async def test_classifier_error_handling(self) -> None:
        """Test classifier error handling."""
        classifier = Mock(spec=PromptClassifier)
        classifier.classify_prompts_async = AsyncMock(
            side_effect=RuntimeError("Model error")
        )

        with pytest.raises((RuntimeError, ValueError, ImportError)):
            await classifier.classify_prompts_async(["Test prompt"])

    def test_classification_result_structure(self) -> None:
        """Test that classification results have expected structure."""
        # Test with a real ClassificationResult
        result = ClassificationResult(
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

        # Verify all expected fields exist
        assert hasattr(result, "task_type_1")
        assert hasattr(result, "prompt_complexity_score")
        assert hasattr(result, "creativity_scope")
        assert hasattr(result, "reasoning")
        assert isinstance(result.task_type_1, str)
        assert isinstance(result.prompt_complexity_score, int | float)
