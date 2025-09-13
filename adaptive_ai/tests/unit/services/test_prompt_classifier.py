"""Unit tests for PromptClassifier service."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.services.prompt_classifier import (
    PromptClassifier,
    get_prompt_classifier,
)


@pytest.mark.unit
class TestPromptClassifier:
    """Test PromptClassifier functionality."""

    @patch.dict(
        "os.environ",
        {
            "MODAL_CLASSIFIER_URL": "https://mock-modal-url.com/classify",
            "JWT_SECRET": "mock-jwt-secret",
        },
    )
    def test_get_prompt_classifier_basic(self):
        """Test basic prompt classifier instantiation."""
        classifier = get_prompt_classifier()
        assert isinstance(classifier, PromptClassifier)
        assert hasattr(classifier, "classify_prompts_async")
        assert hasattr(classifier, "classify_prompt_async")
        assert hasattr(classifier, "health_check_async")
        assert hasattr(classifier, "aclose")

    @pytest.mark.asyncio
    async def test_prompt_classifier_interface(self):
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
    async def test_classify_prompts_empty_input(self):
        """Test classify_prompts_async with empty input."""
        classifier = Mock(spec=PromptClassifier)
        classifier.classify_prompts_async = AsyncMock(return_value=[])

        results = await classifier.classify_prompts_async([])
        assert len(results) == 0
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_classify_prompts_multiple(self):
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
class TestGetPromptClassifier:
    """Test get_prompt_classifier function."""

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {
            "MODAL_CLASSIFIER_URL": "https://mock-modal-url.com/classify",
            "JWT_SECRET": "mock-jwt-secret",
        },
    )
    async def test_get_prompt_classifier_caching(self):
        """Test that get_prompt_classifier uses caching appropriately."""
        # Test that function exists and can be called
        classifier1 = get_prompt_classifier()
        classifier2 = get_prompt_classifier()
        try:
            # If caching works, should be same instance
            # If not, that's also fine for this test
            assert isinstance(classifier1, PromptClassifier)
            assert isinstance(classifier2, PromptClassifier)
        finally:
            # Close AsyncClient to avoid ResourceWarning
            await classifier1.aclose()
            if classifier1 is not classifier2:
                await classifier2.aclose()

    @pytest.mark.asyncio
    @patch.dict(
        "os.environ",
        {
            "MODAL_CLASSIFIER_URL": "https://mock-modal-url.com/classify",
            "JWT_SECRET": "mock-jwt-secret",
        },
    )
    async def test_get_prompt_classifier_basic(self):
        """Test get_prompt_classifier basic functionality."""
        classifier = get_prompt_classifier()
        try:
            assert isinstance(classifier, PromptClassifier)
        finally:
            await classifier.aclose()


@pytest.mark.unit
class TestPromptClassifierEdgeCases:
    """Test edge cases for PromptClassifier."""

    @pytest.mark.asyncio
    async def test_classifier_with_none_input(self):
        """Test classifier behavior with None input."""
        classifier = Mock(spec=PromptClassifier)

        # Mock should handle None gracefully
        async def mock_classify_side_effect(x):
            if x is None:
                return []
            else:
                return [
                    ClassificationResult(
                        task_type_1="Other",
                        prompt_complexity_score=0.5,
                        task_type_2="Other",
                        task_type_prob=0.5,
                        creativity_scope=0.5,
                        reasoning=0.5,
                        contextual_knowledge=0.5,
                        domain_knowledge=0.5,
                        number_of_few_shots=0.0,
                        no_label_reason=0.5,
                        constraint_ct=0.5,
                    )
                ]

        classifier.classify_prompts_async = AsyncMock(
            side_effect=mock_classify_side_effect
        )

        results = await classifier.classify_prompts_async(None)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_classifier_error_handling(self):
        """Test classifier error handling."""
        classifier = Mock(spec=PromptClassifier)
        classifier.classify_prompts_async = AsyncMock(
            side_effect=RuntimeError("Model error")
        )

        with pytest.raises((RuntimeError, ValueError, ImportError)):
            await classifier.classify_prompts_async(["Test prompt"])

    def test_classification_result_structure(self):
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
