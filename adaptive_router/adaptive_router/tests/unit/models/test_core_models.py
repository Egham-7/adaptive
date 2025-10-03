"""Tests for core API models."""

from pydantic import ValidationError
import pytest

from adaptive_router.models.llm_core_models import (
    Alternative,
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)


class TestModelCapability:
    """Test ModelCapability model validation and functionality."""

    def test_minimal_model_capability(self) -> None:
        """Test creating ModelCapability with minimal required fields."""
        model = ModelCapability(provider="openai", model_name="gpt-4")
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.cost_per_1m_input_tokens is None

    def test_full_model_capability(self) -> None:
        """Test creating ModelCapability with all fields."""
        model = ModelCapability(
            provider="openai",
            model_name="gpt-4",
            cost_per_1m_input_tokens=30.0,
            cost_per_1m_output_tokens=60.0,
            max_context_tokens=128000,
            supports_function_calling=True,
            task_type="Code Generation",
            complexity="medium",
            description="GPT-4 model for general tasks",
        )
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"
        assert model.cost_per_1m_input_tokens == 30.0
        assert model.cost_per_1m_output_tokens == 60.0
        assert model.max_context_tokens == 128000
        assert model.supports_function_calling is True
        assert model.task_type == "Code Generation"
        assert model.complexity == "medium"

    def test_partial_model_capability(self) -> None:
        """Test ModelCapability can be created with partial information."""
        model = ModelCapability(
            provider="anthropic",
            cost_per_1m_input_tokens=15.0,
            # Missing model_name and other fields
        )
        assert model.provider == "anthropic"
        assert model.model_name is None
        assert model.cost_per_1m_input_tokens == 15.0

    def test_task_type_string_conversion(self) -> None:
        """Test that task_type accepts both enum and string values."""
        # Test with enum
        model1 = ModelCapability(task_type="Classification")
        assert model1.task_type == "Classification"

        # Test with string
        model2 = ModelCapability(task_type="analysis")
        assert model2.task_type == "analysis"


class TestModelSelectionRequest:
    """Test ModelSelectionRequest model validation."""

    def test_minimal_request(self) -> None:
        """Test creating request with minimal required fields."""
        request = ModelSelectionRequest(prompt="Hello world")
        assert request.prompt == "Hello world"
        assert request.cost_bias is None
        assert request.models is None

    def test_full_request(self) -> None:
        """Test creating request with all fields."""
        models = [
            ModelCapability(provider="openai", model_name="gpt-4"),
            ModelCapability(provider="anthropic", model_name="claude-3-sonnet"),
        ]

        request = ModelSelectionRequest(
            prompt="Write a Python function",
            models=models,
            cost_bias=0.7,
            user_id="test_user",
        )

        assert request.prompt == "Write a Python function"
        if request.models:
            assert len(request.models) == 2
        assert request.cost_bias == 0.7
        assert request.user_id == "test_user"

    def test_empty_prompt_validation(self) -> None:
        """Test that empty prompt raises validation error."""
        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="")

    def test_cost_bias_validation(self) -> None:
        """Test cost_bias must be between 0 and 1."""
        # Valid cost_bias values
        ModelSelectionRequest(prompt="test", cost_bias=0.0)
        ModelSelectionRequest(prompt="test", cost_bias=0.5)
        ModelSelectionRequest(prompt="test", cost_bias=1.0)

        # Invalid cost_bias values should raise ValidationError
        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="test", cost_bias=-0.1)

        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="test", cost_bias=1.1)


class TestModelSelectionResponse:
    """Test ModelSelectionResponse model validation."""

    def test_minimal_response(self) -> None:
        """Test creating response with required fields."""
        response = ModelSelectionResponse(
            provider="openai", model="gpt-4", alternatives=[]
        )
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert response.alternatives == []

    def test_response_with_alternatives(self) -> None:
        """Test response with alternative models."""
        alternatives = [
            Alternative(provider="anthropic", model="claude-3-sonnet"),
            Alternative(provider="openai", model="gpt-3.5-turbo"),
        ]

        response = ModelSelectionResponse(
            provider="openai", model="gpt-4", alternatives=alternatives
        )

        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert len(response.alternatives) == 2
        assert response.alternatives[0].provider == "anthropic"
        assert response.alternatives[1].model == "gpt-3.5-turbo"

    def test_empty_provider_validation(self) -> None:
        """Test that empty provider raises validation error."""
        with pytest.raises(ValidationError):
            ModelSelectionResponse(provider="", model="gpt-4", alternatives=[])

    def test_empty_model_validation(self) -> None:
        """Test that empty model raises validation error."""
        with pytest.raises(ValidationError):
            ModelSelectionResponse(provider="openai", model="", alternatives=[])


class TestAlternative:
    """Test Alternative model validation."""

    def test_valid_alternative(self) -> None:
        """Test creating valid alternative."""
        alt = Alternative(provider="anthropic", model="claude-3-haiku")
        assert alt.provider == "anthropic"
        assert alt.model == "claude-3-haiku"

    def test_alternative_validation(self) -> None:
        """Test alternative field validation."""
        # Valid alternatives
        Alternative(provider="openai", model="gpt-3.5-turbo")

        # Invalid alternatives should raise ValidationError
        with pytest.raises(ValidationError):
            Alternative(provider="", model="gpt-4")

        with pytest.raises(ValidationError):
            Alternative(provider="openai", model="")


@pytest.mark.unit
class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_model_capability_json_serialization(self) -> None:
        """Test ModelCapability can be serialized to/from JSON."""
        original = ModelCapability(
            provider="openai",
            model_name="gpt-4",
            cost_per_1m_input_tokens=30.0,
            task_type="Code Generation",
        )

        # Serialize to dict
        data = original.model_dump()
        assert data["provider"] == "openai"
        assert data["task_type"] == "Code Generation"  # Enum serialized as string

        # Deserialize from dict
        restored = ModelCapability(**data)
        assert restored.provider == original.provider
        assert restored.model_name == original.model_name
        assert restored.cost_per_1m_input_tokens == original.cost_per_1m_input_tokens

    def test_request_response_serialization(self) -> None:
        """Test request/response models can be serialized."""
        # Test request serialization
        request = ModelSelectionRequest(prompt="Test prompt", cost_bias=0.5)
        request_data = request.model_dump()
        restored_request = ModelSelectionRequest(**request_data)
        assert restored_request.prompt == request.prompt
        assert restored_request.cost_bias == request.cost_bias

        # Test response serialization
        response = ModelSelectionResponse(
            provider="openai",
            model="gpt-4",
            alternatives=[Alternative(provider="anthropic", model="claude-3-sonnet")],
        )
        response_data = response.model_dump()
        restored_response = ModelSelectionResponse(**response_data)
        assert restored_response.provider == response.provider
        assert len(restored_response.alternatives) == 1
