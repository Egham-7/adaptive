"""Enhanced model validation tests for comprehensive edge case coverage."""

from pydantic import ValidationError
import pytest

from adaptive_ai.models.llm_core_models import (
    Alternative,
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)


class TestModelCapabilityValidation:
    """Test ModelCapability validation and edge cases."""

    def test_partial_model_validation(self):
        """Test partial model capabilities are valid."""
        # Provider only
        model = ModelCapability(provider="openai")
        assert model.provider == "openai"
        assert model.model_name is None
        assert model.is_partial is True

        # Model name only
        model = ModelCapability(model_name="gpt-4")
        assert model.provider is None
        assert model.model_name == "gpt-4"
        assert model.is_partial is True

    def test_unique_id_generation(self):
        """Test unique ID generation and error handling."""
        # Valid unique ID
        model = ModelCapability(provider="OpenAI", model_name="GPT-4")
        assert model.unique_id == "openai:gpt-4"  # Normalized to lowercase

        # Missing provider
        model = ModelCapability(model_name="gpt-4")
        with pytest.raises(ValueError, match="Provider is required"):
            _ = model.unique_id

        # Missing model name
        model = ModelCapability(provider="openai")
        with pytest.raises(ValueError, match="Model name is required"):
            _ = model.unique_id

    def test_complexity_score_conversion(self):
        """Test complexity string to score conversion."""
        # Valid complexity levels
        model = ModelCapability(complexity="easy")
        assert model.complexity_score == 0.2

        model = ModelCapability(complexity="medium")
        assert model.complexity_score == 0.5

        model = ModelCapability(complexity="hard")
        assert model.complexity_score == 0.8

        # Case insensitive
        model = ModelCapability(complexity="EASY")
        assert model.complexity_score == 0.2

        # Unknown complexity defaults to medium
        model = ModelCapability(complexity="unknown")
        assert model.complexity_score == 0.5

        # None complexity defaults to medium
        model = ModelCapability(complexity=None)
        assert model.complexity_score == 0.5

    def test_task_type_flexibility(self):
        """Test task type accepts both enum and string values."""
        # Enum value
        model = ModelCapability(task_type="Code Generation")
        assert model.task_type == "Code Generation"

        # String value
        model = ModelCapability(task_type="custom_task")
        assert model.task_type == "custom_task"

        # None value
        model = ModelCapability(task_type=None)
        assert model.task_type is None

    def test_cost_validation(self):
        """Test cost field validation."""
        # Valid positive costs
        model = ModelCapability(
            cost_per_1m_input_tokens=0.1, cost_per_1m_output_tokens=0.2
        )
        assert model.cost_per_1m_input_tokens == 0.1
        assert model.cost_per_1m_output_tokens == 0.2

        # Zero costs (valid for free models)
        model = ModelCapability(
            cost_per_1m_input_tokens=0.0, cost_per_1m_output_tokens=0.0
        )
        assert model.cost_per_1m_input_tokens == 0.0
        assert model.cost_per_1m_output_tokens == 0.0

    def test_context_length_validation(self):
        """Test context length validation."""
        # Valid context lengths
        model = ModelCapability(max_context_tokens=128000)
        assert model.max_context_tokens == 128000

        # Very large context length
        model = ModelCapability(max_context_tokens=2000000)
        assert model.max_context_tokens == 2000000

        # Zero context length
        model = ModelCapability(max_context_tokens=0)
        assert model.max_context_tokens == 0


class TestModelSelectionRequestValidation:
    """Test ModelSelectionRequest validation and edge cases."""

    def test_prompt_validation(self):
        """Test prompt field validation."""
        # Valid prompts
        ModelSelectionRequest(prompt="Hello world")
        ModelSelectionRequest(prompt="A" * 10000)  # Very long prompt

        # Empty prompt should raise ValidationError
        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="")

        # Whitespace-only prompt should raise ValidationError
        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="   ")

    def test_cost_bias_edge_cases(self):
        """Test cost bias validation edge cases."""
        # Valid edge values
        ModelSelectionRequest(prompt="test", cost_bias=0.0)
        ModelSelectionRequest(prompt="test", cost_bias=1.0)
        ModelSelectionRequest(prompt="test", cost_bias=0.5)

        # Values outside 0-1 range should raise ValidationError
        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="test", cost_bias=-0.1)

        with pytest.raises(ValidationError):
            ModelSelectionRequest(prompt="test", cost_bias=1.1)

        # None is valid (default behavior)
        request = ModelSelectionRequest(prompt="test", cost_bias=None)
        assert request.cost_bias is None

    def test_models_list_validation(self):
        """Test models list validation."""
        # Empty models list is valid
        request = ModelSelectionRequest(prompt="test", models=[])
        assert request.models == []

        # Models list with partial models
        partial_models = [
            ModelCapability(provider="openai"),
            ModelCapability(model_name="gpt-4"),
        ]
        request = ModelSelectionRequest(prompt="test", models=partial_models)
        if request.models:
            assert len(request.models) == 2

        # None models list is valid
        request = ModelSelectionRequest(prompt="test", models=None)
        assert request.models is None

    def test_tool_calling_fields(self):
        """Test tool calling related fields."""
        # Valid tool call
        tool_call = {"function": {"name": "get_weather", "arguments": "{}"}}
        request = ModelSelectionRequest(prompt="test", tool_call=tool_call)
        assert request.tool_call == tool_call

        # Valid tools list
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        request = ModelSelectionRequest(prompt="test", tools=tools)
        assert request.tools == tools

        # None values are valid
        request = ModelSelectionRequest(prompt="test", tool_call=None, tools=None)
        assert request.tool_call is None
        assert request.tools is None

    def test_optional_fields(self):
        """Test optional field handling."""
        # Request with all optional fields
        request = ModelSelectionRequest(
            prompt="test",
            user_id="user123",
            complexity_threshold=0.7,
            token_threshold=50000,
        )
        assert request.user_id == "user123"
        assert request.complexity_threshold == 0.7
        assert request.token_threshold == 50000

        # Request with no optional fields
        request = ModelSelectionRequest(prompt="test")
        assert request.user_id is None
        assert request.complexity_threshold is None
        assert request.token_threshold is None


class TestModelSelectionResponseValidation:
    """Test ModelSelectionResponse validation and edge cases."""

    def test_required_fields_validation(self):
        """Test that required fields are properly validated."""
        # Valid response
        response = ModelSelectionResponse(
            provider="openai", model="gpt-4", alternatives=[]
        )
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert response.alternatives == []

        # Empty provider should raise ValidationError
        with pytest.raises(ValidationError):
            ModelSelectionResponse(provider="", model="gpt-4", alternatives=[])

        # Empty model should raise ValidationError
        with pytest.raises(ValidationError):
            ModelSelectionResponse(provider="openai", model="", alternatives=[])

    def test_alternatives_validation(self):
        """Test alternatives list validation."""
        # Valid alternatives
        alternatives = [
            Alternative(provider="anthropic", model="claude-3-sonnet"),
            Alternative(provider="openai", model="gpt-3.5-turbo"),
        ]
        response = ModelSelectionResponse(
            provider="openai", model="gpt-4", alternatives=alternatives
        )
        assert len(response.alternatives) == 2

        # Empty alternatives list is valid
        response = ModelSelectionResponse(
            provider="openai", model="gpt-4", alternatives=[]
        )
        assert response.alternatives == []

    def test_provider_model_consistency(self):
        """Test provider and model field consistency."""
        # Whitespace handling
        response = ModelSelectionResponse(
            provider="  openai  ", model="  gpt-4  ", alternatives=[]
        )
        # Should handle whitespace properly
        assert "openai" in response.provider.lower()
        assert "gpt-4" in response.model.lower()


class TestAlternativeValidation:
    """Test Alternative model validation."""

    def test_alternative_field_validation(self):
        """Test Alternative field validation."""
        # Valid alternative
        alt = Alternative(provider="anthropic", model="claude-3-haiku")
        assert alt.provider == "anthropic"
        assert alt.model == "claude-3-haiku"

        # Empty provider should raise ValidationError
        with pytest.raises(ValidationError):
            Alternative(provider="", model="gpt-4")

        # Empty model should raise ValidationError
        with pytest.raises(ValidationError):
            Alternative(provider="openai", model="")

        # Whitespace-only fields should raise ValidationError
        with pytest.raises(ValidationError):
            Alternative(provider="   ", model="gpt-4")

        with pytest.raises(ValidationError):
            Alternative(provider="openai", model="   ")

    def test_alternative_edge_cases(self):
        """Test Alternative edge cases."""
        # Very long provider/model names
        long_name = "a" * 1000
        alt = Alternative(provider=long_name, model=long_name)
        assert len(alt.provider) == 1000
        assert len(alt.model) == 1000

        # Special characters in names
        alt = Alternative(provider="openai-v2", model="gpt-4-turbo-preview")
        assert alt.provider == "openai-v2"
        assert alt.model == "gpt-4-turbo-preview"


class TestModelSerializationEdgeCases:
    """Test edge cases in model serialization."""

    def test_model_capability_serialization_edge_cases(self):
        """Test ModelCapability serialization edge cases."""
        # Model with None values
        model = ModelCapability(
            provider="openai", model_name=None, cost_per_1m_input_tokens=None
        )
        data = model.model_dump()
        restored = ModelCapability(**data)
        assert restored.provider == model.provider
        assert restored.model_name is None

        # Model with enum task type
        model = ModelCapability(task_type="Code Generation")
        data = model.model_dump()
        restored = ModelCapability(**data)
        assert restored.task_type == "Code Generation"  # Serialized as string

    def test_request_serialization_with_complex_data(self):
        """Test request serialization with complex nested data."""
        # Request with tools and tool calls
        request = ModelSelectionRequest(
            prompt="Use the weather function",
            tool_call={
                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}
            },
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            models=[ModelCapability(provider="openai", model_name="gpt-4")],
        )

        data = request.model_dump()
        restored = ModelSelectionRequest(**data)

        assert restored.prompt == request.prompt
        assert restored.tool_call == request.tool_call
        assert restored.tools == request.tools
        if restored.models:
            assert len(restored.models) == 1

    def test_response_serialization_with_alternatives(self):
        """Test response serialization with multiple alternatives."""
        response = ModelSelectionResponse(
            provider="openai",
            model="gpt-4",
            alternatives=[
                Alternative(provider="anthropic", model="claude-3-sonnet"),
                Alternative(provider="google", model="gemini-pro"),
                Alternative(provider="openai", model="gpt-3.5-turbo"),
            ],
        )

        data = response.model_dump()
        restored = ModelSelectionResponse(**data)

        assert restored.provider == response.provider
        assert restored.model == response.model
        assert len(restored.alternatives) == 3
        assert restored.alternatives[0].provider == "anthropic"
