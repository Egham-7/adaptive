"""
Model registry service for validating model names across all providers.
"""

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.services.yaml_model_loader import yaml_model_db


class ModelRegistry:
    """Service for validating and managing model availability across providers.

    This class provides core model lookup and filtering functionality.
    Since yaml_model_loader was removed with Modal migration, this now contains
    basic model definitions.
    """

    def __init__(self) -> None:
        """Initialize the model registry with model definitions from YAML and hardcoded fallbacks."""
        self._models: dict[str, ModelCapability] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load model definitions from YAML files with hardcoded fallbacks."""
        # Try to load from YAML files first
        try:
            yaml_models = yaml_model_db.get_all_models()
            for unique_id, model_data in yaml_models.items():
                # Skip duplicate entries (models are stored both by unique_id and model_name)
                if ":" in unique_id:
                    # model_data is already a ModelCapability object from YAML loader
                    self._models[unique_id] = model_data
        except Exception as e:
            # If YAML loading fails, log but continue with fallbacks
            print(f"Warning: Could not load YAML models, using fallbacks: {e}")

        # Add hardcoded fallback models if not already loaded from YAML
        fallback_models = self._get_fallback_models()
        for unique_id, model_capability in fallback_models.items():
            if unique_id not in self._models:
                self._models[unique_id] = model_capability

    def _get_fallback_models(self) -> dict[str, ModelCapability]:
        """Get hardcoded fallback model definitions."""
        models = {}

        # OpenAI models
        models["openai:gpt-4"] = ModelCapability(
            provider="openai",
            model_name="gpt-4",
            cost_per_1m_input_tokens=30.0,
            cost_per_1m_output_tokens=60.0,
            max_context_tokens=128000,
            supports_function_calling=True,
            complexity="high",
        )

        models["openai:gpt-3.5-turbo"] = ModelCapability(
            provider="openai",
            model_name="gpt-3.5-turbo",
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=6.0,
            max_context_tokens=16385,
            supports_function_calling=True,
            complexity="medium",
        )

        # Anthropic models
        models["anthropic:claude-3-5-sonnet-20241022"] = ModelCapability(
            provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            cost_per_1m_input_tokens=15.0,
            cost_per_1m_output_tokens=75.0,
            max_context_tokens=200000,
            supports_function_calling=True,
            complexity="high",
        )

        models["anthropic:claude-3-haiku-20240307"] = ModelCapability(
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            cost_per_1m_input_tokens=0.25,
            cost_per_1m_output_tokens=1.25,
            max_context_tokens=200000,
            supports_function_calling=True,
            complexity="low",
        )

        return models

    # Core model lookup methods
    def get_model_capability(self, unique_id: str) -> ModelCapability | None:
        """
        Get the full ModelCapability object for a unique_id.

        Args:
            unique_id: The unique_id to get capability for (format: "provider:model_name")

        Returns:
            ModelCapability object if model exists, None otherwise
        """
        return self._models.get(unique_id)

    def get_models_by_name(self, model_name: str) -> list[ModelCapability]:
        """
        Get all model capabilities for a given model name across all providers.

        Args:
            model_name: The model name to search for

        Returns:
            List of ModelCapability objects from all providers that serve this model
        """
        return [
            model for model in self._models.values() if model.model_name == model_name
        ]

    # Simple validation methods
    def is_valid_model(self, unique_id: str) -> bool:
        """Check if a model unique_id is valid (exists in any provider's capabilities)."""
        return unique_id in self._models

    def is_valid_model_name(self, model_name: str) -> bool:
        """Check if a model name is valid (exists in any provider's capabilities)."""
        return len(self.get_models_by_name(model_name)) > 0

    def get_providers_for_model(self, model_name: str) -> set[str]:
        """Get all providers that support a given model name."""
        models = self.get_models_by_name(model_name)
        return {model.provider for model in models if model.provider is not None}

    def get_all_model_names(self) -> list[str]:
        """Get all unique_ids of models in the registry."""
        return list(self._models.keys())

    # Core filtering functionality - this is the main value-add
    def find_models_matching_criteria(
        self, partial_model: ModelCapability
    ) -> list[ModelCapability]:
        """
        Find models that match the criteria specified in a partial ModelCapability.

        This uses functional predicates for clean, composable filtering.

        Args:
            partial_model: ModelCapability with some fields set as criteria

        Returns:
            List of full ModelCapability objects that match the criteria

        Examples:
            # Find OpenAI models with >8K context
            criteria = ModelCapability(
                provider="openai",
                max_context_tokens=8000  # This means "at least 8000"
            )
            models = registry.find_models_matching_criteria(criteria)
        """
        # Use the basic models from our registry
        all_models = list(self._models.values())

        # Simple direct filtering without complex lambda predicates
        matching_models = []

        for model in all_models:
            # Provider filtering (case-insensitive)
            if partial_model.provider:
                if (
                    not model.provider
                    or model.provider.lower() != partial_model.provider.lower()
                ):
                    continue

            # Model name filtering
            if partial_model.model_name:
                if (
                    not model.model_name
                    or model.model_name.lower() != partial_model.model_name.lower()
                ):
                    continue

            # Context filtering (min requirement)
            if partial_model.max_context_tokens is not None:
                if (
                    model.max_context_tokens is None
                    or model.max_context_tokens < partial_model.max_context_tokens
                ):
                    continue

            # Cost filtering (max budget for input)
            if partial_model.cost_per_1m_input_tokens is not None:
                if (
                    model.cost_per_1m_input_tokens is None
                    or model.cost_per_1m_input_tokens
                    > partial_model.cost_per_1m_input_tokens
                ):
                    continue

            # Cost filtering (max budget for output)
            if partial_model.cost_per_1m_output_tokens is not None:
                if (
                    model.cost_per_1m_output_tokens is None
                    or model.cost_per_1m_output_tokens
                    > partial_model.cost_per_1m_output_tokens
                ):
                    continue
            # Function calling support
            if partial_model.supports_function_calling is True:
                if model.supports_function_calling is not True:
                    continue

            # Task type filtering
            if partial_model.task_type is not None:
                if model.task_type is not None:
                    model_task_str = str(model.task_type).strip().lower()
                    partial_task_str = str(partial_model.task_type).strip().lower()
                    if model_task_str != partial_task_str:
                        continue
                # If model has no task_type, it supports all tasks (continue)

            # Complexity filtering
            if partial_model.complexity is not None:
                if (
                    not model.complexity
                    or model.complexity.lower() != partial_model.complexity.lower()
                ):
                    continue

            # If we get here, model matches all criteria
            matching_models.append(model)

        # If no models match the criteria, raise specific error with details
        if not matching_models:
            # Extract only non-None fields from the partial model
            criteria = partial_model.model_dump(exclude_none=True)
            criteria_str = ", ".join(f"{k}={v}" for k, v in criteria.items())
            raise ValueError(f"No models found matching criteria: {criteria_str}")

        return matching_models


# Global instance for use across the application
model_registry = ModelRegistry()
