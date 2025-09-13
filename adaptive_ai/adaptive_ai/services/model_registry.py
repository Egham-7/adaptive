"""
Model registry service for validating model names across all providers.
"""

import logging

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.services.yaml_model_loader import yaml_model_db

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Service for validating and managing model availability across providers.

    This class provides core model lookup and filtering functionality.
    Model definitions are loaded from YAML files using the yaml_model_db.get_all_models()
    function, which remains the canonical source for model metadata and capabilities.
    """

    def __init__(self) -> None:
        """Initialize the model registry with model definitions from YAML files."""
        self._models: dict[str, ModelCapability] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load model definitions from YAML files."""
        # Try to load from YAML files
        try:
            yaml_models = yaml_model_db.get_all_models()
            # YAML loader guarantees unique provider:model_name keys, so we can directly update
            self._models.update(yaml_models)
        except Exception as e:
            # If YAML loading fails, log but continue with empty models
            logger.warning("Could not load YAML models: %s", e)

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
