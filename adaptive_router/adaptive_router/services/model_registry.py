"""
Model registry service for validating model names across all providers.
"""

import logging

from adaptive_router.models.llm_core_models import ModelCapability
from adaptive_router.services.yaml_model_loader import YAMLModelDatabase

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Service for validating and managing model availability across providers.

    This class provides core model lookup and filtering functionality.
    Model definitions are loaded from YAML database.
    """

    def __init__(self, yaml_model_db: YAMLModelDatabase) -> None:
        """Initialize the model registry with model definitions from YAML database.

        Args:
            yaml_model_db: YAML model database instance
        """
        self._models: dict[str, ModelCapability] = {}
        self._load_models(yaml_model_db)

    def _load_models(self, yaml_model_db: YAMLModelDatabase) -> None:
        """Load model definitions from YAML database.

        Raises:
            RuntimeError: If critical error occurs during model loading
        """
        try:
            yaml_models = yaml_model_db.get_all_models()
            self._models.update(yaml_models)

            if not self._models:
                logger.error(
                    "No models loaded from YAML database. Check that YAML files exist "
                    "and are properly formatted in adaptive_router/model_data/data/provider_models/"
                )
        except Exception as e:
            logger.error(
                "Failed to load YAML models",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "hint": "Check that YAML files exist and are valid in model_data/data/provider_models/",
                },
            )
            # Don't raise here - allow service to start with empty models for better debugging

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

    def is_healthy(self) -> bool:
        """Check if model registry is healthy (has models loaded).

        Returns:
            True if models are loaded, False otherwise

        Example:
            >>> registry = ModelRegistry(yaml_db)
            >>> if not registry.is_healthy():
            ...     logger.error("Model registry failed to load models!")
        """
        return len(self._models) > 0

    def get_stats(self) -> dict[str, int]:
        """Get registry statistics.

        Returns:
            Dictionary with model counts by provider
        """
        stats: dict[str, int] = {}
        for model in self._models.values():
            if model.provider:
                stats[model.provider] = stats.get(model.provider, 0) + 1
        return stats

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
            # Provider filtering (case-insensitive) - only filter if provider is provided and non-empty
            if partial_model.provider and partial_model.provider.strip():
                if (
                    not model.provider
                    or model.provider.lower() != partial_model.provider.lower()
                ):
                    continue

            # Model name filtering - only filter if model_name is provided and non-empty
            if partial_model.model_name and partial_model.model_name.strip():
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

            # Get available providers to help user debug
            available_providers = sorted(
                set(m.provider for m in all_models if m.provider)
            )

            raise ValueError(
                f"No models found matching criteria: {criteria_str}\n"
                f"Available providers: {available_providers}\n"
                f"Total models in registry: {len(all_models)}\n"
                f"Hint: Check provider name spelling and model availability. "
                f"Use get_stats() to see models by provider."
            )

        return matching_models
