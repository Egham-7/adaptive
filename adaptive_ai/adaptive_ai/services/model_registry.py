"""
Model registry service for validating model names across all providers.
"""

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.services.yaml_model_loader import yaml_model_db


class ModelRegistry:
    """Service for validating and managing model availability across providers.

    This class focuses on core model lookup and filtering functionality.
    Simple operations are delegated directly to yaml_model_db.
    """

    def __init__(self) -> None:
        """Initialize the model registry with YAML models only."""
        pass

    # Core model lookup methods
    def get_model_capability(self, unique_id: str) -> ModelCapability | None:
        """
        Get the full ModelCapability object for a unique_id.

        Args:
            unique_id: The unique_id to get capability for (format: "provider:model_name")

        Returns:
            ModelCapability object if model exists, None otherwise
        """
        return yaml_model_db.get_model(unique_id)

    def get_models_by_name(self, model_name: str) -> list[ModelCapability]:
        """
        Get all model capabilities for a given model name across all providers.

        Args:
            model_name: The model name to search for

        Returns:
            List of ModelCapability objects from all providers that serve this model
        """
        return yaml_model_db.get_models_by_name(model_name)

    # Simple validation methods (delegate to yaml_model_db)
    def is_valid_model(self, unique_id: str) -> bool:
        """Check if a model unique_id is valid (exists in any provider's capabilities)."""
        return yaml_model_db.has_model(unique_id)

    def is_valid_model_name(self, model_name: str) -> bool:
        """Check if a model name is valid (exists in any provider's capabilities)."""
        return len(self.get_models_by_name(model_name)) > 0

    def get_providers_for_model(self, model_name: str) -> set[str]:
        """Get all providers that support a given model name."""
        models = self.get_models_by_name(model_name)
        return {model.provider for model in models if model.provider is not None}

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
        # Models are already loaded at startup
        all_models = list(yaml_model_db.get_all_models().values())

        # Create filter predicates based on partial model criteria
        filters = []

        # Provider filtering
        if partial_model.provider:
            provider_lower = partial_model.provider.lower()
            filters.append(lambda m: m.provider.lower() == provider_lower)

        # Model name filtering
        if partial_model.model_name:
            name_lower = partial_model.model_name.lower()
            filters.append(lambda m: m.model_name.lower() == name_lower)

        # Context filtering (min requirement)
        if partial_model.max_context_tokens is not None:
            min_tokens = partial_model.max_context_tokens
            filters.append(
                lambda m: (
                    m.max_context_tokens is not None
                    and m.max_context_tokens >= min_tokens
                )
            )

        # Cost filtering (max budget for input)
        if partial_model.cost_per_1m_input_tokens is not None:
            max_cost_input = partial_model.cost_per_1m_input_tokens
            filters.append(
                lambda m: (
                    m.cost_per_1m_input_tokens is not None
                    and m.cost_per_1m_input_tokens <= max_cost_input
                )
            )

        # Cost filtering (max budget for output)
        if partial_model.cost_per_1m_output_tokens is not None:
            max_cost_output = partial_model.cost_per_1m_output_tokens
            filters.append(
                lambda m: (
                    m.cost_per_1m_output_tokens is not None
                    and m.cost_per_1m_output_tokens <= max_cost_output
                )
            )

        # Function calling support
        if partial_model.supports_function_calling is True:
            filters.append(lambda m: m.supports_function_calling is True)

        # Task type filtering
        if partial_model.task_type is not None:
            task_type = partial_model.task_type
            filters.append(
                lambda m: (
                    m.task_type is None  # No restriction means supports all
                    or m.task_type == task_type
                )
            )

        # Complexity filtering
        if partial_model.complexity is not None:
            complexity_lower = partial_model.complexity.lower()
            filters.append(
                lambda m: (
                    m.complexity is not None
                    and m.complexity.lower() == complexity_lower
                )
            )

        # Apply all filters with AND logic using list comprehension
        if not filters:
            # No criteria specified - return all models
            return all_models

        # Filter models by applying all predicates using comprehension
        return [model for model in all_models if all(f(model) for f in filters)]


# Global instance for use across the application
model_registry = ModelRegistry()
