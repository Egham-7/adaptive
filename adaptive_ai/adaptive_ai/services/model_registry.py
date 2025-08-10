"""
Model registry service for validating model names across all providers.
"""

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import ProviderType
from adaptive_ai.services.yaml_model_loader import yaml_model_db


class ModelRegistry:
    """Service for validating and managing model availability across providers."""

    def __init__(self) -> None:
        """Initialize the model registry with YAML models only."""
        pass

    def is_valid_model(self, model_name: str) -> bool:
        """
        Check if a model name is valid (exists in any provider's capabilities).

        Args:
            model_name: The model name to validate

        Returns:
            True if the model exists, False otherwise
        """
        return yaml_model_db.has_model(model_name)

    def validate_models(self, models: list[str]) -> tuple[list[str], list[str]]:
        """
        Validate a list of model names.

        Args:
            models: List of model names to validate

        Returns:
            Tuple of (valid_models, invalid_models)
        """
        valid_models = []
        invalid_models = []

        for model in models:
            if self.is_valid_model(model):
                valid_models.append(model)
            else:
                invalid_models.append(model)

        return valid_models, invalid_models

    def get_all_valid_models(self) -> set[str]:
        """
        Get all valid model names across all providers.

        Returns:
            Set of all valid model names
        """
        # Get all YAML models
        yaml_model_db.load_models()
        all_models = set()
        for model_capability in yaml_model_db._models.values():
            all_models.add(model_capability.model_name)
        return all_models

    def get_providers_for_model(self, model_name: str) -> set[ProviderType]:
        """
        Get all providers that support a given model.

        Args:
            model_name: The model name to check

        Returns:
            Set of providers that support this model
        """
        # Get provider from YAML model capability
        capability = yaml_model_db.get_model(model_name)
        if capability and capability.provider:
            return {capability.provider}
        return set()

    def get_model_count(self) -> int:
        """
        Get the total number of valid models.

        Returns:
            Total count of valid models
        """
        return yaml_model_db.get_model_count()

    def get_model_capability(self, model_name: str) -> ModelCapability | None:
        """
        Get the full ModelCapability object for a model name.

        Args:
            model_name: The model name to get capability for

        Returns:
            ModelCapability object if model exists, None otherwise
        """
        return yaml_model_db.get_model(model_name)

    def convert_names_to_capabilities(
        self, model_names: list[str]
    ) -> tuple[list[ModelCapability], list[str]]:
        """
        Convert a list of model names to ModelCapability objects.

        Args:
            model_names: List of model names to convert

        Returns:
            Tuple of (valid_capabilities, invalid_model_names)
        """
        valid_capabilities = []
        invalid_names = []

        for model_name in model_names:
            capability = self.get_model_capability(model_name)
            if capability:
                valid_capabilities.append(capability)
            else:
                invalid_names.append(model_name)

        return valid_capabilities, invalid_names


# Global instance for use across the application
model_registry = ModelRegistry()
