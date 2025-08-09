"""
Model registry service for validating model names across all providers.
"""

from adaptive_ai.config.providers import provider_model_names
from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import ProviderType
from adaptive_ai.services.yaml_model_loader import yaml_model_db


class ModelRegistry:
    """Service for validating and managing model availability across providers."""

    def __init__(self) -> None:
        """Initialize the model registry with all available models."""
        self._valid_models: set[str] = set()
        self._model_to_providers: dict[str, set[ProviderType]] = {}

        # Build lookup structures from provider model names (lightweight)
        self._build_model_registry()

    def _build_model_registry(self) -> None:
        """Build internal lookup structures for fast model validation."""
        for provider, model_names in provider_model_names.items():
            for model_name in model_names:
                self._valid_models.add(model_name)

                if model_name not in self._model_to_providers:
                    self._model_to_providers[model_name] = set()
                self._model_to_providers[model_name].add(provider)

    def is_valid_model(self, model_name: str) -> bool:
        """
        Check if a model name is valid (exists in any provider's capabilities).

        Args:
            model_name: The model name to validate

        Returns:
            True if the model exists, False otherwise
        """
        # Check YAML database first (primary source)
        if yaml_model_db.has_model(model_name):
            return True

        # Fallback to hardcoded model names
        return model_name in self._valid_models

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
        # Combine hardcoded models with YAML models
        all_models = self._valid_models.copy()

        # Add all YAML models by iterating through the internal dict
        yaml_model_db.load_models()
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
        return self._model_to_providers.get(model_name, set())

    def get_model_count(self) -> int:
        """
        Get the total number of valid models.

        Returns:
            Total count of valid models
        """
        # YAML database now contains all models (including previously hardcoded ones)
        # Add any unique models from hardcoded config that aren't in YAML (should be 0)
        yaml_count = yaml_model_db.get_model_count()
        unique_hardcoded = len(
            self._valid_models
            - {
                model.model_name
                for model in [
                    yaml_model_db.get_model(name) for name in self._valid_models
                ]
                if model is not None
            }
        )
        return yaml_count + unique_hardcoded

    def get_model_capability(self, model_name: str) -> ModelCapability | None:
        """
        Get the full ModelCapability object for a model name.

        Args:
            model_name: The model name to get capability for

        Returns:
            ModelCapability object if model exists, None otherwise
        """
        # Check YAML database first (primary source with full capabilities)
        capability = yaml_model_db.get_model(model_name)
        if capability:
            return capability

        # No fallback needed - YAML now contains all models
        return None

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
