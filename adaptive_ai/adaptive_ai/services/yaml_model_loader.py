"""
Simple YAML model database loader for supported providers.
Loads YAML files once at startup for fast in-memory lookups.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import ProviderType

logger = logging.getLogger(__name__)


class YAMLModelDatabase:
    """Simple in-memory model database loaded from YAML files."""

    def __init__(self) -> None:
        """Initialize empty database."""
        self._models: dict[str, ModelCapability] = {}
        self._loaded = False

    def load_models(self) -> None:
        """Load models from YAML files for supported providers."""
        if self._loaded:
            return

        # Get the model_data directory path
        current_dir = Path(__file__).parent.parent.parent
        yaml_dir = current_dir / "model_data" / "data" / "provider_models"

        if not yaml_dir.exists():
            logger.warning(f"YAML model directory not found: {yaml_dir}")
            self._loaded = True
            return

        # Supported providers (matching ProviderType enum)
        supported_providers = [
            "anthropic",
            "groq",
            "azure",
            "together",
            "openai",
            "google",
            "deepseek",
            "mistral",
            "grok",
        ]

        models_loaded = 0
        for provider in supported_providers:
            yaml_file = yaml_dir / f"{provider}_models_structured.yaml"

            if not yaml_file.exists():
                logger.debug(f"YAML file not found for provider: {provider}")
                continue

            try:
                models_loaded += self._load_provider_yaml(yaml_file, provider.upper())
            except Exception as e:
                logger.error(f"Failed to load {provider} models: {e}")

        self._loaded = True
        logger.info(f"Loaded {models_loaded} models from YAML files")

    def _yaml_to_model_capability(
        self, yaml_data: dict[str, Any], provider_name: str
    ) -> ModelCapability:
        """Convert YAML model data to ModelCapability object."""
        # Map provider name to ProviderType enum
        provider_mapping = {
            "ANTHROPIC": ProviderType.ANTHROPIC,
            "OPENAI": ProviderType.OPENAI,
            "GOOGLE": ProviderType.GOOGLE,
            "GROQ": ProviderType.GROQ,
            "DEEPSEEK": ProviderType.DEEPSEEK,
            "MISTRAL": ProviderType.MISTRAL,
            "GROK": ProviderType.GROK,
        }

        provider_type = provider_mapping.get(provider_name.upper(), ProviderType.OPENAI)

        return ModelCapability(
            description=yaml_data.get("description"),
            provider=provider_type,
            model_name=yaml_data.get("model_name", ""),
            cost_per_1m_input_tokens=yaml_data.get("cost_per_1m_input_tokens"),
            cost_per_1m_output_tokens=yaml_data.get("cost_per_1m_output_tokens"),
            max_context_tokens=yaml_data.get("max_context_tokens"),
            max_output_tokens=yaml_data.get("max_output_tokens"),
            supports_function_calling=yaml_data.get("supports_function_calling"),
            languages_supported=yaml_data.get("languages_supported") or [],
            model_size_params=yaml_data.get("model_size_params"),
            latency_tier=yaml_data.get("latency_tier"),
            task_type=yaml_data.get("task_type"),
            complexity=yaml_data.get("complexity"),
        )

    def _load_provider_yaml(self, yaml_file: Path, provider_name: str) -> int:
        """Load models from a single provider YAML file."""
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            models_section = data.get("models", {})
            models_loaded = 0

            for yaml_key, model_data in models_section.items():
                try:
                    model_capability = self._yaml_to_model_capability(
                        model_data, provider_name
                    )

                    # Store by model name for fast lookup
                    model_name = model_capability.model_name
                    self._models[model_name] = model_capability
                    models_loaded += 1

                except Exception as e:
                    logger.debug(f"Failed to convert model {yaml_key}: {e}")
                    continue

            logger.debug(f"Loaded {models_loaded} models from {provider_name}")
            return models_loaded

        except Exception as e:
            logger.error(f"Failed to parse YAML file {yaml_file}: {e}")
            return 0

    def get_model(self, model_name: str) -> ModelCapability | None:
        """
        Get model by name.

        Args:
            model_name: Model name to lookup

        Returns:
            ModelCapability if found, None otherwise
        """
        if not self._loaded:
            self.load_models()

        # Try exact match first
        if model_name in self._models:
            return self._models[model_name]

        # Try case-insensitive lookup
        for key in self._models.keys():
            if key.lower() == model_name.lower():
                return self._models[key]
        return None

    def has_model(self, model_name: str) -> bool:
        """Check if model exists in database."""
        return self.get_model(model_name) is not None

    def get_model_count(self) -> int:
        """Get total number of unique models loaded."""
        if not self._loaded:
            self.load_models()
        # Count unique models (don't count normalized name duplicates)
        unique_models = set()
        for model_cap in self._models.values():
            unique_models.add(f"{model_cap.provider}:{model_cap.model_name}")
        return len(unique_models)


# Global instance for application use
yaml_model_db = YAMLModelDatabase()
