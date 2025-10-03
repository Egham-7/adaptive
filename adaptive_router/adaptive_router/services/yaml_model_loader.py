"""
Simple YAML model database loader for supported providers.
Loads YAML files once at startup for fast in-memory lookups.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from adaptive_router.models.llm_core_models import ModelCapability

logger = logging.getLogger(__name__)


class YAMLModelDatabase:
    """Simple in-memory model database loaded from YAML files."""

    def __init__(self) -> None:
        """Initialize and immediately load models from YAML files."""
        self._models: dict[str, ModelCapability] = {}  # unique_id -> ModelCapability
        self._models_by_name: dict[str, list[ModelCapability]] = (
            {}
        )  # model_name -> list of providers
        self._load_models_from_yaml()

    def _load_models_from_yaml(self) -> None:
        """Load models from YAML files for supported providers."""
        # Get the model_data directory path
        current_dir = Path(__file__).parent.parent.parent
        yaml_dir = current_dir / "model_data" / "data" / "provider_models"

        if not yaml_dir.exists():
            logger.warning(
                "YAML model directory not found", extra={"yaml_dir": str(yaml_dir)}
            )
            return

        # Supported providers
        supported_providers = [
            "anthropic",
            "groq",
            "openai",
            "gemini",
            "deepseek",
            "grok",
        ]

        # Load models from all providers, handling errors gracefully
        models_loaded_per_provider = []
        for provider in supported_providers:
            yaml_file = yaml_dir / f"{provider}_models_structured.yaml"
            if yaml_file.exists():
                try:
                    models_loaded_per_provider.append(
                        self._load_provider_yaml(yaml_file, provider.upper())
                    )
                except Exception as e:
                    logger.error(
                        "Failed to load provider models",
                        extra={
                            "provider": provider,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    models_loaded_per_provider.append(0)
            else:
                logger.debug(
                    "YAML file not found for provider", extra={"provider": provider}
                )

        total_models_loaded = sum(models_loaded_per_provider)
        logger.info(
            "Loaded models from YAML files",
            extra={"total_models_loaded": total_models_loaded},
        )

    def _yaml_to_model_capability(
        self, yaml_data: dict[str, Any], provider_name: str, yaml_key: str
    ) -> ModelCapability:
        """Convert YAML model data to ModelCapability object.

        Args:
            yaml_data: The model data from YAML
            provider_name: Name of the provider
            yaml_key: The original key from YAML file (used for fallback naming)

        Returns:
            ModelCapability object with validated model_name

        Raises:
            ValueError: If model_name is missing/empty and fallback generation fails
        """
        # Validate and get model_name
        model_name = yaml_data.get("model_name")
        if not model_name or not model_name.strip():
            # Generate a safe fallback name using provider and yaml_key
            fallback_name = f"{provider_name.lower()}_{yaml_key}"
            logger.warning(
                "Missing or empty model_name, using fallback",
                extra={
                    "provider": provider_name,
                    "yaml_key": yaml_key,
                    "fallback_name": fallback_name,
                },
            )
            model_name = fallback_name

        return ModelCapability(
            description=yaml_data.get("description"),
            provider=provider_name.lower(),  # Store consistently in lowercase
            model_name=model_name.strip(),  # Ensure no leading/trailing whitespace
            cost_per_1m_input_tokens=yaml_data.get("cost_per_1m_input_tokens"),
            cost_per_1m_output_tokens=yaml_data.get("cost_per_1m_output_tokens"),
            max_context_tokens=yaml_data.get("max_context_tokens"),
            supports_function_calling=yaml_data.get("supports_function_calling"),
            languages_supported=yaml_data.get("languages_supported") or [],
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
                        model_data, provider_name, yaml_key
                    )

                    # Validate model_name before storing
                    model_name = model_capability.model_name
                    if not model_name or not model_name.strip():
                        logger.error(
                            "Model capability has invalid model_name after processing",
                            extra={"provider": provider_name, "yaml_key": yaml_key},
                        )
                        continue

                    # Store by unique_id (provider:model_name) to allow multi-provider models
                    unique_id = model_capability.unique_id
                    # Normalize unique_id to lowercase for case-insensitive lookup
                    normalized_unique_id = unique_id.lower()
                    if normalized_unique_id in self._models:
                        logger.warning(
                            "Overwriting existing model entry",
                            extra={"unique_id": unique_id},
                        )

                    self._models[normalized_unique_id] = model_capability

                    # Also store by normalized model name for multi-provider lookup
                    normalized_name = model_name.lower().strip()
                    if normalized_name not in self._models_by_name:
                        self._models_by_name[normalized_name] = []
                    self._models_by_name[normalized_name].append(model_capability)

                    models_loaded += 1

                except Exception as e:
                    logger.debug(
                        "Failed to convert model",
                        extra={
                            "yaml_key": yaml_key,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    continue

            logger.debug(
                "Loaded models from provider",
                extra={"models_loaded": models_loaded, "provider": provider_name},
            )
            return models_loaded

        except yaml.YAMLError as e:
            logger.error(
                "Failed to parse YAML file - YAML parsing error",
                extra={
                    "yaml_file": str(yaml_file),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return 0
        except OSError as e:
            logger.error(
                "Failed to parse YAML file - IO error",
                extra={
                    "yaml_file": str(yaml_file),
                    "errno": getattr(e, "errno", None),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return 0
        except Exception as e:
            logger.error(
                "Failed to parse YAML file - unexpected error",
                extra={
                    "yaml_file": str(yaml_file),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    def get_model(self, unique_id: str) -> ModelCapability | None:
        """
        Get model capability by unique_id (provider:model_name).

        Args:
            unique_id: Unique identifier in format "provider:model_name"

        Returns:
            ModelCapability if found, None otherwise
        """
        # Normalize unique_id to lowercase for case-insensitive lookup
        return self._models.get(unique_id.lower())

    def get_models_by_name(self, model_name: str) -> list[ModelCapability]:
        """
        Get all model capabilities for a given model name across all providers.

        Args:
            model_name: The model name to search for

        Returns:
            List of ModelCapability objects from all providers that serve this model
        """
        # Normalize model name for lookup
        normalized_name = model_name.lower().strip()
        return self._models_by_name.get(normalized_name, [])

    def has_model(self, unique_id: str) -> bool:
        """Check if model exists in database by unique_id."""
        # Normalize unique_id to lowercase for case-insensitive lookup
        return unique_id.lower() in self._models

    def get_model_count(self) -> int:
        """Get total number of unique models loaded."""
        return len(self._models)

    def get_all_models(self) -> dict[str, ModelCapability]:
        """Get all loaded models."""
        return self._models.copy()
