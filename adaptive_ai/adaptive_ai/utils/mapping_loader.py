"""
YAML loader utility for model mappings configuration.
Loads provider models, task mappings, and domain mappings from centralized YAML file.
"""

import logging
from pathlib import Path
from threading import Lock
from typing import Any, cast

import yaml

from adaptive_ai.models.llm_classification_models import DomainType
from adaptive_ai.models.llm_core_models import ModelEntry, TaskModelMapping
from adaptive_ai.models.llm_enums import ProviderType, TaskType

logger = logging.getLogger(__name__)


class MappingLoader:
    """Loads model mappings from YAML configuration file."""

    def __init__(self) -> None:
        """Initialize the mapping loader."""
        self._data: dict[str, Any] = {}
        self._loaded = False
        self._load_lock = Lock()

    def load_mappings(self) -> None:
        """Load mappings from YAML file."""
        # Double-checked locking pattern for thread safety
        if self._loaded:
            return

        with self._load_lock:
            # Check again in case another thread loaded while we were waiting
            if self._loaded:
                return

            self._load_yaml_data()
            self._loaded = True

    def _load_yaml_data(self) -> None:
        """Load data from YAML file."""
        # Get the model_data/data directory path
        model_data_dir = Path(__file__).parent.parent.parent / "model_data" / "data"
        yaml_file = model_data_dir / "model_mappings.yaml"

        if not yaml_file.exists():
            logger.error(f"Model mappings YAML file not found: {yaml_file}")
            raise FileNotFoundError(f"Model mappings file not found: {yaml_file}")

        try:
            with open(yaml_file, encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            logger.info(f"Loaded model mappings from {yaml_file}")
        except Exception as e:
            logger.error(f"Failed to load model mappings YAML: {e}")
            raise

    def get_provider_models(self) -> dict[ProviderType, list[str]]:
        """
        Get provider model mappings from YAML.

        Returns:
            Dictionary mapping ProviderType to list of model names
        """
        if not self._loaded:
            self.load_mappings()

        provider_models_yaml = self._data.get("provider_models", {})
        provider_models: dict[ProviderType, list[str]] = {}

        # Map string provider names to ProviderType enums
        provider_name_mapping = {
            "google": ProviderType.GOOGLE,
            "mistral": ProviderType.MISTRAL,
            "openai": ProviderType.OPENAI,
            "deepseek": ProviderType.DEEPSEEK,
            "groq": ProviderType.GROQ,
            "grok": ProviderType.GROK,
            "anthropic": ProviderType.ANTHROPIC,
            "huggingface": ProviderType.HUGGINGFACE,
        }

        for provider_name, models in provider_models_yaml.items():
            provider_type = provider_name_mapping.get(provider_name)
            if provider_type and isinstance(models, list):
                provider_models[provider_type] = models
            else:
                logger.warning(f"Unknown provider or invalid models: {provider_name}")

        return provider_models

    def get_task_mappings(self) -> dict[TaskType, TaskModelMapping]:
        """
        Get task model mappings from YAML.

        Returns:
            Dictionary mapping TaskType to TaskModelMapping
        """
        if not self._loaded:
            self.load_mappings()

        task_mappings_yaml = self._data.get("task_mappings", {})
        task_mappings: dict[TaskType, TaskModelMapping] = {}

        # Map string task names to TaskType enums
        task_name_mapping = {
            "open_qa": TaskType.OPEN_QA,
            "code_generation": TaskType.CODE_GENERATION,
            "summarization": TaskType.SUMMARIZATION,
            "text_generation": TaskType.TEXT_GENERATION,
            "chatbot": TaskType.CHATBOT,
            "classification": TaskType.CLASSIFICATION,
            "closed_qa": TaskType.CLOSED_QA,
            "rewrite": TaskType.REWRITE,
            "brainstorming": TaskType.BRAINSTORMING,
            "extraction": TaskType.EXTRACTION,
            "other": TaskType.OTHER,
        }

        # Provider name mapping
        provider_name_mapping = {
            "google": ProviderType.GOOGLE,
            "mistral": ProviderType.MISTRAL,
            "openai": ProviderType.OPENAI,
            "deepseek": ProviderType.DEEPSEEK,
            "groq": ProviderType.GROQ,
            "grok": ProviderType.GROK,
            "anthropic": ProviderType.ANTHROPIC,
            "huggingface": ProviderType.HUGGINGFACE,
        }

        for task_name, task_data in task_mappings_yaml.items():
            task_type = task_name_mapping.get(task_name)
            if not task_type:
                logger.warning(f"Unknown task type: {task_name}")
                continue

            model_entries = []
            for entry_data in task_data.get("model_entries", []):
                providers = []
                for provider_name in entry_data.get("providers", []):
                    provider_type = provider_name_mapping.get(provider_name)
                    if provider_type:
                        providers.append(provider_type)
                    else:
                        logger.warning(f"Unknown provider: {provider_name}")

                if providers:
                    model_entry = ModelEntry(
                        providers=cast(list[ProviderType | str], providers),
                        model_name=entry_data.get("model_name", ""),
                    )
                    model_entries.append(model_entry)

            task_mappings[task_type] = TaskModelMapping(model_entries=model_entries)

        return task_mappings

    def get_domain_mappings(self) -> dict[DomainType, ModelEntry]:
        """
        Get domain model mappings from YAML with fallback to OTHERDOMAINS.

        Returns:
            Dictionary mapping DomainType to ModelEntry
        """
        if not self._loaded:
            self.load_mappings()

        domain_mappings_yaml = self._data.get("domain_mappings", {})
        domain_mappings: dict[DomainType, ModelEntry] = {}

        # Provider name mapping
        provider_name_mapping = {
            "google": ProviderType.GOOGLE,
            "mistral": ProviderType.MISTRAL,
            "openai": ProviderType.OPENAI,
            "deepseek": ProviderType.DEEPSEEK,
            "groq": ProviderType.GROQ,
            "grok": ProviderType.GROK,
            "anthropic": ProviderType.ANTHROPIC,
            "huggingface": ProviderType.HUGGINGFACE,
        }

        # Load explicitly mapped domains from YAML
        for domain_name, domain_data in domain_mappings_yaml.items():
            # Convert string domain name to DomainType enum
            domain_type = self._get_domain_type_from_name(domain_name)
            if not domain_type:
                logger.warning(f"Unknown domain type: {domain_name}")
                continue

            providers = []
            for provider_name in domain_data.get("providers", []):
                provider_type = provider_name_mapping.get(provider_name)
                if provider_type:
                    providers.append(provider_type)
                else:
                    logger.warning(f"Unknown provider: {provider_name}")

            if providers:
                model_entry = ModelEntry(
                    providers=cast(list[ProviderType | str], providers),
                    model_name=domain_data.get("model_name", ""),
                )
                domain_mappings[domain_type] = model_entry

        # Add fallback mapping for all unmapped domain types to OTHERDOMAINS model
        otherdomains_mapping = domain_mappings.get(DomainType.OTHERDOMAINS)
        if otherdomains_mapping:
            for domain_type in DomainType:
                if domain_type not in domain_mappings:
                    domain_mappings[domain_type] = otherdomains_mapping
                    logger.debug(
                        f"Domain {domain_type} mapped to OTHERDOMAINS fallback"
                    )

        return domain_mappings

    def _get_domain_type_from_name(self, domain_name: str) -> DomainType | None:
        """Convert domain name string to DomainType enum using dynamic lookup."""
        # Try to find matching DomainType by converting names
        domain_name_normalized = domain_name.upper().replace("_", "_")

        for domain_type in DomainType:
            if domain_type.name.lower() == domain_name.lower():
                return domain_type
            # Also try with underscores converted
            if (
                domain_type.name.lower().replace("_", "_")
                == domain_name_normalized.lower()
            ):
                return domain_type

        return None


# Global instance for use across the application
mapping_loader = MappingLoader()
