from typing import Any

from adaptive_ai.config.model_catalog import (
    ACTIVE_PROVIDERS,
    minion_task_model_mappings,
    provider_model_capabilities,
    task_model_mappings_data,
)
from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelCapability,
    ModelSelectionRequest,
    TaskModelEntry,
)
from adaptive_ai.models.llm_enums import ProviderType, TaskType


class LitLoggerProtocol:
    def log(self, key: str, value: Any) -> None: ...


class ModelSelectionService:
    def __init__(self, lit_logger: LitLoggerProtocol | None = None) -> None:
        self._all_model_capabilities_by_id: dict[
            tuple[ProviderType, str], ModelCapability
        ] = {
            (m_cap.provider, m_cap.model_name): m_cap
            for provider_list in provider_model_capabilities.values()
            for m_cap in provider_list
        }

        # Updated default models to only include active providers
        self._default_task_specific_model_entries: list[TaskModelEntry] = [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
        ]
        self.lit_logger: LitLoggerProtocol | None = lit_logger
        self.log(
            "model_selection_service_init",
            {
                "default_models": [
                    e.model_name for e in self._default_task_specific_model_entries
                ]
            },
        )

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def select_candidate_models(
        self,
        request: ModelSelectionRequest,
        classification_result: ClassificationResult,
        prompt_token_count: int,
    ) -> list[ModelCapability]:
        self.log(
            "select_candidate_models_called",
            {
                "provider_constraint": request.provider_constraint,
                "prompt_token_count": prompt_token_count,
                "classification_result": classification_result.model_dump(
                    exclude_unset=True
                ),
            },
        )
        if request.provider_constraint:
            all_known_provider_types = {p.value for p in ProviderType}
            # Filter to only include active providers
            eligible_providers = {
                ProviderType(p)
                for p in request.provider_constraint
                if p in all_known_provider_types and ProviderType(p) in ACTIVE_PROVIDERS
            }
            for p in request.provider_constraint:
                if p not in all_known_provider_types:
                    self.log("invalid_provider_constraint", p)
                elif ProviderType(p) not in ACTIVE_PROVIDERS:
                    self.log("inactive_provider_constraint", p)
        else:
            # Only use active providers even when no constraint is specified
            eligible_providers = ACTIVE_PROVIDERS

        primary_task_type: TaskType = (
            TaskType(classification_result.task_type_1[0])
            if classification_result.task_type_1
            else TaskType.OTHER
        )

        self.log("primary_task_type", primary_task_type.value)

        task_mapping_data = task_model_mappings_data.get(primary_task_type)
        task_specific_model_entries: list[TaskModelEntry] = (
            task_mapping_data.model_entries
            if task_mapping_data
            else self._default_task_specific_model_entries
        )

        seen_model_identifiers: set[tuple[ProviderType, str]] = set()
        candidate_models: list[ModelCapability] = []

        for task_model_entry in task_specific_model_entries:
            model_identifier = (
                task_model_entry.provider,
                task_model_entry.model_name,
            )

            if (
                task_model_entry.provider not in eligible_providers
                or model_identifier in seen_model_identifiers
            ):
                self.log(
                    "model_skipped",
                    {
                        "provider": str(task_model_entry.provider),
                        "model": task_model_entry.model_name,
                    },
                )
                continue

            found_model_cap = self._all_model_capabilities_by_id.get(model_identifier)

            if (
                found_model_cap
                and found_model_cap.max_context_tokens >= prompt_token_count
            ):
                candidate_models.append(found_model_cap)
                seen_model_identifiers.add(model_identifier)
                self.log(
                    "model_candidate_added",
                    {
                        "provider": str(found_model_cap.provider),
                        "model": found_model_cap.model_name,
                    },
                )

        self.log("candidate_models_count", len(candidate_models))
        return candidate_models

    def get_designated_minion(
        self,
        classification_result: ClassificationResult,
    ) -> str:
        """Get the designated HuggingFace minion specialist for the task type."""
        primary_task_type: TaskType = (
            TaskType(classification_result.task_type_1[0])
            if classification_result.task_type_1
            else TaskType.OTHER
        )

        minion_model = minion_task_model_mappings.get(
            primary_task_type,
            minion_task_model_mappings[TaskType.OTHER],  # Fallback to OTHER
        )

        self.log(
            "designated_minion_selected",
            {
                "task_type": primary_task_type.value,
                "minion_model": minion_model,
            },
        )

        return minion_model

    def get_minion_alternatives(
        self,
        primary_minion: str,
    ) -> list[dict[str, str]]:
        """Generate fallback minion alternatives by using other capable minions."""
        alternatives = []

        # Get all other minions from the mapping that could potentially handle the task
        for _task, model in minion_task_model_mappings.items():
            if model != primary_minion:  # Exclude the primary minion
                alternatives.append({"provider": "adaptive", "model": model})

        # Limit to top 3 alternatives to avoid overwhelming
        return alternatives[:3]
