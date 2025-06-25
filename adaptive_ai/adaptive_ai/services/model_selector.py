# adaptive_ai/services/model_selection_service.py


from adaptive_ai.config.model_catalog import (
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


class ModelSelectionService:
    def __init__(self) -> None:
        # Pre-process all model capabilities into a quick-lookup map
        self._all_model_capabilities_by_id: dict[
            tuple[ProviderType, str], ModelCapability
        ] = {
            (m_cap.provider, m_cap.model_name): m_cap
            for provider_list in provider_model_capabilities.values()
            for m_cap in provider_list
        }

        # Define default fallback model entries if no specific task mapping is found
        self._default_task_specific_model_entries: list[TaskModelEntry] = [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GOOGLE, model_name="gemini-2.5-pro"),
            TaskModelEntry(
                provider=ProviderType.ANTHROPIC, model_name="claude-sonnet-4-20250514"
            ),
            TaskModelEntry(provider=ProviderType.GOOGLE, model_name="gemini-2.5-flash"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(
                provider=ProviderType.MISTRAL, model_name="mistral-small-latest"
            ),
        ]

    def select_candidate_models(
        self,
        request: ModelSelectionRequest,
        classification_result: ClassificationResult,
        prompt_token_count: int,
    ) -> list[ModelCapability]:
        """
        Selects and prunes candidate models based on provider constraints, task type,
        and context window limitations.

        Args:
            request: The original ModelSelectionRequest.
            classification_result: The result of prompt classification.
            prompt_token_count: The number of tokens in the prompt.

        Returns:
            A list of eligible ModelCapability objects, ordered by preference.
        """
        # Determine eligible providers based on request constraints
        eligible_providers: list[ProviderType] = (
            [ProviderType(p) for p in request.provider_constraint]
            if request.provider_constraint
            else list(provider_model_capabilities.keys())
        )

        # Determine primary task type from classification result
        primary_task_type: TaskType = (
            TaskType(classification_result.task_type_1[0])
            if classification_result.task_type_1
            else TaskType.OTHER
        )

        # Get task-specific model preference order, fall back to default if not found
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

            # Prune based on provider constraint and avoid duplicates
            if (
                task_model_entry.provider not in eligible_providers
                or model_identifier in seen_model_identifiers
            ):
                continue

            found_model_cap = self._all_model_capabilities_by_id.get(model_identifier)

            # Prune based on context window and add to candidates
            if (
                found_model_cap
                and found_model_cap.max_context_tokens >= prompt_token_count
            ):
                candidate_models.append(found_model_cap)
                seen_model_identifiers.add(model_identifier)

        return candidate_models
