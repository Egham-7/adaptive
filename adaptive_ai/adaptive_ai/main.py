from typing import Any

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import litserve as ls
import tiktoken

from adaptive_ai.config.model_catalog import (
    provider_model_capabilities,
    task_model_mappings_data,
)
from adaptive_ai.core.config import get_settings
from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelCapability,
    ModelSelectionRequest,
    TaskModelEntry,
    TaskModelMapping,  # Import TaskModelMapping for type hint
)
from adaptive_ai.models.llm_enums import ProviderType, TaskType
from adaptive_ai.models.llm_orchestration_models import OrchestratorResponse
from adaptive_ai.services.classification_result_embedding_cache import EmbeddingCache
from adaptive_ai.services.prompt_classifier import get_prompt_classifier


class ProtocolSelectorAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.settings = get_settings()
        self.prompt_classifier = get_prompt_classifier()

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_cache.model_name
        )
        self.embedding_cache = EmbeddingCache(
            embeddings_model=self.embedding_model,
            similarity_threshold=self.settings.embedding_cache.similarity_threshold,
        )

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self._all_model_capabilities_by_id: dict[
            tuple[ProviderType, str], ModelCapability
        ] = {
            (m_cap.provider, m_cap.model_name): m_cap
            for provider_list in provider_model_capabilities.values()
            for m_cap in provider_list
        }

        self._default_task_specific_model_entries = [
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

    def decode_request(self, request: ModelSelectionRequest) -> ModelSelectionRequest:
        return request

    def predict(
        self, requests: list[ModelSelectionRequest]
    ) -> list[OrchestratorResponse]:
        outputs: list[OrchestratorResponse] = []

        for req in requests:
            classification_results_list: list[ClassificationResult] = (
                self.prompt_classifier.classify_prompts([req.prompt])
            )
            current_classification_result: ClassificationResult = (
                classification_results_list[0]
            )

            cached_orchestrator_response: OrchestratorResponse | None = (
                self.embedding_cache.search_cache(current_classification_result)
            )

            if cached_orchestrator_response:
                outputs.append(cached_orchestrator_response)
            else:
                eligible_providers = (
                    [ProviderType(p) for p in req.provider_constraint]
                    if req.provider_constraint
                    else list(provider_model_capabilities.keys())
                )

                primary_task_type = (
                    TaskType(current_classification_result.task_type_1[0])
                    if current_classification_result.task_type_1
                    else TaskType.OTHER
                )

                task_mapping_data: TaskModelMapping | None = (
                    task_model_mappings_data.get(primary_task_type)
                )

                if task_mapping_data:
                    task_specific_model_entries: list[TaskModelEntry] = (
                        task_mapping_data.model_entries
                    )
                else:
                    task_specific_model_entries = (
                        self._default_task_specific_model_entries
                    )

                prompt_token_count = len(self.tokenizer.encode(req.prompt))

                seen_model_identifiers = set()
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
                        continue

                    found_model_cap = self._all_model_capabilities_by_id.get(
                        model_identifier
                    )

                    if (
                        found_model_cap
                        and found_model_cap.max_context_tokens >= prompt_token_count
                    ):
                        candidate_models.append(found_model_cap)
                        seen_model_identifiers.add(model_identifier)

                if not candidate_models:
                    raise ValueError(
                        f"No eligible models found after applying provider and task constraints for prompt: '{req.prompt[:50]}...'"
                    )
                """
                orchestrator_response: OrchestratorResponse = route_model(
                    req, candidate_models
                )
                self.embedding_cache.add_to_cache(
                    current_classification_result, orchestrator_response
                )
                outputs.append(orchestrator_response)
                """

        return outputs

    def encode_response(self, output: OrchestratorResponse) -> dict[str, Any]:
        return output.model_dump()


def create_app() -> ls.LitServer:
    settings = get_settings()
    api = ProtocolSelectorAPI(
        max_batch_size=settings.litserve.max_batch_size,
        batch_timeout=settings.litserve.batch_timeout,
    )

    return ls.LitServer(
        api,
        accelerator=settings.litserve.accelerator,
        devices=settings.litserve.devices,
    )


def main() -> None:
    settings = get_settings()
    app = create_app()
    app.run(
        generate_client_file=False, host=settings.server.host, port=settings.server.port
    )


if __name__ == "__main__":
    main()
