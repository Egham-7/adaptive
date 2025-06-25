from typing import Any

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import litserve as ls
import tiktoken

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelCapability,
    ModelSelectionRequest,
)
from adaptive_ai.models.llm_orchestration_models import OrchestratorResponse
from adaptive_ai.services.classification_result_embedding_cache import EmbeddingCache
from adaptive_ai.services.model_selector import (
    ModelSelectionService,
)
from adaptive_ai.services.prompt_classifier import get_prompt_classifier
from adaptive_ai.services.protocol_manager import ProtocolManager


class ProtocolManagerAPI(ls.LitAPI):
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

        self.model_selection_service = ModelSelectionService()
        self.protocol_manager = ProtocolManager()

    def decode_request(self, request: ModelSelectionRequest) -> ModelSelectionRequest:
        return request

    def predict(
        self, requests: list[ModelSelectionRequest]
    ) -> list[OrchestratorResponse]:
        outputs: list[OrchestratorResponse] = []

        prompts: list[str] = [req.prompt for req in requests]

        all_classification_results: list[ClassificationResult] = (
            self.prompt_classifier.classify_prompts(prompts)
        )

        for i, req in enumerate(requests):
            current_classification_result: ClassificationResult = (
                all_classification_results[i]
            )

            cached_orchestrator_response: OrchestratorResponse | None = (
                self.embedding_cache.search_cache(current_classification_result)
            )

            if cached_orchestrator_response:
                outputs.append(cached_orchestrator_response)
            else:
                try:
                    prompt_token_count = len(self.tokenizer.encode(req.prompt))
                except Exception:
                    # Fallback to character-based estimation
                    prompt_token_count = len(req.prompt) // 4  # Rough approximation

                candidate_models: list[ModelCapability] = (
                    self.model_selection_service.select_candidate_models(
                        request=req,
                        classification_result=current_classification_result,
                        prompt_token_count=prompt_token_count,
                    )
                )

                if not candidate_models:
                    raise ValueError(
                        "No eligible models found after applying provider and task constraints"
                    )
                orchestrator_response: OrchestratorResponse = (
                    self.protocol_manager.select_protocol(
                        candidate_models=candidate_models,
                        classification_result=current_classification_result,
                        prompt=req.prompt,
                    )
                )
                self.embedding_cache.add_to_cache(
                    current_classification_result, orchestrator_response
                )
                outputs.append(orchestrator_response)

        return outputs

    def encode_response(self, output: OrchestratorResponse) -> dict[str, Any]:
        return output.model_dump()


def create_app() -> ls.LitServer:
    settings = get_settings()
    api = ProtocolManagerAPI(
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
