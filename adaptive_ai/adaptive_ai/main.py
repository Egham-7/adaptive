from typing import Any

# Removed HuggingFaceEmbeddings import to avoid model downloads
import litserve as ls
import tiktoken

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelCapability,
    ModelSelectionRequest,
)
from adaptive_ai.models.llm_orchestration_models import OrchestratorResponse

# Removed: from adaptive_ai.services.classification_result_embedding_cache import EmbeddingCache
from adaptive_ai.services.model_selector import (
    ModelSelectionService,
)
from adaptive_ai.services.prompt_classifier import get_prompt_classifier
from adaptive_ai.services.protocol_manager import ProtocolManager


# LitServe Console Logger
class ConsoleLogger(ls.Logger):
    def process(self, key: str, value: Any) -> None:
        print(f"[LitServe] Received {key} with value {value}", flush=True)


class ProtocolManagerAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.settings = get_settings()
        self.prompt_classifier = get_prompt_classifier(lit_logger=self)

        # Cache removed: rule-based routing is fast enough without caching
        # No need for embedding models or cache infrastructure

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.model_selection_service = ModelSelectionService(lit_logger=self)
        self.protocol_manager = ProtocolManager(lit_logger=self, device=device)

    def decode_request(self, request: ModelSelectionRequest) -> ModelSelectionRequest:
        return request

    def predict(
        self, requests: list[ModelSelectionRequest]
    ) -> list[OrchestratorResponse]:
        import time

        outputs: list[OrchestratorResponse] = []

        prompts: list[str] = [req.prompt for req in requests]

        t0 = time.perf_counter()
        all_classification_results: list[ClassificationResult] = (
            self.prompt_classifier.classify_prompts(prompts)
        )
        t1 = time.perf_counter()
        self.log("classification_time", t1 - t0)

        self.log("predict_called", {"batch_size": len(requests)})

        if len(all_classification_results) != len(requests):
            raise ValueError(
                f"Classification results count ({len(all_classification_results)}) "
                f"doesn't match requests count ({len(requests)})"
            )

        for i, req in enumerate(requests):
            current_classification_result: ClassificationResult = (
                all_classification_results[i]
            )
            # Rule-based routing is fast enough - no caching needed
            self.log("cache_disabled", "rule_based_routing_is_fast")

            # Direct routing without cache
            try:
                prompt_token_count = len(self.tokenizer.encode(req.prompt))
            except Exception as e:
                prompt_token_count = len(req.prompt) // 4  # Rough approximation
                self.log(
                    "prompt_token_count_fallback",
                    {"prompt": req.prompt, "error": str(e)},
                )

            select_t0 = time.perf_counter()
            candidate_models: list[ModelCapability] = (
                self.model_selection_service.select_candidate_models(
                    request=req,
                    classification_result=current_classification_result,
                    prompt_token_count=prompt_token_count,
                )
            )
            select_t1 = time.perf_counter()
            self.log("model_selection_time", select_t1 - select_t0)

            # Get designated minion and alternatives
            minion_model = self.model_selection_service.get_designated_minion(
                classification_result=current_classification_result
            )
            minion_alternatives = self.model_selection_service.get_minion_alternatives(
                primary_minion=minion_model
            )

            if not candidate_models:
                self.log("no_eligible_models", {"prompt": req.prompt})
                raise ValueError(
                    "No eligible models found after applying provider and task constraints"
                )
            protocol_t0 = time.perf_counter()
            orchestrator_response: OrchestratorResponse = (
                self.protocol_manager.select_protocol(
                    candidate_models=candidate_models,
                    minion_model=minion_model,
                    minion_alternatives=minion_alternatives,
                    classification_result=current_classification_result,
                    token_count=prompt_token_count,
                    request=req,
                )
            )
            protocol_t1 = time.perf_counter()
            self.log("protocol_selection_time", protocol_t1 - protocol_t0)

            # No caching needed for fast rule-based routing
            outputs.append(orchestrator_response)

        self.log("predict_completed", {"output_count": len(outputs)})
        return outputs

    def encode_response(self, output: OrchestratorResponse) -> dict[str, Any]:
        return output.model_dump()


def create_app() -> ls.LitServer:
    settings = get_settings()
    api = ProtocolManagerAPI(
        max_batch_size=settings.litserve.max_batch_size,
        batch_timeout=settings.litserve.batch_timeout,
    )

    loggers: list[ConsoleLogger] = [ConsoleLogger()]
    callbacks: list[object] = []

    return ls.LitServer(
        api,
        accelerator=settings.litserve.accelerator,
        devices=settings.litserve.devices,
        loggers=loggers,
        callbacks=callbacks,
    )


def main() -> None:
    settings = get_settings()
    app = create_app()
    app.run(
        generate_client_file=False, host=settings.server.host, port=settings.server.port
    )


if __name__ == "__main__":
    main()
