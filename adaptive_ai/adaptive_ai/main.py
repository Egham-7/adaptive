from typing import Any

# Removed HuggingFaceEmbeddings import to avoid model downloads
import litserve as ls
import tiktoken

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.llm_classification_models import (
    ClassificationResult,
    DomainClassificationResult,
)
from adaptive_ai.models.llm_core_models import (
    ModelEntry,
    ModelSelectionRequest,
)
from adaptive_ai.models.llm_orchestration_models import OrchestratorResponse

# Removed: from adaptive_ai.services.classification_result_embedding_cache import EmbeddingCache
from adaptive_ai.services.domain_classifier import get_domain_classifier
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
        self.domain_classifier = get_domain_classifier(lit_logger=self)

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

        # Run both task and domain classification in parallel
        t0 = time.perf_counter()
        all_classification_results: list[ClassificationResult] = (
            self.prompt_classifier.classify_prompts(prompts)
        )
        t1 = time.perf_counter()
        self.log("task_classification_time", t1 - t0)

        # Domain classification
        t2 = time.perf_counter()
        try:
            all_domain_results: list[DomainClassificationResult] = (
                self.domain_classifier.classify_domains(prompts)
            )
            t3 = time.perf_counter()
            self.log("domain_classification_time", t3 - t2)
            self.log(
                "domain_classification_success",
                {
                    "batch_size": len(all_domain_results),
                    "sample_domain": (
                        all_domain_results[0].domain.value
                        if all_domain_results
                        else None
                    ),
                    "sample_confidence": (
                        all_domain_results[0].confidence if all_domain_results else None
                    ),
                },
            )
        except Exception as e:
            t3 = time.perf_counter()
            self.log(
                "domain_classification_failed",
                {
                    "error": str(e),
                    "time_taken": t3 - t2,
                    "batch_size": len(prompts),
                },
            )
            # Re-raise the exception instead of creating fallback results
            raise RuntimeError(f"Domain classification failed: {e!s}") from e

        self.log("predict_called", {"batch_size": len(requests)})

        if len(all_classification_results) != len(requests):
            raise ValueError(
                f"Task classification results count ({len(all_classification_results)}) "
                f"doesn't match requests count ({len(requests)})"
            )

        if len(all_domain_results) != len(requests):
            raise ValueError(
                f"Domain classification results count ({len(all_domain_results)}) "
                f"doesn't match requests count ({len(requests)})"
            )

        for i, req in enumerate(requests):
            current_classification_result: ClassificationResult = (
                all_classification_results[i]
            )
            current_domain_result: DomainClassificationResult = all_domain_results[i]

            # Log both classifications for this request
            self.log(
                "combined_classification_results",
                {
                    "task_type": (
                        current_classification_result.task_type_1[0]
                        if current_classification_result.task_type_1
                        else "Other"
                    ),
                    "domain": current_domain_result.domain.value,
                    "domain_confidence": current_domain_result.confidence,
                },
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
            standard_candidates: list[ModelEntry] = (
                self.model_selection_service.select_candidate_models(
                    request=req,
                    classification_result=current_classification_result,
                    prompt_token_count=prompt_token_count,
                    domain_classification=current_domain_result,
                )
            )

            minion_candidates: list[ModelEntry] = (
                self.model_selection_service.get_minion_candidates(
                    classification_result=current_classification_result,
                    domain_classification=current_domain_result,
                )
            )
            select_t1 = time.perf_counter()
            self.log("model_selection_time", select_t1 - select_t0)

            if not standard_candidates and not minion_candidates:
                self.log("no_eligible_models", {"prompt": req.prompt})
                raise ValueError(
                    "No eligible models found after applying provider and task constraints"
                )

            protocol_t0 = time.perf_counter()
            orchestrator_response: OrchestratorResponse = (
                self.protocol_manager.select_protocol(
                    standard_candidates=standard_candidates,
                    minion_candidates=minion_candidates,
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
