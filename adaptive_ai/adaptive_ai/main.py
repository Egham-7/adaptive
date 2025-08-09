from typing import Any

import litserve as ls  # type: ignore
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
from adaptive_ai.utils.openai_utils import extract_last_message_content


# LitServe Console Logger
class ConsoleLogger(ls.Logger):
    def process(self, key: str, value: Any) -> None:
        print(f"[LitServe] Received {key} with value {value}", flush=True)


class ProtocolManagerAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.settings = get_settings()
        self.prompt_classifier = get_prompt_classifier(lit_logger=self)
        self.domain_classifier = get_domain_classifier(lit_logger=self)

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.model_selection_service = ModelSelectionService(lit_logger=self)
        self.protocol_manager = ProtocolManager(lit_logger=self)

    def decode_request(self, request: dict[str, Any]) -> ModelSelectionRequest:
        # Convert cost_preference strings to cost_bias numbers
        if "protocol_manager_config" in request:
            config = request["protocol_manager_config"]
            if "cost_preference" in config:
                cost_preference = config.pop(
                    "cost_preference"
                )  # Remove the string version
                # Convert cost_preference to cost_bias
                cost_bias_map = {
                    "budget": 0.1,  # Strong preference for cheap models
                    "balanced": 0.5,  # Balanced between cost and performance
                    "performance": 0.8,  # Strong preference for performance
                    "premium": 0.9,  # Premium models preferred
                }
                config["cost_bias"] = cost_bias_map.get(cost_preference, 0.5)

        return ModelSelectionRequest(**request)

    def predict(
        self, requests: list[ModelSelectionRequest]
    ) -> list[OrchestratorResponse]:
        import time

        # Process model validation and conversion for each request
        processed_requests = []
        for req in requests:
            # Enrich partial models if they exist in protocol_manager_config
            if req.protocol_manager_config and req.protocol_manager_config.models:
                req.protocol_manager_config.models = (
                    self.model_selection_service.enrich_partial_models(
                        req.protocol_manager_config.models
                    )
                )
            processed_requests.append(req)

        outputs: list[OrchestratorResponse] = []

        # Get the most recent message content for classification
        prompts: list[str] = [
            extract_last_message_content(req.chat_completion_request)
            for req in processed_requests
        ]

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

        self.log("predict_called", {"batch_size": len(processed_requests)})

        if len(all_classification_results) != len(processed_requests):
            raise ValueError(
                f"Task classification results count ({len(all_classification_results)}) "
                f"doesn't match requests count ({len(processed_requests)})"
            )

        if len(all_domain_results) != len(processed_requests):
            raise ValueError(
                f"Domain classification results count ({len(all_domain_results)}) "
                f"doesn't match requests count ({len(processed_requests)})"
            )

        for i, req in enumerate(processed_requests):
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

            current_prompt = prompts[i]
            try:
                prompt_token_count = len(self.tokenizer.encode(current_prompt))
            except Exception as e:
                prompt_token_count = len(current_prompt) // 4  # Rough approximation
                self.log(
                    "prompt_token_count_fallback",
                    {"prompt": current_prompt, "error": str(e)},
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
                    domain_classification=current_domain_result,
                )
            )
            select_t1 = time.perf_counter()
            self.log("model_selection_time", select_t1 - select_t0)

            if not standard_candidates and not minion_candidates:
                self.log("no_eligible_models", {"prompt": current_prompt})
                raise ValueError(
                    "No eligible models found after applying provider and task constraints"
                )

            protocol_t0 = time.perf_counter()
            # Determine available protocols based on candidates
            available_protocols = []
            if standard_candidates:
                available_protocols.append("standard_llm")
            if minion_candidates:
                available_protocols.append("minion")

            selected_protocol = self.protocol_manager._select_best_protocol(
                classification_result=current_classification_result,
                token_count=prompt_token_count,
                available_protocols=available_protocols,
                request=req,
            )

            # Use the protocol manager's decision instead of defaulting to standard
            from adaptive_ai.models.llm_enums import ProtocolType
            from adaptive_ai.models.llm_orchestration_models import (
                Alternative,
                MinionInfo,
                StandardLLMInfo,
            )

            # Create response based on protocol manager's decision
            if selected_protocol == "minion" and minion_candidates:
                # Create minion protocol response
                orchestrator_response = OrchestratorResponse(
                    protocol=ProtocolType.MINION,
                    minion=MinionInfo(
                        provider=(
                            minion_candidates[0].providers[0].value
                            if hasattr(minion_candidates[0].providers[0], "value")
                            else str(minion_candidates[0].providers[0])
                        ),
                        model=minion_candidates[0].model_name,
                        parameters=self.protocol_manager._get_tuned_parameters(
                            current_classification_result,
                            (
                                current_classification_result.task_type_1[0]
                                if current_classification_result.task_type_1
                                else "general"
                            ),
                        ),
                        alternatives=[
                            Alternative(
                                provider=(
                                    alt.providers[0].value
                                    if hasattr(alt.providers[0], "value")
                                    else str(alt.providers[0])
                                ),
                                model=alt.model_name,
                            )
                            for alt in minion_candidates[
                                1:3
                            ]  # Include top 2 alternatives
                        ],
                    ),
                )
            elif selected_protocol == "standard_llm" and standard_candidates:
                # Create standard protocol response
                orchestrator_response = OrchestratorResponse(
                    protocol=ProtocolType.STANDARD_LLM,
                    standard=StandardLLMInfo(
                        provider=(
                            standard_candidates[0].providers[0].value
                            if hasattr(standard_candidates[0].providers[0], "value")
                            else str(standard_candidates[0].providers[0])
                        ),
                        model=standard_candidates[0].model_name,
                        parameters=self.protocol_manager._get_tuned_parameters(
                            current_classification_result,
                            (
                                current_classification_result.task_type_1[0]
                                if current_classification_result.task_type_1
                                else "general"
                            ),
                        ),
                        alternatives=[
                            Alternative(
                                provider=(
                                    alt.providers[0].value
                                    if hasattr(alt.providers[0], "value")
                                    else str(alt.providers[0])
                                ),
                                model=alt.model_name,
                            )
                            for alt in standard_candidates[
                                1:3
                            ]  # Include top 2 alternatives
                        ],
                    ),
                )
            else:
                raise ValueError(
                    f"No available candidates for selected protocol: {selected_protocol}"
                )
            protocol_t1 = time.perf_counter()
            self.log("protocol_selection_time", protocol_t1 - protocol_t0)

            # No caching needed for fast rule-based routing
            outputs.append(orchestrator_response)

        self.log("predict_completed", {"output_count": len(outputs)})
        return outputs

    def encode_response(self, output: OrchestratorResponse) -> dict[str, Any]:
        result = output.model_dump()
        return result if isinstance(result, dict) else {}


def create_app() -> ls.LitServer:
    settings = get_settings()
    api = ProtocolManagerAPI(
        max_batch_size=settings.litserve.max_batch_size,
        batch_timeout=settings.litserve.batch_timeout,
    )

    loggers: list[ConsoleLogger] = [ConsoleLogger()]
    callbacks: list[object] = []

    # Create LitServer
    server = ls.LitServer(
        api,
        accelerator=settings.litserve.accelerator,
        devices=settings.litserve.devices,
        loggers=loggers,
        callbacks=callbacks,
    )

    return server


def main() -> None:
    settings = get_settings()
    app = create_app()
    app.run(
        generate_client_file=False, host=settings.server.host, port=settings.server.port
    )


if __name__ == "__main__":
    main()
