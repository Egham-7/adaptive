from typing import Any

import litserve as ls
from pydantic import BaseModel
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
from adaptive_ai.services.model_registry import model_registry
from adaptive_ai.services.model_selector import (
    ModelSelectionService,
)
from adaptive_ai.services.prompt_classifier import get_prompt_classifier
from adaptive_ai.services.protocol_manager import ProtocolManager
from adaptive_ai.utils.openai_utils import extract_last_message_content


# Pydantic models for validation endpoint
class ModelValidationRequest(BaseModel):
    models: list[str]


class ModelValidationResponse(BaseModel):
    valid_models: list[str]
    invalid_models: list[str]


# Pydantic models for model conversion endpoint
class ModelConversionRequest(BaseModel):
    model_names: list[str]


class ModelConversionResponse(BaseModel):
    model_capabilities: list[
        dict[str, Any]
    ]  # Will be ModelCapability objects serialized as dicts
    invalid_models: list[str]


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
        self.protocol_manager = ProtocolManager(lit_logger=self, device=device)

    def decode_request(self, request: dict[str, Any]) -> ModelSelectionRequest:
        return ModelSelectionRequest(**request)

    def _process_models_array(
        self, request: ModelSelectionRequest
    ) -> ModelSelectionRequest:
        """
        Process the models array in a request, validate and convert to protocol_manager_config.

        Args:
            request: The original request that may contain a models array

        Returns:
            Modified request with models array converted to protocol_manager_config.models

        Raises:
            ValueError: If any models in the array are invalid
        """
        # If no models array, return request unchanged
        if not request.models:
            return request

        # Validate all models in the array
        valid_models, invalid_models = model_registry.validate_models(request.models)

        # If any models are invalid, raise error
        if invalid_models:
            raise ValueError(f"Invalid model(s): {invalid_models}")

        # Convert valid model names to capabilities
        capabilities, _ = model_registry.convert_names_to_capabilities(valid_models)

        # Create or update protocol_manager_config
        if request.protocol_manager_config:
            # Update existing config
            request.protocol_manager_config.models = capabilities
        else:
            # Create new config
            from adaptive_ai.models.llm_core_models import ProtocolManagerConfig

            request.protocol_manager_config = ProtocolManagerConfig(models=capabilities)

        # Clear the models array since it's now in protocol_manager_config
        request.models = None

        return request

    def predict(
        self, requests: list[ModelSelectionRequest]
    ) -> list[OrchestratorResponse]:
        import time

        # Process model validation and conversion for each request
        processed_requests = []
        for req in requests:
            processed_req = self._process_models_array(req)
            processed_requests.append(processed_req)

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

    # Create LitServer
    server = ls.LitServer(
        api,
        accelerator=settings.litserve.accelerator,
        devices=settings.litserve.devices,
        loggers=loggers,
        callbacks=callbacks,
    )

    # Add custom validation endpoint
    @server.app.post("/validate-models", response_model=ModelValidationResponse)
    async def validate_models(
        request: ModelValidationRequest,
    ) -> ModelValidationResponse:
        """Validate a list of model names and return which are valid/invalid."""
        try:
            valid_models, invalid_models = model_registry.validate_models(
                request.models
            )
            return ModelValidationResponse(
                valid_models=valid_models, invalid_models=invalid_models
            )
        except Exception:
            # Return error in response format that Go middleware expects
            return ModelValidationResponse(
                valid_models=[],
                invalid_models=request.models,
            )

    # Add model conversion endpoint
    @server.app.post("/convert-model-names", response_model=ModelConversionResponse)
    async def convert_model_names(
        request: ModelConversionRequest,
    ) -> ModelConversionResponse:
        """Convert model names to full ModelCapability objects."""
        try:
            valid_capabilities, invalid_names = (
                model_registry.convert_names_to_capabilities(request.model_names)
            )
            # Convert ModelCapability objects to dicts for JSON serialization
            capability_dicts = [cap.model_dump() for cap in valid_capabilities]
            return ModelConversionResponse(
                model_capabilities=capability_dicts, invalid_models=invalid_names
            )
        except Exception as e:
            fiberlog.error(f"Model conversion failed: {e}")
            # Return error in response format that Go middleware expects
            return ModelConversionResponse(
                model_capabilities=[],
                invalid_models=request.model_names,
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
