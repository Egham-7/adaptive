from typing import Any, Protocol

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelEntry,
    ModelSelectionRequest,
)
from adaptive_ai.models.llm_enums import ProtocolType
from adaptive_ai.models.llm_orchestration_models import (
    Alternative,
    MinionInfo,
    OpenAIParameters,
    OrchestratorResponse,
    StandardLLMInfo,
)


class LitLoggerProtocol(Protocol):
    def log(self, key: str, value: Any) -> None: ...


class ProtocolManager:
    def __init__(
        self,
        lit_logger: LitLoggerProtocol | None = None,
        device: (
            str | None
        ) = None,  # Accept but ignore device parameter for compatibility
    ) -> None:
        self.lit_logger: LitLoggerProtocol | None = lit_logger
        self.log(
            "protocol_manager_init",
            {"rule_based": True, "device_ignored": device},
        )

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def _convert_model_entries_to_alternatives(
        self, model_entries: list[ModelEntry]
    ) -> list[Alternative]:
        """Convert ModelEntry objects to Alternative objects with all providers."""
        alternatives = []
        for entry in model_entries:
            for provider in entry.providers:
                alternatives.append(
                    Alternative(provider=provider.value,
                                model=entry.model_name)
                )
        return alternatives

    def select_protocol(
        self,
        standard_candidates: list[ModelEntry],
        minion_candidates: list[ModelEntry],
        classification_result: ClassificationResult,
        token_count: int = 0,
        request: ModelSelectionRequest | None = None,
    ) -> OrchestratorResponse:
        task_type = (
            classification_result.task_type_1[0]
            if classification_result.task_type_1
            else "Other"
        )

        # Extract decision factors from classification result

        complexity_score = (
            classification_result.prompt_complexity_score[0]
            if classification_result.prompt_complexity_score
            else 0.0
        )
        reasoning = (
            classification_result.reasoning[0]
            if classification_result.reasoning
            else 0.0
        )
        number_of_few_shots = (
            classification_result.number_of_few_shots[0]
            if classification_result.number_of_few_shots
            else 0.0
        )

        should_use_standard = (
            complexity_score > 0.40
            or token_count > 3000
            or number_of_few_shots > 4
            or reasoning > 0.55
        )
        protocol_choice = "standard_llm" if should_use_standard else "minion"

        self.log(
            "rule_based_protocol_selection",
            {
                "task_type": task_type,
                "protocol_choice": protocol_choice,
                "complexity_score": complexity_score,
                "token_count": token_count,
                "number_of_few_shots": number_of_few_shots,
                "reasoning": reasoning,
                "decision_factors": {
                    "high_complexity": complexity_score > 0.40,
                    "long_input": token_count > 3000,
                    "many_few_shots": number_of_few_shots > 4,
                    "high_reasoning": reasoning > 0.55,
                },
            },
        )

        # Create OpenAI parameters
        parameters = OpenAIParameters(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            n=1,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        # Create protocol response directly
        if should_use_standard and standard_candidates:
            first_standard = standard_candidates[0]
            primary_provider = first_standard.providers[0].value

            standard = StandardLLMInfo(
                provider=primary_provider,
                model=first_standard.model_name,
                parameters=parameters,
                alternatives=self._convert_model_entries_to_alternatives(
                    standard_candidates[1:]
                ),
            )
            return OrchestratorResponse(
                protocol=ProtocolType.STANDARD_LLM, standard=standard
            )

        elif minion_candidates:
            first_minion = minion_candidates[0]
            primary_provider = first_minion.providers[0].value

            minion = MinionInfo(
                provider=primary_provider,
                model=first_minion.model_name,
                parameters=parameters,
                alternatives=self._convert_model_entries_to_alternatives(
                    minion_candidates[1:]
                ),
            )
            return OrchestratorResponse(protocol=ProtocolType.MINION, minion=minion)

        else:
            raise ValueError("No candidates available for either protocol")
