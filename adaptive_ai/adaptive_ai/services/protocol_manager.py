from typing import Any, Protocol

from pydantic import BaseModel, Field

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelCapability,
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


class ProtocolSelectionOutput(BaseModel):
    protocol: str = Field(description="The protocol to use: standard_llm or minion")
    provider: str = Field(
        description="The provider to use (e.g., OpenAI, DeepSeek, Groq, etc.)"
    )
    model: str = Field(description="The model name to use")
    explanation: str = Field(description="Explanation for the protocol selection")
    temperature: float = Field(
        description="Controls randomness: higher values mean more diverse completions. Range 0.0-2.0."
    )
    top_p: float = Field(
        description="Nucleus sampling: only considers tokens whose cumulative probability "
        "exceeds top_p. Range 0.0-1.0."
    )
    max_tokens: int | None = Field(
        description="The maximum number of tokens to generate in the completion."
    )
    n: int = Field(
        description="How many chat completion choices to generate for each input message."
    )
    stop: str | None = Field(
        description="Sequences where the API will stop generating further tokens."
    )
    frequency_penalty: float = Field(
        description="Penalize new tokens based on their existing frequency in the text "
        "so far. Range -2.0 to 2.0."
    )
    presence_penalty: float = Field(
        description="Penalize new tokens based on whether they appear in the text so "
        "far. Range -2.0 to 2.0."
    )
    standard_alternatives: list[Alternative] = Field(
        default=[],
        description="Alternative models for standard_llm. Each should have provider "
        "and model.",
    )
    minion_alternatives: list[Alternative] = Field(
        default=[],
        description="Alternative models for minion protocol.",
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

    def _format_model_capabilities(
        self, candidate_models: list[ModelCapability]
    ) -> str:
        lines = []
        for i, m in enumerate(candidate_models):
            rank_info = f"  (Rank {i + 1})" if len(candidate_models) > 1 else ""
            lines.append(
                f"- Provider: {m.provider.value}, Model: {m.model_name}{rank_info}"
            )
        return "\n".join(lines)

    def _convert_model_entries_to_alternatives(
        self, model_entries: list[ModelEntry]
    ) -> list[Alternative]:
        """Convert ModelEntry objects to Alternative objects with all providers."""
        alternatives = []
        for entry in model_entries:
            for provider in entry.providers:
                alternatives.append(
                    Alternative(provider=provider.value, model=entry.model_name)
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
        # Check if the REQUEST has tools/functions (not if models support it)
        request_has_tools = False
        if request:
            request_has_tools = bool(request.tools or request.functions)
            # Debug logging
            if self.lit_logger:
                self.lit_logger.log(
                    "debug_tools_check",
                    {
                        "has_tools": bool(request.tools),
                        "has_functions": bool(request.functions),
                        "tools_value": (
                            str(request.tools)[:200] if request.tools else None
                        ),
                        "request_has_tools": request_has_tools,
                    },
                )
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
            request_has_tools
            or complexity_score > 0.40
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
                "request_has_tools": request_has_tools,
                "complexity_score": complexity_score,
                "token_count": token_count,
                "number_of_few_shots": number_of_few_shots,
                "reasoning": reasoning,
                "decision_factors": {
                    "request_has_tools": request_has_tools,
                    "high_complexity": complexity_score > 0.40,
                    "long_input": token_count > 3000,
                    "many_few_shots": number_of_few_shots > 4,
                    "high_reasoning": reasoning > 0.55,
                },
            },
        )

        # Create protocol selection output
        if should_use_standard and standard_candidates:
            first_standard = standard_candidates[0]
            primary_provider = first_standard.providers[0].value
            result = ProtocolSelectionOutput(
                protocol=protocol_choice,
                provider=primary_provider,
                model=first_standard.model_name,
                explanation=f"Rule-based selection: {protocol_choice} due to complexity/requirements",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                n=1,
                stop=None,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                standard_alternatives=self._convert_model_entries_to_alternatives(
                    standard_candidates[1:]
                ),
                minion_alternatives=self._convert_model_entries_to_alternatives(
                    minion_candidates
                ),
            )
        elif minion_candidates:
            first_minion = minion_candidates[0]
            primary_provider = first_minion.providers[0].value
            result = ProtocolSelectionOutput(
                protocol=protocol_choice,
                provider=primary_provider,
                model=first_minion.model_name,
                explanation=f"Rule-based selection: {protocol_choice} for efficiency",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                n=1,
                stop=None,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                standard_alternatives=self._convert_model_entries_to_alternatives(
                    standard_candidates
                ),
                minion_alternatives=self._convert_model_entries_to_alternatives(
                    minion_candidates[1:]
                ),
            )
        else:
            raise ValueError("No candidates available for either protocol")

        protocol_str = result.protocol
        protocol = (
            ProtocolType(protocol_str)
            if protocol_str and protocol_str in ProtocolType.__members__.values()
            else ProtocolType.STANDARD_LLM
        )
        parameters = OpenAIParameters(
            temperature=result.temperature,
            top_p=result.top_p,
            max_tokens=result.max_tokens,
            n=result.n,
            stop=result.stop,
            frequency_penalty=result.frequency_penalty,
            presence_penalty=result.presence_penalty,
        )

        standard_alts = result.standard_alternatives
        minion_alts = result.minion_alternatives

        match protocol:
            case ProtocolType.STANDARD_LLM:
                standard = StandardLLMInfo(
                    provider=result.provider,
                    model=result.model,
                    parameters=parameters,
                    alternatives=standard_alts,
                )
                return OrchestratorResponse(protocol=protocol, standard=standard)
            case ProtocolType.MINION:
                minion = MinionInfo(
                    provider=result.provider,
                    model=result.model,
                    parameters=parameters,
                    alternatives=minion_alts,
                )
                return OrchestratorResponse(protocol=protocol, minion=minion)
            case _:
                return OrchestratorResponse(protocol=protocol)
