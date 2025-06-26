from typing import Any, Protocol, cast

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import ProtocolType
from adaptive_ai.models.llm_orchestration_models import (
    Alternative,
    MinionAlternative,
    MinionInfo,
    OpenAIParameters,
    OrchestratorResponse,
    StandardLLMInfo,
)


class ProtocolSelectionOutput(BaseModel):
    protocol: str = Field(
        description="The protocol to use: standard_llm, minion, or minions_protocol"
    )
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
    standard_alternatives: list[dict[str, str]] | None = Field(
        default=None,
        description="Alternative models for standard_llm. Each should have provider "
        "and model.",
    )
    minion_alternatives: list[dict[str, str]] | None = Field(
        default=None,
        description="Alternative minion task types. Each should have task_type.",
    )


class LitLoggerProtocol(Protocol):
    def log(self, key: str, value: Any) -> None: ...


class ProtocolManager:
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        max_new_tokens: int | None = None,
        lit_logger: LitLoggerProtocol | None = None,
    ) -> None:

        self.base_llm = ChatGroq(
            model=model_name,
            temperature=0.7,
            max_tokens=max_new_tokens,
        )

        self.llm = self.base_llm.with_structured_output(
            schema=ProtocolSelectionOutput,
        )

        self.protocol_descriptions = (
            "Protocols:\n"
            "1. standard_llm: Use a single large language model for the task.\n"
            "   - Advantages: Simplicity, direct response, no orchestration overhead.\n"
            "   - Disadvantages: May not be optimal for complex or multi-step tasks.\n"
            "   - Alternatives: List of alternative models (provider, model).\n"
            "2. minion: Use a specialized smaller model (minion) for a specific subtask.\n"
            "   - Advantages: Efficiency, can be faster and cheaper for narrow tasks.\n"
            "   - Disadvantages: Limited scope, may not handle general or complex queries.\n"
            "   - Alternatives: List of alternative minion task types (task_type).\n"
            "3. minions_protocol: Orchestrate multiple minion models to solve a complex task.\n"
            "   - Advantages: Can break down and solve complex, multi-step, or multi-domain problems.\n"
            "   - Disadvantages: More orchestration overhead, may be slower or more expensive.\n"
            "   - Payload: May include both standard_llm and minion info.\n"
        )
        self.parameter_descriptions = (
            "Parameters to generate for the selected model (explain and set each):\n"
            "- temperature: Controls randomness. Higher values mean more diverse "
            "completions. Range 0.0-2.0.\n"
            "- top_p: Nucleus sampling. Only considers tokens whose cumulative "
            "probability exceeds top_p. Range 0.0-1.0.\n"
            "- max_tokens: The maximum number of tokens to generate in the completion.\n"
            "- n: How many chat completion choices to generate for each input message.\n"
            "- stop: Sequences where the API will stop generating further tokens.\n"
            "- frequency_penalty: Penalize new tokens based on their existing frequency "
            "in the text so far. Range -2.0 to 2.0.\n"
            "- presence_penalty: Penalize new tokens based on whether they appear in the "
            "text so far. Range -2.0 to 2.0.\n"
        )
        self.prompt = PromptTemplate(
            template=(
                "You are a protocol selection expert. Given a user prompt, task type, "
                "and candidate models, choose the best protocol and model. You MUST "
                "respond with the exact JSON format specified.\n"
                "{protocol_descriptions}\n"
                "{parameter_descriptions}\n"
                "For standard_llm, return alternatives as a list of objects with "
                "provider and model.\n"
                "For minion, return alternatives as a list of objects with task_type.\n"
                "For minions_protocol, you may include both standard_llm and minion "
                "info.\n"
                "Prompt: {prompt}\n"
                "Task type: {task_type}\n"
                "Candidate models (with capabilities):\n{model_capabilities}\n"
            ),
            input_variables=[
                "prompt",
                "task_type",
                "model_capabilities",
                "protocol_descriptions",
                "parameter_descriptions",
            ],
        )
        self.lit_logger: LitLoggerProtocol | None = lit_logger
        self.log(
            "protocol_manager_init",
            {"model_name": model_name, "max_new_tokens": max_new_tokens},
        )

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def _format_model_capabilities(
        self, candidate_models: list[ModelCapability]
    ) -> str:
        lines = []
        for m in candidate_models:
            lines.append(
                f"- Provider: {m.provider.value}, Model: {m.model_name}, "
                f"Cost/1M input tokens: {m.cost_per_1m_input_tokens}, "
                f"Cost/1M output tokens: {m.cost_per_1m_output_tokens}, "
                f"Max context: {m.max_context_tokens}, Max output: "
                f"{m.max_output_tokens}, Function calling: "
                f"{m.supports_function_calling}, Languages: "
                f"{', '.join(m.languages_supported)}, Size: {m.model_size_params}, "
                f"Latency: {m.latency_tier}"
            )
        return "\n".join(lines)

    def select_protocol(
        self,
        candidate_models: list[ModelCapability],
        classification_result: ClassificationResult,
        prompt: str,
    ) -> OrchestratorResponse:
        model_capabilities = self._format_model_capabilities(candidate_models)
        task_type = (
            classification_result.task_type_1[0]
            if classification_result.task_type_1
            else "Other"
        )
        self.log(
            "select_protocol_called",
            {
                "prompt": prompt,
                "task_type": task_type,
                "candidate_models_count": len(candidate_models),
            },
        )
        try:
            chain = self.prompt | self.llm

            result = cast(
                ProtocolSelectionOutput,
                chain.invoke(
                    {
                        "prompt": prompt,
                        "task_type": task_type,
                        "model_capabilities": model_capabilities,
                        "protocol_descriptions": self.protocol_descriptions,
                        "parameter_descriptions": self.parameter_descriptions,
                    }
                ),
            )
            self.log(
                "protocol_selection_success",
                {
                    "protocol": result.protocol,
                    "provider": result.provider,
                    "model": result.model,
                },
            )
        except Exception as e:
            self.log("protocol_selection_error", {"error": str(e)})
            raise RuntimeError(f"Protocol selection failed: {e}") from e

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

        standard_alts = None
        if result.standard_alternatives:
            try:
                standard_alts = [
                    Alternative(**alt) for alt in result.standard_alternatives
                ]
            except (TypeError, ValueError) as e:
                self.log(
                    "standard_alternatives_parsing_error",
                    {"error": str(e), "data": result.standard_alternatives},
                )
                standard_alts = None

        minion_alts = None
        if result.minion_alternatives:
            try:
                minion_alts = [
                    MinionAlternative(**alt) for alt in result.minion_alternatives
                ]
            except (TypeError, ValueError) as e:
                self.log(
                    "minion_alternatives_parsing_error",
                    {"error": str(e), "data": result.minion_alternatives},
                )
                minion_alts = None

        if protocol == ProtocolType.STANDARD_LLM:
            standard = StandardLLMInfo(
                provider=result.provider,
                model=result.model,
                parameters=parameters,
                alternatives=standard_alts,
            )
            return OrchestratorResponse(protocol=protocol, standard=standard)
        elif protocol == ProtocolType.MINION:
            minion = MinionInfo(
                task_type=task_type,
                parameters=parameters,
                alternatives=minion_alts,
            )
            return OrchestratorResponse(protocol=protocol, minion=minion)
        elif protocol == ProtocolType.MINIONS_PROTOCOL:
            standard = StandardLLMInfo(
                provider=result.provider,
                model=result.model,
                parameters=parameters,
                alternatives=standard_alts,
            )
            minion = MinionInfo(
                task_type=task_type,
                parameters=parameters,
                alternatives=minion_alts,
            )
            return OrchestratorResponse(
                protocol=protocol, standard=standard, minion=minion
            )
        else:
            return OrchestratorResponse(protocol=protocol)
