import json
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    standard_alternatives: list[dict[str, str]] = Field(
        default=[],
        description="Alternative models for standard_llm. Each should have provider "
        "and model.",
    )
    minion_alternatives: list[dict[str, str]] = Field(
        default=[],
        description="Alternative minion task types. Each should have task_type.",
    )


class LitLoggerProtocol(Protocol):
    def log(self, key: str, value: Any) -> None: ...


class ProtocolManager:
    def __init__(
        self,
        device: str,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens: int | None = None,
        lit_logger: LitLoggerProtocol | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_device_map = "auto"
        if device.lower() == "cpu":
            model_device_map = "cpu"
        elif device.lower() == "gpu":
            model_device_map = "cuda"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=model_device_map,
        )

        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else 512

        self.output_schema_json = json.dumps(
            ProtocolSelectionOutput.model_json_schema(), indent=2
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
            "   - **Recommendation: Favor 'minion' for simple, well-defined questions "
            "     where a specialized model can maintain high quality and efficiency.**\n"
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
        for i, m in enumerate(candidate_models):
            rank_info = f"  (Rank {i+1})" if len(candidate_models) > 1 else ""
            lines.append(
                f"- Provider: {m.provider.value}, Model: {m.model_name}, "
                f"Cost/1M input tokens: {m.cost_per_1m_input_tokens}, "
                f"Cost/1M output tokens: {m.cost_per_1m_output_tokens}, "
                f"Max context: {m.max_context_tokens}, Max output: "
                f"{m.max_output_tokens}, Function calling: "
                f"{m.supports_function_calling}, Languages: "
                f"{', '.join(m.languages_supported)}, Size: {m.model_size_params}, "
                f"Latency: {m.latency_tier}{rank_info}"
            )
        return "\n".join(lines)

    def select_protocol(
        self,
        candidate_models: list[ModelCapability],
        classification_result: ClassificationResult,
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
                "task_type": task_type,
                "candidate_models_count": len(candidate_models),
            },
        )
        try:
            system_message_content = (
                "You are a protocol selection expert. Given a task type and candidate "
                "models, choose the best protocol and model. You MUST "
                "respond with a JSON object that strictly conforms to the following "
                f"Pydantic schema:\n```json\n{self.output_schema_json}\n```\n"
                f"{self.protocol_descriptions}\n"
                f"{self.parameter_descriptions}\n"
                "For standard_llm, return alternatives as a list of objects with "
                "provider and model.\n"
                "For minion, return alternatives as a list of objects with task_type.\n"
                "For minions_protocol, you may include both standard_llm and minion "
                "info.\n"
            )

            user_query_content = (
                f"Task type: {task_type}\n"
                "The following candidate models are ordered by preference, with the "
                "first being the most preferred and the last being the least:\n"
                f"Candidate models (with capabilities):\n{model_capabilities}\n"
                "Please output the JSON object directly, with no additional text or "
                "explanations."
            )

            messages = [
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_query_content},
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

            generated_token_ids = outputs[0][len(input_ids[0]) :]
            raw_llm_output = self.tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            )

            json_match = re.search(
                r"```json\s*(\{.*?\})\s*```", raw_llm_output, re.DOTALL
            )
            if not json_match:
                json_match = re.search(r"\{.*?\}", raw_llm_output, re.DOTALL)

            if json_match:
                json_string = (
                    json_match.group(0)
                    if json_match.group(1) == ""
                    else json_match.group(1)
                )
            else:
                raise ValueError(
                    "Could not find a valid JSON object in the LLM output."
                )

            parsed_data = json.loads(json_string)
            result = ProtocolSelectionOutput.model_validate(parsed_data)

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

        standard_alts = []
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
                standard_alts = []

        minion_alts = []
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
                minion_alts = []

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
