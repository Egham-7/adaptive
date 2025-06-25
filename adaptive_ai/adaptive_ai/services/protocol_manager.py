from typing import Any, Protocol

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import (
    ProtocolType,
    ProviderType,
    TaskType,
)
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
        description="The provider to use (e.g., OpenAI, DeepSeek, etc.)"
    )
    model: str = Field(description="The model name to use")


class LitLoggerProtocol(Protocol):
    def log(self, key: str, value: Any) -> None: ...


class ProtocolManager:
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        max_new_tokens: int = 256,
        lit_logger: LitLoggerProtocol | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        self.parser = PydanticOutputParser(pydantic_object=ProtocolSelectionOutput)
        self.protocol_descriptions = (
            "Protocols:\n"
            "1. standard_llm: Use a single large language model for the task.\n"
            "   - Advantages: Simplicity, direct response, no orchestration overhead.\n"
            "   - Disadvantages: May not be optimal for complex or multi-step tasks.\n"
            "2. minion: Use a specialized smaller model (minion) for a specific subtask.\n"
            "   - Advantages: Efficiency, can be faster and cheaper for narrow tasks.\n"
            "   - Disadvantages: Limited scope, may not handle general or complex queries.\n"
            "3. minions_protocol: Orchestrate multiple minion models to solve a complex "
            "task.\n"
            "   - Advantages: Can break down and solve complex, multi-step, or "
            "multi-domain problems.\n"
            "   - Disadvantages: More orchestration overhead, may be slower or more "
            "expensive.\n"
            "   - Payload: May include both standard_llm and minion info.\n"
        )
        self.prompt = PromptTemplate(
            template=(
                "You are a protocol selection expert. Given a user prompt, task type, "
                "and candidate models, choose the best protocol and model.\n"
                "{protocol_descriptions}\n"
                "{format_instructions}\n"
                "Prompt: {prompt}\n"
                "Task type: {task_type}\n"
                "Candidate models (with capabilities):\n{model_capabilities}\n"
            ),
            input_variables=[
                "prompt",
                "task_type",
                "model_capabilities",
                "protocol_descriptions",
            ],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
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
                f"{m.max_output_tokens}, "
                f"Function calling: {m.supports_function_calling}, "
                f"Languages: {', '.join(m.languages_supported)}, "
                f"Size: {m.model_size_params}, Latency: {m.latency_tier}"
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
            else TaskType.OTHER
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
            chain = self.prompt | self.llm | self.parser
            result: ProtocolSelectionOutput = chain.invoke(
                {
                    "prompt": prompt,
                    "task_type": task_type,
                    "model_capabilities": model_capabilities,
                    "protocol_descriptions": self.protocol_descriptions,
                }
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

        protocol = (
            ProtocolType(result.protocol)
            if result.protocol in ProtocolType.__members__.values()
            else ProtocolType.STANDARD_LLM
        )

        parameters = OpenAIParameters(
            temperature=0.7,
            top_p=1.0,
            max_tokens=256,
            n=1,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        standard_alts = []
        minion_alts = []

        for model_cap in candidate_models:
            if (
                model_cap.provider.value != result.provider
                or model_cap.model_name != result.model
            ):
                if task_type.value in model_cap.languages_supported:
                    standard_alts.append(
                        Alternative(
                            provider=model_cap.provider.value,
                            model=model_cap.model_name,
                        )
                    )

        for task in TaskType:
            if task != task_type:
                minion_alts.append(MinionAlternative(task_type=task.value))

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
