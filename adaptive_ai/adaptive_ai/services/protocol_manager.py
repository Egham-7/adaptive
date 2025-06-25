from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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


# Pydantic schema for structured output
class ProtocolSelectionOutput(BaseModel):
    protocol: str = Field(
        description="The protocol to use: standard_llm, minion, or minions_protocol"
    )
    provider: str = Field(
        description="The provider to use (e.g., OpenAI, DeepSeek, etc.)"
    )
    model: str = Field(description="The model name to use")
    confidence: float = Field(description="Confidence score for the selection")
    explanation: str = Field(description="Explanation for the protocol selection")
    # OpenAIParameters fields
    temperature: float = Field(
        description="Controls randomness: higher values mean more diverse completions. Range 0.0-2.0."
    )
    top_p: float = Field(
        description="Nucleus sampling: only considers tokens whose cumulative probability exceeds top_p. Range 0.0-1.0."
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
        description="Penalize new tokens based on their existing frequency in the text so far. Range -2.0 to 2.0."
    )
    presence_penalty: float = Field(
        description="Penalize new tokens based on whether they appear in the text so far. Range -2.0 to 2.0."
    )
    # Alternatives for standard_llm
    standard_alternatives: list[dict[str, str]] | None = Field(
        default=None,
        description="Alternative models for standard_llm. Each should have provider and model.",
    )
    # Alternatives for minion
    minion_alternatives: list[dict[str, str]] | None = Field(
        default=None,
        description="Alternative minion task types. Each should have task_type.",
    )


class ProtocolManager:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_new_tokens: int = 256,
    ):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
            )
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model {
                               model_name}: {e}"
            ) from e
        self.parser = PydanticOutputParser(pydantic_object=ProtocolSelectionOutput)
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
            "- temperature: Controls randomness. Higher values mean more diverse completions. Range 0.0-2.0.\n"
            "- top_p: Nucleus sampling. Only considers tokens whose cumulative probability exceeds top_p. Range 0.0-1.0.\n"
            "- max_tokens: The maximum number of tokens to generate in the completion.\n"
            "- n: How many chat completion choices to generate for each input message.\n"
            "- stop: Sequences where the API will stop generating further tokens.\n"
            "- frequency_penalty: Penalize new tokens based on their existing frequency in the text so far. Range -2.0 to 2.0.\n"
            "- presence_penalty: Penalize new tokens based on whether they appear in the text so far. Range -2.0 to 2.0.\n"
        )
        self.prompt = PromptTemplate(
            template=(
                "You are a protocol selection expert. Given a user prompt, task type, and candidate models, "
                "choose the best protocol and model.\n"
                "{protocol_descriptions}\n"
                "{parameter_descriptions}\n"
                "For standard_llm, return alternatives as a list of objects with provider and model.\n"
                "For minion, return alternatives as a list of objects with task_type.\n"
                "For minions_protocol, you may include both standard_llm and minion info.\n"
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
                "parameter_descriptions",
            ],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def _format_model_capabilities(
        self, candidate_models: list[ModelCapability]
    ) -> str:
        lines = []
        for m in candidate_models:
            lines.append(
                f"- Provider: {m.provider.value}, Model: {m.model_name}, "
                f"Cost/1M input tokens: {m.cost_per_1m_input_tokens}, "
                f"Cost/1M output tokens: {m.cost_per_1m_output_tokens}, "
                f"Max context: {m.max_context_tokens}, Max output: {
                    m.max_output_tokens
                }, "
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
            else "Other"
        )
        try:
            chain = self.prompt | self.llm | self.parser
            result: ProtocolSelectionOutput = chain.invoke(
                {
                    "prompt": prompt,
                    "task_type": task_type,
                    "model_capabilities": model_capabilities,
                    "protocol_descriptions": self.protocol_descriptions,
                    "parameter_descriptions": self.parameter_descriptions,
                }
            )
        except Exception as e:
            raise RuntimeError(f"Protocol selection failed: {e}") from e
        protocol = (
            ProtocolType(result.protocol)
            if result.protocol in ProtocolType._value2member_map_
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
        minion_alts = None
        if result.standard_alternatives:
            try:
                standard_alts = [
                    Alternative(**alt) for alt in result.standard_alternatives
                ]
            except (TypeError, ValueError):
                # Log the error and continue without alternatives
                standard_alts = None

            standard_alts = [Alternative(**alt) for alt in result.standard_alternatives]
        minion_alts = None
        if result.minion_alternatives:
            try:
                minion_alts = [
                    MinionAlternative(**alt) for alt in result.minion_alternatives
                ]
            except (TypeError, ValueError):
                # Log the error and continue without alternatives
                minion_alts = None
        # Handle each protocol type
        if protocol == ProtocolType.STANDARD_LLM:
            standard = StandardLLMInfo(
                provider=result.provider,
                model=result.model,
                confidence=result.confidence,
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
            # For minions_protocol, expect both standard and minion info in the output
            standard = StandardLLMInfo(
                provider=result.provider,
                model=result.model,
                confidence=result.confidence,
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
        # For future extensibility: handle additional protocols here
        else:
            return OrchestratorResponse(protocol=protocol)
