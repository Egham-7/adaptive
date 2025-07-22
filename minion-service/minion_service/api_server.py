import litserve as ls  # type:ignore
from minion_service.model_manager import ModelManager
import time
from typing import Any, Dict, Generator, List
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion


class LitGPTOpenAIAPI(ls.LitAPI):

    def setup(self, device: str) -> None:
        supported_models = [
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            "Qwen/Qwen2.5-14B-Instruct",  # BUSINESS_AND_INDUSTRIAL/HEALTH
            "Qwen/Qwen2.5-7B-Instruct",  # NEWS / OTHERDOMAINS / REAL_ESTATE
            "codellama/CodeLlama-7b-Instruct-hf",  # COMPUTERS_AND_ELECTRONICS/INTERNET_AND_TELECOM
            "Qwen/Qwen2.5-Math-7B-Instruct",  # FINANCE/SCIENCE
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # JOBS_AND_EDUCATION
            "microsoft/Phi-4-mini-reasoning",  # LAW_AND_GOVERNMENT
            "meta-llama/Meta-Llama-3-8B-Instruct",  # SENSITIVE_SUBJECTS
        ]

        # Auto-unload models after 30 minutes of inactivity with memory management
        self.model_manager = ModelManager(
            preload_models=supported_models,
            inactivity_timeout_minutes=30,
            memory_threshold_percent=85.0,
            memory_reserve_gb=2.0,
        )
        self.model_manager.set_logger_callback(lambda key, value: self.log(key, value))

    def batch(self, inputs: List[ChatCompletion]) -> List[ChatCompletion]:
        """Batch multiple chat requests together."""
        return inputs

    def predict(
        self, prompt: List[ChatCompletionMessageParam], context: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Process chat completion request with batching support.

        Args:
            prompt: List of message dictionaries from ChatCompletionRequest
            context: Request context with OpenAI parameters injected automatically
        """
        start_time = time.perf_counter()

        # Debug: Log the types being received
        print(f"DEBUG: prompt type: {type(prompt)}, context type: {type(context)}")
        print(f"DEBUG: prompt: {prompt}")
        print(f"DEBUG: context: {context}")

        # Handle case where parameters might be swapped
        if isinstance(context, list) and isinstance(prompt, dict):
            # Parameters are swapped - fix them
            prompt, context = context, prompt
            print("DEBUG: Swapped parameters")

        # OpenAI spec automatically injects request parameters into context
        model_name = context.get("model", "") if isinstance(context, dict) else ""

        if not model_name:
            raise ValueError("Model name is required")

        model_load_start = time.perf_counter()
        llm = self.model_manager.get_model(model_name)
        model_load_time = time.perf_counter() - model_load_start
        self.log("model_load_time", model_load_time)

        self.log("request_model", model_name)
        self.log("request_temperature", context.get("temperature", 0.7))
        self.log("request_max_tokens", context.get("max_tokens", 512))

        # Generate response using LitGPT
        inference_start = time.perf_counter()
        generated_text = llm.generate(
            prompt,
            max_new_tokens=context.get("max_tokens", 512),
            temperature=context.get("temperature", 0.7),
            top_p=context.get("top_p", 1.0),
        )
        inference_time = time.perf_counter() - inference_start
        self.log("inference_time", inference_time)

        total_tokens = 0
        if generated_text:
            total_tokens = len(generated_text.split())
            self.log("generated_tokens", total_tokens)

            # Yield tokens for streaming
            yield from (word + " " for word in generated_text.split())
        else:
            yield "I apologize, but I couldn't generate a response."

        total_time = time.perf_counter() - start_time
        self.log("total_request_time", total_time)
        if total_tokens > 0:
            self.log("tokens_per_second", total_tokens / inference_time)

    def unbatch(self, output: Any) -> Any:
        """Unbatch the results for individual responses."""
        return output
