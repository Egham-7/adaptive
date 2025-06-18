from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
import math
import os
from typing import TypedDict, cast

import tiktoken

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.orchestrator import (
    MinionDetails,
    MinionOrchestratorResponse,
    MinionsProtocolDetails,
    MinionsProtocolOrchestratorResponse,
    OrchestratorResponse,
    RemoteLLM,
    StandardLLMDetails,
    StandardLLMOrchestratorResponse,
)
from adaptive_ai.models.parameters import OpenAIParameters
from adaptive_ai.models.requests import ModelSelectionResponse
from adaptive_ai.models.types import (
    VALID_PROVIDERS,
    VALID_TASK_TYPES,
    DifficultyLevel,
    ModelCapability,
    ModelSelectionError,
    TaskDifficultyConfig,
    TaskModelMapping,
    TaskType,
)
from adaptive_ai.services.prompt_classifier import PromptClassifier

logger = logging.getLogger(__name__)


class ModelEconomics(TypedDict, total=False):
    cost: float
    threshold: float


class ModelSelector:
    """
    Service for selecting appropriate LLM models and parameters based on prompt analysis.
    Uses task-specific complexity-based model selection with parallel processing.
    """

    def __init__(
        self, prompt_classifier: PromptClassifier, max_workers: int | None = None
    ) -> None:
        self.prompt_classifier: PromptClassifier = prompt_classifier
        self._settings = get_settings()
        self._model_capabilities = self._get_model_capabilities()
        self._task_mappings = self._get_task_model_mappings()
        self._remote_llm_map = self._get_remote_llm_map()
        cpu_cnt = os.cpu_count() or 1
        self.max_workers = max_workers or min(32, cpu_cnt)

    def _get_model_capabilities(self) -> dict[str, ModelCapability]:
        """Get all model capabilities from config"""
        capabilities = self._settings.get_model_capabilities()
        validated_capabilities: dict[str, ModelCapability] = {}

        for model_name, capability in capabilities.items():
            validated_capabilities[model_name] = cast(
                ModelCapability,
                {
                    "description": str(capability.get("description", "")),
                    "provider": str(capability.get("provider", "Unknown")),
                    "cost_per_1k_tokens": float(
                        capability.get("cost_per_1k_tokens", 0.0)
                    ),
                    "max_tokens": int(capability.get("max_tokens", 4096)),
                    "supports_streaming": bool(
                        capability.get("supports_streaming", False)
                    ),
                    "supports_function_calling": bool(
                        capability.get("supports_function_calling", False)
                    ),
                    "supports_vision": bool(capability.get("supports_vision", False)),
                },
            )
        return validated_capabilities

    def _get_task_model_mappings(self) -> dict[TaskType, TaskModelMapping]:
        """Get task to model mappings from config"""
        mappings = self._settings.get_task_model_mappings()
        validated_mappings: dict[TaskType, TaskModelMapping] = {}

        for task_type, mapping in mappings.items():
            if task_type in VALID_TASK_TYPES:
                validated_mappings[cast(TaskType, task_type)] = cast(
                    TaskModelMapping,
                    {
                        "easy": {
                            "model": str(mapping["easy"]["model"]),
                            "complexity_threshold": float(
                                mapping["easy"]["complexity_threshold"]
                            ),
                        },
                        "medium": {
                            "model": str(mapping["medium"]["model"]),
                            "complexity_threshold": float(
                                mapping["medium"]["complexity_threshold"]
                            ),
                        },
                        "hard": {
                            "model": str(mapping["hard"]["model"]),
                            "complexity_threshold": float(
                                mapping["hard"]["complexity_threshold"]
                            ),
                        },
                    },
                )
        return validated_mappings

    def _get_remote_llm_map(self) -> dict[str, dict[str, str]]:
        """Get the remote LLM map from config settings (config.yaml)."""
        remote_llm_map = getattr(self._settings, "remote_llm_map", None)
        # Ensure the config value is the correct type
        if isinstance(remote_llm_map, dict):
            valid_map: dict[str, dict[str, str]] = {}
            for k, v in remote_llm_map.items():
                if isinstance(v, dict) and "provider" in v and "model" in v:
                    provider = str(v["provider"])
                    model = str(v["model"])
                    valid_map[str(k)] = {"provider": provider, "model": model}
            if valid_map:
                return valid_map
        # Fallback: try to build from task_model_mappings if not present
        task_model_mappings = getattr(self._settings, "get_task_model_mappings", None)
        if callable(task_model_mappings):
            mappings = task_model_mappings()
            result: dict[str, dict[str, str]] = {}
            for task_type, mapping in mappings.items():
                model_name = mapping["hard"]["model"]
                model_cap = self._model_capabilities.get(model_name)
                if (
                    model_cap is not None
                    and "provider" in model_cap
                    and model_cap["provider"] in VALID_PROVIDERS
                ):
                    provider = model_cap["provider"]
                else:
                    provider = "Anthropic"
                result[str(task_type)] = {"provider": provider, "model": model_name}
            return result
        # If all else fails, return a default
        return {"Other": {"provider": "Anthropic", "model": "claude-sonnet-4-0"}}

    def _validate_task_type(self, task_type: str) -> TaskType:
        """Validate and convert string to TaskType"""

        if task_type in VALID_TASK_TYPES:
            return cast(TaskType, task_type)

        logger.warning(f"Unknown task type '{task_type}', defaulting to 'Other'")
        return "Other"

    def _validate_model_selection(self, selected_model: str, task_type: str) -> None:
        """Validate that the selected model exists and is appropriate for the task"""

        if selected_model not in self._model_capabilities:
            raise ModelSelectionError(
                f"Selected model {selected_model} not found in capabilities"
            )

        model_info: ModelCapability = self._model_capabilities[selected_model]
        logger.info(
            f"Selected model {selected_model} ({model_info['provider']}) for task {task_type}"
        )

    def _extract_prompt_scores(
        self, classification: dict[str, list[float]]
    ) -> dict[str, list[float]]:
        """Extract prompt scores in the format expected by OpenAIParameters"""
        return {
            "creativity_scope": classification.get("creativity_scope", [0.0]),
            "reasoning": classification.get("reasoning", [0.0]),
            "contextual_knowledge": classification.get("contextual_knowledge", [0.0]),
            "prompt_complexity_score": classification.get(
                "prompt_complexity_score", [0.0]
            ),
            "domain_knowledge": classification.get("domain_knowledge", [0.0]),
        }

    def _get_default_result(
        self,
        task_type: str,
        prompt_scores: dict[str, list[float]],
        model_name: str = "gpt-4",
    ) -> ModelSelectionResponse:
        """Create default model selection result when no mapping is found"""
        # Get OpenAI parameters
        openai_params = OpenAIParameters(model=model_name)
        openai_params.adjust_parameters(task_type, prompt_scores)

        return ModelSelectionResponse(
            selected_model="gpt-4",
            confidence=0.5,
            reasoning=f"Default model selection for task type: {task_type}",
            alternatives=["gpt-3.5-turbo", "gpt-4-turbo"],
            parameters=openai_params,
            provider="OpenAI",
        )

    def _adjust_difficulty_for_cost_bias(
        self,
        difficulty: DifficultyLevel,
        complexity_score: float,
        task_mapping: TaskModelMapping,
    ) -> DifficultyLevel:
        """Adjust difficulty with sigmoid-scaled cost bias."""
        cost_bias = self._settings.model_selection.cost_bias

        # Neutral bias shortcut
        if math.isclose(cost_bias, 0.5, abs_tol=0.01):
            return difficulty

        # Calculate sigmoid-scaled adjustment
        def sigmoid(x: float) -> float:
            return 1 / (1 + math.exp(-x))

        # Convert cost bias (-1 to 1) to sigmoid-scaled adjustment
        bias_strength = 2 * (cost_bias - 0.5)  # [-1, 1] range
        normalized_strength = (
            3 * bias_strength
        )  # More pronounced curve (adjust multiplier as needed)
        adjustment = (
            sigmoid(normalized_strength) - 0.5
        ) * 0.4  # Scales to [-0.2, 0.2] range

        # Apply non-linear adjustment
        adjusted_score = complexity_score + adjustment

        # Select difficulty using original threshold logic
        if adjusted_score <= task_mapping["easy"]["complexity_threshold"]:
            return "easy"
        elif adjusted_score >= task_mapping["hard"]["complexity_threshold"]:
            return "hard"
        return "medium"

    def _select_difficulty_level(
        self, complexity_score: float, task_mapping: TaskModelMapping
    ) -> DifficultyLevel:
        """Select difficulty level based on complexity score and thresholds"""
        easy_threshold = task_mapping["easy"]["complexity_threshold"]
        hard_threshold = task_mapping["hard"]["complexity_threshold"]

        # Initial difficulty selection
        initial_difficulty: DifficultyLevel
        if complexity_score <= easy_threshold:
            initial_difficulty = "easy"
        elif complexity_score >= hard_threshold:
            initial_difficulty = "hard"
        else:
            initial_difficulty = "medium"

        # Adjust difficulty based on cost bias
        return self._adjust_difficulty_for_cost_bias(
            initial_difficulty, complexity_score, task_mapping
        )

    def _calculate_match_score(
        self, complexity_score: float, selected_config: TaskDifficultyConfig
    ) -> float:
        """Calculate match score based on complexity score and threshold"""
        selected_threshold = selected_config["complexity_threshold"]
        return 1.0 - min(abs(complexity_score - selected_threshold), 1.0)

    def _get_alternatives(
        self, selected_model: str, task_mapping: TaskModelMapping
    ) -> list[str]:
        """Get alternative models from other difficulty levels"""
        alternatives = []
        # Use literal keys to access TypedDict
        for difficulty_key in ["easy", "medium", "hard"]:
            if difficulty_key == "easy":
                model = task_mapping["easy"]["model"]
            elif difficulty_key == "medium":
                model = task_mapping["medium"]["model"]
            else:  # difficulty_key == "hard"
                model = task_mapping["hard"]["model"]

            if model != selected_model and model not in alternatives:
                alternatives.append(model)
        return alternatives[:2]  # Limit to 2 alternatives

    def _get_openai_parameters(
        self, model_name: str, task_type: str, prompt_scores: dict[str, list[float]]
    ) -> OpenAIParameters:
        """Get OpenAI parameters object"""
        try:
            openai_params = OpenAIParameters(model=model_name)
            openai_params.adjust_parameters(task_type, prompt_scores)
            return openai_params
        except Exception as e:
            logger.warning(f"Failed to get OpenAI parameters: {e}, using defaults")
            # Return default OpenAI parameters if adjustment fails
            return OpenAIParameters(model=model_name)

    def _process_single_classification(
        self, classification: dict[str, list[float]], prompt_index: int = 0
    ) -> ModelSelectionResponse:
        """Process a single classification result"""
        try:
            # Get and validate task type from the classification results
            detected_task_type = (
                classification["task_type_1"][0]
                if classification.get("task_type_1") and classification["task_type_1"]
                else "Other"
            )
            validated_task_type = self._validate_task_type(str(detected_task_type))
            logger.debug(
                f"Prompt {prompt_index}: Detected task type: {validated_task_type}"
            )

            # Extract scores in the format expected by OpenAIParameters
            prompt_scores = self._extract_prompt_scores(classification)

            # Get complexity score
            complexity_score: float = classification["prompt_complexity_score"][0]
            logger.debug(f"Prompt {prompt_index}: Complexity score: {complexity_score}")

            # Get task mapping for the validated task type
            task_mapping: TaskModelMapping | None = self._task_mappings.get(
                validated_task_type
            )
            if not task_mapping:
                logger.warning(
                    f"Prompt {prompt_index}: No model mapping found for task type: {validated_task_type}, using default"
                )
                return self._get_default_result(validated_task_type, prompt_scores)

            # Select difficulty level
            selected_difficulty = self._select_difficulty_level(
                complexity_score, task_mapping
            )
            selected_config = task_mapping[selected_difficulty]

            # Validate model selection
            selected_model = selected_config["model"]
            self._validate_model_selection(selected_model, validated_task_type)

            # Calculate match score as confidence
            confidence = self._calculate_match_score(complexity_score, selected_config)

            # Get alternatives
            alternatives = self._get_alternatives(selected_model, task_mapping)

            # Get model info and OpenAI parameters
            model_info = self._model_capabilities[selected_model]
            openai_params = self._get_openai_parameters(
                selected_model, validated_task_type, prompt_scores
            )

            # Create reasoning string
            reasoning = (
                f"Selected {selected_model} for {validated_task_type} task "
                f"with complexity score {complexity_score:.3f} ({selected_difficulty} difficulty)"
            )

            return ModelSelectionResponse(
                selected_model=selected_model,
                confidence=confidence,
                reasoning=reasoning,
                alternatives=alternatives,
                parameters=openai_params,
                provider=model_info["provider"],
            )
        except Exception as e:
            logger.error(
                f"Error processing classification for prompt {prompt_index}: {e!s}"
            )
            # Return default response for failed classifications
            return self._get_default_result(
                "Other",
                {
                    "creativity_scope": [0.5],
                    "reasoning": [0.5],
                    "contextual_knowledge": [0.5],
                    "prompt_complexity_score": [0.5],
                    "domain_knowledge": [0.5],
                },
            )

    def select_model(self, prompts: list[str]) -> list[ModelSelectionResponse]:
        """
        Select the most appropriate model based on prompt analysis and task type.
        Processes prompts in parallel for optimal GPU utilization.

        Args:
            prompts (List[str]): The input prompts

        Returns:
            List[ModelSelectionResponse]: Model selection results for each prompt

        Raises:
            ModelSelectionError: If model selection fails
            ValueError: If input validation fails
        """
        try:
            # Get classifications from the prompt classifier (this should already be batched/parallelized)
            classification_results = self.prompt_classifier.classify_prompts(prompts)
            print(classification_results)
            # Validate results
            if not isinstance(classification_results, list):
                raise ModelSelectionError("Expected list of classification results")

            if not classification_results:
                raise ModelSelectionError("No classification results returned")

            if len(classification_results) != len(prompts):
                logger.warning(
                    f"Mismatch: {len(prompts)} prompts but {len(classification_results)} classifications"
                )

            # Process classifications in parallel using ThreadPoolExecutor
            responses: list[ModelSelectionResponse] = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(
                        self._process_single_classification, classification, i
                    ): i
                    for i, classification in enumerate(classification_results)
                }

                # Create a results dictionary to maintain order
                results: dict[int, ModelSelectionResponse] = {}

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        response = future.result()
                        results[index] = response
                    except Exception as e:
                        logger.error(f"Error processing prompt {index}: {e}")
                        # Add default response for failed processing
                        results[index] = self._get_default_result(
                            "Other",
                            {
                                "creativity_scope": [0.5],
                                "reasoning": [0.5],
                                "contextual_knowledge": [0.5],
                                "prompt_complexity_score": [0.5],
                                "domain_knowledge": [0.5],
                            },
                        )

            # Build the final response list in the correct order
            for i in range(len(classification_results)):
                if i in results:
                    responses.append(results[i])
                else:
                    logger.error(f"Missing response for prompt {i}, using default")
                    responses.append(
                        self._get_default_result(
                            "Other",
                            {
                                "creativity_scope": [0.5],
                                "reasoning": [0.5],
                                "contextual_knowledge": [0.5],
                                "prompt_complexity_score": [0.5],
                                "domain_knowledge": [0.5],
                            },
                        )
                    )

            logger.info(f"Successfully processed {len(responses)} prompts in parallel")
            return responses

        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            raise ModelSelectionError(f"Failed to select model: {e}") from e

    def select_orchestrator_route(self, prompt: str) -> OrchestratorResponse:
        """
        Route the prompt to the appropriate protocol based on complexity and window size.
        - If prompt is 'easy' and complexity < 0.3, select minion.
        - If prompt is 'large' (window size > 2048 tokens), select minions_protocol.
        - Otherwise, select standard_llm.
        """
        # Classify prompt
        classification = self.prompt_classifier.classify_prompts([prompt])[0]
        task_type = classification.get("task_type_1", ["Other"])[0]
        complexity_score = classification.get("prompt_complexity_score", [0.5])[0]
        # For window size, use a simple token count (split by whitespace as a proxy)
        token_count = count_tokens(prompt)
        large_window_threshold = 2  # You can adjust this as needed
        min_window_threshold = 4
        # Minion: easy and low complexity
        if token_count < min_window_threshold or complexity_score < 0.3:
            logger.info(
                f"Routing prompt to MINION protocol (task_type={task_type}, complexity={complexity_score})"
            )
            return MinionOrchestratorResponse(
                protocol="minion", minion_data=MinionDetails(task_type=task_type)
            )
        # Minions Protocol: large prompt
        elif token_count > large_window_threshold:
            logger.info(
                f"Routing prompt to MINIONS_PROTOCOL (task_type={task_type}, token_count={token_count})"
            )
            remote_llm_info = self._remote_llm_map.get(task_type)
            if remote_llm_info is None:
                remote_llm_info = self._remote_llm_map.get("Other")
                if remote_llm_info is None:
                    remote_llm_info = {
                        "provider": "Anthropic",
                        "model": "claude-sonnet-4-0",
                    }
            return MinionsProtocolOrchestratorResponse(
                protocol="minions_protocol",
                minions_protocol_data=MinionsProtocolDetails(
                    task_type=task_type,
                    remote_llm=RemoteLLM(
                        provider=remote_llm_info["provider"],
                        model=remote_llm_info["model"],
                    ),
                ),
            )
        # Standard LLM: default
        else:
            logger.info(
                f"Routing prompt to STANDARD_LLM (task_type={task_type}, complexity={complexity_score}, token_count={token_count})"
            )
            selection = self.select_model([prompt])[0]
            return StandardLLMOrchestratorResponse(
                protocol="standard_llm",
                standard_llm_data=StandardLLMDetails(
                    provider=str(selection.provider), model=selection.selected_model
                ),
                selected_model=selection.selected_model,
                confidence=selection.confidence,
                parameters=selection.parameters,
            )


@lru_cache
def get_model_selector(prompt_classifier: PromptClassifier) -> ModelSelector:
    return ModelSelector(prompt_classifier)


def count_tokens(prompt: str, model_name: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(prompt))
