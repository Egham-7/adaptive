import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import cast

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.requests import ModelSelectionResponse
from adaptive_ai.models.types import (
    VALID_TASK_TYPES,
    DifficultyLevel,
    ModelCapability,
    ModelSelectionError,
    TaskDifficultyConfig,
    TaskModelMapping,
    TaskType,
)
from adaptive_ai.services.llm_parameters import OpenAIParameters
from adaptive_ai.services.prompt_classifier import PromptClassifier

logger = logging.getLogger(__name__)


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

    def _select_difficulty_level(
        self, complexity_score: float, task_mapping: TaskModelMapping
    ) -> DifficultyLevel:
        """Select difficulty level based on complexity score and thresholds"""
        easy_threshold = task_mapping["easy"]["complexity_threshold"]
        hard_threshold = task_mapping["hard"]["complexity_threshold"]

        if complexity_score <= easy_threshold:
            return "easy"
        elif complexity_score >= hard_threshold:
            return "hard"
        else:
            return "medium"

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
                f"Error processing classification for prompt {prompt_index}: {str(e)}"
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


@lru_cache
def get_model_selector(prompt_classifier: PromptClassifier) -> ModelSelector:
    return ModelSelector(prompt_classifier)
