from functools import lru_cache
from typing import List, cast, Optional, Dict
import logging
from models.types import (
    VALID_TASK_TYPES,
    TaskType,
    ModelCapability,
    ModelSelectionError,
    PromptScores,
    ModelParameters,
    TaskModelMapping,
    TaskDifficultyConfig,
    DifficultyLevel,
)
from models.requests import ModelSelectionResponse
from models.config_loader import get_model_capabilities, get_task_model_mappings
from services.prompt_classifier import PromptClassifier
from services.llm_parameters import OpenAIParameters

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Service for selecting appropriate LLM models and parameters based on prompt analysis.
    Uses task-specific complexity-based model selection.
    """

    def __init__(self, prompt_classifier: PromptClassifier) -> None:
        self.prompt_classifier: PromptClassifier = prompt_classifier
        self._model_capabilities = get_model_capabilities()
        self._task_mappings = get_task_model_mappings()
        logger.info("ModelSelector initialized")

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

    def _extract_prompt_scores(self, classification: dict) -> Dict[str, list]:
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
        prompt_scores: Dict[str, list],
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
    ) -> List[str]:
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
        self, model_name: str, task_type: str, prompt_scores: Dict[str, list]
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

    def select_model(self, prompts: List[str]) -> ModelSelectionResponse:
        """
        Select the most appropriate model based on prompt analysis and task type.

        Args:
            prompts (List[str]): The input prompts
            domain (str): The domain context for the prompt

        Returns:
            ModelSelectionResult: Model selection results including scores and parameters

        Raises:
            ModelSelectionError: If model selection fails
            ValueError: If input validation fails
        """
        try:
            classification = self.prompt_classifier.classify_prompt(prompts)

            # Get and validate task type from the classification results
            detected_task_type = (
                classification["task_type_1"][0]
                if classification["task_type_1"]
                else "Other"
            )
            validated_task_type = self._validate_task_type(detected_task_type)
            logger.info(f"Detected task type: {validated_task_type}")

            # Extract scores in the format expected by OpenAIParameters
            prompt_scores = self._extract_prompt_scores(classification)

            # Get complexity score
            complexity_score: float = classification["prompt_complexity_score"][0]
            logger.info(f"Complexity score: {complexity_score}")

            # Get task mapping for the validated task type
            task_mapping: Optional[TaskModelMapping] = self._task_mappings.get(
                validated_task_type
            )
            if not task_mapping:
                logger.warning(
                    f"No model mapping found for task type: {validated_task_type}, using default"
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
            logger.error(f"Error in model selection: {str(e)}")
            raise ModelSelectionError(f"Failed to select model: {str(e)}") from e

    def get_model_parameters(self, prompt: str, task_type: str) -> ModelParameters:
        """
        Get model parameters based on prompt analysis and task type.

        Args:
            prompt (str): The input prompt
            task_type (str): The task type for the prompt

        Returns:
            ModelParameters: Model parameters

        Raises:
            ModelSelectionError: If parameter selection fails
        """
        try:
            classification = self.prompt_classifier.classify_prompt([prompt])

            # Extract scores with type safety
            prompt_scores = PromptScores(
                {
                    "creativity_scope": cast(
                        List[float], classification.get("creativity_scope", [0.0])
                    ),
                    "reasoning": cast(
                        List[float], classification.get("reasoning", [0.0])
                    ),
                    "constraint_ct": cast(
                        List[float], classification.get("constraint_ct", [0.0])
                    ),
                    "contextual_knowledge": cast(
                        List[float], classification.get("contextual_knowledge", [0.0])
                    ),
                    "domain_knowledge": cast(
                        List[float], classification.get("domain_knowledge", [0.0])
                    ),
                }
            )

            return ModelParameters(
                {
                    "task_type": task_type,
                    "prompt_scores": prompt_scores,
                }
            )

        except Exception as e:
            logger.error(f"Error getting model parameters: {str(e)}")
            raise ModelSelectionError(
                f"Failed to get model parameters: {str(e)}"
            ) from e


@lru_cache()
def get_model_selector(prompt_classifier: PromptClassifier) -> ModelSelector:
    return ModelSelector(prompt_classifier)
