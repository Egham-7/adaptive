from typing import Dict, Any, List, TypedDict, cast
import logging
from models.llms import model_capabilities, task_type_model_mapping, ModelCapability

logger = logging.getLogger(__name__)


class ModelInfo(TypedDict):
    model_name: str
    provider: str
    match_score: float


class ModelSelectionError(Exception):
    """Custom exception for model selection errors"""

    pass


class ModelSelector:
    """
    Service for selecting appropriate LLM models and parameters based on prompt analysis.
    Uses task-specific complexity-based model selection.
    """

    def __init__(self, prompt_classifier: Any):
        self.prompt_classifier = prompt_classifier
        logger.info("ModelSelector initialized")

    def _validate_model_selection(self, selected_model: str, task_type: str) -> None:
        """Validate that the selected model exists and is appropriate for the task"""
        if not isinstance(selected_model, str):
            raise ModelSelectionError(f"Invalid model type: {type(selected_model)}")

        if selected_model not in model_capabilities:
            raise ModelSelectionError(
                f"Selected model {selected_model} not found in capabilities"
            )

        model_info = cast(ModelCapability, model_capabilities[selected_model])
        logger.info(
            f"Selected model {selected_model} ({model_info['provider']}) for task {task_type}"
        )

    def select_model(self, prompt: List[str]) -> List[Dict[str, Any]]:
        print(f"DEBUG: select_model received: {type(prompt)} = {prompt}")
        """
        Select the most appropriate model based on prompt analysis and task type.

        Args:
            prompt (List[str]): The input prompts

        Returns:
            List[Dict[str, Any]]: List of model selection results, one for each prompt

        Raises:
            ModelSelectionError: If model selection fails
            ValueError: If input validation fails
        """
        try:
            if (
                not prompt
                or not isinstance(prompt, list)
                or not all(isinstance(p, str) for p in prompt)
            ):
                raise ValueError("Invalid prompt: must be a non-empty list of strings")

            # Get complexity analysis and task type
            classification = self.prompt_classifier.classify_prompt(prompt)

            # Get task types from the classification results
            task_types = classification["task_type_1"]

            # Extract scores with type safety
            prompt_scores = {
                "creativity_scope": cast(
                    List[float], classification.get("creativity_scope", [0.0])
                ),
                "reasoning": cast(List[float], classification.get("reasoning", [0.0])),
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

            # Get complexity scores
            complexity_scores = classification["prompt_complexity_score"]

            # Process each prompt
            results = []
            for i, (task_type, complexity_score) in enumerate(
                zip(task_types, complexity_scores)
            ):
                logger.info(
                    f"Processing prompt {i+1}, task type: {task_type}, complexity: {complexity_score}"
                )

                # Get task difficulties for the current task type
                task_difficulties = task_type_model_mapping.get(task_type, {})
                if not task_difficulties:
                    logger.warning(
                        f"No model mapping found for task type: {task_type}, using default"
                    )
                    results.append(
                        {
                            "selected_model": "gpt-4.1",
                            "provider": "OpenAI",
                            "match_score": 0.0,
                            "task_type": task_type,
                            "difficulty": "medium",
                            "prompt_scores": {
                                k: [v[i]] for k, v in prompt_scores.items()
                            },
                            "complexity_score": complexity_score,
                            "thresholds": {"easy": 0.3, "medium": 0.5, "hard": 0.7},
                        }
                    )
                    continue

                # Get thresholds for the current task type
                easy_threshold = task_difficulties["easy"]["complexity_threshold"]
                medium_threshold = task_difficulties["medium"]["complexity_threshold"]
                hard_threshold = task_difficulties["hard"]["complexity_threshold"]

                # Select difficulty based on complexity score and task-specific thresholds
                selected_difficulty = "medium"  # default
                if complexity_score <= easy_threshold:
                    selected_difficulty = "easy"
                elif complexity_score >= hard_threshold:
                    selected_difficulty = "hard"

                selected_model = str(task_difficulties[selected_difficulty]["model"])
                self._validate_model_selection(selected_model, task_type)

                # Calculate match score based on how close the complexity score is to the selected threshold
                selected_threshold = task_difficulties[selected_difficulty][
                    "complexity_threshold"
                ]
                match_score = 1.0 - min(abs(complexity_score - selected_threshold), 1.0)

                model_info = cast(ModelCapability, model_capabilities[selected_model])

                results.append(
                    {
                        "selected_model": selected_model,
                        "provider": model_info["provider"],
                        "match_score": float(match_score),
                        "task_type": task_type,
                        "difficulty": selected_difficulty,
                        "prompt_scores": {k: [v[i]] for k, v in prompt_scores.items()},
                        "complexity_score": complexity_score,
                        "thresholds": {
                            "easy": easy_threshold,
                            "medium": medium_threshold,
                            "hard": hard_threshold,
                        },
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error in model selection: {str(e)}")
            raise ModelSelectionError(f"Failed to select model: {str(e)}")

    def get_model_parameters(self, prompt: List[str], task_type: str) -> Dict[str, Any]:
        """
        Get model parameters based on prompt analysis and task type.

        Args:
            prompt (List[str]): The input prompts
            task_type (str): The task type for the prompts

        Returns:
            Dict[str, Any]: Model parameters

        Raises:
            ModelSelectionError: If parameter selection fails
        """
        try:
            # Get complexity analysis
            classification = self.prompt_classifier.classify_prompt(prompt)

            # Extract scores with type safety
            prompt_scores = {
                "creativity_scope": cast(
                    List[float], classification.get("creativity_scope", [0.0])
                ),
                "reasoning": cast(List[float], classification.get("reasoning", [0.0])),
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

            return {
                "task_type": task_type,
                "prompt_scores": prompt_scores,
            }

        except Exception as e:
            logger.error(f"Error getting model parameters: {str(e)}")
            raise ModelSelectionError(f"Failed to get model parameters: {str(e)}")
