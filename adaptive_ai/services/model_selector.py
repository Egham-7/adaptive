from typing import Dict, Any, List, TypedDict, Optional, cast
import numpy as np
from models.llms import model_capabilities, task_type_model_mapping, task_weights, ModelCapability
from services.llm_parameters import LLMParametersFactory
from services.prompt_classifier import get_prompt_classifier


class ModelInfo(TypedDict):
    model_name: str
    provider: str
    match_score: float


class ModelSelector:
    """
    Service for selecting appropriate LLM models and parameters based on prompt analysis.
    Uses task-specific complexity-based model selection.
    """

    def __init__(self, prompt_classifier: Any):
        self.prompt_classifier = prompt_classifier

    def select_model(self, prompt: str) -> Dict[str, Any]:
        """
        Select the most appropriate model based on prompt analysis and task type.

        Args:
            prompt (str): The input prompt

        Returns:
            Dict[str, Any]: Model selection results including scores and parameters
        """
        # Get complexity analysis and task type in a single call
        default_domain = "Computers_and_Electronics"
        classification = self.prompt_classifier.classify_prompt(prompt, default_domain)
        
        # Get task type from the classification results
        task_type = classification["task_type_1"][0] if classification["task_type_1"] else "Other"

        # Extract scores
        prompt_scores = {
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

        # Get complexity score
        complexity_score = classification["prompt_complexity_score"][0]

        # Get task difficulties for the current task type
        task_difficulties = task_type_model_mapping.get(task_type, {})
        if not task_difficulties:
            return {
                "selected_model": "gpt-4-turbo",
                "provider": "OpenAI",
                "match_score": 0.0,
                "task_type": task_type,
                "prompt_scores": prompt_scores,
                "complexity_score": complexity_score
            }

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

        selected_model = task_difficulties[selected_difficulty]["model"]
        
        # Calculate match score based on how close the complexity score is to the selected threshold
        selected_threshold = task_difficulties[selected_difficulty]["complexity_threshold"]
        match_score = 1.0 - min(abs(complexity_score - selected_threshold), 1.0)

        # Ensure selected_model is a string and exists in model_capabilities
        if not isinstance(selected_model, str) or selected_model not in model_capabilities:
            selected_model = "gpt-4-turbo"  # Fallback to a safe default

        model_info = cast(ModelCapability, model_capabilities[selected_model])

        return {
            "selected_model": selected_model,
            "provider": model_info["provider"],
            "match_score": float(match_score),
            "task_type": task_type,
            "difficulty": selected_difficulty,
            "prompt_scores": prompt_scores,
            "complexity_score": complexity_score,
            "thresholds": {
                "easy": easy_threshold,
                "medium": medium_threshold,
                "hard": hard_threshold
            }
        }

    def get_model_parameters(
        self, prompt: str, task_type: str
    ) -> Dict[str, Any]:
        """
        Get model parameters based on prompt analysis and task type.

        Args:
            prompt (str): The input prompt
            task_type (str): The task type for the prompt

        Returns:
            Dict[str, Any]: Model parameters
        """
        # Get complexity analysis in a single call
        default_domain = "Computers_and_Electronics"
        classification = self.prompt_classifier.classify_prompt(prompt, default_domain)

        # Extract scores
        prompt_scores = {
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

        return {
            "task_type": task_type,
            "prompt_scores": prompt_scores,
        }