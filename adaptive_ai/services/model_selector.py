from typing import Dict, Any, List, TypedDict, Optional, cast
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from models.llms import model_capabilities, task_type_model_mapping
from services.llm_parameters import LLMParametersFactory


class ModelInfo(TypedDict):
    model_name: str
    provider: str
    match_score: float


class ModelSelector:
    """
    Service for selecting appropriate LLM models and parameters based on prompt analysis.
    Now with vector similarity-based model selection.
    """

    def __init__(
        self, prompt_classifier: Any, task_type_classifier: Any
    ):
        self.prompt_classifier = prompt_classifier
        self.task_type_classifier = task_type_classifier

    def select_model(self, prompt: str) -> Dict[str, Any]:
        """
        Select the most appropriate model based on prompt analysis and task type.

        Args:
            prompt (str): The input prompt

        Returns:
            Dict[str, Any]: Model selection results including scores and parameters
        """
        # Get task type classification
        task_type_result = self.task_type_classifier.classify([prompt])
        task_type = task_type_result[0] if task_type_result else "Other"

        # Get complexity analysis using a default domain
        default_domain = "Computers_and_Electronics"
        complexity = self.prompt_classifier.classify_prompt(prompt, default_domain)

        # Extract scores
        prompt_scores = {
            "creativity_scope": cast(
                List[float], complexity.get("creativity_scope", [0.0])
            ),
            "reasoning": cast(
                List[float], complexity.get("reasoning", [0.0])
            ),
            "constraint_ct": cast(
                List[float], complexity.get("constraint_ct", [0.0])
            ),
            "contextual_knowledge": cast(
                List[float], complexity.get("contextual_knowledge", [0.0])
            ),
        }

        # Get candidate models for the task type
        candidate_models = task_type_model_mapping.get(task_type, [])

        if not candidate_models:
            return {
                "selected_model": "gpt-4-turbo",
                "provider": "OpenAI",
                "match_score": 0.0,
                "task_type": task_type,
                "prompt_scores": prompt_scores
            }

        # Get model capabilities for candidates
        model_vectors = {
            model: model_capabilities[model]["capability_vector"]
            for model in candidate_models
            if model in model_capabilities
        }

        if not model_vectors:
            return {
                "selected_model": "gpt-4-turbo",
                "provider": "OpenAI",
                "match_score": 0.0,
                "task_type": task_type,
                "prompt_scores": prompt_scores
            }

        # Create prompt vector from scores
        prompt_vector = np.array([
            prompt_scores["creativity_scope"][0],
            prompt_scores["reasoning"][0],
            prompt_scores["constraint_ct"][0],
            prompt_scores["contextual_knowledge"][0],
        ])

        # Calculate similarities
        distances = {
            model: euclidean_distances(
                [prompt_vector], [vector]
            )[0][0]
            for model, vector in model_vectors.items()
        }
        max_distance = max(distances.values())
        similarities = {
            model: 1 - (dist / max_distance)
            for model, dist in distances.items()
        }
        print("similarities: ",similarities)
        # Select model with highest similarity
        selected_model = max(similarities.items(), key=lambda x: x[1])[0]
        match_score = similarities[selected_model]

        return {
            "selected_model": selected_model,
            "provider": model_capabilities[selected_model]["provider"],
            "match_score": float(match_score),
            "task_type": task_type,
            "prompt_scores": prompt_scores
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
        # Get task type classification
        task_type_result = self.task_type_classifier.classify([prompt])
        if not task_type_result:
            task_type = "Other"
        else:
            task_type = task_type_result[0]

        # Get complexity analysis using a default domain
        default_domain = "Computers_and_Electronics"
        complexity = self.prompt_classifier.classify_prompt(prompt, default_domain)

        # Extract scores
        prompt_scores = {
            "creativity_scope": cast(
                List[float], complexity.get("creativity_scope", [0.0])
            ),
            "reasoning": cast(
                List[float], complexity.get("reasoning", [0.0])
            ),
            "constraint_ct": cast(
                List[float], complexity.get("constraint_ct", [0.0])
            ),
            "contextual_knowledge": cast(
                List[float], complexity.get("contextual_knowledge", [0.0])
            ),
        }

        return {
            "task_type": task_type,
            "prompt_scores": prompt_scores,
        }
