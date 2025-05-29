from typing import Dict, Any, List, TypedDict, Optional, cast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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
    Now with vector similarity-based model selection.
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

        # Create prompt vector from scores
        prompt_vector = np.array([
            prompt_scores["creativity_scope"][0],
            prompt_scores["reasoning"][0],
            prompt_scores["constraint_ct"][0],
            prompt_scores["contextual_knowledge"][0],
            prompt_scores["domain_knowledge"][0],
        ], dtype=np.float64)

        # Get task weights for the current task type
        task_weight_vector = np.array(task_weights.get(task_type, task_weights["Other"]), dtype=np.float64)
        
        # Multiply prompt vector by task weights
        weighted_prompt_vector = prompt_vector * task_weight_vector

        # Get difficulty levels and their vectors for the task type
        task_difficulties = task_type_model_mapping.get(task_type, {})
        if not task_difficulties:
            return {
                "selected_model": "gpt-4-turbo",
                "provider": "OpenAI",
                "match_score": 0.0,
                "task_type": task_type,
                "prompt_scores": prompt_scores,
                "weighted_vector": weighted_prompt_vector.tolist()
            }

        # Calculate similarities with each difficulty level's vector
        difficulty_similarities = {}
        for difficulty, info in task_difficulties.items():
            difficulty_vector = np.array(info["vector"], dtype=np.float64)
            distance = euclidean_distances([weighted_prompt_vector], [difficulty_vector])[0][0]
            max_distance = np.max([
                float(np.linalg.norm(weighted_prompt_vector)),
                float(np.linalg.norm(difficulty_vector))
            ])
            similarity = 1 - (distance / max_distance) if max_distance > 0 else 1.0
            difficulty_similarities[difficulty] = similarity

        # Select the difficulty level with highest similarity
        selected_difficulty = max(difficulty_similarities.items(), key=lambda x: x[1])[0]
        selected_model = task_difficulties[selected_difficulty]["model"]
        match_score = difficulty_similarities[selected_difficulty]

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
            "weighted_vector": weighted_prompt_vector.tolist()
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