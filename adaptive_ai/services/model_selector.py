from typing import Dict, Any, List, TypedDict, Optional, cast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.llms import model_capabilities, domain_model_mapping
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
        self, prompt_classifier: Any, domain_classifier: Any
    ):  # Add proper type hints
        self.prompt_classifier = prompt_classifier
        self.domain_classifier = domain_classifier

    def select_model(self, prompt: str) -> Dict[str, Any]:
        """
        Select the most appropriate model using vector similarity matching.
        """
        prompt_analysis = self._analyze_prompt(prompt)
        domain = prompt_analysis["domain"]

        # Convert prompt scores to vector for matching
        prompt_vector = np.array(
            [
                prompt_analysis["prompt_scores"]["creativity_scope"][0],
                prompt_analysis["prompt_scores"]["reasoning"][0],
                prompt_analysis["prompt_scores"]["contextual_knowledge"][0],
                prompt_analysis["prompt_scores"]["domain_knowledge"][0],
            ]
        )

        # Find best matching model
        model_info = self._find_suitable_model(domain, prompt_vector)

        return {
            "prompt_scores": prompt_analysis["prompt_scores"],
            "selected_model": model_info["model_name"],
            "provider": model_info["provider"],
            "match_score": model_info["match_score"],
            "domain": domain,
        }

    def get_model_parameters(self, prompt: str) -> Dict[str, Any]:
        """Get optimized parameters for the selected model."""
        prompt_analysis = self._analyze_prompt(prompt)
        domain = prompt_analysis["domain"]

        prompt_vector = np.array(
            [
                prompt_analysis["prompt_scores"]["creativity_scope"][0],
                prompt_analysis["prompt_scores"]["reasoning"][0],
                prompt_analysis["prompt_scores"]["contextual_knowledge"][0],
                prompt_analysis["prompt_scores"]["domain_knowledge"][0],
            ]
        )

        model_info = self._find_suitable_model(domain, prompt_vector)
        parameters = self._get_parameters(
            model_info["provider"],
            model_info["model_name"],
            domain,
            prompt_analysis["prompt_scores"],
        )

        return {
            "parameters": parameters,
            "provider": model_info["provider"],
            "match_score": model_info["match_score"],
        }

    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt to extract scores and domain."""
        domain_result = self.domain_classifier.classify(prompt)
        domain = str(domain_result[0]) if domain_result else ""

        if domain not in domain_model_mapping:
            raise ValueError(f"Domain '{domain}' is not recognized.")

        complexity = self.prompt_classifier.classify_prompt(prompt, domain)

        return {
            "domain": domain,
            "prompt_scores": {
                "creativity_scope": cast(
                    List[float], complexity.get("creativity_scope", [0.0])
                ),
                "reasoning": cast(List[float], complexity.get("reasoning", [0.0])),
                "contextual_knowledge": cast(
                    List[float], complexity.get("contextual_knowledge", [0.0])
                ),
                "domain_knowledge": cast(
                    List[float], complexity.get("domain_knowledge", [0.0])
                ),
                "prompt_complexity_score": cast(
                    List[float], complexity.get("prompt_complexity_score", [0.0])
                ),
            },
        }

    def _find_suitable_model(
        self,
        domain: str,
        prompt_vector: np.ndarray,
        capability_weights: Optional[np.ndarray] = None,
    ) -> ModelInfo:
        """
        Find the best model using vector similarity matching.
        """
        if capability_weights is None:
            capability_weights = np.array([1.0, 1.0, 1.0, 1.0])

        candidate_models = domain_model_mapping.get(domain, [])
        if not candidate_models:
            raise ValueError(f"No models available for domain: {domain}")

        best_model = ""
        best_score = -1.0

        for model_name in candidate_models:
            model_data = model_capabilities.get(model_name)
            if not model_data:
                continue

            model_vec = model_data["capability_vector"][:4]

            similarity = cosine_similarity(
                [prompt_vector * capability_weights], [model_vec * capability_weights]
            )[0][0]

            if similarity > best_score:
                best_model = model_name
                best_score = similarity

        if not best_model:
            raise ValueError(f"No suitable model found for domain: {domain}")

        return {
            "model_name": best_model,
            "provider": model_capabilities[best_model]["provider"],
            "match_score": float(best_score),
        }

    def _get_parameters(
        self,
        provider_name: str,
        model_name: str,
        domain: str,
        prompt_scores: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """Get optimized parameters for the selected model."""
        parameters_provider = LLMParametersFactory.create(provider_name, model_name)
        return parameters_provider.adjust_parameters(domain, prompt_scores)
