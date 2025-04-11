from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.llms import model_capabilities, domain_model_mapping
from services.llm_parameters import LLMParametersFactory

class ModelSelector:
    """
    Service for selecting appropriate LLM models and parameters based on prompt analysis.
    Now with vector similarity-based model selection.
    """

    def __init__(self, prompt_classifier, domain_classifier):
        self.prompt_classifier = prompt_classifier
        self.domain_classifier = domain_classifier

    def select_model(self, prompt: str) -> Dict[str, Any]:
        """
        Select the most appropriate model using vector similarity matching.
        """
        prompt_analysis = self._analyze_prompt(prompt)
        domain = prompt_analysis["domain"]
        
        # Convert prompt scores to vector for matching
        prompt_vector = np.array([
            prompt_analysis["prompt_scores"]["creativity_scope"][0],
            prompt_analysis["prompt_scores"]["reasoning"][0],
            prompt_analysis["prompt_scores"]["contextual_knowledge"][0],
            prompt_analysis["prompt_scores"]["domain_knowledge"][0]
        ])

        # Find best matching model
        model_info = self._find_suitable_model(domain, prompt_vector)
        
        return {
            "prompt_scores": prompt_analysis["prompt_scores"],
            "selected_model": model_info["model_name"],
            "provider": model_info["provider"],
            "match_score": model_info["match_score"],
            "domain": domain
        }

    def get_model_parameters(self, prompt: str) -> Dict[str, Any]:
        """Get optimized parameters for the selected model."""
        prompt_analysis = self._analyze_prompt(prompt)
        domain = prompt_analysis["domain"]
        
        prompt_vector = np.array([
            prompt_analysis["prompt_scores"]["creativity_scope"][0],
            prompt_analysis["prompt_scores"]["reasoning"][0],
            prompt_analysis["prompt_scores"]["contextual_knowledge"][0],
            prompt_analysis["prompt_scores"]["domain_knowledge"][0]
        ])

        model_info = self._find_suitable_model(domain, prompt_vector)
        parameters = self._get_parameters(
            model_info["provider"],
            model_info["model_name"],
            domain,
            prompt_analysis["prompt_scores"]
        )

        return {
            "parameters": parameters,
            "provider": model_info["provider"],
            "match_score": model_info["match_score"]
        }

    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt to extract scores and domain."""
        domain = self.domain_classifier.classify(prompt)[0]
        if domain not in domain_model_mapping:
            raise ValueError(f"Domain '{domain}' is not recognized.")

        complexity = self.prompt_classifier.classify_prompt(prompt, domain)
        
        return {
            "domain": domain,
            "prompt_scores": {
                "creativity_scope": complexity["creativity_scope"],
                "reasoning": complexity["reasoning"],
                "contextual_knowledge": complexity["contextual_knowledge"],
                "domain_knowledge": complexity["domain_knowledge"],
                "prompt_complexity_score": complexity["prompt_complexity_score"]
            }
        }

    def _find_suitable_model(
        self,
        domain: str,
        prompt_vector: np.ndarray,
        capability_weights: np.ndarray = None
    ) -> Dict[str, str]:
        """
        Find the best model using vector similarity matching.
        """
        if capability_weights is None:
            capability_weights = np.array([1.0, 1.0, 1.0, 1.0])
        
        candidate_models = domain_model_mapping[domain]
        best_model, best_score = None, -1
        
        for model_name in candidate_models:
            model_vec = model_capabilities[model_name]["capability_vector"][:4]
            
            similarity = cosine_similarity(
                [prompt_vector * capability_weights],
                [model_vec * capability_weights]
            )[0][0]
            
            if similarity > best_score:
                best_model, best_score = model_name, similarity
        
        return {
            "model_name": best_model,
            "provider": model_capabilities[best_model]["provider"],
            "match_score": float(best_score)
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