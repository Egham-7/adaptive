from typing import Dict, Any, List, Tuple

from models.llms import model_capabilities, domain_model_mapping
from services.llm_parameters import LLMParametersFactory


class ModelSelector:
    """
    Service for selecting appropriate LLM models and parameters based on prompt analysis.
    """

    def __init__(self, prompt_classifier, domain_classifier):
        """
        Initialize the model selection service with required classifiers.

        Args:
            prompt_classifier: Classifier for analyzing prompt characteristics
            domain_classifier: Classifier for determining the domain of a prompt
        """
        self.prompt_classifier = prompt_classifier
        self.domain_classifier = domain_classifier

    def select_model(self, prompt: str) -> Dict[str, Any]:
        """
        Select the most appropriate model for a given prompt with optimized parameters.

        Args:
            prompt: The user's prompt text

        Returns:
            Dict containing selected model name, provider, and parameters

        Raises:
            ValueError: If the classified domain is not recognized
        """
        # Analyze prompt and get domain
        prompt_analysis = self._analyze_prompt(prompt)
        domain = prompt_analysis["domain"]

        complexity_score = prompt_analysis["complexity_score"]

        # Select the most appropriate model for the domain and complexity
        model_info = self._find_suitable_model(domain, complexity_score)
        model_name = model_info["model_name"]
        provider_name = model_info["provider"]

        # Return complete model selection info
        return {
            "prompt_scores": prompt_analysis["prompt_scores"],
            "selected_model": model_name,
            "provider": provider_name,
            "complexity:": complexity_score,
            "domain:": domain,
            # "parameters": parameters,
        }

    def get_model_parameters(self, prompt: str) -> Dict[str, Any]:
        """
        Get optimized parameters for the selected model based on prompt analysis.

        Args:
            prompt: The user's prompt text

        Returns:
            Dict containing model parameters and provider

        Raises:
            ValueError: If the classified domain is not recognized
        """
        # Analyze prompt and get domain
        prompt_analysis = self._analyze_prompt(prompt)
        domain = prompt_analysis["domain"]
        complexity_score = prompt_analysis["complexity_score"]
        prompt_scores = prompt_analysis["prompt_scores"]

        # Select the most appropriate model for the domain and complexity
        model_info = self._find_suitable_model(domain, complexity_score)
        model_name = model_info["model_name"]
        provider_name = model_info["provider"]

        # Get optimized parameters for the selected model
        parameters = self._get_parameters(
            provider_name, model_name, domain, prompt_scores
        )

        return {"parameters": parameters, "provider": provider_name}

    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze the prompt to extract complexity scores and domain.

        Args:
            prompt: The user's prompt text

        Returns:
            Dict with domain, complexity score, and prompt scores

        Raises:
            ValueError: If the classified domain is not recognized
        """

        # Get domain
        domain = self.domain_classifier.classify(prompt)[0]
        if domain not in domain_model_mapping:
            raise ValueError(f"Domain '{domain}' is not recognized.")

        # Get complexity analysis
        complexity = self.prompt_classifier.classify_prompt(prompt, domain)
        complexity_score = float(complexity["prompt_complexity_score"][0])

        # Extract prompt scores for parameter tuning
        prompt_scores = {
            "creativity_scope": complexity["creativity_scope"],
            "reasoning": complexity["reasoning"],
            "contextual_knowledge": complexity["contextual_knowledge"],
            "prompt_complexity_score": complexity["prompt_complexity_score"],
            "domain_knowledge": complexity["domain_knowledge"],
        }

        return {
            "domain": domain,
            "complexity_score": complexity_score,
            "prompt_scores": prompt_scores,
        }

    def _find_suitable_model(
        self, domain: str, complexity_score: float
    ) -> Dict[str, str]:
        """
        Find a suitable model that matches the domain and complexity.

        Args:
            domain: The classified domain
            complexity_score: The prompt complexity score

        Returns:
            Dict with model_name and provider
        """
        suitable_models = domain_model_mapping[domain]

        # Log the models being checked for the given domain
        print(f"Domain: {domain} - Models: {suitable_models}")

        # Find a model that matches the complexity score
        for model_name in suitable_models:
            complexity_range: Tuple[float, float] = model_capabilities[model_name][
                "complexity_range"
            ]
            provider: str = model_capabilities[model_name]["provider"]

            # Explicitly cast complexity range values to float to satisfy mypy
            lower_bound = float(complexity_range[0])
            upper_bound = float(complexity_range[1])

            # Log the complexity range for debugging
            print(
                f"Checking model: {model_name} | Complexity Range: ({lower_bound}, {upper_bound}) | Score: {complexity_score}"
            )

            if lower_bound <= complexity_score <= upper_bound:
                print(f"Selected model: {model_name} within complexity range.")
                return {"model_name": model_name, "provider": provider}

        # If no model matches, return the first suitable model as default
        default_model = suitable_models[0]
        print(f"Returning default model: {default_model}")
        return {
            "model_name": default_model,
            "provider": model_capabilities[default_model]["provider"],
        }

    def _get_parameters(
        self,
        provider_name: str,
        model_name: str,
        domain: str,
        prompt_scores: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Get optimized parameters for a model based on provider, domain and prompt scores.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            domain: The classified domain
            prompt_scores: Dictionary of prompt analysis scores

        Returns:
            Dictionary of optimized parameters
        """
        parameters_provider = LLMParametersFactory.create(provider_name, model_name)
        return parameters_provider.adjust_parameters(domain, prompt_scores)
