from models.llms import model_capabilities, domain_model_mapping
from services.llm_parameters import LLMParametersFactory


class ModelSelectionService:
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

    def select_model(self, prompt):
        """
        Select the most appropriate model for a given prompt.

        Args:
            prompt: The user's prompt text

        Returns:
            Dict containing selected model name and provider

        Raises:
            ValueError: If the classified domain is not recognized
        """
        complexity = self.prompt_classifier.classify_prompt(prompt)
        complexity_score = complexity["prompt_complexity_score"][0]
        domain = self.domain_classifier.classify(prompt)[0]

        if domain not in domain_model_mapping:
            raise ValueError(f"Domain '{domain}' is not recognized.")

        # Filter models suitable for the given domain
        suitable_models = domain_model_mapping[domain]

        # Find a model within the suitable models that matches the complexity score
        for model_name in suitable_models:
            complexity_range = model_capabilities[model_name]["complexity_range"]
            provider = model_capabilities[model_name]["provider"]

            if complexity_range[0] <= complexity_score <= complexity_range[1]:
                return {"selected_model": model_name, "provider": provider}

        # If no model matches the complexity score, return a default model
        return {
            "selected_model": suitable_models[0],
            "provider": model_capabilities[suitable_models[0]]["provider"],
        }

    def get_model_parameters(self, prompt):
        """
        Get optimized parameters for the selected model based on prompt analysis.

        Args:
            prompt: The user's prompt text

        Returns:
            Dict containing model parameters and provider

        Raises:
            ValueError: If the classified domain is not recognized
        """
        complexity = self.prompt_classifier.classify_prompt(prompt)
        complexity_score = complexity["prompt_complexity_score"][0]
        domain = self.domain_classifier.classify(prompt)[0]

        if domain not in domain_model_mapping:
            raise ValueError(f"Domain '{domain}' is not recognized.")

        suitable_models = domain_model_mapping[domain]

        for model_name in suitable_models:
            complexity_range = model_capabilities[model_name]["complexity_range"]
            provider = model_capabilities[model_name]["provider"]

            if complexity_range[0] <= complexity_score <= complexity_range[1]:
                selected_model = {"selected_model": model_name, "provider": provider}
                break
        else:
            selected_model = {
                "selected_model": suitable_models[0],
                "provider": model_capabilities[suitable_models[0]]["provider"],
            }

        prompt_scores = {
            "creativity_scope": complexity["creativity_scope"],
            "reasoning": complexity["reasoning"],
            "contextual_knowledge": complexity["contextual_knowledge"],
            "prompt_complexity_score": complexity["prompt_complexity_score"],
            "domain_knowledge": complexity["domain_knowledge"],
        }

        provider_name = selected_model["provider"]
        model_name = selected_model["selected_model"]

        try:
            # Use the factory to create the appropriate parameters object
            parameters_provider = LLMParametersFactory.create(provider_name, model_name)
            parameters_model = parameters_provider.adjust_parameters(
                domain, prompt_scores
            )
        except ValueError:
            parameters_model = {}

        return {"parameters": parameters_model, "provider": provider_name}
