from functools import lru_cache
from .prompt_classifier import get_prompt_classifier

class DomainClassifier:
    def __init__(self):
        """
        Initialize the DomainClassifier using the prompt classifier's task type classification.
        """
        self.prompt_classifier = get_prompt_classifier()

    def classify(self, texts):
        """
        Classify a list of text samples into their respective task types.

        Args:
            texts (list of str): The text samples to classify.

        Returns:
            list of str: The predicted task types for each text sample.
        """
        # Use a default domain since we only care about task type
        default_domain = "Computers_and_Electronics"
        
        results = []
        for text in texts:
            classification = self.prompt_classifier.classify_prompt(text, default_domain)
            # Get the primary task type (task_type_1)
            results.append(classification["task_type_1"][0])
        
        return results

@lru_cache()
def get_domain_classifier():
    return DomainClassifier()
