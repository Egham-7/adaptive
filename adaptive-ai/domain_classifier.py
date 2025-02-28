import torch
from transformers import AutoConfig, AutoTokenizer
from .nvidiaModel.neom_domain_classifier import CustomModel  

class DomainClassifier:
    def __init__(self, model_name="nvidia/domain-classifier"):
        """
        Initialize the DomainClassifier with the specified model.

        Args:
            model_name (str): The name of the pre-trained model to load.
        """
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CustomModel.from_pretrained(model_name)
        self.model.eval()

    def classify(self, texts):
        """
        Classify a list of text samples into their respective domains.

        Args:
            texts (list of str): The text samples to classify.

        Returns:
            list of str: The predicted domains for each text sample.
        """
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding="longest", truncation=True
        )
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
        predicted_classes = torch.argmax(outputs, dim=1)
        predicted_domains = [
            self.config.id2label[class_idx.item()] for class_idx in predicted_classes
        ]
        return predicted_domains


"""# Example usage:
if __name__ == "__main__":
    classifier = DomainClassifier()
    text_samples = "Sports is a popular domain" 
    predictions = classifier.classify(text_samples)
    print(predictions)"""

