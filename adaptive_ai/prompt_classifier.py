from adaptive_ai.nvidiaModel.neom_prompt_classifier import model,tokenizer

class PromptClassifier:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

    def classify_prompt(self, prompt):
        encoded_texts = self.tokenizer(
            [prompt],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        result = self.model(encoded_texts)
        return result


"""# Example usage:
if __name__ == "__main__":
    classifier = PromptClassifier()
    text_samples = "Sports is a popular domain" 
    predictions = classifier.classify_prompt(text_samples)
    print(predictions)"""