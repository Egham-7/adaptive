from functools import lru_cache
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import nn
from huggingface_hub import PyTorchModelHubMixin


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)


class DomainClassifier:
    def __init__(self):
        self.model = AutoModel.from_pretrained("nvidia/domain-classifier")
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/domain-classifier")
        self.config = AutoConfig.from_pretrained("nvidia/domain-classifier")
        self.model.eval()

    def classify(self, texts):
        if isinstance(texts, str):
            texts = [texts]

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


@lru_cache()
def get_domain_classifier():
    return DomainClassifier()
