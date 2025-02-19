import os
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/models/")
MODEL_PATH = os.path.join(MODEL_CACHE_DIR, MODEL_NAME)

# Ensure the cache directory exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super(CustomModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME)
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map
        self.heads = [MulticlassHead(self.backbone.config.hidden_size, sz) for sz in self.target_sizes]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        self.pool = MeanPooling()

    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()
            top2_strings = [[self.task_type_map[str(idx)] for idx in sample] for sample in top2]
            top2_prob_rounded = [[round(value, 3) for value in sublist] for sublist in top2_prob]
            for i, sublist in enumerate(top2_prob_rounded):
                if sublist[1] < 0.1:
                    top2_strings[i][1] = "NA"
            return ([sublist[0] for sublist in top2_strings], [sublist[1] for sublist in top2_strings], [sublist[0] for sublist in top2_prob_rounded])
        else:
            preds = torch.softmax(preds, dim=1)
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * np.array(self.weights_map[target]), axis=1)
            scores = weighted_sum / self.divisor_map[target]
            return [round(value, decimal) for value in scores]

    def forward(self, batch):
        outputs = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        mean_pooled_representation = self.pool(outputs.last_hidden_state, batch["attention_mask"])
        logits = [head(mean_pooled_representation) for head in self.heads]
        return self.process_logits(logits)

config = AutoConfig.from_pretrained(MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME)
model = CustomModel(
    target_sizes=config.target_sizes,
    task_type_map=config.task_type_map,
    weights_map=config.weights_map,
    divisor_map=config.divisor_map,
).from_pretrained(MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME)
model.eval()

prompt = ["Prompt: Write a Python script that uses a for loop."]
encoded_texts = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, max_length=512, padding="max_length", truncation=True)
result = model(encoded_texts)
print(result)