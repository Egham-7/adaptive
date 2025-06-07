from functools import lru_cache
from typing import Dict, List, Tuple, Any, Union, cast
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer


class MeanPooling(nn.Module):
    def __init__(self) -> None:
        super(MeanPooling, self).__init__()

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MulticlassHead(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super(MulticlassHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        target_sizes: Dict[str, int],
        task_type_map: Dict[str, str],
        weights_map: Dict[str, List[float]],
        divisor_map: Dict[str, float],
    ) -> None:
        super(CustomModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/DeBERTa-v3-base", use_safetensors=True
        )
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map
        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes
        ]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        self.pool = MeanPooling()

    def compute_results(
        self, preds: torch.Tensor, target: str, decimal: int = 4
    ) -> Union[Tuple[List[str], List[str], List[float]], List[float]]:
        if target == "task_type":
            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()
            top2_strings = [
                [self.task_type_map[str(idx)] for idx in sample] for sample in top2
            ]
            top2_prob_rounded = [
                [round(value, 3) for value in sublist] for sublist in top2_prob
            ]
            counter = 0
            for sublist in top2_prob_rounded:
                if sublist[1] < 0.1:
                    top2_strings[counter][1] = "NA"
                counter += 1
            task_type_1 = [sublist[0] for sublist in top2_strings]
            task_type_2 = [sublist[1] for sublist in top2_strings]
            task_type_prob = [sublist[0] for sublist in top2_prob_rounded]
            return (task_type_1, task_type_2, task_type_prob)
        else:
            preds = torch.softmax(preds, dim=1)
            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]
            scores = [round(value, decimal) for value in scores]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return cast(List[float], scores)

    def _extract_classification_results(self, logits: List[torch.Tensor]) -> Dict[str, Union[List[str], List[float]]]:
        """Extract individual classification results from logits."""
        result: Dict[str, Union[List[str], List[float]]] = {}
        
        # Task type classification
        task_type_logits = logits[0]
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        if isinstance(task_type_results, tuple):
            result["task_type_1"] = task_type_results[0]
            result["task_type_2"] = task_type_results[1]
            result["task_type_prob"] = task_type_results[2]

        # Other classifications
        classifications = [
            ("creativity_scope", logits[1]),
            ("reasoning", logits[2]),
            ("contextual_knowledge", logits[3]),
            ("number_of_few_shots", logits[4]),
            ("domain_knowledge", logits[5]),
            ("no_label_reason", logits[6]),
            ("constraint_ct", logits[7])
        ]

        for target, target_logits in classifications:
            target_results = self.compute_results(target_logits, target=target)
            if isinstance(target_results, list):
                result[target] = target_results

        return result

    def _calculate_complexity_scores(
        self, results: Dict[str, Union[List[str], List[float]]], task_types: List[str]
    ) -> List[float]:
        """Calculate complexity scores using task-specific weights."""
        # Task type specific weights for complexity calculation
        TASK_TYPE_WEIGHTS: Dict[str, List[float]] = {
            "Open QA": [0.2, 0.3, 0.15, 0.2, 0.15],
            "Closed QA": [0.1, 0.35, 0.2, 0.25, 0.1],
            "Summarization": [0.2, 0.25, 0.25, 0.1, 0.2],
            "Text Generation": [0.4, 0.2, 0.15, 0.1, 0.15],
            "Code Generation": [0.1, 0.3, 0.2, 0.3, 0.1],
            "Chatbot": [0.25, 0.25, 0.15, 0.1, 0.25],
            "Classification": [0.1, 0.35, 0.25, 0.2, 0.1],
            "Rewrite": [0.2, 0.2, 0.3, 0.1, 0.2],
            "Brainstorming": [0.5, 0.2, 0.1, 0.1, 0.1],
            "Extraction": [0.05, 0.3, 0.3, 0.15, 0.2],
            "Other": [0.25, 0.25, 0.2, 0.15, 0.15],
        }

        # Get required values
        creativity_scope = cast(List[float], results.get("creativity_scope", []))
        reasoning = cast(List[float], results.get("reasoning", []))
        constraint_ct = cast(List[float], results.get("constraint_ct", []))
        domain_knowledge = cast(List[float], results.get("domain_knowledge", []))
        contextual_knowledge = cast(List[float], results.get("contextual_knowledge", []))

        complexity_scores = []
        for i, task_type in enumerate(task_types):
            # Use task-specific weights if available, otherwise use default weights
            weights = TASK_TYPE_WEIGHTS.get(task_type, [0.3, 0.3, 0.2, 0.1, 0.1])
            
            score = round(
                weights[0] * creativity_scope[i] +
                weights[1] * reasoning[i] +
                weights[2] * constraint_ct[i] +
                weights[3] * domain_knowledge[i] +
                weights[4] * contextual_knowledge[i],
                5
            )
            complexity_scores.append(score)

        return complexity_scores

    def process_logits(
        self, logits: List[torch.Tensor], domain: str
    ) -> Dict[str, Union[List[str], List[float], float]]:
        """Main orchestration method for processing logits and calculating complexity scores."""
        results = self._extract_classification_results(logits)
        
        if "task_type_1" in results:
            task_types = cast(List[str], results["task_type_1"])
            complexity_scores = self._calculate_complexity_scores(results, task_types)
            results["prompt_complexity_score"] = complexity_scores
            
        return results

    def forward(
        self, batch: Dict[str, torch.Tensor], domain: str
    ) -> Dict[str, Union[List[str], List[float], float]]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)
        logits = [
            self.heads[k](mean_pooled_representation)
            for k in range(len(self.target_sizes))
        ]
        return self.process_logits(logits, domain)


config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/prompt-task-and-complexity-classifier"
)
model = CustomModel(
    target_sizes=config.target_sizes,
    task_type_map=config.task_type_map,
    weights_map=config.weights_map,
    divisor_map=config.divisor_map,
).from_pretrained("nvidia/prompt-task-and-complexity-classifier")


class PromptClassifier:
    def __init__(self) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def classify_prompt(self, prompt: str, domain: str) -> Dict[str, Any]:
        encoded_texts = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            results = self.model(encoded_texts, domain)
        return cast(Dict[str, Any], results)

    def classify_task_types(self, texts: List[str]) -> List[str]:
        encoded_texts = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            results = self.model(encoded_texts, "general")
        return cast(List[str], results["task_type_1"])


@lru_cache()
def get_prompt_classifier() -> PromptClassifier:
    return PromptClassifier()
