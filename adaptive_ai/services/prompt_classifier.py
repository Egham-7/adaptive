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
        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
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

    def process_logits(
        self, logits: List[torch.Tensor], domain: str
    ) -> Dict[str, Union[List[str], List[float], float]]:
        # Task type specific weights for complexity calculation
        TASK_TYPE_WEIGHTS: Dict[str, List[float]] = {
            "Open QA": [
                0.2,
                0.3,
                0.15,
                0.2,
                0.15,
            ],  # Needs reasoning + some domain/contextual recall
            "Closed QA": [
                0.1,
                0.35,
                0.2,
                0.25,
                0.1,
            ],  # Factual recall + precise reasoning
            "Summarization": [
                0.2,
                0.25,
                0.25,
                0.1,
                0.2,
            ],  # Requires constraint (brevity), context
            "Text Generation": [0.4, 0.2, 0.15, 0.1, 0.15],  # Creativity-driven
            "Code Generation": [
                0.1,
                0.3,
                0.2,
                0.3,
                0.1,
            ],  # High constraint & reasoning + domain knowledge
            "Chatbot": [
                0.25,
                0.25,
                0.15,
                0.1,
                0.25,
            ],  # Creativity, reasoning, and context
            "Classification": [
                0.1,
                0.35,
                0.25,
                0.2,
                0.1,
            ],  # Heavy on reasoning and constraint
            "Rewrite": [
                0.2,
                0.2,
                0.3,
                0.1,
                0.2,
            ],  # Needs adherence to form (constraint) + context
            "Brainstorming": [0.5, 0.2, 0.1, 0.1, 0.1],  # Mostly creativity
            "Extraction": [
                0.05,
                0.3,
                0.3,
                0.15,
                0.2,
            ],  # Reasoning + strict format (constraint) + some context
            "Other": [0.25, 0.25, 0.2, 0.15, 0.15],  # Balanced default
        }

        result: Dict[str, Union[List[str], List[float], float]] = {}
        # Round 1: "task_type"
        task_type_logits = logits[0]
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        if isinstance(task_type_results, tuple):
            result["task_type_1"] = task_type_results[0]
            result["task_type_2"] = task_type_results[1]
            result["task_type_prob"] = task_type_results[2]

        # Round 2: "creativity_scope"
        creativity_scope_logits = logits[1]
        target = "creativity_scope"
        scope_results = self.compute_results(creativity_scope_logits, target=target)
        if isinstance(scope_results, list):
            result[target] = scope_results

        # Round 3: "reasoning"
        reasoning_logits = logits[2]
        target = "reasoning"
        reasoning_results = self.compute_results(reasoning_logits, target=target)
        if isinstance(reasoning_results, list):
            result[target] = reasoning_results

        # Round 4: "contextual_knowledge"
        contextual_knowledge_logits = logits[3]
        target = "contextual_knowledge"
        knowledge_results = self.compute_results(
            contextual_knowledge_logits, target=target
        )
        if isinstance(knowledge_results, list):
            result[target] = knowledge_results

        # Round 5: "number_of_few_shots"
        number_of_few_shots_logits = logits[4]
        target = "number_of_few_shots"
        shots_results = self.compute_results(number_of_few_shots_logits, target=target)
        if isinstance(shots_results, list):
            result[target] = shots_results

        # Round 6: "domain_knowledge"
        domain_knowledge_logits = logits[5]
        target = "domain_knowledge"
        domain_results = self.compute_results(domain_knowledge_logits, target=target)
        if isinstance(domain_results, list):
            result[target] = domain_results

        # Round 7: "no_label_reason"
        no_label_reason_logits = logits[6]
        target = "no_label_reason"
        reason_results = self.compute_results(no_label_reason_logits, target=target)
        if isinstance(reason_results, list):
            result[target] = reason_results

        # Round 8: "constraint_ct"
        constraint_ct_logits = logits[7]
        target = "constraint_ct"
        constraint_results = self.compute_results(constraint_ct_logits, target=target)
        if isinstance(constraint_results, list):
            result[target] = constraint_results

        # Get the primary task type
        task_type_1 = result.get("task_type_1", [])
        if not isinstance(task_type_1, list) or not task_type_1:
            return result

        primary_task_type = task_type_1[0]
        if not isinstance(primary_task_type, str):
            return result

        # Use task-specific weights if available, otherwise use default weights
        weights = TASK_TYPE_WEIGHTS.get(primary_task_type, [0.3, 0.3, 0.2, 0.1, 0.1])

        # Ensure all required values are lists of floats
        creativity_scope = result.get("creativity_scope", [])
        reasoning = result.get("reasoning", [])
        constraint_ct = result.get("constraint_ct", [])
        domain_knowledge = result.get("domain_knowledge", [])
        contextual_knowledge = result.get("contextual_knowledge", [])

        if not all(
            isinstance(x, list)
            for x in [
                creativity_scope,
                reasoning,
                constraint_ct,
                domain_knowledge,
                contextual_knowledge,
            ]
        ):
            return result

        # Type assert that these are lists of floats
        creativity_scope = cast(List[float], creativity_scope)
        reasoning = cast(List[float], reasoning)
        constraint_ct = cast(List[float], constraint_ct)
        domain_knowledge = cast(List[float], domain_knowledge)
        contextual_knowledge = cast(List[float], contextual_knowledge)

        # Calculate complexity score using task-specific weights
        result["prompt_complexity_score"] = [
            round(
                weights[0] * creativity_scope[i]
                + weights[1] * reasoning[i]
                + weights[2] * constraint_ct[i]
                + weights[3] * domain_knowledge[i]
                + weights[4] * contextual_knowledge[i],
                5,
            )
            for i in range(len(creativity_scope))
        ]

        return result

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
