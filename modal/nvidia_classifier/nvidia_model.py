"""NVIDIA prompt classifier model implementation with complete functionality.

This module contains the complete ML model implementation for the NVIDIA 
prompt-task-and-complexity-classifier, including model architecture,
classification logic, and complexity scoring - extracted from the original
working implementation.
"""

from typing import Any, Dict, List
import numpy as np


def get_model_classes():
    """Get all model classes with proper torch imports (only called inside Modal containers)."""
    import torch
    import torch.nn as nn
    from huggingface_hub import PyTorchModelHubMixin
    from transformers import AutoModel
    
    class MeanPooling(nn.Module):
        """Mean pooling layer for transformer outputs."""
        
        def __init__(self) -> None:
            super().__init__()

        def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """Apply mean pooling to transformer hidden states."""
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings

    class MulticlassHead(nn.Module):
        """Multi-class classification head."""
        
        def __init__(self, input_size: int, num_classes: int) -> None:
            super().__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through classification head."""
            return self.fc(x)

    class CustomModel(nn.Module, PyTorchModelHubMixin):
        """Complete NVIDIA prompt classifier model with all functionality."""
        
        def __init__(self, target_sizes: Dict[str, int], task_type_map: Dict[str, str], 
                     weights_map: Dict[str, List[float]], divisor_map: Dict[str, float]) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base", use_safetensors=True)
            self.target_sizes = list(target_sizes.values())
            self.task_type_map = task_type_map
            self.weights_map = weights_map
            self.divisor_map = divisor_map
            
            self.heads = nn.ModuleList([
                MulticlassHead(self.backbone.config.hidden_size, sz) for sz in self.target_sizes
            ])
            self.pool = MeanPooling()

        def compute_results(self, preds: torch.Tensor, target: str, decimal: int = 4):
            """Compute classification results for different target types."""
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
                    int_scores = [max(0, round(x)) for x in scores]
                    return int_scores
                return scores

        def _extract_classification_results(self, logits: List[torch.Tensor]) -> Dict[str, List]:
            """Extract classification results from model logits."""
            result = {}
            task_type_logits = logits[0]
            task_type_results = self.compute_results(task_type_logits, target="task_type")
            if isinstance(task_type_results, tuple):
                result["task_type_1"] = task_type_results[0]
                result["task_type_2"] = task_type_results[1]
                result["task_type_prob"] = task_type_results[2]

            classifications = [
                ("creativity_scope", logits[1]), ("reasoning", logits[2]),
                ("contextual_knowledge", logits[3]), ("number_of_few_shots", logits[4]),
                ("domain_knowledge", logits[5]), ("no_label_reason", logits[6]),
                ("constraint_ct", logits[7]),
            ]

            for target, target_logits in classifications:
                target_results = self.compute_results(target_logits, target=target)
                if isinstance(target_results, list):
                    result[target] = target_results

            return result

        def _calculate_complexity_scores(self, results: Dict[str, List], task_types: List[str]) -> List[float]:
            """Calculate complexity scores based on task type and metrics."""
            task_type_weights = {
                "Open QA": [0.2, 0.3, 0.15, 0.2, 0.15], "Closed QA": [0.1, 0.35, 0.2, 0.25, 0.1],
                "Summarization": [0.2, 0.25, 0.25, 0.1, 0.2], "Text Generation": [0.4, 0.2, 0.15, 0.1, 0.15],
                "Code Generation": [0.1, 0.3, 0.2, 0.3, 0.1], "Chatbot": [0.25, 0.25, 0.15, 0.1, 0.25],
                "Classification": [0.1, 0.35, 0.25, 0.2, 0.1], "Rewrite": [0.2, 0.2, 0.3, 0.1, 0.2],
                "Brainstorming": [0.5, 0.2, 0.1, 0.1, 0.1], "Extraction": [0.05, 0.3, 0.3, 0.15, 0.2],
                "Other": [0.25, 0.25, 0.2, 0.15, 0.15],
            }

            complexity_scores = []
            for i, task_type in enumerate(task_types):
                weights = task_type_weights.get(task_type, [0.3, 0.3, 0.2, 0.1, 0.1])
                score = round(
                    weights[0] * results.get("creativity_scope", [])[i] +
                    weights[1] * results.get("reasoning", [])[i] +
                    weights[2] * results.get("constraint_ct", [])[i] +
                    weights[3] * results.get("domain_knowledge", [])[i] +
                    weights[4] * results.get("contextual_knowledge", [])[i], 5
                )
                complexity_scores.append(score)
            return complexity_scores

        def process_logits(self, logits: List[torch.Tensor]) -> Dict[str, List]:
            """Process model logits into final classification results."""
            batch_results = self._extract_classification_results(logits)
            if "task_type_1" in batch_results:
                task_types = batch_results["task_type_1"]
                complexity_scores = self._calculate_complexity_scores(batch_results, task_types)
                batch_results["prompt_complexity_score"] = complexity_scores
            return batch_results

        def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List]:
            """Complete forward pass with classification and complexity scoring."""
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            mean_pooled_representation = self.pool(last_hidden_state, attention_mask)
            logits = [head(mean_pooled_representation) for head in self.heads]
            return self.process_logits(logits)
    
    return MeanPooling, MulticlassHead, CustomModel


# Export for use in deployment
__all__ = ["get_model_classes"]