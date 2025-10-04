"""NVIDIA Prompt Task Complexity Classifier - Complete ML Implementation.

This module contains the complete NVIDIA prompt-task-and-complexity-classifier
implementation, including model architecture, inference logic, and classification.

The classifier analyzes prompts and provides:
- Task type classification (primary and secondary)
- Complexity scoring across multiple dimensions
- GPU-accelerated inference with PyTorch

Architecture:
- Base model: microsoft/DeBERTa-v3-base
- Custom classification heads for multi-task learning
- Mean pooling for sequence representation
- NVIDIA-specific complexity scoring formula
"""

import logging
import traceback
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ============================================================================
# Model Architecture Components
# ============================================================================


class MeanPooling(nn.Module):
    """Mean pooling layer for transformer outputs.

    Applies attention-weighted mean pooling over the sequence dimension
    of transformer hidden states.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling to transformer hidden states.

        Args:
            last_hidden_state: Transformer hidden states [batch, seq_len, hidden_size]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled embeddings [batch, hidden_size]
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MulticlassHead(nn.Module):
    """Multi-class classification head.

    Simple linear layer for classification tasks.
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head."""
        return self.fc(x)  # type: ignore[no-any-return]


class CustomModel(nn.Module, PyTorchModelHubMixin):
    """Complete NVIDIA prompt classifier model with all functionality.

    Multi-task classification model that predicts:
    - Task type (primary and secondary)
    - Creativity scope
    - Reasoning complexity
    - Contextual knowledge requirements
    - Domain knowledge requirements
    - Few-shot learning needs
    - Classification confidence
    - Constraint complexity
    - Overall prompt complexity score (computed)
    """

    def __init__(
        self,
        target_sizes: Dict[str, int],
        task_type_map: Dict[str, str],
        weights_map: Dict[str, List[float]],
        divisor_map: Dict[str, float],
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/DeBERTa-v3-base", use_safetensors=True
        )
        self.target_sizes = list(target_sizes.values())
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map

        self.heads = nn.ModuleList(
            [
                MulticlassHead(self.backbone.config.hidden_size, sz)
                for sz in self.target_sizes
            ]
        )
        self.pool = MeanPooling()

    def compute_results(
        self, preds: torch.Tensor, target: str, decimal: int = 4
    ) -> Union[Tuple[List[str], List[str], List[float]], List[float]]:
        """Compute classification results for different target types.

        Args:
            preds: Model predictions [batch, num_classes]
            target: Target type (task_type or metric name)
            decimal: Number of decimal places for rounding

        Returns:
            For task_type: Tuple of (primary_types, secondary_types, probabilities)
            For other targets: List of weighted scores
        """
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
            scores_list: List[float] = [
                round(float(value), decimal) for value in scores
            ]

            if target == "number_of_few_shots":
                scores_list = [float(x) if x >= 0.05 else 0.0 for x in scores_list]
            return scores_list

    def _extract_classification_results(
        self, logits: List[torch.Tensor]
    ) -> Dict[str, Union[List[str], List[float]]]:
        """Extract classification results from model logits.

        Args:
            logits: List of logit tensors from classification heads

        Returns:
            Dictionary mapping metric names to classification results
        """
        result: Dict[str, Union[List[str], List[float]]] = {}
        task_type_logits = logits[0]
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        if isinstance(task_type_results, tuple):
            result["task_type_1"] = task_type_results[0]
            result["task_type_2"] = task_type_results[1]
            result["task_type_prob"] = task_type_results[2]

        classifications = [
            ("creativity_scope", logits[1]),
            ("reasoning", logits[2]),
            ("contextual_knowledge", logits[3]),
            ("number_of_few_shots", logits[4]),
            ("domain_knowledge", logits[5]),
            ("no_label_reason", logits[6]),
            ("constraint_ct", logits[7]),
        ]

        for target, target_logits in classifications:
            target_results = self.compute_results(target_logits, target=target)
            if isinstance(target_results, list):
                result[target] = target_results  # type: ignore[assignment]

        return result

    def _calculate_complexity_scores(
        self,
        results: Dict[str, Union[List[str], List[float]]],
        task_types: List[str],
    ) -> List[float]:
        """Calculate complexity scores using NVIDIA's official formula.

        Formula: 0.35*Creativity + 0.25*Reasoning + 0.15*Constraint +
                 0.15*DomainKnowledge + 0.05*ContextualKnowledge + 0.05*FewShots

        Args:
            results: Dictionary of classification results
            task_types: List of task types (for length reference)

        Returns:
            List of complexity scores [0.0-1.0]
        """
        complexity_scores = []
        for i in range(len(task_types)):
            # Get float values from results, with type checking
            creativity = results.get("creativity_scope", [])
            reasoning = results.get("reasoning", [])
            constraint = results.get("constraint_ct", [])
            domain = results.get("domain_knowledge", [])
            contextual = results.get("contextual_knowledge", [])
            few_shots = results.get("number_of_few_shots", [])

            # Extract float values with type assertions
            creativity_val = (
                float(creativity[i])
                if isinstance(creativity, list) and len(creativity) > i
                else 0.0
            )
            reasoning_val = (
                float(reasoning[i])
                if isinstance(reasoning, list) and len(reasoning) > i
                else 0.0
            )
            constraint_val = (
                float(constraint[i])
                if isinstance(constraint, list) and len(constraint) > i
                else 0.0
            )
            domain_val = (
                float(domain[i])
                if isinstance(domain, list) and len(domain) > i
                else 0.0
            )
            contextual_val = (
                float(contextual[i])
                if isinstance(contextual, list) and len(contextual) > i
                else 0.0
            )
            few_shots_val = (
                float(few_shots[i])
                if isinstance(few_shots, list) and len(few_shots) > i
                else 0.0
            )

            # NVIDIA official formula
            score = round(
                0.35 * creativity_val
                + 0.25 * reasoning_val
                + 0.15 * constraint_val
                + 0.15 * domain_val
                + 0.05 * contextual_val
                + 0.05 * few_shots_val,
                5,
            )
            complexity_scores.append(score)
        return complexity_scores

    def process_logits(
        self, logits: List[torch.Tensor]
    ) -> Dict[str, Union[List[str], List[float]]]:
        """Process model logits into final classification results.

        Args:
            logits: List of logit tensors from classification heads

        Returns:
            Dictionary with all classification results including complexity scores
        """
        batch_results = self._extract_classification_results(logits)
        if "task_type_1" in batch_results:
            task_types_raw = batch_results["task_type_1"]
            if isinstance(task_types_raw, list) and all(
                isinstance(t, str) for t in task_types_raw
            ):
                task_types: List[str] = task_types_raw  # type: ignore[assignment]
                complexity_scores = self._calculate_complexity_scores(
                    batch_results, task_types
                )
                batch_results["prompt_complexity_score"] = complexity_scores
        return batch_results

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[List[str], List[float]]]:
        """Complete forward pass with classification and complexity scoring.

        Args:
            batch: Dictionary with input_ids and attention_mask tensors

        Returns:
            Dictionary with all classification results
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)
        logits = [head(mean_pooled_representation) for head in self.heads]
        return self.process_logits(logits)


# ============================================================================
# Inference Service
# ============================================================================


class PromptClassifier:
    """Prompt task complexity classifier service for ML inference.

    This service loads the NVIDIA classifier model and provides inference
    methods for analyzing prompts. Supports GPU acceleration when available.

    The classifier analyzes prompts across multiple dimensions:
    - Task type classification
    - Complexity scoring
    - Domain knowledge requirements
    - Reasoning and creativity needs
    """

    def __init__(
        self, model_name: str = "nvidia/prompt-task-and-complexity-classifier"
    ) -> None:
        """Initialize the classifier with model loading.

        Args:
            model_name: HuggingFace model identifier
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.torch = torch

        # Load model configuration
        config = AutoConfig.from_pretrained(model_name)

        self.logger.info(f"Loading model: {model_name}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
        else:
            self.logger.info("GPU: CPU")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model and create custom model
        self.model = CustomModel(
            target_sizes=config.target_sizes,
            task_type_map=config.task_type_map,
            weights_map=config.weights_map,
            divisor_map=config.divisor_map,
        ).from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.logger.info("Model loaded on GPU")
        else:
            self.logger.info("Model loaded on CPU")

        self.model.eval()
        self.logger.info("Model ready for inference")

    def classify_prompt(self, prompt: str) -> Dict[str, Any]:
        """Classify a single prompt.

        Args:
            prompt: Text prompt to classify

        Returns:
            Dictionary with classification results including:
            - task_type_1, task_type_2: Primary and secondary task types
            - task_type_prob: Confidence score
            - creativity_scope, reasoning, contextual_knowledge, etc.: Metric scores
            - prompt_complexity_score: Overall complexity

        Raises:
            RuntimeError: If inference fails
        """
        # Tokenize
        encoded = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to GPU if available
        if self.torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Run inference
        with self.torch.no_grad():
            try:
                result = self.model(encoded)
                # Extract single result from batch
                return {
                    key: values[0] if isinstance(values, list) else values
                    for key, values in result.items()
                }
            except Exception as e:
                # Log the full error with context and stacktrace
                self.logger.error(
                    f"Model inference failed for prompt (length: {len(prompt)}): {str(e)}"
                )
                self.logger.error(f"Full stacktrace: {traceback.format_exc()}")

                raise RuntimeError(f"Model inference failed: {str(e)}") from e
