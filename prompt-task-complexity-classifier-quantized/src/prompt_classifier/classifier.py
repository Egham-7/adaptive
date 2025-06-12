"""
Quantized Prompt Task and Complexity Classifier

Main classifier module providing the QuantizedPromptClassifier class
for efficient inference with the quantized ONNX model.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig

try:
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSequenceClassification
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


class QuantizedPromptClassifier:
    """
    Quantized ONNX implementation of the prompt task and complexity classifier.

    This class provides an efficient, quantized version of the original model
    optimized for fast CPU inference while maintaining compatibility with
    the original API.

    Attributes:
        model_path: Path to the model directory
        config: Model configuration
        tokenizer: Tokenizer for text preprocessing
        session: ONNX Runtime inference session
        target_sizes: Output dimensions for each classification head
        task_type_map: Mapping from indices to task type names
        weights_map: Weights for computing weighted scores
        divisor_map: Divisors for normalizing scores
        task_type_weights: Task-specific weights for complexity calculation
    """

    def __init__(self, model_path: Union[str, Path] = "./"):
        """
        Initialize the quantized model.

        Args:
            model_path: Path to the model directory containing the ONNX file
                       and configuration files

        Raises:
            FileNotFoundError: If required model files are not found
            ImportError: If required dependencies are not available
        """
        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "ONNX Runtime and optimum are required. "
                "Install with: pip install optimum[onnxruntime]"
            )

        self.model_path = Path(model_path)

        # Load configuration
        self.config = AutoConfig.from_pretrained(str(self.model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # Extract configuration values
        self.target_sizes = self.config.target_sizes
        self.task_type_map = self.config.task_type_map
        self.weights_map = self.config.weights_map
        self.divisor_map = self.config.divisor_map

        # Load ONNX model
        onnx_path = self.model_path / "model_quantized.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        self.session = ort.InferenceSession(str(onnx_path))

        # Task-specific complexity weights
        self.task_type_weights = {
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

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        **kwargs
    ) -> "QuantizedPromptClassifier":
        """
        Load a quantized model from a directory or Hugging Face Hub.

        Args:
            model_path: Path to model directory or HF Hub model ID
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Initialized QuantizedPromptClassifier instance
        """
        return cls(model_path=model_path)

    def tokenize_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Tokenize input texts for the model."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        return {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

    def run_inference(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run ONNX inference and return raw logits."""
        outputs = self.session.run(None, inputs)
        return outputs

    def compute_results(self, preds: np.ndarray, target: str, decimal: int = 4) -> Any:
        """
        Compute results for a specific classification target.

        Args:
            preds: Raw logits from the model
            target: Target classification type
            decimal: Decimal places for rounding

        Returns:
            Processed results based on target type
        """
        if target == "task_type":
            # Get top 2 predictions
            top2_indices = np.argsort(preds, axis=1)[:, -2:][:, ::-1]  # Descending order
            softmax_probs = self._softmax(preds)
            top2_probs = np.take_along_axis(softmax_probs, top2_indices, axis=1)

            # Convert to strings and probabilities
            top2_strings = []
            top2_prob_rounded = []

            for i in range(len(top2_indices)):
                sample_strings = [self.task_type_map[str(idx)] for idx in top2_indices[i]]
                sample_probs = [round(prob, 3) for prob in top2_probs[i]]

                # Filter low confidence second predictions
                if sample_probs[1] < 0.1:
                    sample_strings[1] = "NA"

                top2_strings.append(sample_strings)
                top2_prob_rounded.append(sample_probs)

            task_type_1 = [strings[0] for strings in top2_strings]
            task_type_2 = [strings[1] for strings in top2_strings]
            task_type_prob = [probs[0] for probs in top2_prob_rounded]

            return task_type_1, task_type_2, task_type_prob
        else:
            # For other targets, compute weighted scores
            preds_softmax = self._softmax(preds)
            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(preds_softmax * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]
            scores = [round(score, decimal) for score in scores]

            if target == "number_of_few_shots":
                scores = [max(0, x) if x >= 0.05 else 0 for x in scores]

            return scores

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to numpy array."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def calculate_complexity_scores(self, results: Dict[str, Any], task_types: List[str]) -> List[float]:
        """Calculate overall complexity scores using task-specific weights."""
        complexity_scores = []

        for i, task_type in enumerate(task_types):
            weights = self.task_type_weights.get(task_type, [0.25, 0.25, 0.2, 0.15, 0.15])

            score = round(
                weights[0] * results["creativity_scope"][i] +
                weights[1] * results["reasoning"][i] +
                weights[2] * results["constraint_ct"][i] +
                weights[3] * results["domain_knowledge"][i] +
                weights[4] * results["contextual_knowledge"][i],
                5
            )
            complexity_scores.append(score)

        return complexity_scores

    def classify_prompts(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple prompts and return comprehensive results.

        Args:
            prompts: List of text prompts to classify

        Returns:
            List of classification results, one per prompt
        """
        # Tokenize inputs
        inputs = self.tokenize_texts(prompts)

        # Run inference
        logits_list = self.run_inference(inputs)

        # Process each classification head
        batch_results = {}

        # Task type classification
        task_type_results = self.compute_results(logits_list[0], "task_type")
        batch_results["task_type_1"] = task_type_results[0]
        batch_results["task_type_2"] = task_type_results[1]
        batch_results["task_type_prob"] = task_type_results[2]

        # Other classifications
        classifications = [
            ("creativity_scope", logits_list[1]),
            ("reasoning", logits_list[2]),
            ("contextual_knowledge", logits_list[3]),
            ("number_of_few_shots", logits_list[4]),
            ("domain_knowledge", logits_list[5]),
            ("no_label_reason", logits_list[6]),
            ("constraint_ct", logits_list[7]),
        ]

        for target, target_logits in classifications:
            batch_results[target] = self.compute_results(target_logits, target)

        # Calculate complexity scores
        complexity_scores = self.calculate_complexity_scores(
            batch_results, batch_results["task_type_1"]
        )
        batch_results["prompt_complexity_score"] = complexity_scores

        # Convert batch results to individual sample results
        individual_results = []
        batch_size = len(prompts)

        for i in range(batch_size):
            sample_result = {}
            for key, value in batch_results.items():
                if isinstance(value, list) and len(value) > i:
                    if isinstance(value[i], str):
                        sample_result[key] = [value[i]]
                    else:
                        sample_result[key] = [float(value[i])]
                else:
                    sample_result[key] = value
            individual_results.append(sample_result)

        return individual_results

    def classify_single_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Classify a single prompt for convenience.

        Args:
            prompt: Text prompt to classify

        Returns:
            Classification results for the prompt
        """
        results = self.classify_prompts([prompt])
        return results[0]

    def get_task_types(self, prompts: List[str]) -> List[str]:
        """
        Get just the primary task types for a list of prompts.

        Args:
            prompts: List of text prompts

        Returns:
            List of primary task type predictions
        """
        results = self.classify_prompts(prompts)
        return [result["task_type_1"][0] for result in results]

    def get_complexity_scores(self, prompts: List[str]) -> List[float]:
        """
        Get just the complexity scores for a list of prompts.

        Args:
            prompts: List of text prompts

        Returns:
            List of complexity scores
        """
        results = self.classify_prompts(prompts)
        return [result["prompt_complexity_score"][0] for result in results]


# Testing and validation functions moved to separate testing module
