"""
Quantized Prompt Task and Complexity Classifier

Main classifier module providing the QuantizedPromptClassifier class
for efficient inference with the quantized ONNX model.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from transformers import AutoConfig, AutoTokenizer

# Conditional import for ONNX Runtime and Optimum
try:
    import onnxruntime as ort

    # F401: ORTModelForSequenceClassification imported but unused; consider using importlib.util.find_spec to test for availability
    from optimum.onnxruntime import ORTModelForSequenceClassification  # noqa: F401

    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


# Type aliases for numpy arrays to improve readability and satisfy mypy
NPInt64Array = np.ndarray[Any, np.dtype[np.int64]]
NPFloatArray = NDArray[np.floating[Any]]


class QuantizedPromptClassifier:
    """
    Quantized ONNX implementation of the prompt task and complexity classifier.

    This class provides an efficient, quantized version of the original model
    optimized for fast CPU inference while maintaining compatibility with
    the original API.
    """

    def __init__(self, model_path: str | Path = "./"):
        """
        Initialize the quantized model.

        Args:
            model_path: Path to the model directory containing the ONNX file
                        and configuration files.

        Raises:
            FileNotFoundError: If required model files are not found.
            ImportError: If required dependencies are not available.
        """
        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "ONNX Runtime and optimum are required. "
                "Install with: pip install optimum[onnxruntime]"
            )

        self.model_path = Path(model_path)

        # Ensure model_quantized.onnx exists
        self._onnx_model_file = self.model_path / "model_quantized.onnx"
        if not self._onnx_model_file.exists():
            raise FileNotFoundError(f"ONNX model not found at {self._onnx_model_file}")

        # Load configuration and tokenizer
        self.config = AutoConfig.from_pretrained(str(self.model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # Extract configuration values (already validated by AutoConfig)
        self.target_names: list[str] = list(
            self.config.target_sizes.keys()
        )  # Ensure order is maintained
        self.task_type_map: dict[str, str] = self.config.task_type_map
        self.weights_map: dict[str, list[float]] = self.config.weights_map
        self.divisor_map: dict[str, float] = self.config.divisor_map

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(str(self._onnx_model_file))
        self._input_names: list[str] = [inp.name for inp in self.session.get_inputs()]

        # Task-specific complexity weights (as provided by the original model)
        self.task_type_weights: dict[str, list[float]] = {
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
        cls, model_path: str | Path, **kwargs: Any
    ) -> "QuantizedPromptClassifier":
        """
        Load a quantized model from a directory or Hugging Face Hub.

        Args:
            model_path: Path to model directory or HF Hub model ID.
            **kwargs: Additional arguments (for compatibility with HF methods, unused).

        Returns:
            Initialized QuantizedPromptClassifier instance.
        """
        # Hugging Face Hub paths are typically directories for local loading
        # The __init__ handles both string and Path
        return cls(model_path=model_path)

    def tokenize_texts(self, texts: list[str]) -> dict[str, NPInt64Array]:
        """
        Tokenize input texts for the model, ensuring correct tensor types.

        Args:
            texts: A list of input strings.

        Returns:
            A dictionary containing 'input_ids' and 'attention_mask'
            as numpy arrays of type int64.
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        return {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

    def run_inference(self, inputs: dict[str, NPInt64Array]) -> list[NPFloatArray]:
        """
        Run ONNX inference and return raw logits.

        Args:
            inputs: A dictionary containing 'input_ids' and 'attention_mask'
                    numpy arrays, prepared by `tokenize_texts`.

        Returns:
            A list of numpy arrays, where each array corresponds to the
            raw logits output from one of the classification heads.
        """
        # Filter inputs to only include what the ONNX model expects
        input_feed = {
            name: inputs[name] for name in self._input_names if name in inputs
        }
        # ONNX Runtime's run method is typed to return List[Any], so we cast it
        outputs = self.session.run(None, input_feed)
        return cast(list[NPFloatArray], outputs)

    def _softmax(self, x: NPFloatArray) -> NPFloatArray:
        """Apply softmax function to a numpy array along the last axis."""
        exp_x: NPFloatArray = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return cast(NPFloatArray, exp_x / np.sum(exp_x, axis=-1, keepdims=True))

    def _compute_task_type_results(
        self, preds: NPFloatArray
    ) -> tuple[list[str], list[str], list[float]]:
        """
        Compute results for the 'task_type' classification head.

        Args:
            preds: Raw logits (numpy array) from the 'task_type' head.

        Returns:
            A tuple containing:
            - List of primary task type strings.
            - List of secondary task type strings (or "NA").
            - List of primary task type probabilities.
        """
        # Get top 2 predictions (indices and probabilities)
        top2_indices = np.argsort(preds, axis=1)[:, -2:][:, ::-1]  # Descending
        softmax_probs = self._softmax(preds)
        top2_probs = np.take_along_axis(softmax_probs, top2_indices, axis=1)

        task_type_1_list: list[str] = []
        task_type_2_list: list[str] = []
        task_type_prob_list: list[float] = []

        for i in range(len(top2_indices)):
            primary_idx, secondary_idx = top2_indices[i]
            primary_prob, secondary_prob = top2_probs[i]

            primary_string = self.task_type_map[str(primary_idx)]
            secondary_string = self.task_type_map[str(secondary_idx)]

            # Filter low confidence second predictions
            if secondary_prob < 0.1:
                secondary_string = "NA"

            task_type_1_list.append(primary_string)
            task_type_2_list.append(secondary_string)
            task_type_prob_list.append(round(float(primary_prob), 3))

        return task_type_1_list, task_type_2_list, task_type_prob_list

    def _compute_score_results(self, preds_softmax: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        num_classes = preds_softmax.shape[1]
        
        # Provide default uniform weights if none are given
        if weights is None:
            weights = np.ones(num_classes) / num_classes
        else:
            weights = np.array(weights)
            if len(weights) != num_classes:
                # If weights don't match, use uniform weights instead of raising error
                print(f"Warning: Shape mismatch in scoring: preds_softmax {preds_softmax.shape}, weights {weights.shape}. Using uniform weights.")
                weights = np.ones(num_classes) / num_classes
        
        score = float(np.dot(preds_softmax, weights).squeeze())
        top_class_index = int(np.argmax(preds_softmax))
        confidence = float(np.max(preds_softmax))
        
        return {
            "task_type_1": self.target_names[top_class_index] if top_class_index < len(self.target_names) else f"unknown_class_{top_class_index}",
            "prompt_complexity_score": round(score, 2),
            "task_type_prob": confidence
        }

    def _post_process_logits(
        self, logits_list: list[NPFloatArray]
    ) -> dict[str, list[Any]]:
        """
        Apply post-processing to raw ONNX logits to get structured results.

        Args:
            logits_list: A list of numpy arrays, where each array corresponds
                         to the raw logits output from one of the classification heads.

        Returns:
            A dictionary where keys are classification names and values are lists
            of scores/results for each sample in the batch.
        """
        batch_results: dict[str, list[Any]] = {}

        # The order of logits_list directly corresponds to the order of target_names
        # defined in the CustomModel's __init__ (which comes from config.target_sizes)
        for i, target_name in enumerate(self.target_names):
            current_logits = logits_list[i]

            if target_name == "task_type":
                task_type_1, task_type_2, task_type_prob = (
                    self._compute_task_type_results(current_logits)
                )
                batch_results["task_type_1"] = task_type_1
                batch_results["task_type_2"] = task_type_2
                batch_results["task_type_prob"] = task_type_prob
            else:
                # Get the appropriate weights for this target
                weights = self.weights_map.get(target_name)
                if weights is None:
                    # If no weights specified, use uniform weights
                    weights = np.ones(current_logits.shape[1]) / current_logits.shape[1]
                
                # Convert weights to numpy array if it isn't already
                weights = np.array(weights)
                
                # Compute scores with the weights
                scores = self._compute_score_results(current_logits, weights)
                batch_results[target_name] = scores

        return batch_results

    def classify_prompts(self, prompts: list[str]) -> list[dict[str, Any]]:
        """
        Classify multiple prompts and return comprehensive results.

        Args:
            prompts: List of text prompts to classify.

        Returns:
            List of classification results, one dictionary per prompt.
        """
        if not prompts:
            return []

        # Tokenize and run inference
        inputs = self.tokenize_texts(prompts)
        logits_list = self.run_inference(inputs)

        # Post-process logits into structured batch results
        batch_results = self._post_process_logits(logits_list)

        # Calculate overall complexity scores using task-specific weights
        complexity_scores: list[float] = []
        for i, task_type in enumerate(batch_results["task_type_1"]):
            weights = self.task_type_weights.get(
                task_type, [0.25, 0.25, 0.2, 0.15, 0.15]
            )  # Default weights if task_type not found

            score = round(
                weights[0] * batch_results["creativity_scope"][i]
                + weights[1] * batch_results["reasoning"][i]
                + weights[2] * batch_results["constraint_ct"][i]
                + weights[3] * batch_results["domain_knowledge"][i]
                + weights[4] * batch_results["contextual_knowledge"][i],
                5,
            )
            complexity_scores.append(score)
        batch_results["prompt_complexity_score"] = complexity_scores

        # Transpose batch results into a list of individual sample results
        individual_results: list[dict[str, Any]] = []
        for i in range(len(prompts)):
            sample_result: dict[str, Any] = {}
            for key, values in batch_results.items():
                if isinstance(values, list) and len(values) > i:
                    sample_result[key] = values[i]
                else:
                    # Should not happen if all results are batched lists
                    sample_result[key] = values
            individual_results.append(sample_result)

        return individual_results

    def classify_single_prompt(self, prompt: str) -> dict[str, Any]:
        """
        Classify a single prompt for convenience.

        Args:
            prompt: Text prompt to classify.

        Returns:
            Classification results for the prompt as a dictionary.
        """
        results = self.classify_prompts([prompt])
        return results[0]

    def get_task_types(self, prompts: list[str]) -> list[str]:
        """
        Get just the primary task types for a list of prompts.

        Args:
            prompts: List of text prompts.

        Returns:
            List of primary task type predictions (strings).
        """
        results = self.classify_prompts(prompts)
        return [result["task_type_1"] for result in results]

    def get_complexity_scores(self, prompts: list[str]) -> list[float]:
        """
        Get just the complexity scores for a list of prompts.

        Args:
            prompts: List of text prompts.

        Returns:
            List of complexity scores (floats).
        """
        results = self.classify_prompts(prompts)
        return [result["prompt_complexity_score"] for result in results]
