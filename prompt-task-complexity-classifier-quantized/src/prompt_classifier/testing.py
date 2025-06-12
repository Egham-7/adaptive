#!/usr/bin/env python3
"""
Test script for the quantized prompt task and complexity classifier model.

This script demonstrates how to use the quantized ONNX model for inference
and provides examples of processing the outputs to get meaningful results.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig

try:
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSequenceClassification
    OPTIMUM_AVAILABLE = True
except ImportError:
    print("Warning: optimum not available. Only direct ONNX Runtime usage will work.")
    OPTIMUM_AVAILABLE = False


class QuantizedPromptClassifier:
    """
    Wrapper class for the quantized prompt task and complexity classifier.

    Provides the same interface as the original CustomModel but optimized
    for the quantized ONNX version.
    """

    def __init__(self, model_path: str = "./"):
        """
        Initialize the quantized model.

        Args:
            model_path: Path to the model directory containing the ONNX file
        """
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


def test_quantized_model():
    """Test the quantized model with example prompts."""

    print("🚀 Testing Quantized Prompt Task and Complexity Classifier")
    print("=" * 60)

    # Initialize classifier
    try:
        classifier = QuantizedPromptClassifier("./")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Write a Python function to implement quicksort algorithm with detailed comments explaining each step",
        "Summarize the main themes in Shakespeare's Hamlet in 3 paragraphs",
        "Create a marketing campaign for a new eco-friendly water bottle targeting millennials",
        "Debug this code and explain what's wrong: print('Hello World'",
        "Explain quantum entanglement in simple terms that a 10-year-old could understand"
    ]

    print(f"\n📝 Testing with {len(test_prompts)} example prompts")
    print("-" * 60)

    # Measure inference time
    start_time = time.time()
    results = classifier.classify_prompts(test_prompts)
    inference_time = time.time() - start_time

    print(f"⚡ Inference completed in {inference_time:.3f} seconds")
    print(f"📊 Average time per prompt: {inference_time/len(test_prompts):.3f} seconds")
    print()

    # Display results
    for i, (prompt, result) in enumerate(zip(test_prompts, results)):
        print(f"📋 Prompt {i+1}:")
        print(f"   Text: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"   Task Type: {result['task_type_1'][0]} (confidence: {result['task_type_prob'][0]:.3f})")
        if result['task_type_2'][0] != "NA":
            print(f"   Secondary: {result['task_type_2'][0]}")
        print(f"   Complexity Score: {result['prompt_complexity_score'][0]:.3f}")
        print(f"   Creativity: {result['creativity_scope'][0]:.2f}")
        print(f"   Reasoning: {result['reasoning'][0]:.2f}")
        print(f"   Domain Knowledge: {result['domain_knowledge'][0]:.2f}")
        print()


def benchmark_performance():
    """Benchmark the quantized model performance."""

    print("⚡ Performance Benchmark")
    print("=" * 40)

    try:
        classifier = QuantizedPromptClassifier("./")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Test with different batch sizes
    test_prompt = "Write a comprehensive analysis of machine learning algorithms and their applications"

    batch_sizes = [1, 5, 10, 20]

    for batch_size in batch_sizes:
        prompts = [test_prompt] * batch_size

        # Warm-up
        classifier.classify_prompts([test_prompt])

        # Benchmark
        times = []
        for _ in range(5):
            start_time = time.time()
            classifier.classify_prompts(prompts)
            times.append(time.time() - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time

        print(f"Batch size {batch_size:2d}: {avg_time:.3f}±{std_time:.3f}s ({throughput:.1f} prompts/sec)")


def validate_outputs():
    """Validate that model outputs are in expected format."""

    print("🔍 Output Validation")
    print("=" * 30)

    try:
        classifier = QuantizedPromptClassifier("./")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    test_prompt = ["Explain how neural networks work"]
    results = classifier.classify_prompts(test_prompt)
    result = results[0]

    # Validate structure
    expected_keys = [
        "task_type_1", "task_type_2", "task_type_prob",
        "creativity_scope", "reasoning", "contextual_knowledge",
        "number_of_few_shots", "domain_knowledge", "no_label_reason",
        "constraint_ct", "prompt_complexity_score"
    ]

    print("✅ Output validation:")
    for key in expected_keys:
        if key in result:
            value = result[key]
            if isinstance(value, list) and len(value) > 0:
                print(f"   {key}: {type(value[0]).__name__} = {value[0]}")
            else:
                print(f"   {key}: {type(value).__name__} = {value}")
        else:
            print(f"   ❌ Missing key: {key}")

    # Validate ranges
    print("\n📏 Value range validation:")

    # Task type should be valid
    task_type = result["task_type_1"][0]
    valid_tasks = list(classifier.task_type_map.values())
    print(f"   Task type '{task_type}' valid: {task_type in valid_tasks}")

    # Scores should be in [0, 1] range
    score_keys = ["creativity_scope", "reasoning", "contextual_knowledge",
                  "domain_knowledge", "constraint_ct"]

    for key in score_keys:
        if key in result:
            score = result[key][0]
            valid_range = 0 <= score <= 1
            print(f"   {key} in [0,1]: {valid_range} (value: {score:.3f})")

    # Complexity score validation
    complexity = result["prompt_complexity_score"][0]
    print(f"   Complexity score: {complexity:.3f}")


if __name__ == "__main__":
    print("🧪 Quantized Model Test Suite")
    print("=" * 50)

    # Run all tests
    test_quantized_model()
    print("\n")
    benchmark_performance()
    print("\n")
    validate_outputs()

    print("\n✨ All tests completed!")
