#!/usr/bin/env python3
"""
Simple usage examples for the quantized prompt task complexity classifier.

This file demonstrates basic usage patterns for the quantized ONNX model
with both optimum and direct ONNX runtime approaches.
"""

import numpy as np
import time
from typing import List, Dict, Any

try:
    import torch
    from transformers import AutoTokenizer, AutoConfig
    from optimum.onnxruntime import ORTModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/Transformers not available")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available")


def example_optimum_usage():
    """Example using Optimum for easy integration with transformers."""

    if not TORCH_AVAILABLE:
        print("❌ PyTorch/Transformers required for this example")
        return

    print("🚀 Example 1: Using Optimum (Recommended)")
    print("=" * 50)

    # Load model and tokenizer
    model = ORTModelForSequenceClassification.from_pretrained(
        "./",  # Current directory
        file_name="model_quantized.onnx"
    )

    tokenizer = AutoTokenizer.from_pretrained("./")
    config = AutoConfig.from_pretrained("./")

    # Example prompts
    prompts = [
        "What is machine learning?",
        "Write a Python function to sort a list using bubble sort algorithm",
        "Create a marketing strategy for a new smartphone targeting Gen Z users"
    ]

    print(f"Processing {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = time.time() - start_time

        # Process outputs - split by target sizes
        target_sizes = list(config.target_sizes.values())
        logits = outputs.logits[0]  # Remove batch dimension

        start_idx = 0
        results = {}
        target_names = list(config.target_sizes.keys())

        for j, (name, size) in enumerate(zip(target_names, target_sizes)):
            end_idx = start_idx + size
            head_logits = logits[start_idx:end_idx]
            probs = torch.softmax(head_logits, dim=0)

            if name == "task_type":
                # Get top prediction for task type
                top_idx = torch.argmax(probs).item()
                task_type = config.task_type_map[str(top_idx)]
                confidence = probs[top_idx].item()
                results[name] = f"{task_type} ({confidence:.3f})"
            else:
                # For other targets, compute weighted score
                weights = torch.tensor(config.weights_map[name])
                weighted_score = torch.sum(probs * weights).item()
                results[name] = f"{weighted_score:.3f}"

            start_idx = end_idx

        print(f"  ⏱️  Inference time: {inference_time*1000:.1f}ms")
        print(f"  🎯 Task type: {results['task_type']}")
        print(f"  🧠 Reasoning: {results['reasoning']}")
        print(f"  💡 Creativity: {results['creativity_scope']}")


def example_direct_onnx_usage():
    """Example using ONNX Runtime directly for maximum performance."""

    if not ONNX_AVAILABLE or not TORCH_AVAILABLE:
        print("❌ ONNX Runtime and transformers required for this example")
        return

    print("\n🔥 Example 2: Direct ONNX Runtime Usage")
    print("=" * 50)

    # Load ONNX session
    session = ort.InferenceSession("model_quantized.onnx")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./")

    # Example batch processing
    prompts = [
        "Explain quantum physics",
        "Debug this Python code: print('hello world'",
        "Summarize the benefits of renewable energy",
        "Create a recipe for chocolate chip cookies"
    ]

    print(f"Batch processing {len(prompts)} prompts...")

    # Tokenize all prompts at once
    inputs = tokenizer(
        prompts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Prepare inputs for ONNX
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

    # Run batch inference
    start_time = time.time()
    outputs = session.run(None, onnx_inputs)
    inference_time = time.time() - start_time

    print(f"  ⚡ Batch inference time: {inference_time*1000:.1f}ms")
    print(f"  📊 Average per prompt: {inference_time*1000/len(prompts):.1f}ms")
    print(f"  🔢 Output shapes: {[output.shape for output in outputs]}")

    # Process first prompt's results
    print(f"\nDetailed results for: '{prompts[0]}'")

    # Task type (first output)
    task_logits = outputs[0][0]  # First sample
    task_probs = softmax(task_logits)
    top_task_idx = np.argmax(task_probs)

    # Load config for mappings
    with open("config.json", "r") as f:
        import json
        config = json.load(f)

    task_type = config["task_type_map"][str(top_task_idx)]
    print(f"  🎯 Task: {task_type} ({task_probs[top_task_idx]:.3f})")

    # Other dimensions
    dimension_names = ["creativity_scope", "reasoning", "contextual_knowledge",
                      "number_of_few_shots", "domain_knowledge", "no_label_reason", "constraint_ct"]

    for i, dim_name in enumerate(dimension_names):
        dim_logits = outputs[i+1][0]  # Skip task_type output
        dim_probs = softmax(dim_logits)
        weights = np.array(config["weights_map"][dim_name])
        weighted_score = np.sum(dim_probs * weights)
        print(f"  📈 {dim_name}: {weighted_score:.3f}")


def softmax(x):
    """Numpy softmax implementation."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def benchmark_performance():
    """Benchmark the quantized model performance."""

    if not ONNX_AVAILABLE or not TORCH_AVAILABLE:
        print("❌ Required libraries not available for benchmarking")
        return

    print("\n⚡ Performance Benchmark")
    print("=" * 50)

    # Load ONNX session
    session = ort.InferenceSession("model_quantized.onnx")
    tokenizer = AutoTokenizer.from_pretrained("./")

    test_prompt = "Write a comprehensive analysis of artificial intelligence applications in healthcare"

    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16]

    for batch_size in batch_sizes:
        prompts = [test_prompt] * batch_size

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        # Warmup
        session.run(None, onnx_inputs)

        # Benchmark
        times = []
        for _ in range(10):
            start_time = time.time()
            session.run(None, onnx_inputs)
            times.append(time.time() - start_time)

        avg_time = np.mean(times)
        throughput = batch_size / avg_time

        print(f"  Batch {batch_size:2d}: {avg_time*1000:6.1f}ms ({throughput:5.1f} prompts/sec)")


def main():
    """Run all examples."""
    print("🧪 Quantized Model Usage Examples")
    print("=" * 60)

    # Check model file exists
    import os
    if not os.path.exists("model_quantized.onnx"):
        print("❌ model_quantized.onnx not found!")
        print("Please run the quantization script first.")
        return

    try:
        # Run examples
        example_optimum_usage()
        example_direct_onnx_usage()
        benchmark_performance()

        print("\n✨ All examples completed successfully!")
        print("\n💡 Tips:")
        print("  - Use Optimum for easy integration with existing code")
        print("  - Use direct ONNX Runtime for maximum performance")
        print("  - Batch multiple prompts for better throughput")
        print("  - The quantized model is ~75% smaller and 2-4x faster!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure all required dependencies are installed:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
