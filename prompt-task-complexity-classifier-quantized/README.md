# Prompt Task Complexity Classifier - Quantized

High-performance, statically quantized ONNX implementation of NVIDIA's prompt task and complexity classifier optimized for fast CPU inference.

## Overview

A standalone Python package providing a statically quantized version of the [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) with 76% size reduction and 3-5x speed improvement while maintaining accuracy.

### Key Features

- **Fast Inference**: 3-5x faster than original model on CPU
- **Compact Size**: 76% smaller model footprint using INT8 precision
- **Comprehensive Analysis**: 8 classification dimensions + complexity scoring
- **Easy Integration**: Drop-in replacement with familiar API
- **Production Ready**: Optimized for server deployment and batch processing

## What This Model Does

The quantized classifier analyzes text prompts across **8 key dimensions**:

| Dimension | Description | Classes |
|-----------|-------------|---------|
| **Task Type** | Primary task category | 11 types (QA, Generation, Summarization, etc.) |
| **Creativity Scope** | Creative thinking requirements | 5 levels (0.0 - 1.0) |
| **Reasoning** | Logical reasoning complexity | 5 levels (0.0 - 1.0) |
| **Contextual Knowledge** | Context understanding needs | 5 levels (0.0 - 1.0) |
| **Few-shot Learning** | Examples needed | 5 levels (0-4+ shots) |
| **Domain Knowledge** | Specialized expertise required | 5 levels (0.0 - 1.0) |
| **Label Reasoning** | Classification reasoning needs | 5 levels (0.0 - 1.0) |
| **Constraint Handling** | Rule/constraint complexity | 5 levels (0.0 - 1.0) |

Plus a **task-weighted complexity score** that combines all dimensions.

## Quick Start

### Installation

```bash
# Install the package with Poetry
cd prompt-task-complexity-classifier-quantized
poetry install

# Or install dependencies directly
pip install torch transformers onnxruntime optimum[onnxruntime] huggingface-hub numpy
```

### Basic Usage

```python
from prompt_classifier import QuantizedPromptClassifier

# Load the quantized model
classifier = QuantizedPromptClassifier.from_pretrained("./")

# Classify a single prompt
result = classifier.classify_single_prompt(
    "Write a Python function to implement quicksort with detailed comments"
)

print(f"Task: {result['task_type_1'][0]}")           # "Code Generation"
print(f"Complexity: {result['prompt_complexity_score'][0]:.3f}")  # 0.652
print(f"Reasoning: {result['reasoning'][0]:.3f}")    # 0.750
```

### Batch Processing

```python
# Process multiple prompts efficiently
prompts = [
    "What is the capital of France?",
    "Explain quantum computing and write simulation code",
    "Create a marketing strategy for eco-friendly products"
]

results = classifier.classify_prompts(prompts)

for prompt, result in zip(prompts, results):
    task_type = result['task_type_1'][0]
    complexity = result['prompt_complexity_score'][0]
    print(f"{task_type}: {complexity:.3f} - {prompt[:50]}...")
```

### Command Line Interface

```bash
# Quantize the original model
prompt-classifier quantize --output-dir ./my_quantized_model

# Classify prompts from command line
prompt-classifier classify "Explain machine learning" "Write a sorting algorithm"

# Upload to Hugging Face Hub
prompt-classifier upload your-username/my-quantized-model --private
```

## Performance Benchmarks

| Metric | Original Model | Quantized Model | Improvement |
|--------|---------------|-----------------|-------------|
| **Model Size** | 734 MB | 178 MB | 76% smaller |
| **Inference Speed** | 45ms/prompt | 9ms/prompt | 5x faster |
| **Memory Usage** | ~1.2 GB | ~280 MB | 77% reduction |
| **Accuracy** | Baseline | -0.8% typical | Minimal loss |

*Static quantization provides better performance than dynamic quantization.*

## Advanced Usage

### Custom Model Path

```python
# Load from custom directory
classifier = QuantizedPromptClassifier.from_pretrained("/path/to/model")

# Load from Hugging Face Hub
classifier = QuantizedPromptClassifier.from_pretrained("username/model-name")
```

### Direct ONNX Runtime Usage

```python
import onnxruntime as ort
from transformers import AutoTokenizer

# For maximum performance
session = ort.InferenceSession("model_quantized.onnx")
tokenizer = AutoTokenizer.from_pretrained("./")

# Run inference directly
inputs = tokenizer("Your prompt", return_tensors="np", padding=True, truncation=True)
outputs = session.run(None, {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64)
})
```

## Development

### Setup

```bash
# Install with development dependencies  
poetry install --with dev

# Activate environment
poetry shell
```

### Quantize Your Own Model

```bash
# Run static quantization with calibration data
python -m prompt_classifier.scripts.quantization \
    --model-id nvidia/prompt-task-and-complexity-classifier \
    --output-dir ./quantized_output \
    --num_calibration_samples 5000
```

### Testing

```bash
# Run comprehensive tests
python -m prompt_classifier.testing

# Or use pytest
pytest tests/ -v --cov=prompt_classifier
```

## API Reference

### `QuantizedPromptClassifier`

Main class for prompt classification with quantized ONNX backend.

#### Methods

- `from_pretrained(model_path)` - Load model from directory or HF Hub
- `classify_prompts(prompts: List[str])` - Classify multiple prompts  
- `classify_single_prompt(prompt: str)` - Classify one prompt
- `get_task_types(prompts: List[str])` - Get just task types
- `get_complexity_scores(prompts: List[str])` - Get just complexity scores

## Requirements

- Python 3.9+
- PyTorch 1.9+
- Transformers 4.21+
- ONNX Runtime 1.12+
- Optimum 1.12+
- NumPy 1.21+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes and add tests
4. Run tests and linting
5. Submit a Pull Request

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NVIDIA** for the original prompt task and complexity classifier
- **Microsoft** for ONNX Runtime quantization framework
- **Hugging Face** for Optimum and Transformers libraries