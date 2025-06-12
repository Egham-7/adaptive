# Adaptive AI Service

Python ML service that intelligently selects optimal LLM models based on prompt analysis.

## Features

- **Prompt Classification**: Multi-dimensional prompt analysis (creativity, reasoning, context, domain)
- **Model Selection**: Vector similarity matching to find optimal models
- **Domain Classification**: Specialized routing for different content domains
- **Parameter Optimization**: Automatic tuning of model parameters
- **High Performance**: LitServe for fast inference serving

## Quick Start

```bash
# Install dependencies
poetry install

# Set environment variables
cp .env.example .env.local

# Run service
poetry run python main.py
```

## Environment Variables

```bash
# Optional: for enhanced features
OPENAI_API_KEY=sk-xxxxx
HUGGINGFACE_TOKEN=hf_xxxxx
```

## API

### Model Selection

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "prompt": "Write a Python function to sort a list"
}
```

**Response:**
```json
{
  "selected_model": "gpt-4o",
  "provider": "openai",
  "match_score": 0.94,
  "domain": "programming",
  "prompt_scores": {
    "creativity_scope": [0.3],
    "reasoning": [0.8],
    "contextual_knowledge": [0.6],
    "domain_knowledge": [0.9]
  }
}
```

## How It Works

1. **Prompt Analysis**: Uses NVIDIA's prompt classifier to extract complexity dimensions
2. **Domain Detection**: Classifies prompt into specialized domains (code, writing, analysis, etc.)
3. **Vector Matching**: Cosine similarity between prompt vector and model capability vectors
4. **Model Selection**: Returns best matching model with confidence score

## Project Structure

```
services/
├── model_selector.py      # Main selection logic
├── prompt_classifier.py   # Prompt complexity analysis
├── domain_classifier.py   # Domain classification
└── llm_parameters.py     # Parameter optimization

models/
├── llms.py               # Model definitions and capabilities
└── domain_mappings.py    # Domain-to-model mappings

core/
└── utils.py              # Utility functions
```

## Adding New Models

1. Define model capabilities in `models/llms.py`:

```python
model_capabilities = {
    "new-model": {
        "provider": "provider-name",
        "capability_vector": [0.8, 0.9, 0.7, 0.8],  # [creativity, reasoning, context, domain]
        "cost_per_token": 0.00001,
        "max_tokens": 4096
    }
}
```

2. Update domain mappings in `models/domain_mappings.py`

## Model Quantization (ONNX)

This section describes how to convert the internal prompt classifier model to ONNX format and quantize it for potentially faster inference and reduced model size. The service can then be configured to use this quantized ONNX model.

### Generating a Quantized ONNX Model

The `scripts/quantize_model.py` script is provided to export the `CustomModel` (used for prompt classification) to the ONNX format and then apply dynamic quantization.

**Prerequisites:**

- A Python environment with all project dependencies installed. Ensure you have run:
  ```bash
  poetry install
  ```
- For GPU-accelerated quantization and ONNX runtime, ensure you have the necessary CUDA toolkit installed and that `optimum[onnxruntime-gpu]` is included in your dependencies. If it's not part of the default install (check `pyproject.toml`), you might need to install it via an extras group if defined, or add it to the main dependencies:
  ```bash
  # Example if 'onnxruntime-gpu' was an extra group (not currently the case)
  # poetry install --extras "onnxruntime-gpu"
  # Or, ensure optimum = {extras = ["onnxruntime-gpu"], version = "..."} is in pyproject.toml
  ```
  Currently, `optimum = {extras = ["onnxruntime-gpu"], version = "^1.19.0"}` should be in the main dependencies in `pyproject.toml` for GPU support.

**Script Usage:**

The script `scripts/quantize_model.py` accepts the following command-line arguments:

- `model_id_or_path`: (Required) Hugging Face model ID (e.g., "nvidia/prompt-task-and-complexity-classifier") or a path to a local directory from which to load the `AutoConfig` for the model. This config provides parameters like `target_sizes` and default maps for the `CustomModel`.
- `--onnx_output_path`: (Required) Output path for the exported (non-quantized) ONNX model (e.g., `models/prompt_classifier.onnx`).
- `--quantized_output_path`: (Required) Output path for the quantized ONNX model (e.g., `models/prompt_classifier.quant.onnx`).
- `--task_type_map_json PATH`: (Optional) Path to a JSON file containing the `task_type_map`. If provided, this will override the map loaded from the config.
- `--weights_map_json PATH`: (Optional) Path to a JSON file for the `weights_map`. Overrides the config version.
- `--divisor_map_json PATH`: (Optional) Path to a JSON file for the `divisor_map`. Overrides the config version.
- `--batch_size INT`: (Optional) Batch size to use for dummy inputs during ONNX export. Default is 1.
- `--seq_length INT`: (Optional) Sequence length for dummy inputs during ONNX export. Default is 128.

**Example Command:**

```bash
python scripts/quantize_model.py \
  "nvidia/prompt-task-and-complexity-classifier" \
  --onnx_output_path "models/prompt_classifier.onnx" \
  --quantized_output_path "models/prompt_classifier.quant.onnx"
```

This command will:
1. Load the configuration from "nvidia/prompt-task-and-complexity-classifier".
2. Instantiate the `CustomModel` (which internally uses "microsoft/DeBERTa-v3-base" as its backbone).
3. Export this `CustomModel` to `models/prompt_classifier.onnx`.
4. Apply dynamic quantization (QDQ, QInt8) to the ONNX model and save it to `models/prompt_classifier.quant.onnx`.

## Testing

```bash
poetry run pytest
```

## Docker

```bash
docker build -t adaptive-ai .
docker run -p 8000:8000 adaptive-ai
```
