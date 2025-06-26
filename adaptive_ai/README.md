````markdown
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
# Install uv (if you don't have it)
pip install uv

# Create a virtual environment and install dependencies
uv venv
uv sync

# Set environment variables (see next section)
cp .env.example .env.local

# Run service
uv run adaptive_ai/python main.py
```
````

## Environment Variables

```bash
# HUGGINGFACE_TOKEN=hf_xxxxx

# Example for a private Hugging Face model download or rate-limit bypass:
# HF_HOME=/path/to/cache # To specify Hugging Face cache directory
# HF_TOKEN=hf_xxxxx # Your Hugging Face token if downloading private models or for higher rate limits
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

The response structure varies based on the `protocol` chosen by the AI.

**Example for `standard_llm` protocol:**

This protocol indicates that a single large language model has been selected for the task.

```json
{
  "protocol": "standard_llm",
  "standard": {
    "provider": "openai",
    "model": "gpt-4o",
    "parameters": {
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 512,
      "n": 1,
      "stop": null,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    },
    "alternatives": [
      {
        "provider": "google",
        "model": "gemini-1.5-pro"
      },
      {
        "provider": "anthropic",
        "model": "claude-3-opus"
      }
    ]
  },
  "minion": null
}
```

**Example for `minion` protocol:**

This protocol indicates that a specialized "minion" model is recommended for a specific subtask.

````json
{
  "protocol": "minion",
  "standard": null,
  "minion": {
    "task_type": "code_generation",
    "parameters": {
      "temperature": 0.3,
      "top_p": 0.5,
      "max_tokens": 256,
      "n": 1,
      "stop": "```",
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    },
    "alternatives": [
      {
        "task_type": "text_summarization"
      }
    ]
  }
}
````

**Example for `minions_protocol`:**

This protocol suggests orchestrating multiple minion models. It will include details for both a standard LLM (for orchestration/high-level reasoning) and a minion (for a specific sub-task).

```json
{
  "protocol": "minions_protocol",
  "standard": {
    "provider": "openai",
    "model": "gpt-4o",
    "parameters": {
      "temperature": 0.5,
      "top_p": 0.8,
      "max_tokens": 1024,
      "n": 1,
      "stop": null,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    },
    "alternatives": null
  },
  "minion": {
    "task_type": "data_extraction",
    "parameters": {
      "temperature": 0.1,
      "top_p": 0.1,
      "max_tokens": 128,
      "n": 1,
      "stop": "}",
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    },
    "alternatives": null
  }
}
```

## How It Works

1.  **Prompt Analysis**: Uses NVIDIA's prompt classifier to extract complexity dimensions.
2.  **Domain Detection**: Classifies prompt into specialized domains (code, writing, analysis, etc.).
3.  **Model Selection**: An internal LLM, loaded directly via Hugging Face Transformers, determines the best `protocol` (e.g., `standard_llm`, `minion`, `minions_protocol`) and specific models/parameters based on the analyzed prompt, available model capabilities, and specified preferences (e.g., favoring specialized 'minion' models for simple questions to balance quality and efficiency). It generates a structured output based on a predefined Pydantic schema, which is then parsed to orchestrate the response.

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

1.  Define model capabilities in `models/llms.py`:

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

2.  Update domain mappings in `models/domain_mappings.py`

## Testing

```bash
uv run pytest
```

## Docker

```bash
docker build -t adaptive-ai .
docker run -p 8000:8000 adaptive-ai
```

