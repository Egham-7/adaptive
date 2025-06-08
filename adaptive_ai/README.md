# Adaptive AI Service

Python service that analyzes prompts and selects the optimal LLM model.

## Features

- **Prompt Analysis** - Multi-dimensional analysis (creativity, reasoning, context, domain)
- **Model Selection** - Vector similarity matching to find optimal models
- **Domain Classification** - Specialized routing for code, writing, analysis, etc.
- **Fast Inference** - LitServe for high-performance serving

## Quick Start

```bash
poetry install
poetry run python main.py
```

Runs on port 8000 by default.

## API

**Endpoint:** `POST /predict`

```json
// Request
{
  "prompt": "Write a Python function to sort a list"
}

// Response
{
  "selected_model": "gpt-4o",
  "provider": "openai",
  "match_score": 0.94,
  "domain": "programming"
}
```

## How It Works

1. Analyzes prompt complexity across multiple dimensions
2. Classifies domain (code, writing, analysis, etc.)
3. Uses vector similarity to match with model capabilities
4. Returns optimal model selection with confidence score

## Adding Models

Edit `models/llms.py`:

```python
model_capabilities = {
    "new-model": {
        "provider": "provider-name",
        "capability_vector": [0.8, 0.9, 0.7, 0.8],  # [creativity, reasoning, context, domain]
        "cost_per_token": 0.00001
    }
}
```

## Docker

```bash
docker build -t adaptive-ai .
docker run -p 8000:8000 adaptive-ai
```