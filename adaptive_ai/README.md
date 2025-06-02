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

## Testing

```bash
poetry run pytest
```

## Docker

```bash
docker build -t adaptive-ai .
docker run -p 8000:8000 adaptive-ai
```