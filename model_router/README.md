# Adaptive AI Service

Python ML service that intelligently selects optimal LLM models for each request.

## Quick Start

```bash
uv sync
cp .env.example .env.local  # Add HuggingFace token
uv run adaptive-ai
```

## Features

- **7-Dimensional Analysis** - Analyzes prompts across creativity, reasoning, context, etc.
- **Task Classification** - Maps prompts to optimal model types
- **Cost Optimization** - Balances performance vs cost based on bias settings
- **Protocol Selection** - Chooses between standard, minion, or minions protocols
- **Smart Caching** - Caches similar prompt classifications
- **High Performance** - FastAPI for fast API serving with async processing

## API

### Model Selection
`POST /predict`

```json
{
  "chat_completion_request": {
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7
  },
  "model_router": {
    "cost_bias": 0.3,
    "models": [{"provider": "openai"}, {"provider": "anthropic"}]
  }
}
```

### Health Check
`GET /health`

## How It Works

1. **Prompt Classification** - Custom DeBERTa model analyzes 7 dimensions
2. **Model Mapping** - Maps task types to optimal model candidates  
3. **Cost Filtering** - Applies cost bias and provider constraints
4. **Protocol Selection** - Chooses optimal protocol (standard/minion/minions)

## Tech Stack

- **FastAPI** for high-performance API serving with async processing
- **Modal** for GPU-accelerated prompt classification
- **HuggingFace Transformers** for ML models (via Modal)
- **httpx** for async HTTP client communication

## Development

```bash
uv run pytest              # Run tests
uv run black .             # Format code  
uv run ruff check .        # Lint code
uv run mypy .              # Type checking
```