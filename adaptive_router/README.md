# Adaptive Router

Intelligent LLM model selection library with integrated ML-powered prompt classification.

## Quick Start

### As a Library

```python
from adaptive_router import ModelRouter, ModelSelectionRequest

# Initialize router (handles all dependencies internally)
router = ModelRouter()

# Make a selection request
request = ModelSelectionRequest(
    prompt="Write a Python function to sort a list",
    cost_bias=0.5  # 0.0 = cheapest, 1.0 = most capable
)

# Get the best model
response = router.select_model(request)
print(f"Selected: {response.provider}/{response.model}")
print(f"Alternatives: {response.alternatives}")
```

### As a Service

```bash
# Install dependencies
uv install

# Start FastAPI server
uv run adaptive-router

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Modal Deployment

```bash
# Deploy to Modal for serverless GPU inference
modal deploy deploy.py

# Use deployed function
from modal import Function

select_model = Function.lookup("prompt-task-complexity-classifier", "select_model")
response = select_model.remote(request)
```

## Features

- **Integrated ML Classifier**: Built-in NVIDIA prompt-task-complexity-classifier
- **Direct GPU/CPU Inference**: Local PyTorch inference, no external API calls
- **Task Classification**: Categorizes prompts by complexity and task type
- **Cost Optimization**: Balances performance vs. cost based on user preferences
- **Multiple Deployment Modes**: Library, FastAPI server, or Modal serverless
- **Smart Model Selection**: Considers task complexity, type, and model capabilities
- **YAML-Based Model Database**: Easy configuration of models and pricing

## Architecture

```
adaptive_router/
├── adaptive_router/          # Main package
│   ├── models/              # Pydantic models
│   ├── services/            # Core business logic
│   │   ├── model_router.py              # Main router (public API)
│   │   ├── model_registry.py            # Model metadata
│   │   ├── yaml_model_loader.py         # YAML config loader
│   │   └── prompt_task_complexity_classifier.py  # ML classifier
│   └── utils/               # JWT, helpers
├── model_data/              # YAML model definitions
├── deploy.py               # Modal deployment
├── config.py              # Configuration
└── tests/                 # Unit & integration tests
```

## API

### Library API

```python
from adaptive_router import ModelRouter, ModelSelectionRequest, ModelCapability

router = ModelRouter()

# Basic usage
request = ModelSelectionRequest(prompt="Explain quantum computing")
response = router.select_model(request)

# With specific models
request = ModelSelectionRequest(
    prompt="Write a sorting algorithm",
    models=[
        ModelCapability(provider="openai", model_name="gpt-4"),
        ModelCapability(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
    ],
    cost_bias=0.3
)
response = router.select_model(request)
```

### HTTP API

**POST /select_model**

Request:
```json
{
  "prompt": "Write a Python function to calculate factorial",
  "cost_bias": 0.5,
  "models": [
    {
      "provider": "openai",
      "model_name": "gpt-4"
    }
  ]
}
```

Response:
```json
{
  "provider": "openai",
  "model": "gpt-4",
  "alternatives": [
    {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022"
    }
  ]
}
```

**GET /health**

```json
{
  "status": "healthy"
}
```

## Configuration

Environment variables:

```bash
# Server
HOST=0.0.0.0
PORT=8000

# FastAPI
FASTAPI_WORKERS=1
FASTAPI_ACCESS_LOG=true
FASTAPI_LOG_LEVEL=info

# ML Classifier
CLASSIFIER__MODEL_NAME=nvidia/prompt-task-and-complexity-classifier
CLASSIFIER__DEVICE=cuda  # or cpu
CLASSIFIER__MAX_LENGTH=512

# Modal (for serverless deployment)
MODAL_DEPLOYMENT__APP_NAME=prompt-task-complexity-classifier
MODAL_DEPLOYMENT__GPU_TYPE=T4
MODAL_DEPLOYMENT__TIMEOUT=600
JWT_SECRET=your_jwt_secret
```

## How It Works

1. **Prompt Classification**: NVIDIA DeBERTa-based model analyzes prompt complexity and task type
2. **Model Filtering**: Filters available models based on task requirements
3. **Cost-Performance Ranking**: Ranks models using cost bias and complexity score
4. **Selection**: Returns best model with alternatives

### Complexity Scoring

The classifier analyzes prompts across multiple dimensions:
- Task type (code, creative, analysis, etc.)
- Complexity score (0.0-1.0)
- Domain knowledge requirements
- Reasoning and creativity needs
- Contextual knowledge requirements

### Cost Bias

- `0.0`: Prefer cheapest models
- `0.5`: Balance cost and capability
- `1.0`: Prefer most capable models

## Development

### Setup

```bash
# Install dependencies
uv install

# Install dev dependencies
uv install --all-extras
```

### Testing

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest -m unit

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/unit/services/test_model_router.py
```

### Code Quality

```bash
# Format code
uv run black .

# Lint
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Type checking
uv run mypy .
```

### Make Commands

```bash
make test          # Run all tests
make test-unit     # Run unit tests
make test-cov      # Run with coverage
make lint          # Check with ruff
make lint-fix      # Fix issues
make format        # Format with black
make typecheck     # Type checking
make quality       # All quality checks
```

## Model Configuration

Models are defined in YAML files under `model_data/data/provider_models/`:

```yaml
# openai_models_structured.yaml
models:
  gpt-4:
    model_name: "gpt-4"
    cost_per_1m_input_tokens: 30.0
    cost_per_1m_output_tokens: 60.0
    max_context_tokens: 8192
    supports_function_calling: true
    task_type: "general"
    complexity: "high"
```

Supported providers:
- OpenAI
- Anthropic
- Groq
- DeepSeek
- Google AI (Gemini)
- xAI (Grok)

## Tech Stack

- **ML Framework**: PyTorch 2.2+ with CUDA support
- **Transformers**: HuggingFace Transformers 4.52+
- **Model**: NVIDIA DeBERTa-based classifier
- **API Framework**: FastAPI 0.104+
- **ASGI Server**: Hypercorn 0.17+
- **Modal**: Serverless GPU deployment
- **Configuration**: Pydantic Settings
- **Testing**: pytest with coverage

## Performance

- **Classification**: <50ms per request
- **Model Selection**: <10ms
- **Memory**: ~2-4GB with ML model loaded
- **Throughput**: 500+ requests/second

## Contributing

1. Create feature branch from `dev`
2. Implement changes with tests
3. Run quality checks: `make quality`
4. Submit PR with documentation updates

## License

MIT
