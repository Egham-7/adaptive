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


## Features

- **Cluster-Based Routing**: UniRouter algorithm with K-means clustering
- **Local Feature Extraction**: Sentence transformers + TF-IDF (no external API calls)
- **Cost Optimization**: Balances performance vs. cost based on user preferences
- **Two Deployment Modes**: Python library or FastAPI HTTP server
- **MinIO S3 Integration**: Loads cluster profiles from Railway-hosted storage
- **Smart Model Selection**: Per-cluster error rates for optimal routing
- **YAML-Based Model Database**: Easy configuration of models and pricing

## Architecture

```
adaptive_router/
├── adaptive_router/          # Main package
│   ├── api/                 # FastAPI server
│   │   └── app.py           # HTTP endpoints
│   ├── cli.py               # CLI entry point
│   ├── models/              # Pydantic models
│   ├── services/            # Core routing logic
│   │   ├── model_router.py              # Public API
│   │   ├── router_service.py            # Router integration
│   │   ├── router.py                    # Core routing
│   │   ├── cluster_engine.py            # K-means clustering
│   │   ├── feature_extractor.py         # ML feature extraction
│   │   └── storage_profile_loader.py    # MinIO S3 loader
│   └── config/              # YAML configurations
├── tests/                   # Unit & integration tests
└── pyproject.toml           # Dependencies + CLI script
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
# MinIO S3 Storage (Railway deployment)
S3_BUCKET_NAME=adaptive-router-profiles
MINIO_PRIVATE_ENDPOINT=http://localhost:9000
MINIO_PUBLIC_ENDPOINT=https://minio.railway.app
MINIO_ROOT_USER=your_minio_user
MINIO_ROOT_PASSWORD=your_minio_password

# Logging
LOG_LEVEL=INFO
```

If both `MINIO_PRIVATE_ENDPOINT` and `MINIO_PUBLIC_ENDPOINT` are provided, the service prefers the private endpoint; otherwise it falls back to the public endpoint, and finally to `http://localhost:9000` for local development.

## How It Works

1. **Feature Extraction**: Sentence transformers (384D embeddings) + TF-IDF (5000D features)
2. **Cluster Assignment**: K-means predicts which of K clusters the prompt belongs to
3. **Model Ranking**: For assigned cluster, ranks models by minimizing: `routing_score = error_rate + λ × normalized_cost` (lower is better)
4. **Selection**: Returns best model with alternatives

### Cluster-Based Routing

The router uses pre-trained clusters stored in MinIO:
- **Clusters**: K clusters learned from historical prompts (typically K=10-50)
- **Error Rates**: Each model has per-cluster error rates from past performance
- **Features**: Combined semantic + lexical features for robust clustering
- **Scaling**: StandardScaler normalization for consistent clustering

### Cost Preference (λ)

- `0.0`: Prefer cheapest models (prioritize cost savings)
- `0.5`: Balance accuracy and cost
- `1.0`: Prefer most accurate models (prioritize quality)

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

- **ML Framework**: PyTorch 2.2+ with sentence-transformers
- **Clustering**: scikit-learn K-means
- **API Framework**: FastAPI 0.118+
- **ASGI Server**: Hypercorn 0.17+
- **Storage**: boto3 for MinIO S3
- **Configuration**: Pydantic Settings
- **Testing**: pytest with coverage

## Performance

- **Feature Extraction**: 20-50ms per request
- **Cluster Assignment + Selection**: <10ms
- **Total Latency**: 30-60ms per request
- **Memory**: 2-4GB (sentence transformers + cluster profiles)
- **Throughput**: 100-500 requests/second

## Contributing

1. Create feature branch from `dev`
2. Implement changes with tests
3. Run quality checks: `make quality`
4. Submit PR with documentation updates

## References

This implementation is based on the UniRouter algorithm described in:

**Jitkrittum, W., et al. (2025).** "Universal Model Routing for Efficient LLM Inference." *arXiv preprint arXiv:2502.08773*. Available at: https://arxiv.org/abs/2502.08773

## License

MIT
