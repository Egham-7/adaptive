# Adaptive Router - Intelligent Model Selection Service

## Memory and Documentation

**IMPORTANT**: When working on this service, remember to:

### Memory Management
Use ByteRover MCP for persistent memory across sessions:
- **Before adding memories**: Always search first with `mcp__byterover-mcp__byterover-retrieve-knowledge` to avoid duplicates
- **Add memories**: Use `mcp__byterover-mcp__byterover-store-knowledge` for ML model configurations, training results, troubleshooting solutions
- **Search memories**: Use `mcp__byterover-mcp__byterover-retrieve-knowledge` to recall previous conversations and solutions
- **Best practices for memory storage**: Only commit meaningful, reusable information like ML model patterns, PyTorch configurations, classification algorithms, cost optimization strategies, and implementation details that provide value beyond common knowledge

### Documentation
For documentation needs, use Ref MCP tools:
- **Search docs**: Use `mcp__Ref__ref_search_documentation` for Python, FastAPI, PyTorch, HuggingFace, scikit-learn documentation
- **Read specific docs**: Use `mcp__Ref__ref_read_url` to read documentation pages

## Overview

The adaptive_router service is a unified Python ML package that provides intelligent model selection for the Adaptive LLM infrastructure. It uses cluster-based intelligent routing with per-cluster error rates to select optimal LLM models. The service supports two deployment modes: Library (import and use directly in Python code) and FastAPI (HTTP API server with local GPU/CPU inference).

## Key Features

- **Cluster-Based Routing**: UniRouter algorithm with K-means clustering and per-cluster error rates
- **Flexible Deployment**: Python library import or FastAPI HTTP server
- **Cost Optimization**: Balances performance vs. cost based on configurable preferences
- **High-Performance API**: FastAPI framework with Hypercorn ASGI server, OpenAPI documentation
- **MinIO S3 Storage**: Loads cluster profiles from Railway-hosted MinIO storage
- **Local ML Inference**: Sentence transformers and scikit-learn for feature extraction and clustering

## Technology Stack

- **ML Framework**: PyTorch 2.2+ with sentence-transformers for semantic embeddings
- **Clustering**: scikit-learn for K-means clustering and feature scaling
- **API Framework**: FastAPI 0.118+ for HTTP server mode
- **ASGI Server**: Hypercorn 0.17+ with HTTP/1.1, HTTP/2, and WebSocket support
- **Storage**: boto3 for MinIO S3 integration (Railway deployment)
- **LLM Integration**: LangChain for orchestration and provider abstraction
- **Configuration**: Pydantic Settings for type-safe configuration management
- **Logging**: Standard Python logging with structured JSON output

## Project Structure

```
adaptive_router/
├── adaptive_router/
│   ├── __init__.py                           # Library exports for Python import
│   ├── cli.py                                # CLI entry point (starts FastAPI server)
│   ├── api/                                  # FastAPI server implementation
│   │   ├── __init__.py
│   │   └── app.py                            # FastAPI application with endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── storage_config.py                 # MinIO S3 configuration
│   ├── models/                               # Data models and schemas
│   │   ├── __init__.py
│   │   ├── llm_core_models.py                # Core request/response models
│   │   └── routing_schemas.py                # Routing decision models
│   ├── services/                             # Core routing services
│   │   ├── __init__.py
│   │   ├── model_router.py                   # Public API (uses RouterService)
│   │   ├── router_service.py                 # Router integration layer
│   │   ├── router.py                         # Core routing logic
│   │   ├── cluster_engine.py                 # K-means clustering
│   │   ├── feature_extractor.py              # Sentence transformers + TF-IDF
│   │   ├── storage_profile_loader.py         # MinIO S3 profile loading
│   │   └── yaml_model_loader.py              # YAML configuration loader
│   ├── config/                               # Configuration files
│   │   └── unirouter_models.yaml             # Model definitions and routing config
│   └── data/                                 # (Local data not used in production)
├── tests/                                    # Test suite (unit + integration)
├── pyproject.toml                            # Dependencies and CLI entry point
├── uv.lock                                   # Dependency lock file
└── README.md                                 # Service documentation
```

## Environment Configuration

### Required Environment Variables

```bash
# Server Configuration
HOST=0.0.0.0                     # Hypercorn host
PORT=8000                        # Hypercorn port

# FastAPI Configuration
FASTAPI_WORKERS=1                # Number of workers (ignored in programmatic mode)
FASTAPI_RELOAD=false             # Auto-reload on code changes (dev only - not supported)
FASTAPI_ACCESS_LOG=true          # Enable access logging
FASTAPI_LOG_LEVEL=info           # Hypercorn log level

# MinIO S3 Storage Configuration (Railway deployment)
S3_BUCKET_NAME=adaptive-router-profiles  # MinIO bucket name
MINIO_PUBLIC_ENDPOINT=https://minio.railway.app  # MinIO endpoint URL
MINIO_ROOT_USER=your_minio_user          # MinIO access key
MINIO_ROOT_PASSWORD=your_minio_password  # MinIO secret key
```

### Optional Configuration

```bash
# Debugging
DEBUG=false                      # Enable debug logging
LOG_LEVEL=INFO                  # Logging level

# Hypercorn-Specific Configuration (Optional)
# For HTTP/2 support (requires SSL/TLS):
# HYPERCORN_CERTFILE=cert.pem     # SSL certificate file
# HYPERCORN_KEYFILE=key.pem       # SSL private key file
```

## Deployment Modes

### 1. Library Mode (Python Import)
Use adaptive_router as a Python library in your code:

```python
from adaptive_router import ModelRouter, ModelSelectionRequest

# Initialize router (loads cluster profiles from MinIO automatically)
router = ModelRouter()

# Select optimal model based on prompt
request = ModelSelectionRequest(
    prompt="Write a Python function to sort a list",
    cost_bias=0.5  # 0.0 = cheapest, 1.0 = most capable
)
response = router.select_model(request)
print(f"Selected: {response.provider} / {response.model}")
print(f"Alternatives: {response.alternatives}")
```

### 2. FastAPI Server Mode
Run as HTTP API server for production deployment:

```bash
# Install dependencies
uv install

# Start FastAPI server
uv run adaptive-router

# Server starts on http://0.0.0.0:8000
# API docs available at http://localhost:8000/docs
# ReDoc available at http://localhost:8000/redoc

# Custom port
PORT=8001 uv run adaptive-router

# Enable debug logging
DEBUG=true uv run adaptive-router
```

**API Endpoints:**
- `POST /select_model` - Select optimal model based on prompt
- `GET /health` - Health check endpoint

Access interactive API docs at `http://localhost:8000/docs`

## Development Commands

### Local Development
```bash
# Install dependencies
uv install

# Start the FastAPI server
uv run adaptive-router

# Start with development settings
DEBUG=true uv run adaptive-router

# Start on custom port
PORT=8001 uv run adaptive-router

# Note: The CLI starts Hypercorn programmatically
# For multi-process deployment, use a process manager like supervisord
```

### Code Quality
```bash
# Format code with Black
uv run black .

# Lint with Ruff
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Type checking with mypy
uv run mypy .

# Run all quality checks
uv run black . && uv run ruff check . && uv run mypy .
```

### Testing

We provide multiple convenient ways to run tests:

#### Using Make Commands (Recommended)
```bash
# Show all available commands
make help

# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run with coverage report
make test-cov

# Run with HTML coverage report
make test-cov-html

# Run specific test categories
make test-config      # Configuration tests
make test-services    # Service tests
make test-models      # Model tests
make test-routing     # Routing integration tests

# Code quality
make lint            # Check with ruff
make lint-fix        # Fix issues
make format          # Format with black
make typecheck       # Type checking
make quality         # All quality checks
```

#### Using Shell Script
```bash
# Run all tests
./scripts/test.sh

# Run unit tests only  
./scripts/test.sh unit

# Run integration tests only
./scripts/test.sh integration

# Run with coverage
./scripts/test.sh coverage

# Clean test artifacts
./scripts/test.sh clean
```

#### Using uv run directly
```bash
# Run unit tests only (CI-safe, no external dependencies)
uv run pytest -m "unit"

# Run integration tests (requires running AI service on localhost:8000)
uv run pytest -m "integration"

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest adaptive_router/tests/unit/core/test_config.py

# Run with verbose output
uv run pytest -v

# Run tests and generate HTML coverage report
uv run pytest --cov --cov-report=html
```

## API Interface

The service exposes a FastAPI REST API that accepts model selection requests and returns orchestration responses. The API includes automatic OpenAPI documentation available at `/docs`.

### Request Format
```python
{
    "chat_completion_request": {
        "messages": [
            {"role": "user", "content": "Write a Python function to sort a list"}
        ],
        "model": "gpt-4",
        "temperature": 0.7
    },
    "adaptive_router": {
        "models": [
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "cost_per_1m_input_tokens": 30.0,
                "cost_per_1m_output_tokens": 60.0,
                "max_context_tokens": 128000,
                "supports_function_calling": true
            },
            {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet-20240229",
                "cost_per_1m_input_tokens": 15.0,
                "cost_per_1m_output_tokens": 75.0,
                "max_context_tokens": 200000,
                "supports_function_calling": true
            }
        ],
        "cost_bias": 0.5,
        "complexity_threshold": 0.7,
        "token_threshold": 1000
    },
    "user_preferences": {
        "preferred_providers": ["openai", "anthropic"],
        "cost_optimization": true,
        "quality_preference": "high"
    }
}
```

### Response Format
```python
{
    "selection": {
        "provider": "openai",
        "model": "gpt-4",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "alternatives": [
            {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "cost_ratio": 0.85,
                "reason": "fallback_option"
            }
        ],
        "reasoning": "Selected GPT-4 for balanced cost-performance trade-off based on prompt complexity analysis"
    }
}
```

## Core Services

### Model Router Service
**File**: `adaptive_router/services/model_router.py`

- Intelligent model routing with complexity-aware selection
- Selects optimal LLM models based on task complexity and cost optimization
- Integrates with prompt classifier for task analysis
- Provides model capability matching and filtering

### Prompt Task Complexity Classifier Service  
**File**: `adaptive_router/services/prompt_task_complexity_classifier.py`

- Complete NVIDIA transformer-based classifier implementation
- Uses PyTorch with local GPU/CPU inference capability
- Classifies prompt task types (code, math, creative, etc.)
- Determines complexity levels and processing requirements
- Provides confidence scores for classification decisions
- Supports batch processing for high throughput

### Model Registry Service
**File**: `adaptive_router/services/model_registry.py`

- Validates model names across all supported providers
- Manages model availability and capability lookups
- Provides core model filtering functionality
- Integrates with YAML model database for metadata

### YAML Model Database Service
**File**: `adaptive_router/services/yaml_model_loader.py`

- Loads provider model configurations from YAML files
- Provides fast in-memory model metadata lookups
- Handles model capability definitions and pricing data
- Supports dynamic provider configuration updates
- Implements smart fallback strategies
- Tracks usage patterns and optimizes over time
- Provides cost savings metrics and recommendations


## Routing Algorithm

### Cluster-Based Selection (UniRouter)
- **Algorithm**: K-means clustering of prompts based on semantic features
- **Features**: Sentence transformer embeddings (384D) + TF-IDF features (5000D)
- **Clusters**: Prompts grouped into K clusters (configurable, typically K=10-50)
- **Error Rates**: Each model has per-cluster error rates learned from historical data
- **Selection**: Combines error rates + cost + user preferences to rank models

### Feature Extraction
- **Embeddings**: Sentence transformers (all-MiniLM-L6-v2) for semantic similarity
- **TF-IDF**: Term frequency-inverse document frequency for lexical patterns
- **Scaling**: StandardScaler normalization for both feature types
- **Concatenation**: Combined 5384D feature vector per prompt

### Cost-Performance Trade-off
- **Cost Preference**: λ parameter (0.0 = cheapest, 1.0 = most accurate)
- **Routing Score**: Weighted combination of predicted accuracy and normalized cost
- **Formula**: `score = predicted_accuracy - λ * normalized_cost`
- **Optimization**: Selects model with highest routing score for assigned cluster

## Caching and Performance

### API Performance
- **Framework**: FastAPI with async/await for optimal performance
- **ASGI Server**: Hypercorn with HTTP/1.1 and HTTP/2 support
- **CORS**: Configured middleware for cross-origin requests
- **Error Handling**: Global exception handlers with structured logging

### Storage Integration
- **MinIO S3**: Loads cluster profiles from Railway-hosted MinIO on startup
- **Profile Caching**: Cluster centers, error rates, and scalers loaded once at init
- **Connection Pooling**: boto3 handles S3 connection pooling automatically
- **Health Checks**: Validates MinIO connectivity during startup

### Request Processing
- **Feature Extraction**: Local sentence transformers + TF-IDF (no external API calls)
- **Cluster Assignment**: Fast K-means predict using pre-loaded centroids
- **Model Selection**: In-memory scoring of models based on cluster assignment
- **Response Time**: <100ms end-to-end for typical requests
- **Throughput**: 100+ requests/second on standard hardware

## Configuration

### Provider Configuration
**File**: `adaptive_router/config/providers.py`

```python
PROVIDERS = {
    "openai": {
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "pricing": {"input": 0.03, "output": 0.06},
        "capabilities": ["general", "code", "analysis"],
        "max_tokens": 4096
    },
    "anthropic": {
        "models": ["claude-3-sonnet", "claude-3-haiku"],
        "pricing": {"input": 0.015, "output": 0.075},
        "capabilities": ["general", "analysis", "creative"],
        "max_tokens": 8192
    }
}
```

### Task Mappings
**File**: `adaptive_router/config/task_mappings.py`

```python
TASK_MODEL_MAPPINGS = {
    "code": {
        "preferred_models": ["gpt-4", "claude-3-sonnet"],
        "parameters": {"temperature": 0.1, "max_tokens": 2048}
    },
    "creative": {
        "preferred_models": ["claude-3-sonnet", "gpt-4"],
        "parameters": {"temperature": 0.8, "max_tokens": 1024}
    }
}
```

## Monitoring and Observability

### Metrics
- **Classification Accuracy**: Track prediction confidence and user feedback
- **Model Selection Success**: Monitor downstream API success rates
- **Cost Optimization**: Track actual vs. projected cost savings
- **Performance**: Request latency, throughput, error rates

### Logging
- **Structured Logging**: JSON-formatted logs with request correlation
- **Classification Results**: Log task types, domains, and confidence scores
- **Model Selection**: Log selected providers, models, and reasoning
- **Performance Metrics**: Log inference times and batch processing stats

### Health Checks
- **Model Loading**: Verify all ML models are loaded and functional
- **Memory Usage**: Monitor memory consumption and detect leaks
- **Cache Performance**: Track cache hit rates and eviction patterns
- **Dependency Health**: Verify HuggingFace Hub connectivity

## Deployment

### Docker
```dockerfile
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --all-extras --no-dev

COPY . .
EXPOSE 8000

CMD ["uv", "run", "adaptive-ai"]
```

### Docker Compose
The service is included in the root `docker-compose.yml` with proper networking and resource allocation.

### Resource Requirements

#### Library Mode
- **CPU**: 2-4 cores for feature extraction and clustering
- **GPU**: Optional (macOS uses CPU, Linux can use CUDA if available)
- **Memory**: 2-4GB RAM for sentence transformers and scikit-learn
- **Storage**: 1-2GB for HuggingFace model cache (~500MB for all-MiniLM-L6-v2)

#### FastAPI Server Mode
- **CPU**: 2-4 cores for concurrent requests + feature extraction
- **GPU**: Optional (sentence transformers work well on CPU)
- **Memory**: 2-4GB RAM for models, API server, and request processing
- **Storage**: 1-2GB for models, cluster profiles, and logs
- **Network**: Connection to MinIO S3 (Railway) for loading cluster profiles

### Hypercorn Benefits
- **Protocol Support**: HTTP/1.1, HTTP/2, WebSockets out of the box
- **Future-Ready**: HTTP/3 support available with `hypercorn[h3]` extra
- **Graceful Shutdown**: Built-in signal handling and graceful shutdown support
- **Sans-IO Architecture**: Modern implementation using sans-io hyper libraries

## Troubleshooting

### Common Issues

**Service won't start**
- Check Python version (3.10+ required)
- Verify all dependencies installed: `uv install`
- Check port availability (default: 8000)
- Review environment variable configuration (especially MinIO settings)
- Verify Hypercorn is correctly installed: `uv run hypercorn --version`

**MinIO connection failures**
- Verify S3_BUCKET_NAME environment variable is set
- Check MINIO_PUBLIC_ENDPOINT is accessible from your deployment
- Validate MINIO_ROOT_USER and MINIO_ROOT_PASSWORD credentials
- Test MinIO connectivity: use boto3 client to list buckets
- Check Railway MinIO service status if using Railway deployment

**Model loading failures**
- Verify sentence-transformers is installed: `uv sync`
- Check HuggingFace Hub connectivity for model downloads
- Verify CUDA drivers if using GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Check available disk space for model cache (~500MB for all-MiniLM-L6-v2)
- On macOS, CPU mode is used automatically (no CUDA required)

**Routing errors**
- Verify input format matches ModelSelectionRequest schema
- Check prompt length is reasonable (no hard limit, but very long prompts are slower)
- Ensure MinIO profile data contains required cluster centers and error rates
- Enable debug logging: `DEBUG=true uv run adaptive-router`

**Performance issues**
- Monitor CPU usage during feature extraction
- Check memory usage (sentence transformers ~1-2GB)
- Verify cluster profile loaded successfully at startup (check logs)
- For production, consider using CUDA if available (Linux only)
- Review batch processing if handling multiple requests concurrently

### Debug Commands

**Library Mode:**
```bash
# Test model router
python -c "
from adaptive_router import ModelRouter, ModelSelectionRequest
router = ModelRouter()
request = ModelSelectionRequest(prompt='Explain quantum computing', cost_bias=0.5)
response = router.select_model(request)
print(f'Provider: {response.provider}, Model: {response.model}')
"

# Check CUDA availability (Linux only)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

**FastAPI Server Mode:**
```bash
# Enable debug logging
DEBUG=true uv run adaptive-router

# Check service health
curl -X GET http://localhost:8000/health

# Test model selection endpoint
curl -X POST http://localhost:8000/select_model \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a sorting algorithm", "cost_bias": 0.5}'

# Check MinIO connectivity
python -c "
from adaptive_router.core.storage_config import MinIOSettings
from adaptive_router.services.storage_profile_loader import StorageProfileLoader
settings = MinIOSettings()
loader = StorageProfileLoader.from_minio_settings(settings)
print('MinIO connection successful')
"
```

## Performance Benchmarks

### Library/FastAPI Mode
- **Feature Extraction**: 20-50ms per request (sentence transformers + TF-IDF)
- **Cluster Assignment**: <5ms (K-means predict on pre-loaded centroids)
- **Model Selection**: <5ms (scoring and ranking models)
- **Total Latency**: 30-60ms end-to-end per request
- **Throughput**: 100-500 requests/second (depends on CPU cores)
- **Memory Usage**: 2-4GB (sentence transformers + cluster profiles)
- **First Request**: Slower (~2-5s) due to loading models from HuggingFace

### Startup Performance
- **MinIO Profile Load**: 1-3 seconds (downloads cluster centers, error rates, scalers)
- **Model Download**: 5-10 seconds first time (HuggingFace cache), instant after that
- **Total Startup**: 5-15 seconds depending on network and cache state

### Routing Quality
- **Cost Savings**: 30-70% compared to always using most capable models
- **Accuracy Retention**: >90% of optimal model selection vs. oracle routing
- **Cluster Silhouette**: Typically 0.3-0.5 (good cluster separation)
- **Per-Cluster Accuracy**: Varies by cluster, tracked in profile metadata

## Contributing

### Code Style
- **Formatting**: Black with 88-character line length
- **Linting**: Ruff with comprehensive rule set
- **Type Checking**: mypy with strict configuration
- **Import Sorting**: Ruff isort with first-party module recognition

### Testing Requirements
- **Unit Tests**: All services and utilities must have unit tests
- **Integration Tests**: End-to-end testing with mock ML models
- **Performance Tests**: Benchmark classification speed and accuracy
- **Coverage**: Minimum 80% test coverage required

### Documentation Updates
**IMPORTANT**: When making changes to this service, always update documentation:

1. **Update this CLAUDE.md** when:
   - Adding new ML models or classification algorithms
   - Modifying API interfaces or request/response formats
   - Changing environment variables or configuration settings
   - Adding new providers, task types, or domain classifications
   - Updating Python dependencies or ML framework versions
   - Adding new services or modifying existing service logic

2. **Update root CLAUDE.md** when:
   - Changing service ports, commands, or basic service description
   - Modifying the service's role in the intelligent routing architecture
   - Adding new ML capabilities or performance characteristics

3. **Update adaptive-docs/** when:
   - Adding new model selection features
   - Changing cost optimization algorithms
   - Modifying provider integration or routing logic

### Pull Request Process
1. Create feature branch from `dev`
2. Implement changes with comprehensive tests
3. Run full quality checks: `uv run black . && uv run ruff check . && uv run mypy . && uv run pytest --cov`
4. **Update relevant documentation** (CLAUDE.md files, adaptive-docs/, README)
5. Submit PR with performance impact analysis and documentation updates