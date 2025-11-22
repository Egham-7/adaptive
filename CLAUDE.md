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
├── adaptive_router/                          # Core ML library package
│   ├── __init__.py                           # Library exports for Python import
│   ├── core/                                 # Core ML routing components
│   │   ├── __init__.py
│   │   ├── router.py                         # ModelRouter - main routing logic
│   │   ├── cluster_engine.py                 # ClusterEngine - K-means clustering
│   │   └── feature_extractor.py              # FeatureExtractor - sentence transformers + TF-IDF
│   ├── loaders/                              # Profile loading implementations
│   │   ├── __init__.py
│   │   ├── base.py                           # ProfileLoader base class
│   │   ├── local.py                          # LocalFileProfileLoader
│   │   └── minio.py                          # MinIOProfileLoader - S3 profile loading
│   ├── models/                               # Pydantic data models and schemas
│   │   ├── __init__.py
│   │   ├── api.py                            # Request/response models
│   │   ├── config.py                         # YAML configuration models
│   │   ├── evaluation.py                     # Evaluation metrics models
│   │   ├── health.py                         # Health check models
│   │   ├── registry.py                       # Model registry models
│   │   ├── routing.py                        # Routing decision models
│   │   └── storage.py                        # Storage/profile models
│   ├── utils/                                # Utility modules
│   │   ├── __init__.py
│   │   └── model_parser.py                   # Model name parsing utilities
│   └── tests/                                # Test suite (inside package)
│       ├── fixtures/                         # Test fixtures
│       │   ├── ml_fixtures.py
│       │   ├── model_fixtures.py
│       │   └── request_fixtures.py
│       ├── integration/                      # Integration tests
│       │   ├── test_api_endpoints.py
│       │   ├── test_cost_optimization.py
│       │   ├── test_model_selection_flows.py
│       │   └── test_task_routing.py
│       └── unit/                             # Unit tests
│           ├── models/
│           └── services/
├── app/                                      # FastAPI HTTP server (separate package)
│   ├── __init__.py
│   ├── main.py                               # FastAPI application factory (create_app)
│   ├── config.py                             # App configuration (env vars)
│   ├── health.py                             # Health check endpoints
│   ├── models.py                             # API-specific models
│   ├── registry/                             # External model registry integration
│   │   ├── __init__.py
│   │   ├── client.py                         # HTTP client for registry API
│   │   └── models.py                         # Registry model cache
│   └── utils/                                # App-specific utilities
│       ├── fuzzy_matching.py                 # Fuzzy model name matching
│       └── model_resolver.py                 # Model resolution logic
├── scripts/                                  # Utility scripts
│   ├── models/                               # Model management scripts
│   ├── training/                             # Training scripts
│   └── utils/                                # Utility scripts
├── pyproject.toml                            # Dependencies and package configuration
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
from core.router import ModelRouter
from models.routing import RoutingRequest
from loaders.minio import MinIOProfileLoader

# Initialize profile loader from MinIO
loader = MinIOProfileLoader.from_env()

# Initialize router with loaded profile
router = ModelRouter(loader)

# Select optimal model based on prompt
request = RoutingRequest(
    prompt="Write a Python function to sort a list",
    cost_preference=0.5  # 0.0 = cheapest, 1.0 = most capable
)
response = router.route(request)
print(f"Selected: {response.provider} / {response.model}")
print(f"Reasoning: {response.reasoning}")
```

### 2. FastAPI Server Mode

Run as HTTP API server for production deployment:

```bash
# Install dependencies
uv install

# Start FastAPI server (development mode with auto-reload)
fastapi dev app/main.py

# Or use Hypercorn for production
hypercorn app.main:create_app() --bind 0.0.0.0:8000

# Server starts on http://0.0.0.0:8000
# API docs available at http://localhost:8000/docs
# ReDoc available at http://localhost:8000/redoc
# Health check at http://localhost:8000/health

# Custom port
PORT=8001 hypercorn app.main:create_app() --bind 0.0.0.0:8001

# Enable debug logging
DEBUG=true hypercorn app.main:create_app() --bind 0.0.0.0:8000
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

# Start the FastAPI server (development mode with auto-reload)
fastapi dev app/main.py

# Or use Hypercorn directly (production mode)
hypercorn app.main:create_app() --bind 0.0.0.0:8000

# Start with custom configuration
HOST=0.0.0.0 PORT=8001 hypercorn app.main:create_app() --bind 0.0.0.0:8001

# Start with debug logging
DEBUG=true hypercorn app.main:create_app() --bind 0.0.0.0:8000

# For multi-process deployment
hypercorn app.main:create_app() --bind 0.0.0.0:8000 --workers 4
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

### Model Router

**File**: `adaptive_router/core/router.py`

The `ModelRouter` class is the main entry point for intelligent model selection:

- Cluster-based routing using UniRouter algorithm
- Accepts `RoutingRequest` with prompt and cost preference
- Returns `RoutingResponse` with selected model and reasoning
- Uses pre-loaded cluster profiles from MinIO S3
- Combines feature extraction, cluster assignment, and cost optimization
- Supports multiple provider models with per-cluster error rates

### Cluster Engine

**File**: `adaptive_router/core/cluster_engine.py`

The `ClusterEngine` handles K-means clustering operations:

- Loads pre-trained cluster centers from storage profiles
- Assigns prompts to clusters using K-means prediction
- Manages cluster metadata and silhouette scores
- Provides fast cluster assignment (<5ms per request)
- Supports configurable number of clusters (typically 10-50)

### Feature Extractor

**File**: `adaptive_router/core/feature_extractor.py`

The `FeatureExtractor` converts prompts to feature vectors:

- Sentence transformer embeddings using `all-MiniLM-L6-v2` (384D)
- TF-IDF features for lexical patterns (5000D)
- StandardScaler normalization for both feature types
- Concatenated 5384D feature vectors
- Local GPU/CPU inference (no external API calls)
- Cached models for fast subsequent requests

### Profile Loaders

**Files**: `adaptive_router/loaders/`

Profile loading system with multiple implementations:

**Base Loader** (`loaders/base.py`):

- `ProfileLoader` abstract base class
- Defines interface for loading cluster profiles
- Returns `ClusterProfile` with centers, error rates, scalers

**MinIO Loader** (`loaders/minio.py`):

- `MinIOProfileLoader` for S3-compatible storage
- Loads profiles from Railway-hosted MinIO
- Supports environment-based configuration
- Connection pooling and retry logic

**Local Loader** (`loaders/local.py`):

- `LocalFileProfileLoader` for file-based profiles
- Used for testing and offline development
- Supports pickle and JSON formats

### External Model Registry Integration

**Files**: `app/registry/`

Integration with external model registry service for model metadata:

**Registry Client** (`app/registry/client.py`):

- HTTP client for model registry API
- Fetches model metadata (pricing, capabilities, context limits)
- Supports provider/model validation
- Caching layer for performance

**Registry Models** (`app/registry/models.py`):

- Model cache with TTL expiration
- Provider model listings
- Fuzzy matching support for model resolution
- Replaces legacy YAML-based model database

### Model Resolution Utilities

**Files**: `app/utils/`

**Fuzzy Matching** (`app/utils/fuzzy_matching.py`):

- Fuzzy model name matching using string similarity
- Handles common typos and variations
- Provider-specific model name normalization

**Model Resolver** (`app/utils/model_resolver.py`):

- Resolves model names to canonical forms
- Integrates registry client with fuzzy matching
- Provides fallback strategies for unknown models

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

### Model Metadata Configuration

Model metadata (pricing, capabilities, context limits) is now fetched from an **external model registry service** via HTTP API, replacing the legacy YAML-based configuration system.

**Registry Client Configuration** (`app/config.py`):

- `REGISTRY_API_URL` - Base URL for model registry service
- `REGISTRY_CACHE_TTL` - Cache time-to-live for model metadata (seconds)
- `REGISTRY_TIMEOUT` - HTTP request timeout for registry calls (seconds)

**Example Registry Response**:

```json
{
  "provider": "openai",
  "model_name": "gpt-4",
  "cost_per_1m_input_tokens": 30.0,
  "cost_per_1m_output_tokens": 60.0,
  "max_context_tokens": 128000,
  "supports_function_calling": true,
  "capabilities": ["general", "code", "analysis"]
}
```

### Cluster Profile Configuration

Cluster profiles (centers, error rates, scalers) are loaded from **MinIO S3 storage** via the profile loader system:

**MinIO Configuration** (Environment Variables):

- `S3_BUCKET_NAME` - MinIO bucket name for cluster profiles
- `MINIO_PUBLIC_ENDPOINT` - MinIO endpoint URL
- `MINIO_ROOT_USER` - MinIO access key
- `MINIO_ROOT_PASSWORD` - MinIO secret key

**Profile Structure**:

- `cluster_centers.pkl` - K-means cluster centroids (numpy array)
- `error_rates.json` - Per-cluster error rates for each model
- `scaler.pkl` - StandardScaler for feature normalization
- `metadata.json` - Cluster quality metrics and metadata

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
- Verify Hypercorn is correctly installed: `hypercorn --version`
- Ensure you're using the correct command: `fastapi dev app/main.py` or `hypercorn app.main:create_app() --bind 0.0.0.0:8000`

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

- Verify input format matches RoutingRequest schema
- Check prompt length is reasonable (no hard limit, but very long prompts are slower)
- Ensure MinIO profile data contains required cluster centers and error rates
- Enable debug logging: `DEBUG=true fastapi dev app/main.py`

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
from core.router import ModelRouter
from models.routing import RoutingRequest
from loaders.minio import MinIOProfileLoader

loader = MinIOProfileLoader.from_env()
router = ModelRouter(loader)
request = RoutingRequest(prompt='Explain quantum computing', cost_preference=0.5)
response = router.route(request)
print(f'Provider: {response.provider}, Model: {response.model}')
print(f'Reasoning: {response.reasoning}')
"

# Check CUDA availability (Linux only)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

**FastAPI Server Mode:**

```bash
# Start with debug logging
DEBUG=true hypercorn app.main:create_app() --bind 0.0.0.0:8000

# Check service health
curl -X GET http://localhost:8000/health

# Test model selection endpoint
curl -X POST http://localhost:8000/select_model \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a sorting algorithm", "cost_preference": 0.5}'

# Check MinIO connectivity
python -c "
from loaders.minio import MinIOProfileLoader
loader = MinIOProfileLoader.from_env()
profile = loader.load_profile()
print(f'MinIO connection successful - loaded {len(profile.cluster_centers)} clusters')
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
