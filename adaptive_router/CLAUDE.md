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

The adaptive_router service is a unified Python ML package that provides intelligent model selection for the Adaptive LLM infrastructure. It includes the complete NVIDIA prompt-task-complexity-classifier implementation with both local GPU/CPU inference and remote Modal serverless deployment options. The service supports three deployment modes: Library (import and use directly), FastAPI (HTTP API server with local inference), and Modal (serverless GPU deployment with JWT authentication).

## Key Features

- **Integrated ML Classifier**: Complete NVIDIA prompt-task-complexity-classifier built-in with local PyTorch inference
- **Flexible Deployment**: Local GPU/CPU inference, FastAPI server, or Modal serverless deployment
- **Task Classification**: Categorizes prompts by complexity and task type (code, math, creative, etc.)
- **Multiple Deployment Modes**: Library import, FastAPI HTTP server, or Modal serverless with JWT authentication
- **Cost Optimization**: Balances performance vs. cost based on user preferences and prompt analysis
- **High-Performance API**: FastAPI framework with OpenAPI documentation and structured logging
- **Serverless Scale**: Optional Modal deployment for auto-scaling GPU inference with sub-100ms latency

## Technology Stack

- **ML Framework**: PyTorch 2.2+ with CUDA support for GPU acceleration
- **Transformers**: HuggingFace Transformers 4.52+ for model loading
- **Model**: NVIDIA DeBERTa-based classifier (microsoft/DeBERTa-v3-base backbone)
- **API Framework**: FastAPI 0.104+ for HTTP server mode
- **ASGI Server**: Hypercorn 0.17+ with HTTP/1.1, HTTP/2, and WebSocket support
- **Serverless Deployment**: Modal 1.1+ for GPU-accelerated serverless inference
- **HTTP Client**: httpx for Modal API communication with connection pooling
- **Authentication**: python-jose and PyJWT for JWT token handling in Modal mode
- **LLM Integration**: LangChain for orchestration and provider abstraction
- **Configuration**: Pydantic Settings for type-safe configuration management
- **Logging**: Standard Python logging with structured JSON output

## Project Structure

```
adaptive_router/
├── adaptive_router/
│   ├── __init__.py                           # Library exports for Python import
│   ├── main.py                               # FastAPI server entry point
│   ├── deploy.py                             # Modal deployment entry point
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                         # Unified configuration (classifier + API + Modal)
│   ├── models/                               # Data models and schemas
│   │   ├── __init__.py
│   │   ├── llm_core_models.py                # Core request/response models
│   │   ├── llm_classification_models.py      # Classification result models (shared)
│   │   └── ...
│   ├── services/                             # Core services and ML
│   │   ├── __init__.py
│   │   ├── prompt_task_complexity_classifier.py  # Complete NVIDIA classifier (ML)
│   │   ├── model_registry.py                 # Model metadata and capabilities
│   │   ├── model_router.py                   # Model selection logic
│   │   └── yaml_model_loader.py              # YAML configuration loader
│   └── utils/                                # Utility functions
│       ├── __init__.py
│       └── jwt.py                            # JWT authentication for Modal
├── deploy.py                                 # Modal deployment script (root level)
├── tests/                                    # Test suite (unit + integration)
├── pyproject.toml                            # Dependencies (PyTorch + FastAPI + Modal)
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

# ML Classifier Configuration (for local inference)
CLASSIFIER__MODEL_NAME=nvidia/prompt-task-and-complexity-classifier  # HuggingFace model
CLASSIFIER__DEVICE=cuda          # Device: cuda or cpu (auto-detected by default)
CLASSIFIER__MAX_LENGTH=512       # Maximum token length

# Modal Deployment Configuration (for Modal mode only)
MODAL_DEPLOYMENT__APP_NAME=prompt-task-complexity-classifier  # Modal app name
MODAL_DEPLOYMENT__GPU_TYPE=T4    # GPU type: T4, A10G, A100, etc.
MODAL_DEPLOYMENT__TIMEOUT=600    # Container timeout in seconds
MODAL_DEPLOYMENT__MIN_CONTAINERS=0  # Minimum containers to keep warm
MODAL_DEPLOYMENT__MAX_CONTAINERS=1  # Maximum containers for auto-scaling
JWT_SECRET=your_jwt_secret       # JWT secret for Modal authentication
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
from adaptive_router import get_prompt_classifier, ModelRouter, model_registry

# Initialize classifier
classifier = get_prompt_classifier()

# Classify a prompt
result = classifier.classify_prompt("Write a Python function to sort a list")
print(f"Task: {result['task_type_1']}, Complexity: {result['prompt_complexity_score']}")

# Use router for model selection
router = ModelRouter(model_registry)
models = router.select_models(
    task_complexity=result["prompt_complexity_score"],
    task_type=result["task_type_1"],
    cost_bias=0.5
)
print(f"Recommended model: {models[0].provider} / {models[0].model_name}")
```

### 2. FastAPI Server Mode
Run as HTTP API server with local GPU/CPU inference:

```bash
# Install dependencies
uv install

# Start FastAPI server (local GPU/CPU inference)
uv run adaptive-ai

# Or with custom settings
PORT=8001 uv run adaptive-ai
```

Access API docs at `http://localhost:8000/docs`

### 3. Modal Serverless Mode (Optional)
Deploy to Modal for serverless GPU inference with auto-scaling:

```bash
# Deploy to Modal
modal deploy deploy.py

# Modal will provide URLs for:
# - POST /classify - Single prompt classification
# - POST /classify_batch - Batch classification

# Authenticate requests with JWT Bearer token
curl -X POST https://your-modal-url/classify \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a sorting algorithm"}'
```

## Development Commands

### Local Development
```bash
# Install dependencies
uv install

# Start the service
uv run adaptive-ai

# Start with development settings
DEBUG=true uv run adaptive-ai

# Start on custom port
PORT=8001 uv run adaptive-ai

# Note: Hypercorn workers setting is ignored in programmatic mode
# For multi-process deployment, use a process manager like gunicorn or supervisord
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


## ML Models and Classification (via Modal)

### Task Classification
- **Model Type**: NVIDIA transformer model deployed on Modal GPU infrastructure
- **Input**: Last message content from chat completion request
- **Output**: Task categories (code, math, creative, analysis, etc.) with confidence scores
- **Deployment**: Modal T4 GPU with JWT authentication
- **Performance**: Sub-100ms inference time including network latency

### Domain Classification
- **Model Type**: NVIDIA model with contextual understanding
- **Input**: Full conversation context and prompt content
- **Output**: Domain categories (software, science, business, etc.) with confidence
- **Features**: GPU-accelerated inference on Modal infrastructure
- **Accuracy**: >90% on domain-specific benchmarks

### Cost-Performance Modeling
- **Approach**: Multi-objective optimization using historical performance data
- **Metrics**: Cost per token, response quality, latency, success rate
- **Learning**: Continuous adaptation based on user feedback and outcomes
- **Optimization**: Pareto-optimal trade-offs between cost and quality

## Caching and Performance

### API Performance
- **Framework**: FastAPI with async/await for optimal performance
- **HTTP Client**: httpx with connection pooling for Modal API calls
- **Authentication**: JWT token caching to minimize overhead
- **Retry Logic**: Exponential backoff for resilient Modal communication

### Modal Integration
- **Connection**: HTTP/HTTPS with JWT authentication
- **Timeout**: Configurable request timeout (default: 30s)
- **Retries**: Configurable retry attempts with exponential backoff
- **Health Checks**: Regular Modal service health monitoring

### Request Processing
- **Async Processing**: Native FastAPI async for concurrent requests
- **Error Handling**: Graceful fallbacks when Modal API is unavailable
- **Response Time**: <100ms end-to-end including Modal API latency
- **Throughput**: Limited by Modal service capacity, not local processing

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

Resource requirements vary significantly by deployment mode:

#### Library Mode (Local Inference)
- **CPU**: 2-4 cores recommended for ML inference
- **GPU**: NVIDIA GPU with CUDA support recommended (optional, falls back to CPU)
- **Memory**: 4-8GB RAM for PyTorch models and transformers
- **Storage**: 2-3GB for HuggingFace model cache and application code

#### FastAPI Server Mode (Local Inference)
- **CPU**: 2-4 cores for concurrent request handling + ML inference
- **GPU**: NVIDIA GPU with CUDA support recommended for production workloads
- **Memory**: 4-8GB RAM for ML models, API server, and request processing
- **Storage**: 2-3GB for models, application, and logs

#### Modal Serverless Mode (Remote Inference)
- **CPU**: 1-2 cores sufficient (no local ML inference, only API routing)
- **Memory**: 512MB-1GB RAM (minimal local processing)
- **Storage**: 1GB for application and logs
- **Network**: Reliable internet connection for Modal API calls
- **Modal Resources**: T4 GPU (or configured GPU type) managed by Modal

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
- Review environment variable configuration
- Verify Hypercorn is correctly installed: `uv run hypercorn --version`

**Local ML model loading failures** (Library/FastAPI modes)
- Verify PyTorch and transformers are installed: `uv sync`
- Check HuggingFace Hub connectivity for model downloads
- Verify CUDA drivers if using GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Check available disk space for model cache (~2-3GB required)
- Review CLASSIFIER__DEVICE setting (auto, cuda, or cpu)

**Modal API connection failures** (Modal mode only)
- Verify Modal deployment is active: `modal app list`
- Check JWT_SECRET environment variable matches Modal secret
- Verify network connectivity to Modal service
- Check Modal service health status

**Classification errors** (All modes)
- Verify input format matches expected schema
- Check prompt length is within model limits (default: 512 tokens)
- Review classification result structure
- Enable debug logging: `DEBUG=true uv run adaptive-ai`

**Performance issues**

*Library/FastAPI modes (local inference):*
- Monitor GPU utilization: `nvidia-smi` (if using CUDA)
- Check CPU usage and available cores
- Verify model is loaded in memory (first inference is slower)
- Consider reducing CLASSIFIER__MAX_LENGTH for faster processing
- Review batch size if processing multiple prompts

*Modal mode (remote inference):*
- Check Modal API response times
- Verify JWT token generation overhead
- Monitor network latency to Modal service
- Consider adjusting request timeout settings

### Debug Commands

**Library Mode:**
```bash
# Test local classifier
python -c "
from adaptive_router import get_prompt_classifier
classifier = get_prompt_classifier()
result = classifier.classify_prompt('Write a Python sorting function')
print(result)
"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

**FastAPI Server Mode:**
```bash
# Enable debug logging
DEBUG=true uv run adaptive-ai

# Check service health
curl -X GET http://localhost:8000/health

# Test model selection endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a sorting algorithm", "cost_bias": 0.5}'

# Monitor GPU usage (if CUDA enabled)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

**Modal Serverless Mode:**
```bash
# Check Modal deployment status
modal app list

# Test Modal function
modal run deploy.py::select_model --prompt "Hello world"

# View Modal logs
modal app logs prompt-task-complexity-classifier
```

## Performance Benchmarks

Performance characteristics vary by deployment mode:

### Library/FastAPI Mode (Local Inference)
- **Single Request**: 50-200ms (depends on CPU/GPU, faster after model warmup)
- **Batch Processing**: 100-500 requests/second (GPU-accelerated)
- **Memory Usage**: 4-8GB with all models loaded in memory
- **First Request Latency**: 2-5 seconds (model loading time)
- **GPU Speedup**: 3-5x faster than CPU inference

### Modal Serverless Mode (Remote Inference)
- **Single Request**: 100-300ms end-to-end (including network latency)
- **Cold Start**: 5-10 seconds for first request (container initialization)
- **Warm Request**: <100ms after container warmup
- **Concurrent Requests**: Auto-scales based on load (up to max_containers)
- **Network Overhead**: 20-50ms typical latency to Modal infrastructure

### Cost Optimization Results
- **Typical Savings**: 30-70% compared to always using premium models
- **Quality Retention**: >95% user satisfaction with selected models
- **Accuracy**: >90% optimal model selection for classified tasks
- **Latency Impact**: <100ms additional latency vs. direct routing

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