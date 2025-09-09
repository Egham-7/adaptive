# Adaptive AI - Intelligent Model Selection Service

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
- **Search docs**: Use `mcp__Ref__ref_search_documentation` for Python, LitServe, PyTorch, HuggingFace, scikit-learn documentation
- **Read specific docs**: Use `mcp__Ref__ref_read_url` to read documentation pages

## Overview

The adaptive_ai service is a Python-based ML microservice that provides intelligent model selection for the Adaptive LLM infrastructure. Built with FastAPI web framework, it analyzes prompts using Modal-deployed machine learning classifiers to select the optimal provider and model for each request, enabling significant cost savings and performance optimization.

## Key Features

- **ML-Powered Model Selection**: Uses Modal-deployed NVIDIA classifiers for intelligent routing
- **Task Classification**: Categorizes prompts by complexity and task type (code, math, creative, etc.) via Modal API
- **Domain Classification**: Identifies specialized domains for targeted model selection
- **Cost Optimization**: Balances performance vs. cost based on user preferences and prompt analysis
- **Protocol Management**: Decides between standard LLM calls vs. specialized "minion" protocols
- **High-Performance API**: FastAPI framework with OpenAPI documentation and async endpoints

## Technology Stack

- **Framework**: FastAPI 0.104+ for high-performance API serving
- **ASGI Server**: Hypercorn 0.17+ with HTTP/1.1, HTTP/2, and WebSocket support
- **HTTP Client**: httpx for async API calls to Modal service
- **Authentication**: python-jose for JWT token handling
- **Model Registry**: Static model metadata (no local ML models)
- **LLM Integration**: LangChain for orchestration and provider abstraction
- **Configuration**: Pydantic Settings for type-safe configuration management

## Project Structure

```
adaptive_ai/
├── adaptive_ai/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py             # Pydantic settings and configuration
│   ├── models/                   # Data models and schemas
│   │   ├── __init__.py
│   │   ├── llm_core_models.py    # Core request/response models
│   │   ├── llm_classification_models.py  # Classification result models
│   │   ├── llm_enums.py          # Enums for providers, tasks, domains
│   │   ├── llm_orchestration_models.py   # Orchestration response models
│   │   └── unified_model.py      # Unified model interfaces
│   ├── services/                 # Core ML services and business logic
│   │   ├── __init__.py
│   │   ├── model_selector.py     # Model selection algorithms
│   │   ├── prompt_classifier.py  # Task type classification
│   │   ├── domain_classifier.py  # Domain-specific classification
│   │   ├── cost_optimizer.py     # Cost optimization logic
│   │   ├── model_registry.py     # Model metadata and capabilities
│   │   ├── classification_result_embedding_cache.py # Caching layer
│   │   └── unified_model_selector.py  # Unified selection interface
│   ├── config/                   # Provider and task configuration
│   │   ├── __init__.py
│   │   ├── providers.py          # Provider metadata and pricing
│   │   ├── task_mappings.py      # Task-to-model mappings
│   │   └── domain_mappings.py    # Domain-specific configurations
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── openai_utils.py       # OpenAI API utilities
├── tests/                        # Test suite
├── pyproject.toml               # Dependencies and tool configuration
├── uv.lock                      # Dependency lock file
├── Dockerfile                   # Container configuration
└── README.md                    # Service documentation
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

# Modal API Configuration
MODAL_CLASSIFIER_URL=https://...  # Modal classifier service URL
JWT_SECRET=your_jwt_secret        # JWT secret for Modal authentication
MODAL_REQUEST_TIMEOUT=30          # Request timeout in seconds
MODAL_MAX_RETRIES=3              # Maximum retry attempts
MODAL_RETRY_DELAY=1.0            # Base retry delay in seconds
```

### Optional Configuration

```bash
# Debugging
DEBUG=false                      # Enable debug logging
LOG_LEVEL=INFO                  # Logging level

# Additional Modal Configuration
MODAL_HEALTH_CHECK_INTERVAL=60   # Health check interval in seconds

# Hypercorn-Specific Configuration (Optional)
# For HTTP/2 support (requires SSL/TLS):
# HYPERCORN_CERTFILE=cert.pem     # SSL certificate file
# HYPERCORN_KEYFILE=key.pem       # SSL private key file
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
uv run pytest adaptive_ai/tests/unit/core/test_config.py

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
    "model_router": {
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

### Model Selection Service
**File**: `adaptive_ai/services/model_selector.py`

- Analyzes prompt characteristics and user preferences
- Selects optimal models based on task classification and domain
- Provides cost-performance trade-off analysis
- Supports both standard LLM and specialized "minion" protocols

### Prompt Classification Service
**File**: `adaptive_ai/services/prompt_classifier.py`

- Uses ML models to classify prompt task types (code, math, creative, etc.)
- Determines complexity levels and processing requirements
- Provides confidence scores for classification decisions
- Supports batch processing for high throughput

### Domain Classification Service
**File**: `adaptive_ai/services/domain_classifier.py`

- Identifies specialized domains requiring specific model capabilities
- Maps domains to optimal provider/model combinations
- Enables domain-specific parameter tuning
- Supports custom domain extensions

### Cost Optimization Service
**File**: `adaptive_ai/services/cost_optimizer.py`

- Analyzes cost-performance trade-offs across providers
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
**File**: `adaptive_ai/config/providers.py`

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
**File**: `adaptive_ai/config/task_mappings.py`

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
- **CPU**: 1-2 cores sufficient (no local ML inference)
- **Memory**: 512MB-1GB RAM (minimal local processing)
- **Storage**: 1GB for application and logs
- **Network**: Reliable internet connection for Modal API calls

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

**Modal API connection failures**
- Verify MODAL_CLASSIFIER_URL is correct and accessible
- Check JWT_SECRET environment variable is set
- Verify network connectivity to Modal service
- Check Modal service health status

**Classification errors**
- Verify input format matches expected schema
- Check Modal API response format
- Review JWT token generation and validation
- Monitor network latency to Modal service

**Performance issues**
- Check Modal API response times
- Verify JWT token caching is working
- Monitor httpx connection pool usage
- Consider adjusting request timeout settings

### Debug Commands
```bash
# Enable debug logging
DEBUG=true uv run adaptive-ai

# Check Modal API health
curl -X GET http://localhost:8000/health

# Test classification endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"chat_completion_request": {"messages": [{"role": "user", "content": "Hello"}]}}'

# Monitor resource usage
pip install psutil && python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

## Performance Benchmarks

### Classification Speed
- **Single Request**: <50ms end-to-end
- **Batch Processing**: 500+ requests/second
- **Memory Usage**: ~2-4GB with all models loaded
- **Cache Hit Rate**: 60-80% in production

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