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

The adaptive_ai service is a Python-based ML microservice that provides intelligent model selection for the Adaptive LLM infrastructure. Built with LitServe ML serving framework, it analyzes prompts using machine learning classifiers to select the optimal provider and model for each request, enabling significant cost savings and performance optimization.

## Key Features

- **ML-Powered Model Selection**: Uses PyTorch and scikit-learn classifiers for intelligent routing
- **Task Classification**: Categorizes prompts by complexity and task type (code, math, creative, etc.)
- **Domain Classification**: Identifies specialized domains for targeted model selection
- **Cost Optimization**: Balances performance vs. cost based on user preferences and prompt analysis
- **Protocol Management**: Decides between standard LLM calls vs. specialized "minion" protocols
- **High-Performance Serving**: LitServe framework for sub-100ms inference with batch processing

## Technology Stack

- **Framework**: LitServe 0.2.10+ for ML model serving
- **ML Libraries**: PyTorch 2.2+, scikit-learn 1.7+, HuggingFace Transformers
- **NLP**: Sentence Transformers for embeddings, Tiktoken for tokenization
- **LLM Integration**: LangChain for orchestration and provider abstraction
- **Caching**: In-memory caching with cachetools for classification results
- **Configuration**: Pydantic Settings for type-safe configuration management

## Project Structure

```
adaptive_ai/
├── adaptive_ai/
│   ├── __init__.py
│   ├── main.py                    # LitServe application entry point
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
HOST=0.0.0.0                     # LitServe host
PORT=8000                        # LitServe port

# LitServe Configuration
LITSERVE_MAX_BATCH_SIZE=32       # Maximum batch size for inference
LITSERVE_BATCH_TIMEOUT=0.01      # Batch timeout in seconds
LITSERVE_ACCELERATOR=auto        # Hardware accelerator (auto/cpu/gpu)
LITSERVE_DEVICES=auto           # Device configuration

# ML Model Configuration
MODEL_CACHE_DIR=/tmp/models      # Directory for cached models
ENABLE_MODEL_CACHING=true        # Enable model caching
CLASSIFICATION_THRESHOLD=0.7     # Confidence threshold for classification

# Performance Configuration
ENABLE_EMBEDDINGS_CACHE=true     # Enable embedding caching
CACHE_TTL=3600                   # Cache TTL in seconds
MAX_CACHE_SIZE=10000            # Maximum cache entries
```

### Optional Configuration

```bash
# Debugging
DEBUG=false                      # Enable debug logging
LOG_LEVEL=INFO                  # Logging level

# Model Paths (will download from HuggingFace if not specified)
TASK_CLASSIFIER_PATH=/path/to/task/model
DOMAIN_CLASSIFIER_PATH=/path/to/domain/model
EMBEDDINGS_MODEL_PATH=/path/to/embeddings/model
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
```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_model_selector.py

# Run with verbose output
uv run pytest -v

# Run tests and generate HTML coverage report
uv run pytest --cov --cov-report=html
```

## API Interface

The service exposes a LitServe API that accepts model selection requests and returns orchestration responses.

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
        "candidates": [
            {"provider": "openai", "model": "gpt-4"},
            {"provider": "anthropic", "model": "claude-3-sonnet"}
        ],
        "preference": "balanced",
        "max_cost_per_token": 0.001
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


## ML Models and Classification

### Task Classification
- **Model Type**: Transformer-based text classifier
- **Input**: Last message content from chat completion request
- **Output**: Task categories (code, math, creative, analysis, etc.) with confidence scores
- **Training**: Fine-tuned on diverse prompt datasets
- **Performance**: Sub-50ms inference time

### Domain Classification
- **Model Type**: Sentence transformer with cosine similarity
- **Input**: Full conversation context and prompt content
- **Output**: Domain categories (software, science, business, etc.) with confidence
- **Features**: Contextual understanding of specialized domains
- **Accuracy**: >90% on domain-specific benchmarks

### Cost-Performance Modeling
- **Approach**: Multi-objective optimization using historical performance data
- **Metrics**: Cost per token, response quality, latency, success rate
- **Learning**: Continuous adaptation based on user feedback and outcomes
- **Optimization**: Pareto-optimal trade-offs between cost and quality

## Caching and Performance

### Classification Caching
- **Cache Key**: Hash of prompt content and configuration
- **TTL**: Configurable (default: 1 hour)
- **Storage**: In-memory with LRU eviction
- **Hit Rate**: Typically 60-80% for production workloads

### Model Caching
- **Strategy**: Lazy loading with persistent storage
- **Location**: Local filesystem or shared storage
- **Models**: HuggingFace models cached after first download
- **Size**: ~2-5GB total for all classification models

### Batch Processing
- **Batch Size**: Configurable (default: 32 requests)
- **Timeout**: Configurable (default: 10ms)
- **Benefits**: 5-10x throughput improvement for batch workloads
- **Latency**: <100ms end-to-end including ML inference

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
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv install --frozen

COPY adaptive_ai/ ./adaptive_ai/
EXPOSE 8000

CMD ["uv", "run", "adaptive-ai"]
```

### Docker Compose
The service is included in the root `docker-compose.yml` with proper networking and resource allocation.

### Resource Requirements
- **CPU**: 2-4 cores recommended for production
- **Memory**: 4-8GB RAM (includes ML model storage)
- **Storage**: 10GB for model caching and logs
- **GPU**: Optional, can accelerate inference by 2-3x

## Troubleshooting

### Common Issues

**Service won't start**
- Check Python version (3.10+ required)
- Verify all dependencies installed: `uv install`
- Check port availability (default: 8000)
- Review environment variable configuration

**Model loading failures**
- Verify internet connection for HuggingFace downloads
- Check available disk space (models require ~5GB)
- Review cache directory permissions
- Ensure sufficient memory (8GB+ recommended)

**Classification errors**
- Verify input format matches expected schema
- Check tokenization errors with tiktoken
- Review model file integrity
- Monitor memory usage during inference

**Performance issues**
- Enable model caching: `ENABLE_MODEL_CACHING=true`
- Adjust batch size: `LITSERVE_MAX_BATCH_SIZE=16`
- Monitor cache hit rates
- Consider GPU acceleration

### Debug Commands
```bash
# Enable debug logging
DEBUG=true uv run adaptive-ai

# Check model loading
python -c "from adaptive_ai.services.prompt_classifier import get_prompt_classifier; print('Models loaded successfully')"

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