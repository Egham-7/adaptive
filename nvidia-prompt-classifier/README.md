# NVIDIA Prompt Classifier

High-performance Modal deployment of NVIDIA's prompt task and complexity classifier with GPU acceleration and production-ready features.

## Overview

This project provides a clean, production-ready deployment of the NVIDIA prompt classifier model using Modal's serverless GPU infrastructure. The classifier analyzes prompts to determine task types, complexity scores, and various metrics useful for intelligent model routing.

## Features

- **🎮 GPU Acceleration**: Deployed on NVIDIA T4 GPUs for fast inference
- **⚡ High Performance**: Simple, direct GPU processing
- **🔐 Secure**: JWT authentication with token caching
- **📦 Clean Architecture**: Organized package structure following Modal best practices
- **🚀 Auto-scaling**: Dynamic container scaling based on load
- **🛡️ Production Ready**: Comprehensive error handling, retries, and monitoring

## Project Structure

```
nvidia-prompt-classifier/
├── deploy.py                    # Modal deployment script (single file)
├── nvidia_classifier/           # Minimal ML package
│   ├── __init__.py             # Package exports
│   └── nvidia_model.py         # Complete NVIDIA model implementation
├── tests/                      # Test suite
│   ├── integration/            # Integration tests
│   └── unit/                   # Unit tests (placeholder)
├── pyproject.toml              # Dependencies and configuration
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.10+
- Modal account and CLI installed
- NVIDIA model access (automatic via HuggingFace)

### Installation

```bash
# Install dependencies
uv install

# Set up Modal (if not already done)
modal setup
```

### Deployment

```bash
# Deploy to Modal
modal deploy deploy.py

# The service will be available at the Modal-provided URL
```

### Environment Setup

Create a Modal secret named `jwt` with your JWT authentication key:

```bash
modal secret create jwt jwt_auth="your-secret-key-here"
```

## API Endpoints

### POST `/classify`
Batch classification of multiple prompts.

**Request:**
```json
{
  "prompts": ["Write a Python function", "Solve 2+2"]
}
```

**Response:**
```json
{
  "task_type_1": ["Code Generation", "Closed QA"],
  "task_type_2": ["NA", "NA"],
  "task_type_prob": [0.95, 0.87],
  "creativity_scope": [0.3, 0.1],
  "reasoning": [0.7, 0.2],
  "contextual_knowledge": [0.4, 0.1],
  "prompt_complexity_score": [0.65, 0.25],
  "domain_knowledge": [0.8, 0.1],
  "number_of_few_shots": [0, 0],
  "no_label_reason": [0.05, 0.13],
  "constraint_ct": [0.2, 0.1]
}
```

### POST `/classify/single`
Single prompt classification (same response format as batch).

### GET `/health`
Basic health check (no authentication required).

## Performance Characteristics

- **Latency**: ~200-500ms per batch including GPU processing
- **Throughput**: Up to 20 concurrent containers under load
- **Batch Size**: Supports up to 100 prompts per request
- **Concurrency**: Modal handles concurrent requests automatically

## Model Details

- **Model**: `nvidia/prompt-task-and-complexity-classifier`
- **Architecture**: DeBERTa-v3-base backbone with multi-head classification
- **GPU**: NVIDIA T4 (16GB VRAM) 
- **Framework**: PyTorch + Transformers + Accelerate

## Configuration

Key configuration options in `deploy.py`:

```python
MODEL_NAME = "nvidia/prompt-task-and-complexity-classifier"
GPU_TYPE = "T4"
APP_NAME = "nvidia-prompt-classifier"

# ML Container Configuration
container_idle_timeout=300      # 5 min warm containers
max_containers=20              # Scale up to 20 GPU containers
min_containers=1               # Keep 1 container warm

# Web Container Configuration  
container_idle_timeout=60       # Web containers scale down faster
max_containers=10              # Limit web containers (lightweight)
cpu=2                          # 2 CPUs for JSON processing
```

## Authentication

All protected endpoints require JWT authentication:

```bash
curl -X POST "https://your-modal-url/classify" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello world"]}'
```

## Development

### Local Testing

```bash
# Run integration tests
uv run pytest tests/integration/ -v

# Test specific endpoint
uv run pytest tests/integration/test_classify_endpoint.py::test_classify_endpoint -v
```

### Code Quality

```bash
# Format code
uv run black .

# Type checking
uv run mypy .

# Linting
uv run ruff check .
```

## Architecture

### Dual Image Strategy

Following Modal best practices, the deployment uses two optimized images:

1. **ML Image** (`ml_image`): GPU-enabled container with PyTorch and ML dependencies
2. **Web Image** (`web_image`): Lightweight FastAPI container for API endpoints

### Resource Optimization

- **GPU Containers**: Models download automatically on first use and are cached
- **Web Containers**: Minimal dependencies for lightweight FastAPI serving
- **Scaling**: Independent scaling of ML vs web containers based on load
- **Caching**: JWT token caching and efficient model loading

### Performance Features

- **Direct Processing**: Simple, synchronous GPU inference
- **Connection Pooling**: Efficient HTTP client configuration
- **Error Handling**: Comprehensive error handling with fallback responses

## Monitoring

The deployment includes several monitoring endpoints:

- Health checks at `/health` and `/health/detailed`
- Performance benchmarking at `/benchmark`
- Structured logging with request correlation
- Container metrics via Modal dashboard

## Cost Optimization

- **Dynamic Scaling**: Containers scale down when idle to minimize costs
- **Resource Right-sizing**: Separate resource allocation for ML vs web containers
- **Efficient Batching**: Process multiple prompts per GPU invocation
- **Smart Caching**: Model caching and efficient container reuse reduce cold start times

## Contributing

1. Make changes to the codebase
2. Update tests as needed
3. Run code quality checks: `uv run black . && uv run mypy . && uv run ruff check .`
4. Test the deployment: `modal deploy deploy.py`
5. Submit pull request

## License

This project follows the same license as the parent adaptive repository.