# Adaptive AI Service

Python ML service that intelligently selects optimal LLM models based on prompt analysis. Built with LitServe for high-performance inference.

## Overview

The brain of the Adaptive platform, responsible for analyzing prompts and selecting the optimal language model for each request using advanced ML techniques.

### Key Features

- **Multi-Dimensional Analysis**: Analyzes prompts across 7 dimensions (creativity, reasoning, context, etc.)
- **Task-Specific Selection**: Maps tasks to optimal models based on task type
- **Provider Constraints**: Supports filtering by specific providers
- **Cost Bias Control**: Configurable cost vs. performance trade-offs
- **Protocol Selection**: Supports standard_llm, minion, and minions_protocol
- **High Performance**: LitServe for fast batch inference
- **Embedding Cache**: Intelligent caching for similar prompts

## Quick Start

```bash
# Install dependencies
pip install uv
uv venv
uv sync

# Set environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Run service
uv run python adaptive_ai/main.py
```

### Docker

```bash
docker build -t adaptive-ai .
docker run -p 8000:8000 --env-file .env.local adaptive-ai
```

## Configuration

### Required Environment Variables

```bash
# Hugging Face Configuration
HUGGINGFACE_TOKEN=hf_xxxxx

# Service Configuration
ADAPTIVE_AI_ENVIRONMENT=production
ADAPTIVE_AI_LOG_LEVEL=INFO
ADAPTIVE_AI_HOST=0.0.0.0
ADAPTIVE_AI_PORT=8000

# LitServe Configuration
LITSERVE_MAX_BATCH_SIZE=10
LITSERVE_BATCH_TIMEOUT=1.0
LITSERVE_ACCELERATOR=cpu
```

## API Usage

### Model Selection

**POST** `/predict`

```json
{
  "prompt": "Write a Python function to sort a list",
  "user_id": "user123",
  "provider_constraint": ["openai", "anthropic"],
  "cost_bias": 0.3
}
```

**Parameters:**
- `prompt`: The input prompt to analyze
- `provider_constraint`: Array of allowed providers (optional)
- `cost_bias`: Float 0-1 (0 = cost-optimized, 1 = performance-optimized)

### Health Check

**GET** `/health`

Returns service health status and model loading state.

## How It Works

1. **Prompt Analysis**: Uses custom DeBERTa model for 7-dimensional classification
2. **Task Mapping**: Maps task types to optimal model candidates
3. **Provider Filtering**: Applies provider constraints and cost bias
4. **Protocol Selection**: Uses internal LLM to select optimal protocol
5. **Caching**: Caches results for similar prompts

### Classification Dimensions

- **Task Type**: Open QA, Code Generation, Summarization, etc.
- **Creativity Scope**: Creative vs. factual content needs
- **Reasoning**: Simple vs. complex reasoning requirements
- **Contextual Knowledge**: Context length and complexity
- **Domain Knowledge**: Specialized domain requirements
- **Few-shot Learning**: Examples needed for the task
- **Constraint Handling**: Rule/constraint complexity

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=adaptive_ai

# Development mode with hot reload
uv run watchfiles adaptive_ai.main:main
```

## Adding New Models

1. **Add Model Capability** in `config/model_catalog.py`
2. **Add Provider Type** in `models/llm_enums.py`
3. **Update Task Mappings** in `config/model_catalog.py`

## Performance

- **Prompt Classification**: <50ms per request
- **Model Selection**: <20ms per request
- **Cache Hit Rate**: 60-80% for similar prompts
- **Throughput**: 100+ requests/second with batching

## Security

- **Input Validation**: All inputs validated with Pydantic models
- **No Data Storage**: Prompts not stored after processing
- **Encrypted Communication**: All external API calls encrypted
- **Rate Limiting**: Built-in rate limiting via LitServe