# Adaptive Backend

OpenAI-compatible API server providing intelligent LLM routing. Built with Go for high performance and reliability.

## Overview

High-performance Go server that acts as a unified gateway to multiple LLM providers while maintaining full OpenAI API compatibility and adding intelligent model selection.

### Key Features

- **OpenAI Compatible**: Drop-in replacement, just change the base URL
- **Smart Routing**: AI service selects optimal model for each prompt
- **Multi-Provider**: OpenAI, Anthropic, Groq, DeepSeek, Google AI
- **Unified Streaming**: Converts all provider responses to OpenAI format
- **Cost Optimization**: Routes to cheaper models when appropriate
- **High Performance**: Built with Go for low latency and high throughput

## Quick Start

```bash
# Install dependencies
go mod tidy

# Set environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Run server
go run cmd/api/main.go
```

### Docker

```bash
docker build -t adaptive-backend .
docker run -p 8080:8080 --env-file .env.local adaptive-backend
```

## Configuration

### Required Environment Variables

```bash
# Provider API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
GOOGLE_AI_API_KEY=...

# Database Configuration
DB_SERVER=localhost
DB_NAME=adaptive
DB_USER=sa
DB_PASSWORD=your-password
DB_PORT=5432

# Service Configuration
AI_SERVICE_URL=http://localhost:8000
ADDR=:8080
```

## API Usage

### Chat Completions

**POST** `/v1/chat/completions`

```json
{
  "model": "adaptive",
  "messages": [
    {"role": "user", "content": "Explain quantum computing"}
  ],
  "stream": false,
  "temperature": 0.7,
  "provider_constraints": ["openai", "anthropic"],
  "cost_bias": 0.3
}
```

**Parameters:**
- `model`: Must be "adaptive" for intelligent routing
- `provider_constraints`: Array of allowed providers (optional)
- `cost_bias`: Float 0-1 (0 = cost-optimized, 1 = performance-optimized)

### List Models

**GET** `/v1/models`

Returns available models across all providers.

### Health Check

**GET** `/health`

Returns service health status and dependencies.

## Development

```bash
# Run tests
go test ./...

# Run with race detection
go test -race ./...

# Build for production
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o bin/adaptive-backend cmd/api/main.go
```

## Monitoring

- **Metrics**: Exposed at `/metrics` (Prometheus format)
- **Logging**: Structured JSON logging with configurable levels
- **Health Checks**: `/health`, `/health/live`, `/health/startup`

## Architecture

The server receives OpenAI-compatible requests, sends prompts to the AI service for model selection, routes requests to the selected provider, and returns unified responses in OpenAI format.

## Security

- API key-based authentication
- Rate limiting per API key
- TLS encryption for all communications
- No sensitive data logging