# Adaptive Backend

Go API server providing OpenAI-compatible endpoints with intelligent LLM routing.

## Quick Start

```bash
go mod tidy
cp .env.example .env.local  # Add your API keys
go run cmd/api/main.go
```

## Features

- **OpenAI Compatible** - Drop-in replacement, just change base URL
- **Smart Routing** - AI service selects optimal models automatically  
- **Multi-Provider** - OpenAI, Anthropic, Groq, DeepSeek, Gemini
- **Advanced Caching** - Dual-layer caching (prompt + semantic)
- **Circuit Breakers** - Automatic fallback on provider failures
- **Streaming Support** - SSE streaming for all providers

## API

### Chat Completions
`POST /v1/chat/completions`

```json
{
  "model": "",  // Empty string enables intelligent routing
  "messages": [{"role": "user", "content": "Hello"}],
  "protocol_manager": {
    "cost_bias": 0.3  // 0 = cheapest, 1 = best performance
  },
  "prompt_cache": {
    "enabled": true,
    "ttl": 3600
  }
}
```

### Other Endpoints
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Tech Stack

- **Go 1.24** with Fiber web framework
- **OpenAI Go SDK** for provider integrations
- **Redis** for caching layer
- **Prometheus** metrics

## Development

```bash
go test ./...           # Run tests
go build cmd/api/main.go # Build binary
go fmt ./...            # Format code
go vet ./...            # Static analysis
```