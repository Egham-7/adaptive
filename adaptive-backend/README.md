# Adaptive Backend

High-performance Go API server providing OpenAI-compatible endpoints with intelligent LLM routing and 60-80% cost savings.

## 🚀 Quick Start

```bash
go mod tidy
cp .env.example .env.local  # Add your API keys
go run cmd/api/main.go
```

## ⭐ Key Features

- **OpenAI Drop-in Replacement** - Same API, 60-80% cost reduction
- **Intelligent Model Routing** - AI-powered selection based on prompt complexity
- **Multi-Provider Support** - OpenAI, Anthropic, Groq, DeepSeek, Gemini, Grok
- **Dual Caching System** - Prompt-response cache + semantic caching
- **Circuit Breaker Pattern** - Automatic provider health monitoring and failover
- **High Performance** - 1000+ req/s with <100ms overhead
- **Redis Integration** - Circuit breaker state and prompt caching

## 💡 API Usage

### Chat Completions
`POST /v1/chat/completions`

**Simple intelligent routing:**
```json
{
  "model": "",  // Empty string = smart routing + cost savings
  "messages": [{"role": "user", "content": "Hello"}]
}
```

**Advanced configuration:**
```json
{
  "model": "",
  "messages": [{"role": "user", "content": "Complex analysis task"}],
  "model_router": {
    "cost_bias": 0.3,  // 0 = cheapest, 1 = best performance
    "models": [
      {"provider": "openai"},
      {"provider": "anthropic", "model_name": "claude-3-sonnet"}
    ]
  },
  "prompt_response_cache": {
    "enabled": true,
    "semantic_threshold": 0.85
  },
  "fallback": {
    "mode": "sequential"  // or "race"
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

## 🛠️ Development

### Quick Commands
```bash
go test ./...                    # Run tests
go build -o main cmd/api/main.go # Build binary
```

### Code Quality (Required Before Commit)
```bash
# REQUIRED: Format and lint code
gofmt -w .
golangci-lint run

# REQUIRED: Modernize and clean
go mod tidy
gofumpt -w .
go run golang.org/x/tools/cmd/goimports@latest -w .

# Verify
go vet ./...
go test ./...
```

### Install Quality Tools
```bash
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install mvdan.cc/gofumpt@latest
```

### Pre-commit Checklist
- [ ] `gofmt -w .` - Code formatted
- [ ] `golangci-lint run` - No linter issues  
- [ ] `go test ./...` - All tests pass
- [ ] `go mod tidy` - Dependencies clean