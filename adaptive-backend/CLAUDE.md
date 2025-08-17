# Adaptive Backend - Go API Server

## Overview

The adaptive-backend is a high-performance Go API server that provides an OpenAI-compatible chat completions API with intelligent multi-provider routing. Built with Fiber web framework, it serves as the main entry point for the Adaptive LLM infrastructure.

## Key Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/chat/completions` endpoint
- **Multi-Provider Support**: Routes requests to OpenAI, Anthropic, Gemini, Groq, DeepSeek, Grok, HuggingFace
- **Circuit Breaker Pattern**: Automatic failover and health monitoring
- **High Performance**: Built for 1000+ req/s with <100ms overhead
- **Rate Limiting**: API key-based rate limiting with sliding window algorithm
- **Semantic Caching**: Response caching for improved performance and cost reduction

## Technology Stack

- **Framework**: Go 1.24+ with Fiber v2 (Express-like HTTP framework)
- **HTTP Client**: FastHTTP for high-performance networking
- **Concurrency**: Pond v2 worker pools for request handling
- **Caching**: Custom semantic cache integration
- **Authentication**: API key-based authentication middleware
- **Monitoring**: Built-in health checks and observability

## Project Structure

```
adaptive-backend/
├── cmd/api/main.go           # Application entry point
├── internal/
│   ├── api/                  # HTTP handlers and routes
│   │   ├── completions.go    # Chat completions handler
│   │   └── health.go         # Health check handler
│   ├── middleware/           # HTTP middleware
│   │   └── auth.go          # API key authentication
│   ├── models/              # Data models and protocols
│   │   ├── completions.go   # OpenAI-compatible request/response models
│   │   ├── prompt_cache.go  # Caching models
│   │   └── protocol_manager.go # Protocol management models
│   ├── services/            # Business logic and providers
│   │   ├── auth/            # Authentication services
│   │   ├── cache/           # Caching services
│   │   ├── chat/            # Chat completion services
│   │   ├── circuitbreaker/  # Circuit breaker implementation
│   │   ├── protocol_manager/ # Protocol management
│   │   └── providers/       # LLM provider implementations
│   └── utils/               # Shared utilities
├── go.mod                   # Go module dependencies
└── Dockerfile              # Container configuration
```

## Environment Configuration

Required environment variables:

```bash
# Server Configuration
ADDR=:8080                              # Server listen address
ENV=development                         # Environment (development/production)
LOG_LEVEL=info                         # Log level (trace, debug, info, warn, error)
ALLOWED_ORIGINS=http://localhost:3000   # CORS allowed origins

# Provider API Keys
OPENAI_API_KEY=sk-...                  # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-...           # Anthropic API key
GEMINI_API_KEY=...                     # Gemini API key
GROQ_API_KEY=gsk_...                   # Groq API key
DEEPSEEK_API_KEY=sk-...                # DeepSeek API key
# ... other provider keys

# Cache Configuration (optional)
REDIS_URL=redis://localhost:6379       # Redis URL for caching
CACHE_TTL=3600                         # Cache TTL in seconds
```

## Configuration via YAML

**YAML configuration is the preferred method** for configuring the backend service. Create a `config.yaml` file in the project root:

```yaml
---
# Adaptive Backend Configuration
# Environment variables can be referenced using ${VAR} or ${VAR:-default} syntax

server:
  addr: "${ADDR:-:8080}"
  allowed_origins: "${ALLOWED_ORIGINS:-http://localhost:3000}"
  environment: "${ENV:-development}"
  log_level: "${LOG_LEVEL:-info}"

# Provider configurations
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"  # Optional custom base URL
  
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com/v1"
  
  gemini:
    api_key: "${GEMINI_API_KEY}"
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"

services:
  adaptive_ai:
    base_url: "${ADAPTIVE_AI_BASE_URL:-http://localhost:8000}"
  redis:
    url: "${REDIS_URL:-redis://localhost:6379}"
```

**Environment Variable Support**: Environment variables can still be used by referencing them from YAML using `${VAR_NAME}` or `${VAR_NAME:-default_value}` syntax. This allows for flexible configuration while maintaining the structured YAML format.

## Development Commands

### Local Development
```bash
# Start development server
go run cmd/api/main.go

# Build binary
go build -o main cmd/api/main.go

# Run with hot reload (install air first: go install github.com/air-verse/air@latest)
air
```

### Testing
```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run tests with coverage
go test -cover ./...

# Run specific test package
go test ./internal/services/providers/...
```

### Code Quality
```bash
# Format code
go fmt ./...

# Run static analysis
go vet ./...

# Clean up dependencies
go mod tidy

# Check for security vulnerabilities
go list -json -deps ./... | nancy sleuth
```

## API Endpoints

### Chat Completions
```http
POST /v1/chat/completions
Content-Type: application/json
X-Stainless-API-Key: your-api-key

{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}
```

### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "services": {
    "adaptive_ai": "healthy",
    "redis": "healthy"
  }
}
```

## Provider Integration

The backend supports multiple LLM providers through a unified interface:

### Supported Providers
- **OpenAI**: GPT-4, GPT-3.5, etc.
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku, etc.
- **Gemini**: Gemini Pro, Gemini Flash
- **Groq**: Llama models with ultra-fast inference
- **DeepSeek**: DeepSeek Coder, DeepSeek Chat
- **Grok**: xAI's Grok models
- **HuggingFace**: Various open-source models

### Provider Selection
The backend coordinates with the adaptive_ai service to intelligently select the best provider based on:
- Prompt complexity and type
- Cost optimization preferences
- Provider availability and health
- Performance requirements

## Circuit Breaker & Resilience

### Circuit Breaker States
- **Closed**: Normal operation, requests flow through
- **Open**: Provider unavailable, requests fail fast
- **Half-Open**: Testing if provider has recovered

### Fallback Strategy
1. Primary provider selection via adaptive_ai service
2. Automatic fallback to secondary providers on failure
3. Circuit breaker prevents cascading failures
4. Real-time health monitoring and recovery

## Performance Optimizations

### Connection Pooling
- FastHTTP for efficient HTTP connections
- Connection reuse across requests
- Configurable pool sizes per provider

### Caching
- Semantic cache for similar prompts
- Redis-backed distributed caching
- Configurable TTL and cache keys
- Cache invalidation strategies

### Rate Limiting
- Sliding window rate limiting per API key
- Configurable limits (default: 1000 req/min)
- Graceful degradation under load
- Rate limit headers in responses

## Monitoring & Observability

### Health Checks
- Service health endpoints
- Provider availability monitoring
- Dependency health verification
- Graceful startup waiting for dependencies

### Logging
- Structured logging with configurable levels
- Request/response logging in development
- Error tracking and stack traces
- Performance metrics logging

### Metrics (Future)
- Request latency and throughput
- Provider performance metrics
- Error rates and success rates
- Cache hit/miss ratios

## Deployment

### Docker
```dockerfile
# Build stage
FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o main cmd/api/main.go

# Runtime stage
FROM alpine:3.18
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/main .
EXPOSE 8080
CMD ["./main"]
```

### Docker Compose
The service is included in the root `docker-compose.yml` with proper networking and health checks.

## Security Considerations

### API Key Authentication
- All `/v1/*` endpoints require API key authentication
- Keys validated against the database
- Rate limiting per API key
- Request logging for audit trails

### CORS Configuration
- Configurable allowed origins
- Comprehensive header allowlist
- Credentials support for authenticated requests
- Security headers in responses

### Input Validation
- Request payload validation
- Model name validation against supported providers
- Parameter sanitization and bounds checking
- Error message sanitization

## Troubleshooting

### Common Issues

**Server won't start**
- Check required environment variables (ADDR, ALLOWED_ORIGINS)
- Verify port is not already in use
- Check log level configuration

**Provider API errors**
- Verify API keys are correct and have sufficient credits
- Check provider status and rate limits
- Review circuit breaker states

**Performance issues**
- Monitor connection pool utilization
- Check Redis cache connectivity
- Review rate limiting configuration
- Analyze request latency patterns

### Debug Commands
```bash
# Check service health
curl http://localhost:8080/health

# Enable debug logging
export LOG_LEVEL=debug

# Monitor requests in development
export ENV=development

# Test API endpoint
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Stainless-API-Key: your-key" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}]}'
```

## Contributing

### Code Style
- Follow standard Go formatting (`go fmt`)
- Use descriptive variable names
- Implement proper error handling
- Write comprehensive tests
- Document public APIs

### Testing Requirements
- Unit tests for all services
- Integration tests for provider endpoints
- Mocked external dependencies
- Table-driven tests for multiple scenarios
- Minimum 80% test coverage

### Documentation Updates
**IMPORTANT**: When making changes to this service, always update documentation:

1. **Update this CLAUDE.md** when:
   - Adding new API endpoints or changing existing ones
   - Modifying environment variables or configuration
   - Adding new providers or changing provider integration
   - Updating dependencies or Go version requirements
   - Adding new development commands or deployment procedures

2. **Update root CLAUDE.md** when:
   - Changing service ports, commands, or basic service description
   - Modifying the service's role in the overall architecture
   - Adding new service dependencies or external integrations

3. **Update adaptive-docs/** when:
   - Making API changes that affect external users
   - Adding new features that need public documentation
   - Changing authentication or usage patterns

### Pull Request Process
1. Create feature branch from `dev`
2. Implement changes with tests
3. Run `go test ./...` and `go vet ./...`
4. **Update relevant documentation** (CLAUDE.md files, adaptive-docs/, README)
5. Submit PR with clear description and documentation impact summary