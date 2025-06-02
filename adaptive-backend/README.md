# Adaptive Backend

Go API server for the Adaptive AI platform - handles authentication, conversations, and intelligent LLM routing.

## Features

- **Smart Model Selection**: Integrates with AI service for automatic model selection
- **Semantic Caching**: LRU + vector similarity caching for performance
- **Multi-Provider Support**: OpenAI, Anthropic, Groq, DeepSeek, Google AI
- **Real-time Streaming**: SSE-based chat completion streaming
- **Authentication**: Clerk integration with middleware
- **Monitoring**: Prometheus metrics and health checks

## Quick Start

```bash
# Install dependencies
go mod tidy

# Set environment variables
cp .env.example .env.local

# Run server
go run cmd/api/main.go
```

## Environment Variables

```bash
ADDR=:8080
ALLOWED_ORIGINS=http://localhost:3000
DB_SERVER=localhost
DB_NAME=adaptive
DB_USER=sa
DB_PASSWORD=your-password
CLERK_SECRET_KEY=sk_test_xxxxx
OPENAI_API_KEY=sk-xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx
AI_SERVICE_URL=http://localhost:8000
```

## API Endpoints

### Chat Completions
- `POST /api/chat/completions` - Create completion
- `POST /api/chat/completions/stream` - Stream completion

### Conversations
- `GET /api/conversations` - List conversations
- `POST /api/conversations` - Create conversation
- `GET /api/conversations/{id}/messages` - Get messages

### API Keys
- `GET /api/api_keys/{userId}` - List user API keys
- `POST /api/api_keys` - Create API key
- `DELETE /api/api_keys/{id}` - Delete API key

## Project Structure

```
cmd/api/           # Application entry point
internal/
├── api/           # HTTP handlers
├── middleware/    # Auth and API key middleware
├── models/        # Data models
├── services/      # Business logic
└── repositories/  # Data access layer
```

## Building

```bash
# Development
go run cmd/api/main.go

# Production build
go build -o bin/adaptive-backend cmd/api/main.go

# Docker
docker build -t adaptive-backend .
```

## Testing

```bash
go test ./...
```