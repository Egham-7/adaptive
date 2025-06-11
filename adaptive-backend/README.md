# Adaptive Backend

OpenAI-compatible API server that provides intelligent LLM routing through a familiar interface.

## Features

- **OpenAI Compatible** - Drop-in replacement, just change the base URL
- **Smart Routing** - AI service selects optimal model for each prompt
- **Multi-Provider** - OpenAI, Anthropic, Groq, DeepSeek, Google AI
- **Unified Streaming** - Converts all provider responses to OpenAI format
- **Cost Optimization** - Routes to cheaper models when appropriate

## Quick Start

```bash
# Install dependencies
go mod tidy

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run server
go run cmd/api/main.go
```

## Usage with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8080/v1"
)

# Route across all providers
response = client.chat.completions.create(
    model="adaptive",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Use only OpenAI models
client.base_url = "http://localhost:8080/v1/openai"
response = client.chat.completions.create(
    model="adaptive",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

```javascript
import OpenAI from 'openai';

// Route across all providers
const client = new OpenAI({
  apiKey: 'your-api-key',
  baseURL: 'http://localhost:8080/v1'
});

// Use only Anthropic models
const anthropicClient = new OpenAI({
  apiKey: 'your-api-key', 
  baseURL: 'http://localhost:8080/v1/anthropic'
});
```

## Environment Variables

```bash
# Provider API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
GOOGLE_AI_API_KEY=...

# Database
DB_SERVER=localhost
DB_NAME=adaptive
DB_USER=sa
DB_PASSWORD=your-password

# Service URLs
AI_SERVICE_URL=http://localhost:8000
ADDR=:8080
```

## API Endpoints

### OpenAI-Compatible
- `POST /v1/chat/completions` - Chat completions (with streaming) - Routes across all providers
- `GET /v1/models` - List available models

### Provider-Specific
- `POST /v1/openai/chat/completions` - OpenAI-only chat completions
- `POST /v1/anthropic/chat/completions` - Anthropic-only chat completions  
- `POST /v1/groq/chat/completions` - Groq-only chat completions
- `POST /v1/deepseek/chat/completions` - DeepSeek-only chat completions
- `POST /v1/gemini/chat/completions` - Gemini-only chat completions

### Management
- `GET /api/conversations` - List conversations
- `POST /api/conversations` - Create conversation
- `GET /api/api_keys/{userId}` - List API keys
- `POST /api/api_keys` - Create API key

## How It Works

1. Receives OpenAI-compatible request
2. Sends prompt to AI service for model selection (optionally constrained by provider)
3. Routes request to selected provider
4. Converts response to OpenAI format
5. Returns unified response

### Main Endpoint (`/v1/chat/completions`)
- Routes across **all providers** for optimal model selection
- AI service chooses best model from OpenAI, Anthropic, Groq, DeepSeek, Gemini

### Provider Endpoints (`/v1/{provider}/chat/completions`)
- Routes only within **specific provider** models
- Same interface, constrained model selection
- Useful for staying within provider ecosystems

## Project Structure

```
cmd/api/                    # Application entry point
internal/
├── api/                   # HTTP handlers
├── middleware/            # Auth and API key middleware
├── models/               # Data models
├── services/             # Business logic
│   ├── providers/        # LLM provider implementations
│   └── stream_readers/   # Response conversion to OpenAI format
└── repositories/         # Data access layer
```

## Building

```bash
# Development
go run cmd/api/main.go

# Production
go build -o bin/adaptive-backend cmd/api/main.go

# Docker
docker build -t adaptive-backend .
```

## Testing

```bash
go test ./...
```
