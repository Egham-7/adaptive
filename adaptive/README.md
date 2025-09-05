# Adaptive

Intelligent LLM infrastructure that automatically selects the optimal model for each request. Drop-in OpenAI replacement with 30-70% cost savings.

## Features

- 🧠 **Smart routing** - AI selects optimal models automatically
- 💰 **Cost optimization** - 30-70% savings vs direct provider usage  
- ⚡ **Fast caching** - Dual-layer cache for instant responses
- 🔄 **Multi-provider** - OpenAI, Anthropic, Groq, DeepSeek, Google AI
- 📊 **Analytics** - Usage tracking and cost insights

## Quick Start

### 1. Deploy

```bash
git clone https://github.com/your-org/adaptive.git
cd adaptive
cp .env.example .env  # Add your API keys
docker-compose up -d
```

### 2. Use with any OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-adaptive-key",
    base_url="https://your-deployment-url/v1"
)

response = client.chat.completions.create(
    model="",  # Empty string enables intelligent routing
    messages=[{"role": "user", "content": "Hello world"}]
)
```

## Architecture

- **adaptive-backend/** - Go API server (Fiber, OpenAI SDK, Redis)
- **adaptive_ai/** - Python ML service (LitServe, HuggingFace, scikit-learn)
- **adaptive-app/** - Next.js web app (React 19, Prisma, tRPC, Clerk)
- **adaptive-docs/** - Documentation (Mintlify)

## Development

```bash
# Backend
cd adaptive-backend && go run cmd/api/main.go

# AI Service  
cd adaptive_ai && uv run adaptive-ai

# Frontend
cd adaptive-app && bun dev
```

## Documentation

See [adaptive-docs/](./adaptive-docs/) for complete documentation and API reference.