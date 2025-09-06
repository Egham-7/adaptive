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

## Examples

### TypeScript Examples

Complete TypeScript examples demonstrating various integration patterns:

#### 🚀 [OpenAI SDK Example](./examples/ts/examples/basic-openai.ts)
- Drop-in OpenAI SDK replacement with intelligent routing
- Streaming and non-streaming responses
- 60-80% cost reduction with zero code changes

```bash
cd examples/ts && bun basic-openai.ts
```

#### 🏛️ [Anthropic SDK Example](./examples/ts/examples/basic-anthropic.ts) 
- Native Claude Messages API format preserved
- Streaming and non-streaming with intelligent routing
- Type-safe Anthropic SDK patterns

```bash
cd examples/ts && bun basic-anthropic.ts
```

#### ⚡ [Vercel AI SDK Example](./examples/ts/examples/basic-vercel-ai-sdk.ts)
- Modern AI patterns with `generateText`, `streamText`, and tools
- Perfect for React apps with `ai/react` hooks
- Seamless Vercel AI SDK integration

```bash
cd examples/ts && bun basic-vercel-ai-sdk.ts
```

#### 🎯 [Model Selection Example](./examples/ts/examples/basic-select-model.ts)
- Test routing decisions without inference
- Cost vs performance optimization with `cost_bias`
- Function calling model prioritization
- Custom model specifications for enterprise/local models

```bash
cd examples/ts && bun basic-select-model.ts
```

### Development Setup

```bash
cd examples/ts
bun install          # Install dependencies
bun run check        # Check formatting and linting
bun run check:write  # Auto-fix issues
bun run format       # Format code only
```

## Documentation

See [adaptive-docs/](./adaptive-docs/) for complete documentation and API reference.