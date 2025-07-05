# Adaptive - Intelligent LLM Infrastructure

AI-powered model selection and cost optimization with OpenAI-compatible API. Automatically routes requests to the optimal LLM for each task.

## Key Features

- 🧠 **Smart Model Selection** - AI analyzes prompts and selects optimal models
- 💰 **Cost Optimization** - Automatically routes to cheaper models when appropriate
- ⚡ **OpenAI Compatible** - Drop-in replacement, just change the base URL
- 🔄 **Multi-Provider** - OpenAI, Anthropic, Groq, DeepSeek, Google AI
- 📊 **Usage Analytics** - Track spending and optimization opportunities

## Architecture

- **Backend** (Go) - OpenAI-compatible API server
- **AI Service** (Python) - Intelligent model selection
- **Frontend** (Next.js) - Web interface and analytics
- **Monitoring** - Prometheus & Grafana dashboards

## Quick Start

### 1. Deploy with Docker

```bash
git clone https://github.com/your-org/adaptive.git
cd adaptive
cp .env.example .env  # Configure API keys
docker-compose up -d
```

### 2. Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-adaptive-api-key",
    base_url="https://api.adaptive.ai/v1"
)

response = client.chat.completions.create(
    model="adaptive",  # Auto-routing, or use 'gpt-4o', 'claude-3-5-sonnet', etc.
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

### 3. Use with JavaScript

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'your-adaptive-api-key',
  baseURL: 'https://api.adaptive.ai/v1'
});

const response = await client.chat.completions.create({
  model: 'adaptive',  // Auto-routing, or use 'gpt-4o', 'claude-3-5-sonnet', etc.
  messages: [{ role: 'user', content: 'Write a Python function' }]
});
```

### 4. Use with Vercel AI SDK

```typescript
import { adaptive } from '@ai-sdk/adaptive';
import { generateText } from 'ai';

const { text } = await generateText({
  model: adaptive(),  // Auto-routing, or adaptive('gpt-4o'), etc.
  prompt: 'Write a Python function to sort a list'
});
```

## Configuration

Configure environment variables in `.env`:

```bash
# Provider API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
GOOGLE_AI_API_KEY=...

# Database & Services
DB_SERVER=localhost
DB_NAME=adaptive
AI_SERVICE_URL=http://localhost:8000
ADDR=:8080
```

## API Reference

### Main Endpoint

`POST /v1/chat/completions` - OpenAI-compatible chat completions with intelligent model selection

```json
{
  "model": "adaptive",
  "messages": [{"role": "user", "content": "Explain quantum computing"}],
  "provider_constraints": ["openai", "anthropic"],  // Optional
  "cost_bias": 0.3  // 0.0 = cost-optimized, 1.0 = performance-optimized
}
```

### Response includes selection metadata

```json
{
  "choices": [...],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 150,
    "total_tokens": 192,
    "cost_saved": 0.35
  },
  "provider": "openai",
  "model": "gpt-4o"
}
```

## How It Works

1. **Prompt Analysis** - AI analyzes complexity, domain, and context requirements
2. **Model Selection** - Selects optimal model based on task requirements and cost considerations
3. **Provider Routing** - Routes request while maintaining OpenAI-compatible format
4. **Response Enhancement** - Returns response with selection metadata and cost savings info

## Project Structure

```
adaptive/
├── adaptive-backend/    # Go API server (OpenAI-compatible)
├── adaptive_ai/        # Python AI service (model selection)
├── adaptive-app/       # Next.js frontend (web interface)
├── monitoring/         # Prometheus & Grafana
└── analysis/          # Cost and performance analysis
```

## Development

### Prerequisites
- Docker and Docker Compose
- Go 1.21+ (backend), Python 3.11+ (AI service), Node.js 18+ (frontend)

### Local Development

```bash
# Backend (Go)
cd adaptive-backend && go run cmd/api/main.go

# AI Service (Python)
cd adaptive_ai && uv sync && uv run python adaptive_ai/main.py

# Frontend (Next.js)
cd adaptive-app && bun install && bun run dev
```

### Testing

```bash
cd adaptive-backend && go test ./...     # Backend tests
cd adaptive_ai && uv run pytest         # AI service tests
cd adaptive-app && bun run test         # Frontend tests
```

## Performance

- **Latency**: <100ms model selection overhead
- **Throughput**: 1000+ requests/second
- **Cost Savings**: 30-70% typical reduction
- **Availability**: 99.9% uptime target

## Contributing

See [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

Business Source License 1.1 - Free for non-commercial use, commercial license required for production.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/adaptive/issues)
- **Documentation**: [docs.adaptive.ai](https://docs.adaptive.ai)