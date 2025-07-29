# Claude Assistant Configuration

## Documentation and Best Practices

When you're unsure about documentation, APIs, or best practices for any library, framework, or technology:

1. **First, check if Context7 MCP server is available** by using the `mcp__context7__resolve-library-id` and `mcp__context7__get-library-docs` tools
2. **Only use Context7 if available** - if the MCP server tools are not accessible, fall back to other research methods
3. **Use Context7 for up-to-date documentation** on any programming language, framework, or library when you need:
   - Current API documentation
   - Code examples and snippets
   - Best practices and patterns
   - Version-specific information

## When to Use Context7

- When implementing features with unfamiliar libraries
- When debugging issues with specific APIs
- When you need current documentation (your knowledge cutoff is January 2025)
- When looking for idiomatic code patterns
- When working with any technology stack

## Example Usage Pattern

```
1. Identify the library/technology you need help with
2. Use resolve-library-id to find the correct library identifier
3. Use get-library-docs with relevant topic/context
4. Apply the documentation to solve the problem
```

This ensures you always have access to the most current and accurate documentation when available.

## Project Structure - Adaptive LLM Infrastructure

This is a multi-service intelligent LLM infrastructure project with the following architecture:

### Services Overview

**🔥 Core Services:**
- **adaptive-backend/** - Go API server (main entry point)
  - OpenAI-compatible chat completions API
  - Multi-provider routing (OpenAI, Anthropic, Groq, DeepSeek, Google AI)
  - Circuit breaker and fallback mechanisms
  - Built with Fiber framework (Go 1.24+)

- **adaptive_ai/** - Python AI service 
  - Intelligent model selection using ML classifiers
  - Prompt analysis and task complexity detection
  - Cost optimization and provider routing decisions
  - Uses transformers, scikit-learn, sentence-transformers

- **adaptive-app/** - Next.js frontend
  - Web interface for chat and analytics
  - Usage dashboards and cost tracking
  - Built with React 19, Prisma, tRPC, Clerk auth
  - Real-time chat with AI SDK integration

**📊 Supporting Services:**
- **monitoring/** - Prometheus & Grafana observability stack
- **analysis/** - Cost analysis and performance benchmarking tools
- **benchmarks/** - Performance testing (MMLU, genai-perf)
- **adaptive-docs/** - Documentation site

### Technology Stack

**Backend (Go):**
- Fiber web framework
- OpenAI, Anthropic, Google AI SDKs  
- Redis for caching
- Circuit breaker patterns

**AI Service (Python):**
- LitServe for ML serving
- HuggingFace transformers
- LangChain for LLM orchestration
- Sentence transformers for embeddings

**Frontend (TypeScript/React):**
- Next.js 15 with App Router
- Prisma ORM with PostgreSQL
- tRPC for type-safe APIs
- Clerk for authentication
- Vercel AI SDK for chat

**Infrastructure:**
- Docker containerization
- Prometheus metrics
- Grafana dashboards
- Redis caching layer

### Key Features
- 🧠 Smart model selection based on prompt analysis
- 💰 Cost optimization with 30-70% typical savings
- ⚡ OpenAI-compatible API (drop-in replacement)
- 🔄 Multi-provider fallback and circuit breaking
- 📊 Real-time usage analytics and cost tracking
- 🚀 High throughput (1000+ req/s) with <100ms overhead

When working with this codebase, consider the service interactions and data flow between the Go backend, Python AI service, and Next.js frontend.