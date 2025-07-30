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

## Common Commands

### Frontend (adaptive-app/)
- `bun dev` - Start development server with Turbo
- `bun run build` - Build for production (includes Prisma generate and type checking)
- `bun run check` - Run Biome linter/formatter
- `bun run check:write` - Run Biome with auto-fix
- `bun run typecheck` - Run TypeScript type checking
- `bun run db:generate` - Generate Prisma client and run migrations
- `bun run db:push` - Push schema changes to database
- `bun run db:studio` - Open Prisma Studio
- `bun run db:seed-providers` - Seed provider data
- `bun run stripe` - Start Stripe webhook listener

### Backend Go (adaptive-backend/)
- `go run cmd/api/main.go` - Start Go API server
- `go build -o main cmd/api/main.go` - Build binary
- `go test ./...` - Run all tests
- `go mod tidy` - Clean up dependencies
- `go fmt ./...` - Format Go code
- `go vet ./...` - Run Go vet analyzer

### AI Service (adaptive_ai/)
- `uv run adaptive-ai` - Start AI service
- `uv run black .` - Format Python code
- `uv run ruff check .` - Run linter
- `uv run ruff check --fix .` - Run linter with auto-fix
- `uv run mypy .` - Run type checker
- `uv run pytest` - Run tests
- `uv run pytest --cov` - Run tests with coverage

### Docker & Infrastructure
- `docker-compose up -d` - Start all services in background
- `docker-compose logs -f [service]` - Follow logs for specific service
- `docker-compose down` - Stop all services
- `docker-compose build` - Rebuild containers
- `docker-compose exec [service] bash` - Access service shell

### Development Workflow
- `bun run dev` - Start frontend development (from adaptive-app/)
- `go run cmd/api/main.go` - Start backend API (from adaptive-backend/)
- `uv run adaptive-ai` - Start AI service (from adaptive_ai/)

## Code Style Guidelines

### TypeScript/React (Frontend)
- **IMPORTANT**: Use ES modules (import/export) syntax, NOT CommonJS (require)
- Use Biome for consistent code formatting and linting
- Destructure imports when possible: `import { foo } from 'bar'`
- Follow React 19 patterns with Server Components where appropriate
- Use TypeScript strict mode - all types must be properly defined
- Prefer `interface` over `type` for object shapes
- Use `const` assertions for immutable data
- Self-closing JSX elements: `<Component />` not `<Component></Component>`
- Organize imports: React imports first, then third-party, then local

### Go (Backend)
- Follow standard Go formatting with `go fmt`
- Use descriptive variable names, avoid abbreviations
- Implement proper error handling - never ignore errors
- Use interfaces for abstractions and testing
- Follow Go naming conventions (PascalCase for exports, camelCase for private)
- Keep functions small and focused
- Use context.Context for request scoping and cancellation
- Prefer dependency injection over global variables

### Python (AI Service)
- **IMPORTANT**: Use Black for code formatting (line length: 88)
- Use Ruff for linting and import sorting
- All code must pass mypy type checking
- Use type hints for all function parameters and return values
- Follow PEP 8 naming conventions
- Use dataclasses or Pydantic models for structured data
- Prefer f-strings for string formatting
- Use descriptive variable names, avoid single-letter variables
- Handle exceptions explicitly, don't use bare `except:`

## Testing Instructions

### Frontend Testing
- Currently using manual testing and type checking
- **TODO**: Set up React Testing Library for component tests
- Run `bun run typecheck` before committing
- Test database operations with `bun run db:studio`

### Backend Go Testing
- Use standard Go testing: `go test ./...`
- Write unit tests for services and handlers
- Use table-driven tests for multiple scenarios
- Mock external dependencies using interfaces
- Test file naming: `*_test.go`

### AI Service Testing
- Use pytest: `uv run pytest`
- Run with coverage: `uv run pytest --cov`
- Test file naming: `test_*.py`
- Mock ML models and external APIs in tests
- Use fixtures for common test data

### Integration Testing
- Use Docker Compose for full stack testing
- Test API endpoints with realistic payloads
- Verify provider integrations work correctly

## Developer Environment Setup

### Prerequisites
- **Node.js** 18+ and **Bun** latest
- **Go** 1.24+ (specified in go.mod)
- **Python** 3.10+ with **uv** package manager
- **Docker** and **Docker Compose**
- **PostgreSQL** (or use Docker)
- **Redis** (or use Docker)

### First-time Setup
1. Clone repository and install dependencies:
   ```bash
   # Frontend
   cd adaptive-app && bun install
   
   # Backend Go - dependencies auto-installed
   cd ../adaptive-backend
   
   # AI Service
   cd ../adaptive_ai && uv install
   ```

2. Set up environment files:
   - Copy `.env.example` to `.env` in each service directory
   - Configure database URLs, API keys, etc.

3. Initialize database:
   ```bash
   cd adaptive-app
   bun run db:generate
   bun run db:seed-providers
   ```

4. Start services:
   ```bash
   # Option 1: Docker Compose (recommended)
   docker-compose up -d
   
   # Option 2: Individual services
   # Terminal 1: Frontend
   cd adaptive-app && bun run dev
   
   # Terminal 2: Go Backend
   cd adaptive-backend && go run cmd/api/main.go
   
   # Terminal 3: AI Service
   cd adaptive_ai && uv run adaptive-ai
   ```

### Required Tools
- **Biome**: Code formatting for TypeScript/React
- **Prisma**: Database ORM and migrations
- **Stripe CLI**: For webhook testing
- **uv**: Fast Python package manager
- **Context7 MCP**: For up-to-date documentation (when available)

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

## Repository Workflow

### Git Workflow
- **Main branch**: `main` (production-ready code)
- **Development branch**: `dev` (active development)
- Create feature branches from `dev`: `feature/your-feature-name`
- Always create pull requests to merge into `dev`
- Use conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`

### Before Committing
**IMPORTANT**: Always run these checks before committing:

**Frontend:**
```bash
cd adaptive-app
bun run typecheck  # Must pass
bun run check      # Must pass
```

**Backend Go:**
```bash
cd adaptive-backend
go test ./...      # Must pass
go vet ./...       # Must pass
go fmt ./...       # Auto-format
```

**AI Service:**
```bash
cd adaptive_ai
uv run mypy .      # Must pass
uv run ruff check . # Must pass
uv run black .     # Auto-format
uv run pytest     # Must pass
```

### Pull Request Guidelines
- Write clear, descriptive PR titles
- Include testing steps in PR description
- Link related issues with `Fixes #123`
- Request reviews from relevant team members
- Ensure all CI checks pass

## Core Files and Utilities

### Frontend Key Files
- `src/server/api/` - tRPC API routes and routers
- `src/lib/` - Shared utilities and configurations
- `src/types/` - TypeScript type definitions
- `prisma/schema.prisma` - Database schema
- `src/middleware.ts` - Clerk authentication middleware
- `biome.jsonc` - Code style configuration

### Backend Key Files
- `internal/api/` - HTTP handlers and routes
- `internal/services/` - Business logic and providers
- `internal/models/` - Data models and protocols
- `cmd/api/main.go` - Application entry point
- `internal/middleware/auth.go` - Authentication middleware

### AI Service Key Files
- `adaptive_ai/main.py` - LitServe application entry
- `adaptive_ai/services/` - ML models and classification logic
- `adaptive_ai/models/` - Data models and enums
- `adaptive_ai/config/` - Provider and task mappings
- `pyproject.toml` - Dependencies and tool configuration

### Utility Functions
- **Frontend**: `src/lib/utils.ts` - Common utilities, cn() for styling
- **Go**: `internal/utils/` - Message utilities and helpers
- **Python**: `adaptive_ai/utils/` - OpenAI utilities and helpers

## Debugging and Common Issues

### Frontend Issues
- **Database connection errors**: Check DATABASE_URL in .env
- **Clerk auth issues**: Verify CLERK_* environment variables
- **Build failures**: Run `bun run typecheck` to identify TypeScript errors
- **Prisma errors**: Run `bun run db:generate` after schema changes

### Backend Go Issues
- **Port conflicts**: Default port 8080, check if already in use
- **Provider API errors**: Verify API keys in .env.local
- **Redis connection**: Ensure Redis is running (Docker or local)

### AI Service Issues
- **Model loading**: First run downloads HuggingFace models (slow)
- **Memory issues**: ML models require significant RAM
- **Port conflicts**: Default port 8000
- **Dependencies**: Use `uv install` not pip for consistency

### Docker Issues
- **Port conflicts**: Check if ports 3000, 8000, 8080 are available
- **Build failures**: Run `docker-compose build --no-cache`
- **Health check failures**: Check service logs with `docker-compose logs [service]`

### Performance Tips
- Use Docker Compose for consistent development environment
- Frontend: Enable Turbo mode with `--turbo` flag
- Go: Use `go run` for development, build binary for production
- Python: Models cache after first load for faster subsequent requests

### API Testing
- Test OpenAI-compatible endpoint: `POST http://localhost:8080/v1/chat/completions`
- Use Postman collection or curl for API testing
- Check Grafana dashboards for monitoring: `http://localhost:3001`

**IMPORTANT**: When you encounter errors, always check service logs first:
```bash
# Docker logs
docker-compose logs -f [service-name]

# Direct service logs
# Check stdout/stderr from running services
```