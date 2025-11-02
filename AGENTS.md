# Repository Guidelines

## Project Structure & Module Organization
- `adaptive_router/` hosts the Python router; routing lives under `services/` and tests mirror it in `tests/`.
- `examples/ts/` provides Bun demos of the API; update only when routing contracts change.
- `benchmarks/` offers profiling notebooks; run them with `uv` inside a throwaway environment.
- `docker-compose.yml` plus `scripts/installers/` spin up Redis and other dev dependencies.

## Build, Test, and Development Commands
- `uv run adaptive-router` starts the FastAPI service at `http://localhost:8000`.
- `uv run pytest --cov adaptive_router` executes the Python tests with coverage.
- `uv run ruff check` and `uv run black .` keep linting and formatting consistent.
- `bun run basic-openai.ts` in `examples/ts` smoke-tests the OpenAI SDK integration.
- `docker-compose up -d` provisions the cache and supporting services required for integration tests.

## Coding Style & Naming Conventions
- Python code follows PEP 8 with 4-space indents, snake_case modules, and type hints; document public entry points.
- Run `black` before committing and avoid hand-editing generated YAML in `adaptive_router/config/`.
- TypeScript snippets use Biome defaults: camelCase variables, PascalCase components, and `bun run check` ahead of reviews.

## Testing Guidelines
- Prefer focused `pytest` modules mirroring the source tree and include regression cases for misrouted prompts.
- Keep coverage near 90% for `adaptive_router/services/`; justify gaps with `# pragma: no cover`.
- When extending TypeScript samples, add small smoke scripts and run `bun run typecheck`.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style (`feat:`, `fix:`, `refactor:`); scope details can call out touched modules (`fix(test_model_router.py): …`).
- Squash WIP commits; PRs need passing `uv run pytest` and `uv run ruff check`, refreshed docs when behavior shifts, and linked issues or context.
- Include screenshots or terminal captures when altering developer tooling or HTTP responses to help reviewers validate end-to-end changes quickly.

## Configuration & Security Notes
- Copy `.env.example` to `.env` for local runs and never commit secrets; `docker-compose.yml` expects values via env vars.
- Rotate keys used in benchmarks and examples; treat MinIO and Redis endpoints in `scripts/installers/` as ephemeral dev services only.
