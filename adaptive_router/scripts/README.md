# OpenRouter Model Sync Scripts

## sync_openrouter_models.py

High-performance async pipeline to sync OpenRouter models with endpoints to PostgreSQL.

### Features

- **Async/parallel processing** using asyncio and httpx
- **SQLAlchemy ORM** for type-safe database operations
- **Strict typing** with Pydantic models
- **Endpoint fetching** - fetches all available endpoints for each model
- **Parallel endpoint fetching** - fetches endpoints for hundreds of models concurrently
- **Incremental sync** - only inserts new models, skips existing ones
- **24-hour cache** - caches OpenRouter API responses to reduce API calls
- **Multiple export formats** - JSON, Parquet, and CSV

### Installation

```bash
# Install dependencies
uv sync --group sync

# Or manually
pip install httpx sqlalchemy[asyncio] asyncpg pydantic polars
```

### Usage

```bash
# Basic usage (PostgreSQL sync only) - uses 24h cache
uv run python scripts/sync_openrouter_models.py \
  --db-url "postgresql://user:pass@localhost:5432/adaptive"

# Force refresh cache (ignore existing cache)
uv run python scripts/sync_openrouter_models.py \
  --db-url "postgresql://user:pass@localhost:5432/adaptive" \
  --no-cache

# With Parquet export (recommended for analytics)
uv run python scripts/sync_openrouter_models.py \
  --db-url "postgresql://user:pass@localhost:5432/adaptive" \
  --output-parquet models.parquet

# With CSV export
uv run python scripts/sync_openrouter_models.py \
  --db-url "postgresql://user:pass@localhost:5432/adaptive" \
  --output-csv models.csv

# With JSON export (full OpenRouter data)
uv run python scripts/sync_openrouter_models.py \
  --db-url "postgresql://user:pass@localhost:5432/adaptive" \
  --output-json enriched_models.json

# All exports at once
uv run python scripts/sync_openrouter_models.py \
  --db-url "postgresql://user:pass@localhost:5432/adaptive" \
  --output-parquet models.parquet \
  --output-csv models.csv \
  --output-json models.json
```

### Caching

24-hour cache for models and endpoints in `~/.cache/adaptive_router/`. Use `--no-cache` to force refresh.

### How It Works

1. Fetch models from OpenRouter API (with caching)
2. Fetch endpoints for each model in parallel
3. Sync to PostgreSQL
4. Optional export to JSON/Parquet/CSV

### Database Schema

```sql
CREATE TABLE llm_models (
    id SERIAL PRIMARY KEY,

    -- OpenRouter ID (e.g., "google/gemini-2.5-pro")
    openrouter_id VARCHAR(255) NOT NULL UNIQUE,

    -- Provider info (parsed from openrouter_id)
    provider VARCHAR(50) NOT NULL,
    model_name VARCHAR(255) NOT NULL,

    -- OpenRouter model data (stored as JSONB)
    display_name VARCHAR(255),
    description TEXT,
    context_length INTEGER,
    pricing JSONB,
    architecture JSONB,
    top_provider JSONB,
    supported_parameters JSONB,
    default_parameters JSONB,

    -- Endpoints from OpenRouter (array of endpoint objects)
    endpoints JSONB,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE,
    last_updated TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_llm_models_provider ON llm_models(provider);
CREATE INDEX idx_llm_models_openrouter_id ON llm_models(openrouter_id);
```

### Performance

First run: ~10s (fetches all endpoints). Cached runs: ~1-2s (loads from cache).

### Example Output

```
2025-10-07 09:00:00 - INFO - 🚀 Starting OpenRouter model sync pipeline
2025-10-07 09:00:00 - INFO - ✓ Loaded 324 models from cache (age: 1.0h)
2025-10-07 09:00:00 - INFO - Fetching endpoints for 324 models in parallel...
2025-10-07 09:00:00 - INFO -   • Cache: 310 hits, 14 misses
2025-10-07 09:00:01 - INFO - ✓ Fetched endpoints for 310 models (from 324 total)
2025-10-07 09:00:01 - INFO - Syncing models to database...
2025-10-07 09:00:01 - INFO - ✓ Inserted 0 new models
2025-10-07 09:00:01 - INFO -   • Skipped: 310 existing models
2025-10-07 09:00:01 - INFO - ✅ Pipeline completed successfully
```

### Notes

- Syncs 300+ models with endpoint data from OpenRouter
- 24-hour cache for both models and endpoints
- Run periodically to keep data fresh
