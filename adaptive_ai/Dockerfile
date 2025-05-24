# --- Stage 1: Build Stage ---
FROM python:3.12-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="/app/.venv/bin:/opt/poetry/bin:$PATH"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libffi-dev \
        libssl-dev \
        git \
        bash \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

# Install only runtime dependencies (not dev)
RUN poetry install --only main --no-root --no-interaction

COPY . .

# --- Stage 2: Final Stage ---
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/hf_cache" \
    HOME="/app"

WORKDIR /app

COPY --from=builder /app /app

# ✅ Install only what’s available in Debian Bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    libssl-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["python", "/app/main.py"]

