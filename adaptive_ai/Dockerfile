# Multi-stage build for optimized FastAPI service

# Build stage - Install dependencies and prepare app
FROM python:3.11-slim AS builder

# Build arguments
ARG HOST=::
ARG PORT=8000

# Environment variables for build stage
ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/root/.local/bin:$PATH"

WORKDIR /app
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies needed for building
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy dependency files
COPY pyproject.toml uv.lock* README.md ./

# Install dependencies in virtual environment
RUN uv sync --no-dev --frozen

# Copy application code
COPY adaptive_ai/ ./adaptive_ai/
COPY config.yaml entrypoint.sh ./

# Production stage - Minimal runtime image
FROM python:3.11-slim AS runtime

# Build arguments passed to runtime
ARG HOST=::
ARG PORT=8000

# Runtime environment variables
ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/app/.venv/bin:$PATH" \
  HOST=${HOST} \
  PORT=${PORT}

WORKDIR /app

# Install only essential runtime dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  ca-certificates \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && groupadd -r appuser \
  && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy virtual environment and application from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/adaptive_ai /app/adaptive_ai
COPY --from=builder --chown=appuser:appuser /app/config.yaml /app/entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user for security
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD sh -c "python -m hypercorn adaptive_ai.main:app --bind \${HOST:-::}:\${PORT:-8000}"
