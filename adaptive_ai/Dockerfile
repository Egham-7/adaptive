# -------------------
# STAGE 1: Builder
# -------------------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION=1.8.3
ENV PATH="$POETRY_HOME/bin:$PATH"
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libffi-dev \
    libssl-dev \
    openssl \
    ca-certificates \
    && curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION \
    && ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure Poetry
RUN poetry config virtualenvs.create false && \
    poetry config virtualenvs.in-project false

# Copy dependency files first for layer caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Copy the rest of the application code
COPY . .

# Install the package
RUN poetry install --only-root --no-interaction --no-ansi

# -------------------
# STAGE 2: Runtime
# -------------------
FROM python:3.11-slim

ENV HOME="/app"
ENV HF_HOME="/app/hf_cache"
ENV PYTHONPATH="/app"
ENV ADAPTIVE_AI_CONFIG_FILE="/app/config/config.yaml"
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and configuration
COPY --from=builder /app/adaptive_ai /app/adaptive_ai
COPY --from=builder /app/config /app/config
COPY --from=builder /app/entrypoint.sh /entrypoint.sh

# Clean up cache directories
RUN rm -rf /root/.cache /app/.cache /app/.mypy_cache /tmp/hf_cache

# Create non-root user
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --home /app appuser && \
    mkdir -p /app/hf_cache && \
    chown -R appuser:appgroup /app /app/hf_cache

# Make entrypoint executable
RUN chmod +x /entrypoint.sh && chown appuser:appgroup /entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
