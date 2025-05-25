# =======================
#    STAGE 1: BUILDER
# =======================
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04 AS builder

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV HF_HOME="/tmp/hf_cache"
ENV HOME="/app"

# Install build tools, Python, Poetry
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        libffi-dev libssl-dev build-essential git curl && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    curl -sSL https://install.python-poetry.org | python3

WORKDIR /app

# Copy only poetry dependency files first (better cache efficiency)
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies only (main group)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# Copy the rest of your source code (ensure .dockerignore is robust!)
COPY . .

# Clean up all caches/artifacts/bloat in builder stage
RUN poetry cache clear --all pypi && \
    rm -rf /root/.cache /app/.cache /app/.mypy_cache /tmp/hf_cache

# =======================
#    STAGE 2: RUNTIME
# =======================
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV HF_HOME="/tmp/hf_cache"
ENV HOME="/app"

# Only Python runtime, not dev tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy site-packages from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

# Copy poetry CLI (for entrypoint, if needed)
COPY --from=builder /opt/poetry /opt/poetry
COPY --from=builder /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy your cleaned code only (no caches, .dockerignore must be correct)
COPY --from=builder /app /app

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Secure user config
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app /entrypoint.sh

USER appuser

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]

