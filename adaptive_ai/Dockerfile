# ==========================
#    STAGE 1: BUILDER
# ==========================
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04 AS builder

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV HF_HOME="/tmp/hf_cache"
ENV HOME="/app"

# Install build tools, Python, Poetry, and SSL dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        libffi-dev libssl-dev build-essential git curl \
        openssl libssl3 ca-certificates \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only dependencies first for better build caching
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# Copy app code
COPY . .

# Clean up Poetry and pip cache to minimize image size
RUN poetry cache clear --all pypi && \
    rm -rf /root/.cache /app/.cache /app/.mypy_cache /tmp/hf_cache

# ==========================
#    STAGE 2: RUNTIME
# ==========================
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV HF_HOME="/tmp/hf_cache"
ENV HOME="/app"

# Install Python runtime, SSL, and essential libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        libssl3 libssl-dev openssl ca-certificates libffi-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python dependencies and Poetry from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /opt/poetry /opt/poetry
COPY --from=builder /usr/local/bin/poetry /usr/local/bin/poetry
COPY --from=builder /app /app

# Entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app /entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

