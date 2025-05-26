# -------------------
# STAGE 1: Builder
# -------------------
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libffi-dev \
        libssl-dev \
        openssl \
        ca-certificates \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# Copy the rest of the application code
COPY . .

# -------------------
# STAGE 2: Runtime
# -------------------
FROM python:3.12-slim

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV HOME="/app"
ENV HF_HOME="/app/hf_cache"
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

RUN rm -rf /root/.cache /app/.cache /app/.mypy_cache /tmp/hf_cache

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --home /app appuser && \
    mkdir -p /app/hf_cache && \
    chown -R appuser:appgroup /app /app/hf_cache

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown appuser:appgroup /entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

