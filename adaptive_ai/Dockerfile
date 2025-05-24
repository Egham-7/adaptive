FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="/app/.venv/bin:/opt/poetry/bin:$PATH"

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
RUN poetry install --only main --no-root --no-interaction
COPY . .

# Pre-download Hugging Face models
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/deberta-v3-base')" && \
    python -c "from transformers import AutoModel; AutoModel.from_pretrained('nvidia/domain-classifier')"

# --- Final stage ---
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME="/app/hf_cache" \
    HOME="/app"

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /root/.cache/huggingface /app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    libssl-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create app user, chown only app code (not large venv/cache)
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser && \
    find /app -maxdepth 1 ! -name ".venv" ! -name "hf_cache" -exec chown -R appuser:appgroup {} +

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

