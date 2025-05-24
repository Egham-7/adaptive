FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=2.1.3 \
    PATH="/opt/poetry/bin:$PATH" \
    HF_HOME="/tmp/hf_cache" \
    HOME="/app"

# Install Poetry and system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    bash \
    libffi-dev \
    libssl-dev && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Poetry setup for system site-packages
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction

COPY . .

# Optional: Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

USER appuser

EXPOSE 8000

CMD ["python", "/app/main.py"]

