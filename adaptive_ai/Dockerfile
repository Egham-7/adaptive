FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Build arguments for host and port configuration
ARG HOST=::
ARG PORT=8000

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/root/.local/bin:$PATH" \
  HF_HOME=/app/cache/huggingface \
  HF_HUB_CACHE=/app/cache/huggingface \
  HOST=${HOST} \
  PORT=${PORT}

WORKDIR /app
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Create cache directory
RUN mkdir -p /app/cache/huggingface

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* README.md ./

# Install dependencies (this layer will be cached if deps don't change)
RUN uv sync --all-extras --no-dev

# Copy application code
COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uv run hypercorn adaptive_ai.main:app --bind ${HOST:-::}:${PORT:-8000}"]
