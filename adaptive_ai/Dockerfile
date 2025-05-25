# Use NVIDIA's CUDA base image with Python 3.12
FROM nvidia/cuda:12.4.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=2.1.3 \
    PATH="/opt/poetry/bin:$PATH" \
    HF_HOME="/tmp/hf_cache" \
    HOME="/app"

# Install system dependencies and Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    bash \
    libffi-dev \
    libssl-dev \
    python3.12 \
    python3.12-venv \
    python3.12-dev && \
    ln -s /usr/bin/python3.12 /usr/bin/python3 && \
    curl -sSL https://install.python-poetry.org | python3 && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction

# Copy application code
COPY . .

# Create a non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# Expose application port
EXPOSE 8000

# Set default command
CMD ["python", "/app/main.py"]

