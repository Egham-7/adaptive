# Use CUDA runtime image for LitGPT
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/app/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        curl \
        git \
        build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev \
        python$PYTHON_VERSION-venv \
    && ln -sf /usr/bin/python$PYTHON_VERSION /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml README.md ./

# Install project dependencies
RUN uv pip install --system -e .

# Create cache directory for models
RUN mkdir -p /app/hf_cache

# Copy application code
COPY minion_service/ ./minion_service/

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "minion_service.main"]
