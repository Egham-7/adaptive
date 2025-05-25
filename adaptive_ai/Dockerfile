FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=2.1.3 \
    PATH="/opt/poetry/bin:$PATH" \
    HF_HOME="/tmp/hf_cache" \
    HOME="/app"

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    python3.12 \
    python3.12-venv \
    python3.12-dev && \
    ln -s /usr/bin/python3.12 /usr/bin/python3 && \
    curl -sSL https://install.python-poetry.org | python3 && \
    ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi



COPY . .

# Copy and prepare entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create user and set permissions
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app /entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

