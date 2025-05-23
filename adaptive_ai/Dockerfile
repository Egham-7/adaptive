FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  POETRY_HOME="/opt/poetry" \
  POETRY_VERSION=2.1.3 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  PATH="/app/.venv/bin:$PATH"

# Python & basic deps
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  software-properties-common wget git && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  python${PYTHON_VERSION}-venv && \
  wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
  python${PYTHON_VERSION} get-pip.py && \
  rm get-pip.py && \
  ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
  ln -sf /usr/local/bin/pip${PYTHON_VERSION} /usr/local/bin/pip && \
  apt-get purge -y --auto-remove software-properties-common && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Poetry install
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

# Copy dependency files and install dependencies
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root --no-interaction

# Copy rest of the app
COPY . .

# Optional but recommended: Non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "/app/main.py"]

