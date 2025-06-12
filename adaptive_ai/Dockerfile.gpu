FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/root/.local/bin:$PATH"

WORKDIR /app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  software-properties-common \
  curl && \
  add-apt-repository -y ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
  python3.11 \
  python3.11-venv \
  python3.11-dev \
  python3-pip && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
  ln -sf /usr/bin/python3.11 /usr/bin/python

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY pyproject.toml poetry.lock* ./
COPY README.md ./
COPY adaptive_ai/ ./adaptive_ai/

RUN poetry env use python3 && \
  poetry config virtualenvs.create false && \
  poetry install --only=main --no-interaction --no-ansi

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

