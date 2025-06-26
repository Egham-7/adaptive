FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/root/.local/bin:$PATH"

WORKDIR /app # This sets the working directory inside the container

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
  apt-get install -y --no-install-recommends curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . . 

RUN uv sync --all-extras --dev

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
