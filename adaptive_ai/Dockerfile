FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime 

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/root/.local/bin:$PATH"

WORKDIR /app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
  apt-get install -y --no-install-recommends curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install Poetry using the pre-installed Python
RUN curl -sSL https://install.python-poetry.org | python3 -

COPY . .

# Use python3 (which points to 3.10 in this base image)
RUN poetry env use python3 && \
  poetry config virtualenvs.create false && \
  poetry install --only=main --no-interaction --no-ansi

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
