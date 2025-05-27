# Use Python 3.13 as the base image
FROM python:3.11.10-slim AS builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  POETRY_HOME="/opt/poetry" \
  POETRY_VERSION=2.1.1 \
  POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Poetry with pip (more direct than curl method)
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Set the working directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Install dependencies into the project's virtualenv
RUN poetry install --only main --no-root --no-interaction

# Final stage
FROM python:3.11.10-slim

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create a non-root user to run the application
RUN adduser --disabled-password --gecos "" appuser && \
  chown -R appuser:appuser /app

# Copy the virtual environment from the builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy the application code
COPY --chown=appuser:appuser . .

# Expose the port the app runs on
EXPOSE 8000

# Switch to non-root user
USER appuser



CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
