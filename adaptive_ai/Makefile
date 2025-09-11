# Makefile for adaptive_ai development

.PHONY: help install dev clean test test-unit test-integration test-cov test-cov-html test-fast test-config test-services test-models test-routing all-tests lint lint-fix format format-check typecheck quality quality-fix

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install dependencies"
	@echo "  dev           Start development server"
	@echo "  clean         Clean build artifacts and cache"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  test-cov-html Run tests with HTML coverage report"
	@echo "  test-fast     Run tests in fast mode (stop on first failure)"
	@echo "  test-config   Run configuration tests only"
	@echo "  test-services Run service tests only"
	@echo "  test-models   Run model tests only"
	@echo "  test-routing  Run routing integration tests"
	@echo "  all-tests     Run both unit and integration tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          Check code with ruff"
	@echo "  lint-fix      Fix code issues with ruff"
	@echo "  format        Format code with black"
	@echo "  format-check  Check code formatting"
	@echo "  typecheck     Check types with mypy"
	@echo "  quality       Run all quality checks"
	@echo "  quality-fix   Fix all quality issues"

# Setup commands
install:
	uv sync

dev:
	uv run adaptive-ai

clean:
	@echo "Cleaning build artifacts and cache..."
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf test-results/
	@rm -rf dist/
	@rm -rf build/
	@rm -rf *.egg-info/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Test commands
test:
	uv run pytest

test-unit:
	uv run pytest -m unit --cov=. --cov-report=xml

test-integration:
	uv run pytest -m integration

test-cov:
	uv run pytest --cov=adaptive_ai --cov-report=term-missing

test-cov-html:
	uv run pytest --cov=adaptive_ai --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	uv run pytest -x

test-config:
	uv run pytest adaptive_ai/tests/unit/core/

test-services:
	uv run pytest adaptive_ai/tests/unit/services/

test-models:
	uv run pytest adaptive_ai/tests/unit/models/

test-routing:
	uv run pytest adaptive_ai/tests/integration/

all-tests: test-unit test-integration

# Code quality commands
lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run black . --fast

format-check:
	uv run black --check . --fast

typecheck:
	uv run mypy adaptive_ai/

quality: lint format-check typecheck

quality-fix: lint-fix format typecheck
