#!/usr/bin/env python3
"""Adaptive Router Training Script.

This script trains an Adaptive Router profile from labeled dataset files.
It fetches model information and pricing from the Adaptive Models API,
configures provider API keys from environment variables, and trains
a router profile using the Trainer class.

The script supports CSV, JSON, and Parquet dataset formats with customizable
column names for input and expected output data.

Requirements:
- ADAPTIVE_API_KEY environment variable (for API authentication)
- Provider API keys (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY) as needed
- Dataset file with input and expected output columns

Dataset Format:
The dataset should contain at least two columns:
- Input column: Contains the prompts/questions to be routed
- Expected output column: Contains the correct answers/responses

Examples:
  # Train from CSV with default columns
  python train.py --input-path data.csv --dataset-type csv \\
    --output-path profile.json --models openai/gpt-4 openai/gpt-3.5-turbo

  # Train from JSON with custom columns
  python train.py --input-path data.json --dataset-type json \\
    --output-path profile.json --models anthropic/claude-3-5-sonnet-20241022 \\
    --input-column question --expected-column answer

  # Train from Parquet with custom clusters
  python train.py --input-path data.parquet --dataset-type parquet \\
    --output-path profile.json --n-clusters 30 \\
    --models openai/gpt-4 anthropic/claude-3-5-sonnet-20241022 deepseek/deepseek-chat
"""

# Standard library imports
import sys
import os
import argparse
import logging
from typing import Dict, List

# Third-party imports
import httpx
import polars as pl

# Local imports
from adaptive_router.core.trainer import Trainer
from adaptive_router.core.provider_registry import default_registry
from adaptive_router.models.api import Model
from adaptive_router.models.train import ProviderConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdaptiveModelsAPIClient:
    """HTTP client for the Adaptive Models API.

    Handles authentication and communication with the Adaptive Models API
    to fetch model information and pricing data.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.llmadaptive.uk/v1",
        timeout: float = 30.0,
    ):
        """Initialize the API client.

        Args:
            api_key: Adaptive API key for authentication
            base_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_model_info(self, provider: str, model_name: str) -> dict:
        """Fetch model information from the API.

        Args:
            provider: Model provider (e.g., "openai", "anthropic")
            model_name: Model name (e.g., "gpt-4", "claude-3-5-sonnet-20241022")

        Returns:
            Model information dictionary from API

        Raises:
            ValueError: If model not found or API errors occur
        """
        url = f"{self.base_url}/models/{provider}/{model_name}"

        try:
            response = self.client.get(url, headers=self.headers)

            if response.status_code == 404:
                raise ValueError(f"Model not found: {provider}/{model_name}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            raise ValueError(f"API error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Network error: {str(e)}")

    def list_models(self, provider: str | None = None) -> List[dict]:
        """List models from the API.

        Args:
            provider: Optional provider filter

        Returns:
            List of model dictionaries
        """
        url = f"{self.base_url}/models"
        params = {"author": provider} if provider else {}

        try:
            response = self.client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ValueError(f"API error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Network error: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close HTTP client."""
        self.client.close()


def parse_model_spec(model_spec: str) -> tuple[str, str]:
    """Parse model specification in provider/model_name format.

    Args:
        model_spec: Model specification string

    Returns:
        Tuple of (provider, model_name)

    Raises:
        ValueError: If format is invalid
    """
    model_spec = model_spec.strip()
    if not model_spec:
        raise ValueError("Model specification cannot be empty")

    parts = model_spec.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid model format: '{model_spec}'. "
            f"Expected format: provider/model_name (e.g., openai/gpt-4)"
        )

    provider, model_name = parts
    provider = provider.strip().lower()
    model_name = model_name.strip()

    if not provider or not model_name:
        raise ValueError("Provider and model name cannot be empty")

    return provider, model_name


def create_model_from_api(
    api_client: AdaptiveModelsAPIClient, model_spec: str
) -> Model:
    """Create a Model object by fetching information from the API.

    Args:
        api_client: Initialized API client
        model_spec: Model specification in provider/model_name format

    Returns:
        Model object with pricing information

    Raises:
        ValueError: If model not found or pricing data is invalid
    """
    # Parse specification
    provider, model_name = parse_model_spec(model_spec)

    # Fetch from API
    data = api_client.get_model_info(provider, model_name)

    # Extract pricing
    pricing = data.get("pricing")
    if not pricing:
        raise ValueError(f"No pricing information for {model_spec}")

    prompt_cost = pricing.get("prompt_cost")
    completion_cost = pricing.get("completion_cost")

    if not prompt_cost or not completion_cost:
        raise ValueError(
            f"Incomplete pricing data for {model_spec}. "
            f"Both prompt_cost and completion_cost are required."
        )

    # Convert pricing (API returns per-token as strings like "0.00015")
    # Model class expects per-1M-tokens as floats
    cost_per_1m_input = float(prompt_cost) * 1_000_000
    cost_per_1m_output = float(completion_cost) * 1_000_000

    # Log pricing info
    logger.info(
        f"Loaded {model_spec}: "
        f"${cost_per_1m_input:.4f}/1M input tokens, "
        f"${cost_per_1m_output:.4f}/1M output tokens"
    )

    return Model(
        provider=provider,
        model_name=model_name,
        cost_per_1m_input_tokens=cost_per_1m_input,
        cost_per_1m_output_tokens=cost_per_1m_output,
    )


def load_provider_configs(models: List[Model]) -> Dict[str, ProviderConfig]:
    """Load provider configurations from environment variables.

    Args:
        models: List of Model objects

    Returns:
        Dictionary mapping provider names to ProviderConfig objects

    Raises:
        ValueError: If providers are unsupported or API keys are missing
    """
    # Extract unique providers
    providers = {m.provider for m in models}

    # Check each provider is registered
    unsupported = []
    for provider in providers:
        if not default_registry.is_registered(provider):
            unsupported.append(provider)

    if unsupported:
        available = default_registry.list_providers()
        raise ValueError(
            f"Unsupported providers: {', '.join(unsupported)}\n"
            f"Available providers: {', '.join(available)}\n"
            f"Please register custom providers before training."
        )

    # Environment variable mapping
    env_var_map = {
        "azure": "AZURE_OPENAI_API_KEY",
        "azure-openai": "AZURE_OPENAI_API_KEY",
        "amazon-bedrock": "AWS_ACCESS_KEY_ID",
        "bedrock": "AWS_ACCESS_KEY_ID",
        "ollama": "OLLAMA_HOST",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "grok": "XAI_API_KEY",
    }

    # Load API keys
    provider_configs = {}
    missing = []

    for provider in providers:
        # Skip 'local' provider - doesn't need API key
        if provider == "local":
            provider_configs[provider] = ProviderConfig(api_key="")
            continue

        # Get env var name
        env_var = env_var_map.get(provider, f"{provider.upper()}_API_KEY")

        # Read from environment
        api_key = os.getenv(env_var)

        if not api_key:
            missing.append((provider, env_var))
        else:
            provider_configs[provider] = ProviderConfig(api_key=api_key)

    if missing:
        error_lines = ["Missing required API keys:"]
        for provider, env_var in missing:
            error_lines.append(f"  - {provider}: Set {env_var} environment variable")
        raise ValueError("\n".join(error_lines))

    return provider_configs


def train_router(
    input_path: str,
    dataset_type: str,
    output_path: str,
    model_specs: List[str],
    api_key: str,
    n_clusters: int,
    input_column: str,
    expected_column: str,
) -> None:
    """Orchestrate the complete training workflow.

    Args:
        input_path: Path to input dataset file
        dataset_type: Type of dataset ('csv', 'json', 'parquet')
        output_path: Path to save trained profile
        model_specs: List of model specifications
        api_key: Adaptive API key
        n_clusters: Number of clusters for training
        input_column: Name of input column in dataset
        expected_column: Name of expected output column
    """
    # Log training header
    logger.info("=" * 80)
    logger.info("ADAPTIVE ROUTER TRAINING")
    logger.info("=" * 80)
    logger.info(f"Input path: {input_path}")
    logger.info(f"Dataset type: {dataset_type}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Models: {', '.join(model_specs)}")
    logger.info(f"Number of clusters: {n_clusters}")
    logger.info("=" * 80)

    # Initialize API client
    with AdaptiveModelsAPIClient(api_key) as api_client:
        # Fetch models from API
        models = []
        failed = []

        for model_spec in model_specs:
            try:
                model = create_model_from_api(api_client, model_spec)
                models.append(model)
            except Exception as e:
                logger.error(f"Failed to load {model_spec}: {e}")
                failed.append((model_spec, str(e)))

        # Check for failures
        if failed:
            logger.error(f"\nFailed to load {len(failed)} model(s):")
            for spec, error in failed:
                logger.error(f"  - {spec}: {error}")
            sys.exit(1)

        # Log success
        logger.info(f"\nSuccessfully loaded {len(models)} model(s)")

        # Load provider configurations
        try:
            provider_configs = load_provider_configs(models)
            logger.info(
                f"Loaded configurations for {len(provider_configs)} provider(s)"
            )
        except Exception as e:
            logger.error(f"Failed to load provider configurations: {e}")
            sys.exit(1)

        # Initialize trainer
        trainer = Trainer(
            models=models, provider_configs=provider_configs, n_clusters=n_clusters
        )

        # Train based on dataset type
        logger.info(f"\nStarting training from {dataset_type} file...")

        try:
            if dataset_type == "csv":
                result = trainer.train_from_csv(
                    input_path, input_column, expected_column
                )
            elif dataset_type == "json":
                result = trainer.train_from_json(
                    input_path, input_column, expected_column
                )
            elif dataset_type == "parquet":
                df = pl.read_parquet(input_path)
                result = trainer.train_from_dataframe(df, input_column, expected_column)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

        except FileNotFoundError:
            logger.error(f"Dataset file not found: {input_path}")
            sys.exit(1)
        except KeyError as e:
            logger.error(f"Column not found in dataset: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            sys.exit(1)

        # Display training results
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total samples: {result.total_samples}")
        logger.info(f"Number of clusters: {result.n_clusters}")
        logger.info(f"Silhouette score: {result.silhouette_score:.4f}")
        logger.info(f"Training time: {result.training_time:.2f} seconds")
        if result.inference_time:
            logger.info(f"Inference time: {result.inference_time:.2f} seconds")

        # Display error rates
        logger.info("\nModel Error Rates:")
        for model_id, error_rates in result.error_rates.items():
            avg_error = sum(error_rates) / len(error_rates) if error_rates else 0
            logger.info(f"  {model_id}: {avg_error * 100:.2f}%")

        # Save profile
        logger.info(f"\nSaving profile to: {output_path}")
        try:
            trainer.save_profile(output_path)
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            sys.exit(1)

        # Log success
        logger.info(f"Profile saved successfully to: {output_path}")

        # Log completion
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)


def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train an Adaptive Router profile from labeled data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from CSV with default columns
  python train.py --input-path data.csv --dataset-type csv \\
    --output-path profile.json --models openai/gpt-4 openai/gpt-3.5-turbo

  # Train from JSON with custom columns
  python train.py --input-path data.json --dataset-type json \\
    --output-path profile.json --models anthropic/claude-3-5-sonnet-20241022 \\
    --input-column question --expected-column answer

  # Train from Parquet with custom clusters
  python train.py --input-path data.parquet --dataset-type parquet \\
    --output-path profile.json --n-clusters 30 \\
    --models openai/gpt-4 anthropic/claude-3-5-sonnet-20241022 deepseek/deepseek-chat
""",
    )

    # Required arguments
    parser.add_argument(
        "--input-path",
        required=True,
        type=str,
        help="Path to input dataset file (CSV, JSON, or Parquet)",
    )

    parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["csv", "json", "parquet"],
        help="Format of the input dataset",
    )

    parser.add_argument(
        "--output-path",
        required=True,
        type=str,
        help="Path where trained profile will be saved (JSON format)",
    )

    parser.add_argument(
        "--models",
        required=True,
        nargs="+",
        type=str,
        help="Models in provider/model_name format (e.g., openai/gpt-4)",
    )

    # Optional arguments
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=20,
        help="Number of clusters for training (default: 20)",
    )

    parser.add_argument(
        "--input-column",
        type=str,
        default="input",
        help="Name of input column in dataset (default: input)",
    )

    parser.add_argument(
        "--expected-column",
        type=str,
        default="expected_output",
        help="Name of expected output column (default: expected_output)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv("ADAPTIVE_API_KEY")
    if not api_key:
        logger.error(
            "ADAPTIVE_API_KEY environment variable not set.\n"
            "Please set it with your Adaptive API key:\n"
            "  export ADAPTIVE_API_KEY='your-api-key-here'"
        )
        sys.exit(1)

    # Call training function
    train_router(
        input_path=args.input_path,
        dataset_type=args.dataset_type,
        output_path=args.output_path,
        model_specs=args.models,
        api_key=api_key,
        n_clusters=args.n_clusters,
        input_column=args.input_column,
        expected_column=args.expected_column,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
