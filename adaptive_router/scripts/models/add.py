#!/usr/bin/env python3
"""End-to-end workflow to add a new LLM to UniRouter.

This script combines evaluation and profiling into a single command:
1. Evaluates the model on validation set (calls LLM API)
2. Computes per-cluster error rates (profiling)
3. Updates unirouter_models.yaml with new model configuration
4. Updates llm_profiles.json with error rates
5. Tests routing with the new model

Usage:
    # OpenAI model
    uv run python scripts/models/add.py \
        --provider openai \
        --model gpt-4o-mini \
        --cost 0.15 \
        --api-key $OPENAI_API_KEY

    # Anthropic model
    uv run python scripts/models/add.py \
        --provider anthropic \
        --model claude-3-5-sonnet-20241022 \
        --cost 3.0 \
        --api-key $ANTHROPIC_API_KEY

    # Groq model
    uv run python scripts/models/add.py \
        --provider groq \
        --model llama-3.1-70b-versatile \
        --cost 0.0 \
        --api-key $GROQ_API_KEY
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

# Add adaptive_router to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
SCRIPTS_DIR = Path(__file__).parent.parent
ADAPTIVE_ROUTER_DIR = SCRIPTS_DIR.parent
CONFIG_DIR = ADAPTIVE_ROUTER_DIR / "config"
MODELS_CONFIG_FILE = CONFIG_DIR / "unirouter_models.yaml"
CLUSTERS_DIR = (
    ADAPTIVE_ROUTER_DIR / "adaptive_router" / "data" / "unirouter" / "clusters"
)
PROFILES_FILE = CLUSTERS_DIR / "llm_profiles.json"


def run_command(cmd: list[str], description: str) -> bool:
    """Run a shell command and handle errors.

    Args:
        cmd: Command to run as list of strings
        description: Description of the command for logging

    Returns:
        True if command succeeded, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"{description}")
    logger.info(f"{'='*80}")
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False


def update_models_config(
    provider: str,
    model: str,
    cost_per_1m_tokens: float,
    description: str | None = None,
):
    """Update unirouter_models.yaml with new model.

    Args:
        provider: Provider name
        model: Model name
        cost_per_1m_tokens: Cost per 1M tokens
        description: Optional model description
    """
    logger.info(f"\n{'='*80}")
    logger.info("Updating unirouter_models.yaml")
    logger.info(f"{'='*80}")

    # Load existing config
    with open(MODELS_CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    # Create model entry
    model_id = f"{provider}:{model}"
    new_model = {
        "id": model_id,
        "name": model,
        "provider": provider,
        "cost_per_1m_tokens": cost_per_1m_tokens,
        "description": description or f"Auto-added {provider} model",
    }

    # Check if model already exists
    existing_models = config.get("gpt5_models", [])
    model_exists = False
    for i, existing in enumerate(existing_models):
        if existing.get("id") == model_id:
            logger.info(f"Model {model_id} already exists, updating...")
            existing_models[i] = new_model
            model_exists = True
            break

    if not model_exists:
        logger.info(f"Adding new model {model_id}...")
        existing_models.append(new_model)

    config["gpt5_models"] = existing_models

    # Save updated config
    with open(MODELS_CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✅ Updated {MODELS_CONFIG_FILE}")


def verify_profile(model_id: str) -> bool:
    """Verify that model profile was created successfully.

    Args:
        model_id: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        True if profile exists and is valid
    """
    if not PROFILES_FILE.exists():
        logger.error(f"Profiles file not found: {PROFILES_FILE}")
        return False

    with open(PROFILES_FILE) as f:
        profiles = json.load(f)

    if model_id not in profiles:
        logger.error(f"Model {model_id} not found in profiles")
        return False

    error_rates = profiles[model_id]
    if not isinstance(error_rates, list) or len(error_rates) == 0:
        logger.error(f"Invalid error rates for {model_id}: {error_rates}")
        return False

    logger.info(f"✅ Profile verified for {model_id}")
    logger.info(f"   Error rates: {[f'{e:.3f}' for e in error_rates]}")
    return True


def test_routing(model_id: str):
    """Test routing with the new model.

    Args:
        model_id: Model identifier (e.g., "openai:gpt-4o-mini")
    """
    logger.info(f"\n{'='*80}")
    logger.info("Testing UniRouter with new model")
    logger.info(f"{'='*80}")

    try:
        from adaptive_router.core.router import ModelRouter
        from adaptive_router.models.api import ModelSelectionRequest

        # Initialize ModelRouter
        router = ModelRouter()

        # Test routing with a sample question
        test_prompt = "Write a Python function to calculate the factorial of a number."

        request = ModelSelectionRequest(
            prompt=test_prompt,
            cost_bias=0.5,  # Balanced cost preference
        )

        response = router.select_model(request)

        logger.info("\n✅ Routing test successful!")
        logger.info(f"   Test prompt: {test_prompt}")
        logger.info(f"   Selected model: {response.provider}/{response.model}")
        logger.info(
            f"   Alternatives: {[f'{a.provider}/{a.model}' for a in response.alternatives]}"
        )

        # Check if new model appears in routing
        all_models = [f"{response.provider}:{response.model}"]
        all_models.extend([f"{a.provider}:{a.model}" for a in response.alternatives])

        if model_id in all_models:
            logger.info(f"\n✅ New model {model_id} is available for routing!")
        else:
            logger.warning(
                f"\n⚠️  New model {model_id} was not selected for this test prompt"
            )
            logger.info(
                "   This is normal - routing depends on cluster and cost preferences"
            )

    except Exception as e:
        logger.error(f"Routing test failed: {e}", exc_info=True)


def main():
    """Main workflow to add new model."""
    parser = argparse.ArgumentParser(
        description="Add a new LLM to UniRouter (evaluation + profiling + config update)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["openai", "anthropic", "groq", "deepseek"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-4o-mini, claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        required=True,
        help="Cost per 1M tokens (input + output combined average)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or use environment variable)",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Optional model description",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step (use existing predictions)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("UniRouter: Add New LLM (End-to-End Workflow)")
    logger.info("=" * 80)
    logger.info(f"Provider: {args.provider}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Cost: ${args.cost} per 1M tokens")
    logger.info("=" * 80)

    model_id = f"{args.provider}:{args.model}"

    # Step 1: Evaluate model (unless skipped)
    if not args.skip_evaluation:
        eval_cmd = [
            "uv",
            "run",
            "python",
            str(SCRIPTS_DIR / "models" / "evaluate.py"),
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--rate-limit-delay",
            str(args.rate_limit_delay),
        ]

        if args.api_key:
            eval_cmd.extend(["--api-key", args.api_key])

        if not run_command(eval_cmd, "Step 1: Evaluate Model on Validation Set"):
            logger.error("❌ Evaluation failed. Aborting.")
            sys.exit(1)
    else:
        logger.info("\n⚠️  Skipping evaluation (using existing predictions)")

    # Step 2: Profile model (compute error rates)
    profile_cmd = [
        "uv",
        "run",
        "python",
        str(SCRIPTS_DIR / "models" / "profile.py"),
        "--model",
        model_id,
        "--update-config",
    ]

    if not run_command(profile_cmd, "Step 2: Profile Model (Compute Error Rates)"):
        logger.error("❌ Profiling failed. Aborting.")
        sys.exit(1)

    # Step 3: Update models config
    try:
        update_models_config(
            provider=args.provider,
            model=args.model,
            cost_per_1m_tokens=args.cost,
            description=args.description,
        )
    except Exception as e:
        logger.error(f"❌ Failed to update models config: {e}")
        sys.exit(1)

    # Step 4: Verify profile
    if not verify_profile(model_id):
        logger.error("❌ Profile verification failed. Check llm_profiles.json")
        sys.exit(1)

    # Step 5: Test routing
    try:
        test_routing(model_id)
    except Exception as e:
        logger.warning(f"⚠️  Routing test failed: {e}")
        logger.info("   This is not critical - model has been added successfully")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ SUCCESS! New model added to UniRouter")
    logger.info("=" * 80)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Config file: {MODELS_CONFIG_FILE}")
    logger.info(f"Profiles file: {PROFILES_FILE}")
    logger.info("\nNext steps:")
    logger.info("1. Review the updated configuration files")
    logger.info("2. Test routing with different prompts and cost preferences")
    logger.info("3. Monitor model performance in production")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
