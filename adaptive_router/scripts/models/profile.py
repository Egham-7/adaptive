#!/usr/bin/env python3
"""Profile a new LLM by computing per-cluster error rates.

Paper Section 5.1: "For any LLM h ∈ H_all, compute Ψ_clust(h) ∈ [0,1]^K using its
per-cluster validation errors: Ψ_clust,k(h) := (1/|C_k|) Σ 1[y ≠ h(x)]"

This creates a K-dimensional feature vector for each LLM representing its
performance profile across different prompt types.

Usage:
    uv run python scripts/models/profile.py --model openai:gpt-4o-mini
    uv run python scripts/models/profile.py --model anthropic:claude-3-5-sonnet-20241022 --update-config
"""

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add adaptive_router to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaptive_router.services.unirouter.cluster_engine import ClusterEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
ADAPTIVE_ROUTER_DIR = Path(__file__).parent.parent.parent / "adaptive_router"
VALIDATION_FILE = (
    ADAPTIVE_ROUTER_DIR / "data" / "unirouter" / "validation" / "validation.json"
)
PREDICTIONS_DIR = ADAPTIVE_ROUTER_DIR / "data" / "unirouter" / "predictions"
CLUSTERS_DIR = ADAPTIVE_ROUTER_DIR / "data" / "unirouter" / "clusters"
CLUSTER_ENGINE_FILE = CLUSTERS_DIR / "cluster_engine.pkl"
METADATA_FILE = CLUSTERS_DIR / "metadata.json"
PROFILES_FILE = CLUSTERS_DIR / "llm_profiles.json"


class ModulePathUnpickler(pickle.Unpickler):
    """Custom unpickler that handles module path changes from unirouter.* to adaptive_router.services.unirouter.*"""

    def find_class(self, module, name):
        """Override find_class to redirect old unirouter paths."""
        if module.startswith("unirouter."):
            # Map old paths to new paths
            old_to_new = {
                "unirouter.clustering.cluster_engine": "adaptive_router.services.unirouter.cluster_engine",
                "unirouter.clustering.feature_extractor": "adaptive_router.services.unirouter.feature_extractor",
                "unirouter.models.schemas": "adaptive_router.services.unirouter.schemas",
            }

            new_module = old_to_new.get(
                module,
                module.replace("unirouter.", "adaptive_router.services.unirouter."),
            )
            logger.debug(f"Remapping pickle module: {module} -> {new_module}")
            return super().find_class(new_module, name)

        return super().find_class(module, name)


def load_validation_questions() -> List[Dict]:
    """Load validation questions from JSON."""
    logger.info(f"Loading validation questions from {VALIDATION_FILE}")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}\n"
            f"Run Phase 1 setup first to copy validation data."
        )

    with open(VALIDATION_FILE) as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} validation questions")
    return questions


def load_cluster_engine() -> ClusterEngine:
    """Load trained cluster engine from pickle."""
    logger.info(f"Loading cluster engine from {CLUSTER_ENGINE_FILE}")

    if not CLUSTER_ENGINE_FILE.exists():
        raise FileNotFoundError(
            f"Cluster engine not found: {CLUSTER_ENGINE_FILE}\n"
            f"Run clustering first or check data directory."
        )

    # Load cluster engine using custom unpickler
    with open(CLUSTER_ENGINE_FILE, "rb") as f:
        cluster_engine = ModulePathUnpickler(f).load()

    if not isinstance(cluster_engine, ClusterEngine):
        raise ValueError(
            f"Loaded cluster engine has wrong type: {type(cluster_engine)}"
        )

    # WORKAROUND for macOS: Reload the embedding model fresh to avoid version conflicts
    try:
        import platform

        if platform.system() == "Darwin":
            logger.info("Reloading embedding model for macOS compatibility...")
            from sentence_transformers import SentenceTransformer

            device = "cpu"
            model_name = cluster_engine.feature_extractor.embedding_model_name
            cluster_engine.feature_extractor.embedding_model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True
            )
            logger.info(f"Reloaded embedding model on {device}")
    except Exception as e:
        logger.warning(f"Could not reload embedding model, continuing anyway: {e}")

    # Load metadata
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    logger.info(
        f"Loaded cluster engine: K={metadata['n_clusters']} clusters, "
        f"silhouette score: {metadata.get('silhouette_score', 'N/A')}"
    )

    return cluster_engine


def assign_to_clusters(
    questions: List[Dict], cluster_engine: ClusterEngine
) -> Dict[int, List[Dict]]:
    """Assign validation questions to clusters using trained engine.

    This implements: C_k := {(x,y) : (x,y) ∈ S_val, Φ_clust,k(x) = 1}
    """
    logger.info("Assigning validation questions to clusters...")

    # Convert questions to CodeQuestion objects for cluster_engine
    from adaptive_router.services.unirouter.schemas import CodeQuestion

    code_questions = [
        CodeQuestion(
            question_id=q["question_id"],
            question=q["question"],
            choices=q["choices"],
            answer=q["answer"],
            category=q.get("category"),
            difficulty=q.get("difficulty"),
        )
        for q in questions
    ]

    # Get cluster assignments for validation set
    cluster_assignments = cluster_engine.assign_clusters(code_questions)

    # Group questions by cluster (keep original dict format)
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    for q, cluster_id in zip(questions, cluster_assignments):
        clusters[int(cluster_id)].append(q)

    logger.info("\nValidation set cluster distribution:")
    for cluster_id in sorted(clusters.keys()):
        cluster_questions = clusters[cluster_id]
        pct = (len(cluster_questions) / len(questions)) * 100
        logger.info(
            f"  Cluster {cluster_id:2d}: {len(cluster_questions):4d} questions ({pct:5.1f}%)"
        )

    return clusters


def load_predictions(model_id: str) -> Dict[str, str]:
    """Load LLM predictions from file.

    Args:
        model_id: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Dictionary mapping question_id to predicted answer
    """
    # Convert model_id to filename: provider:model -> provider_model_predictions.json
    filename = model_id.replace(":", "_") + "_predictions.json"
    predictions_file = PREDICTIONS_DIR / filename

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}\n"
            f"Run scripts/models/evaluate.py first to generate predictions."
        )

    with open(predictions_file) as f:
        predictions = json.load(f)

    logger.info(f"Loaded {len(predictions)} predictions from {predictions_file}")
    return predictions


def compute_per_cluster_errors(
    predictions: Dict[str, str],
    clusters: Dict[int, List[Dict]],
) -> np.ndarray:
    """Compute Ψ_clust(h) - per-cluster error rates for an LLM.

    Args:
        predictions: {question_id: predicted_answer}
        clusters: {cluster_id: [questions]}

    Returns:
        K-dimensional vector of error rates
    """
    K = len(clusters)
    error_rates = np.zeros(K)

    logger.info("\nPer-cluster coverage analysis:")
    logger.info(
        f"{'Cluster':<10} {'Total':<10} {'Evaluated':<12} {'Coverage %':<12} {'Errors':<10} {'Error Rate':<12}"
    )
    logger.info("-" * 80)

    for cluster_id, questions in clusters.items():
        total_in_cluster = len(questions)
        if total_in_cluster == 0:
            continue

        # Count errors in this cluster - ONLY for questions with predictions
        errors = 0
        evaluated = 0
        for q in questions:
            pred = predictions.get(q["question_id"])
            if pred is None:
                # Skip questions that weren't evaluated
                continue
            evaluated += 1
            if pred != q["answer"]:
                errors += 1

        # Error rate for this cluster (only among evaluated questions)
        if evaluated > 0:
            error_rates[cluster_id] = errors / evaluated
            coverage_pct = (evaluated / total_in_cluster) * 100
            logger.info(
                f"{cluster_id:<10} {total_in_cluster:<10} {evaluated:<12} {coverage_pct:<12.1f} {errors:<10} {error_rates[cluster_id]:<12.3f}"
            )
        else:
            # No evaluated questions in this cluster - use 0 error rate
            error_rates[cluster_id] = 0.0
            logger.warning(
                f"{cluster_id:<10} {total_in_cluster:<10} {evaluated:<12} {'0.0':<12} {errors:<10} {'N/A (no samples)':<12}"
            )

    total_questions = sum(len(q) for q in clusters.values())
    total_evaluated = sum(
        1
        for q_list in clusters.values()
        for q in q_list
        if predictions.get(q["question_id"]) is not None
    )
    overall_coverage = (
        (total_evaluated / total_questions) * 100 if total_questions > 0 else 0
    )
    logger.info("-" * 80)
    logger.info(
        f"Overall: {total_questions} questions, {total_evaluated} evaluated ({overall_coverage:.1f}% coverage)"
    )

    return error_rates


def update_llm_profiles(model_id: str, error_rates: np.ndarray):
    """Update llm_profiles.json with new model's error rates.

    Args:
        model_id: Model identifier (e.g., "openai:gpt-4o-mini")
        error_rates: K-dimensional vector of per-cluster error rates
    """
    # Load existing profiles
    if PROFILES_FILE.exists():
        with open(PROFILES_FILE) as f:
            profiles = json.load(f)
    else:
        profiles = {}

    # Update with new model
    profiles[model_id] = error_rates.tolist()

    # Save updated profiles
    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)

    logger.info(f"✅ Updated {PROFILES_FILE} with {model_id} profile")


def main():
    """Main profiling pipeline."""
    parser = argparse.ArgumentParser(
        description="Profile LLM by computing per-cluster error rates"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., openai:gpt-4o-mini, anthropic:claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update llm_profiles.json automatically",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("UniRouter: Profile LLM via Per-Cluster Error Rates")
    logger.info("=" * 80)

    try:
        # Load validation questions
        questions = load_validation_questions()

        # Load cluster engine
        cluster_engine = load_cluster_engine()

        # Assign questions to clusters
        clusters = assign_to_clusters(questions, cluster_engine)

        # Load predictions for this model
        predictions = load_predictions(args.model)

        # Compute per-cluster error rates
        logger.info(f"\n{'='*80}")
        logger.info(f"Computing per-cluster error rates for {args.model}")
        logger.info(f"{'='*80}")

        error_rates = compute_per_cluster_errors(predictions, clusters)

        logger.info(f"\n{args.model} Error Profile:")
        logger.info(f"  Error rates per cluster: {[f'{e:.3f}' for e in error_rates]}")
        logger.info(f"  Average error: {np.mean(error_rates):.3f}")
        logger.info(f"  Std dev: {np.std(error_rates):.3f}")

        # Show best/worst clusters
        best_cluster = int(np.argmin(error_rates))
        worst_cluster = int(np.argmax(error_rates))
        logger.info(
            f"\n  Best cluster: {best_cluster} (error: {error_rates[best_cluster]:.3f})"
        )
        logger.info(
            f"  Worst cluster: {worst_cluster} (error: {error_rates[worst_cluster]:.3f})"
        )

        # Update profiles file if requested
        if args.update_config:
            update_llm_profiles(args.model, error_rates)
        else:
            logger.info(f"\n⚠️  Profile NOT saved to {PROFILES_FILE}")
            logger.info("   Run with --update-config to save automatically")

        logger.info("\n" + "=" * 80)
        logger.info("✅ LLM profiling complete!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
