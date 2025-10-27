#!/usr/bin/env python3
"""Re-cluster validation dataset with optimized TF-IDF dimensions.

This script:
1. Loads validation questions (990 questions already local)
2. Extracts hybrid features (384-D embeddings + 192-D TF-IDF)
3. Performs K-means clustering with spherical normalization
4. Saves lightweight cluster model (~500KB vs 100MB)
5. NO heavy data downloads - uses existing validation.json

Usage:
    # Use existing validation set (990 questions)
    uv run python scripts/cluster_dataset.py

    # Custom parameters
    uv run python scripts/cluster_dataset.py --k 20 --tfidf-dim 192 --random-seed 42

    # Stream from HuggingFace (requires internet, no local storage)
    uv run python scripts/cluster_dataset.py --source huggingface --max-questions 2000
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add adaptive_router to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_router.models import CodeQuestion
from adaptive_router.services.cluster_engine import ClusterEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
ADAPTIVE_ROUTER_DIR = Path(__file__).parent.parent / "adaptive_router"
VALIDATION_FILE = (
    ADAPTIVE_ROUTER_DIR / "data" / "unirouter" / "validation" / "validation.json"
)
CLUSTERS_DIR = ADAPTIVE_ROUTER_DIR / "data" / "unirouter" / "clusters"
CONFIG_FILE = ADAPTIVE_ROUTER_DIR / "config" / "unirouter_models.yaml"


def load_validation_questions() -> List[CodeQuestion]:
    """Load validation questions from local file (NO download).

    Returns:
        List of CodeQuestion objects

    Raises:
        FileNotFoundError: If validation.json not found
    """
    logger.info(f"Loading validation questions from {VALIDATION_FILE}")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}\n"
            f"Run Phase 1 setup first to copy validation data."
        )

    with open(VALIDATION_FILE) as f:
        data = json.load(f)

    questions = [
        CodeQuestion(
            question_id=q["question_id"],
            question=q["question"],
            choices=q["choices"],
            answer=q["answer"],
            category=q.get("category"),
            difficulty=q.get("difficulty"),
        )
        for q in data
    ]

    logger.info(f"✅ Loaded {len(questions)} questions from local file")
    return questions


def load_from_huggingface(max_questions: Optional[int] = None) -> List[CodeQuestion]:
    """Load questions from HuggingFace with streaming (NO local storage).

    Args:
        max_questions: Maximum number of questions to load

    Returns:
        List of CodeQuestion objects
    """
    logger.info("Loading questions from HuggingFace with streaming...")
    logger.info("⚠️  This requires internet connection but NO local storage")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )

    questions: list[CodeQuestion] = []

    # 1. Load MMLU subsets (streaming)
    logger.info("\n1. Streaming MMLU subsets...")
    mmlu_subsets = [
        "college_computer_science",
        "high_school_computer_science",
        "college_mathematics",
        "high_school_mathematics",
        "formal_logic",
        "machine_learning",
    ]

    for subset in mmlu_subsets:
        logger.info(f"   Streaming: {subset}")
        try:
            dataset = load_dataset("cais/mmlu", subset, split="test", streaming=True)

            for idx, item in enumerate(dataset):
                if max_questions and len(questions) >= max_questions:
                    break

                try:
                    question_text = item.get("question", "")
                    choices = item.get("choices", [])
                    answer_idx = item.get("answer", 0)

                    # Convert to A/B/C/D format
                    if isinstance(answer_idx, int) and 0 <= answer_idx < 4:
                        answer = chr(65 + answer_idx)
                    else:
                        continue

                    if not isinstance(choices, list) or len(choices) != 4:
                        continue

                    question = CodeQuestion(
                        question_id=f"mmlu_{subset}_{idx}",
                        question=question_text,
                        choices=choices,
                        answer=answer,
                        category=f"mmlu_{subset}",
                        difficulty=None,
                    )

                    questions.append(question)

                except Exception as e:
                    logger.warning(f"Failed to parse question: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to load {subset}: {e}")
            continue

    # 2. Load ARC Challenge (streaming)
    if not max_questions or len(questions) < max_questions:
        logger.info("\n2. Streaming ARC Challenge...")
        try:
            arc_dataset = load_dataset(
                "allenai/ai2_arc", "ARC-Challenge", split="test", streaming=True
            )

            for idx, item in enumerate(arc_dataset):
                if max_questions and len(questions) >= max_questions:
                    break

                try:
                    question_text = item.get("question", "")
                    choices_dict = item.get("choices", {})
                    answer_key = item.get("answerKey", "")

                    if isinstance(choices_dict, dict):
                        choice_texts = choices_dict.get("text", [])
                    else:
                        continue

                    if len(choice_texts) < 2:
                        continue

                    answer = answer_key.upper().strip() if answer_key else "A"

                    # Pad to 4 choices
                    while len(choice_texts) < 4:
                        choice_texts.append(f"[Option {chr(65+len(choice_texts))}]")

                    question = CodeQuestion(
                        question_id=f"arc_{idx}",
                        question=question_text,
                        choices=choice_texts[:4],
                        answer=answer,
                        category="arc_challenge",
                        difficulty="challenge",
                    )

                    questions.append(question)

                except Exception as e:
                    logger.warning(f"Failed to parse ARC question: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to load ARC: {e}")

    logger.info(f"\n✅ Streamed {len(questions)} questions from HuggingFace")
    logger.info("   (Data NOT stored locally - only loaded to memory)")

    return questions


def save_cluster_model(
    engine: ClusterEngine,
    output_dir: Path,
    silhouette_score: float,
) -> Dict[str, str]:
    """Save cluster model in lightweight JSON format.

    Args:
        engine: Fitted ClusterEngine
        output_dir: Directory to save model
        silhouette_score: Clustering quality score

    Returns:
        Dictionary with saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving lightweight cluster model to {output_dir}")

    # 1. Save as pickle for compatibility with existing code
    logger.info("  Saving cluster_engine.pkl (for compatibility)...")
    pickle_file = output_dir / "cluster_engine.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(engine, f)

    pickle_size = pickle_file.stat().st_size / 1024 / 1024
    logger.info(f"  ✅ Saved cluster_engine.pkl ({pickle_size:.1f} MB)")

    # 2. Save metadata
    logger.info("  Saving metadata.json...")
    metadata = {
        "n_clusters": engine.n_clusters,
        "n_train_questions": len(engine.questions),
        "silhouette_score": silhouette_score,
        "embedding_model": engine.feature_extractor.embedding_model_name,
        "tfidf_max_features": engine.feature_extractor.tfidf_vectorizer.max_features,
        "embedding_dim": engine.feature_extractor.embedding_dim,
        "total_features": engine.feature_extractor.embedding_dim
        + engine.feature_extractor.tfidf_vectorizer.max_features,
        "cluster_sizes": [
            int(count) for count in np.bincount(engine.cluster_assignments)
        ],
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    metadata_size = metadata_file.stat().st_size / 1024
    logger.info(f"  ✅ Saved metadata.json ({metadata_size:.1f} KB)")

    # 3. Log what we're NOT saving (for efficiency)
    logger.info("\n  📊 Storage efficiency:")
    logger.info(
        f"     Cluster centers: {engine.n_clusters} × {metadata['total_features']}-D"
    )
    logger.info(f"     TF-IDF vocab: {metadata['tfidf_max_features']} terms")
    logger.info("     Embedding model: Reloaded fresh (not saved)")

    return {
        "cluster_engine": str(pickle_file),
        "metadata": str(metadata_file),
    }


def run_clustering(
    questions: List[CodeQuestion],
    k: int,
    tfidf_dim: int,
    random_seed: int,
) -> tuple[ClusterEngine, float]:
    """Run K-means clustering on questions.

    Args:
        questions: List of questions to cluster
        k: Number of clusters
        tfidf_dim: TF-IDF feature dimensions
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (fitted ClusterEngine, silhouette_score)
    """
    logger.info("\n" + "=" * 80)
    logger.info("CLUSTERING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"  Questions: {len(questions)}")
    logger.info(f"  Clusters (K): {k}")
    logger.info(f"  TF-IDF dimensions: {tfidf_dim}")
    logger.info("  Embedding dimensions: 384")
    logger.info(f"  Total features: {384 + tfidf_dim}")
    logger.info(
        f"  Feature ratio: {384/(384+tfidf_dim):.1%} embeddings, {tfidf_dim/(384+tfidf_dim):.1%} TF-IDF"
    )
    logger.info(f"  Random seed: {random_seed}")
    logger.info("=" * 80)

    # Initialize clustering engine
    logger.info("\nInitializing ClusterEngine...")
    engine = ClusterEngine(
        n_clusters=k,
        max_iter=300,
        random_state=random_seed,
        n_init=10,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        tfidf_max_features=tfidf_dim,
        tfidf_ngram_range=(1, 2),
    )

    # Perform clustering
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMING CLUSTERING")
    logger.info("=" * 80)

    engine.fit(questions)

    # Get cluster info
    cluster_info = engine.get_cluster_info()

    logger.info("\n" + "=" * 80)
    logger.info("CLUSTERING RESULTS")
    logger.info("=" * 80)
    logger.info(f"  Silhouette score: {cluster_info['silhouette_score']:.6f}")
    logger.info(
        f"  Cluster size range: {cluster_info['min_cluster_size']} - {cluster_info['max_cluster_size']}"
    )
    logger.info(f"  Average cluster size: {cluster_info['avg_cluster_size']:.1f}")
    logger.info("\nCluster distribution:")
    for cluster_id, size in cluster_info["cluster_sizes"].items():
        pct = (size / len(questions)) * 100
        logger.info(f"  Cluster {cluster_id}: {size:4d} questions ({pct:5.1f}%)")
    logger.info("=" * 80)

    return engine, cluster_info["silhouette_score"]


def main():
    """Main clustering workflow."""
    parser = argparse.ArgumentParser(
        description="Re-cluster validation dataset with optimized dimensions"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["local", "huggingface"],
        default="local",
        help="Data source (default: local validation.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters (default: 5)",
    )
    parser.add_argument(
        "--tfidf-dim",
        type=int,
        default=192,
        help="TF-IDF feature dimensions (default: 192)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Max questions to load from HuggingFace (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CLUSTERS_DIR,
        help=f"Output directory (default: {CLUSTERS_DIR})",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("UniRouter: Re-Cluster with Optimized Dimensions")
    logger.info("=" * 80)

    try:
        # 1. Load questions
        if args.source == "local":
            questions = load_validation_questions()
        else:
            questions = load_from_huggingface(max_questions=args.max_questions)

        if len(questions) == 0:
            logger.error("❌ No questions loaded!")
            sys.exit(1)

        # 2. Run clustering
        engine, silhouette = run_clustering(
            questions=questions,
            k=args.k,
            tfidf_dim=args.tfidf_dim,
            random_seed=args.random_seed,
        )

        # 3. Save cluster model
        saved_files = save_cluster_model(
            engine=engine,
            output_dir=args.output_dir,
            silhouette_score=silhouette,
        )

        # 4. Test cluster assignment
        logger.info("\n" + "=" * 80)
        logger.info("TESTING CLUSTER ASSIGNMENT")
        logger.info("=" * 80)

        test_prompts = [
            "Write a Python function to implement quicksort",
            "Explain quantum entanglement in physics",
            "Design a REST API with authentication",
        ]

        for prompt in test_prompts:
            cluster_id, distance = engine.assign_question(prompt)
            logger.info(f"\nPrompt: '{prompt}'")
            logger.info(f"  → Cluster {cluster_id} (distance: {distance:.3f})")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ CLUSTERING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"  Clusters: {engine.n_clusters}")
        logger.info(f"  Questions: {len(questions)}")
        logger.info(f"  Silhouette score: {silhouette:.6f}")
        logger.info(
            f"  Features: {384 + args.tfidf_dim}-D ({384}-D emb + {args.tfidf_dim}-D tfidf)"
        )
        logger.info(f"  Output: {args.output_dir}")
        logger.info("\nSaved files:")
        for key, path in saved_files.items():
            logger.info(f"  {key}: {path}")
        logger.info("=" * 80)
        logger.info("\n📌 NEXT STEPS:")
        logger.info("1. Re-profile all models:")
        logger.info(
            "   uv run python scripts/profile_new_model.py --model openai:gpt-5-codex --update-config"
        )
        logger.info(
            "   uv run python scripts/profile_new_model.py --model openai:gpt-5-mini --update-config"
        )
        logger.info(
            "   uv run python scripts/profile_new_model.py --model openai:gpt-5-nano --update-config"
        )
        logger.info(
            "   uv run python scripts/profile_new_model.py --model openai:gpt-4.1-nano --update-config"
        )
        logger.info("\n2. Test routing with new clusters:")
        logger.info(
            '   python -c "from adaptive_router.services.unirouter_service import UniRouterService; ..."'
        )
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Clustering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
