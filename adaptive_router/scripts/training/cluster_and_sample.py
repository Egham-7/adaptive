#!/usr/bin/env python3
"""Cluster large dataset and sample representatives for model profiling.

This implements the UniRouter methodology:
1. Stream LARGE dataset from HuggingFace (2000-5000 questions, NO local storage)
2. Test K values (5-20), find optimal silhouette score
3. Cluster with optimal K using K-means on hybrid features
4. Stratified sampling: Select N representatives per cluster
5. Save ONLY sampled questions to validation.json (~100-600 questions)
6. Save cluster_engine.pkl (cluster centers for inference routing)

Key insight: Cluster centers learned from LARGE dataset give comprehensive
coverage, but we only need SMALL sampled representatives for model profiling.

Usage:
    # Default: Stream 3000 questions, K=5-20, 30 samples/cluster
    uv run python scripts/cluster_and_sample.py

    # Custom parameters
    uv run python scripts/cluster_and_sample.py \\
        --stream-size 5000 \\
        --k-min 8 --k-max 15 \\
        --samples-per-cluster 25

    # Dry run (don't save)
    uv run python scripts/cluster_and_sample.py --dry-run
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np

# Add adaptive_router to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_router.models import CodeQuestion
from adaptive_router.core.cluster_engine import ClusterEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
ADAPTIVE_ROUTER_DIR = Path(__file__).parent.parent
VALIDATION_FILE = (
    ADAPTIVE_ROUTER_DIR
    / "adaptive_router"
    / "data"
    / "unirouter"
    / "validation"
    / "validation.json"
)
CLUSTERS_DIR = (
    ADAPTIVE_ROUTER_DIR / "adaptive_router" / "data" / "unirouter" / "clusters"
)
CONFIG_FILE = ADAPTIVE_ROUTER_DIR / "config" / "unirouter_models.yaml"

# MMLU subjects to include
MMLU_SUBJECTS = [
    # Existing subjects (from current validation.json)
    "college_computer_science",
    "high_school_computer_science",
    "college_mathematics",
    "high_school_mathematics",
    "formal_logic",
    "machine_learning",
    # NEW Option A: Technical/scientific subjects
    "college_physics",
    "high_school_physics",
    "computer_security",
    "electrical_engineering",
    "abstract_algebra",
    "astronomy",
    "elementary_mathematics",
    "college_chemistry",
    "high_school_chemistry",
    "conceptual_physics",
]


def stream_questions_from_huggingface(max_questions: int) -> List[CodeQuestion]:
    """Stream questions from HuggingFace datasets (NO local storage).

    Args:
        max_questions: Maximum total questions to stream

    Returns:
        List of CodeQuestion objects
    """
    logger.info(f"{'='*80}")
    logger.info("STREAMING QUESTIONS FROM HUGGINGFACE")
    logger.info(f"{'='*80}")
    logger.info(f"Target: {max_questions} total questions")
    logger.info(f"MMLU subjects: {len(MMLU_SUBJECTS)}")
    logger.info("ARC: Challenge + Easy")
    logger.info("⚠️  Streaming only - NO local storage")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )

    questions: List[CodeQuestion] = []
    questions_per_mmlu = max_questions // (len(MMLU_SUBJECTS) + 2)  # Reserve for ARC
    arc_challenge_budget = max_questions // 4
    arc_easy_budget = max_questions // 4

    # 1. Stream MMLU subjects
    logger.info(f"\n1. Streaming MMLU subjects ({questions_per_mmlu} per subject)...")
    for subject in MMLU_SUBJECTS:
        logger.info(f"   Streaming: {subject}")
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test", streaming=True)
            count = 0

            for idx, item in enumerate(dataset):
                if count >= questions_per_mmlu:
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
                        question_id=f"mmlu_{subject}_{idx}",
                        question=question_text,
                        choices=choices,
                        answer=answer,
                        category=f"mmlu_{subject}",
                        difficulty=None,
                    )

                    questions.append(question)
                    count += 1

                except Exception as e:
                    logger.warning(f"Failed to parse MMLU question: {e}")
                    continue

            logger.info(f"     ✅ Streamed {count} questions from {subject}")

        except Exception as e:
            logger.error(f"Failed to load {subject}: {e}")
            continue

    # 2. Stream ARC-Challenge
    logger.info(f"\n2. Streaming ARC-Challenge ({arc_challenge_budget} questions)...")
    try:
        arc_dataset = load_dataset(
            "allenai/ai2_arc", "ARC-Challenge", split="test", streaming=True
        )
        count = 0

        for idx, item in enumerate(arc_dataset):
            if count >= arc_challenge_budget:
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
                    question_id=f"arc_challenge_{idx}",
                    question=question_text,
                    choices=choice_texts[:4],
                    answer=answer,
                    category="arc_challenge",
                    difficulty="challenge",
                )

                questions.append(question)
                count += 1

            except Exception as e:
                logger.warning(f"Failed to parse ARC-Challenge question: {e}")
                continue

        logger.info(f"   ✅ Streamed {count} questions from ARC-Challenge")

    except Exception as e:
        logger.error(f"Failed to load ARC-Challenge: {e}")

    # 3. Stream ARC-Easy
    logger.info(f"\n3. Streaming ARC-Easy ({arc_easy_budget} questions)...")
    try:
        arc_easy_dataset = load_dataset(
            "allenai/ai2_arc", "ARC-Easy", split="test", streaming=True
        )
        count = 0

        for idx, item in enumerate(arc_easy_dataset):
            if count >= arc_easy_budget:
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
                    question_id=f"arc_easy_{idx}",
                    question=question_text,
                    choices=choice_texts[:4],
                    answer=answer,
                    category="arc_easy",
                    difficulty="easy",
                )

                questions.append(question)
                count += 1

            except Exception as e:
                logger.warning(f"Failed to parse ARC-Easy question: {e}")
                continue

        logger.info(f"   ✅ Streamed {count} questions from ARC-Easy")

    except Exception as e:
        logger.error(f"Failed to load ARC-Easy: {e}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Total questions streamed: {len(questions)}")
    logger.info("⚠️  Data NOT stored locally - only in memory")
    logger.info(f"{'='*80}")

    return questions


def find_optimal_k(
    questions: List[CodeQuestion],
    k_min: int,
    k_max: int,
    tfidf_dim: int,
    random_seed: int,
) -> tuple[int, List[tuple[int, float]]]:
    """Find optimal K value by testing multiple values.

    Args:
        questions: List of questions to cluster
        k_min: Minimum K to test
        k_max: Maximum K to test
        tfidf_dim: TF-IDF feature dimensions
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (optimal_k, [(k, silhouette_score), ...])
    """
    logger.info(f"\n{'='*80}")
    logger.info("FINDING OPTIMAL K VALUE")
    logger.info(f"{'='*80}")
    logger.info(f"Testing K range: {k_min} to {k_max}")
    logger.info(f"Questions: {len(questions)}")

    results: List[tuple[int, float]] = []

    for k in range(k_min, k_max + 1):
        logger.info(f"\nTesting K={k}...")

        # Create cluster engine
        engine = ClusterEngine(
            n_clusters=k,
            max_iter=300,
            random_state=random_seed,
            n_init=10,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            tfidf_max_features=tfidf_dim,
            tfidf_ngram_range=(1, 2),
        )

        # Fit and get silhouette score
        engine.fit(questions)
        cluster_info = engine.get_cluster_info()
        silhouette = cluster_info["silhouette_score"]

        results.append((k, silhouette))
        logger.info(f"  K={k}: silhouette={silhouette:.6f}")

    # Find optimal K
    optimal_k = max(results, key=lambda x: x[1])[0]
    optimal_silhouette = max(results, key=lambda x: x[1])[1]

    logger.info(f"\n{'='*80}")
    logger.info("K OPTIMIZATION RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Optimal K: {optimal_k}")
    logger.info(f"Silhouette score: {optimal_silhouette:.6f}")
    logger.info("\nAll results:")
    for k, silhouette in results:
        marker = "👈 BEST" if k == optimal_k else ""
        logger.info(f"  K={k:2d}: {silhouette:.6f} {marker}")
    logger.info(f"{'='*80}")

    return optimal_k, results


def cluster_with_k(
    questions: List[CodeQuestion],
    k: int,
    tfidf_dim: int,
    random_seed: int,
) -> ClusterEngine:
    """Cluster questions with specified K value.

    Args:
        questions: List of questions to cluster
        k: Number of clusters
        tfidf_dim: TF-IDF feature dimensions
        random_seed: Random seed

    Returns:
        Fitted ClusterEngine
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"CLUSTERING WITH K={k}")
    logger.info(f"{'='*80}")

    engine = ClusterEngine(
        n_clusters=k,
        max_iter=300,
        random_state=random_seed,
        n_init=10,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        tfidf_max_features=tfidf_dim,
        tfidf_ngram_range=(1, 2),
    )

    engine.fit(questions)
    cluster_info = engine.get_cluster_info()

    logger.info("\nClustering complete!")
    logger.info(f"  Silhouette score: {cluster_info['silhouette_score']:.6f}")
    logger.info(
        f"  Cluster size range: {cluster_info['min_cluster_size']} - {cluster_info['max_cluster_size']}"
    )
    logger.info(f"  Average cluster size: {cluster_info['avg_cluster_size']:.1f}")

    return engine


def stratified_sample_from_clusters(
    questions: List[CodeQuestion],
    cluster_assignments: np.ndarray,
    n_clusters: int,
    samples_per_cluster: int,
    random_seed: int,
) -> List[CodeQuestion]:
    """Stratified sampling: Select N representatives from each cluster.

    Args:
        questions: All questions
        cluster_assignments: Cluster IDs for each question
        n_clusters: Number of clusters
        samples_per_cluster: Number of samples per cluster
        random_seed: Random seed

    Returns:
        List of sampled representative questions
    """
    logger.info(f"\n{'='*80}")
    logger.info("STRATIFIED SAMPLING")
    logger.info(f"{'='*80}")
    logger.info(f"Samples per cluster: {samples_per_cluster}")

    np.random.seed(random_seed)
    sampled_questions: List[CodeQuestion] = []

    for cluster_id in range(n_clusters):
        # Get questions in this cluster
        cluster_mask = cluster_assignments == cluster_id
        cluster_questions = [q for q, mask in zip(questions, cluster_mask) if mask]

        # Sample from this cluster
        n_samples = min(samples_per_cluster, len(cluster_questions))
        sampled_indices = np.random.choice(
            len(cluster_questions), size=n_samples, replace=False
        )
        cluster_samples = [cluster_questions[i] for i in sampled_indices]

        sampled_questions.extend(cluster_samples)

        logger.info(
            f"  Cluster {cluster_id:2d}: {len(cluster_questions):4d} total → {n_samples:3d} sampled"
        )

    logger.info(f"\n{'='*80}")
    logger.info(f"Total sampled: {len(sampled_questions)} questions")
    logger.info(f"{'='*80}")

    return sampled_questions


def save_validation_set(questions: List[CodeQuestion], output_file: Path) -> None:
    """Save sampled validation questions to JSON.

    Args:
        questions: Sampled questions
        output_file: Path to save file
    """
    logger.info(f"\nSaving validation set to {output_file}")

    # Ensure directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict format
    questions_dict = [
        {
            "question_id": q.question_id,
            "question": q.question,
            "choices": q.choices,
            "answer": q.answer,
            "category": q.category,
            "difficulty": q.difficulty,
        }
        for q in questions
    ]

    # Save with pretty formatting
    with open(output_file, "w") as f:
        json.dump(questions_dict, f, indent=2)

    file_size = output_file.stat().st_size / 1024 / 1024
    logger.info(f"✅ Saved {len(questions)} questions ({file_size:.2f} MB)")


def save_cluster_engine(engine: ClusterEngine, output_dir: Path) -> None:
    """Save cluster engine to JSON files (+ pickle for local use).

    Saves lightweight JSON files for Git tracking and full pickle for local use.

    Args:
        engine: Fitted ClusterEngine
        output_dir: Directory to save model
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving cluster engine to {output_dir}")

    # 1. Save cluster centers as JSON (Git-friendly)
    cluster_centers_file = output_dir / "cluster_centers.json"
    cluster_data = {
        "cluster_centers": engine.kmeans.cluster_centers_.tolist(),
        "n_clusters": engine.n_clusters,
        "feature_dim": engine.kmeans.cluster_centers_.shape[1],
    }
    with open(cluster_centers_file, "w") as f:
        json.dump(cluster_data, f, indent=2)

    cluster_centers_size = cluster_centers_file.stat().st_size / 1024
    logger.info(f"✅ Saved cluster_centers.json ({cluster_centers_size:.1f} KB)")

    # 2. Save TF-IDF vocabulary as JSON (Git-friendly)
    tfidf_vocab_file = output_dir / "tfidf_vocabulary.json"

    # Convert vocabulary to native Python types (numpy int64 -> int)
    vocabulary_native = {
        str(k): int(v)
        for k, v in engine.feature_extractor.tfidf_vectorizer.vocabulary_.items()
    }

    tfidf_data = {
        "vocabulary": vocabulary_native,
        "idf": engine.feature_extractor.tfidf_vectorizer.idf_.tolist(),
        "max_features": int(engine.feature_extractor.tfidf_vectorizer.max_features),
        "ngram_range": list(engine.feature_extractor.tfidf_vectorizer.ngram_range),
    }
    with open(tfidf_vocab_file, "w") as f:
        json.dump(tfidf_data, f, indent=2)

    tfidf_vocab_size = tfidf_vocab_file.stat().st_size / 1024
    logger.info(f"✅ Saved tfidf_vocabulary.json ({tfidf_vocab_size:.1f} KB)")

    # 2b. Save scaler parameters as JSON (Git-friendly)
    logger.info("Saving scaler parameters...")
    scaler_params_file = output_dir / "scaler_parameters.json"
    scaler_data = {
        "embedding_scaler": {
            "mean": engine.feature_extractor.embedding_scaler.mean_.tolist(),
            "scale": engine.feature_extractor.embedding_scaler.scale_.tolist(),
        },
        "tfidf_scaler": {
            "mean": engine.feature_extractor.tfidf_scaler.mean_.tolist(),
            "scale": engine.feature_extractor.tfidf_scaler.scale_.tolist(),
        },
    }
    with open(scaler_params_file, "w") as f:
        json.dump(scaler_data, f, indent=2)

    scaler_params_size = scaler_params_file.stat().st_size / 1024
    logger.info(f"✅ Saved scaler_parameters.json ({scaler_params_size:.1f} KB)")

    # 3. Save metadata with enhanced config info
    cluster_info = engine.get_cluster_info()
    metadata = {
        "n_clusters": cluster_info["n_clusters"],
        "n_train_questions": cluster_info["n_questions"],
        "silhouette_score": cluster_info["silhouette_score"],
        "embedding_model": engine.feature_extractor.embedding_model_name,
        "embedding_dim": engine.feature_extractor.embedding_dim,
        "tfidf_max_features": engine.feature_extractor.tfidf_vectorizer.max_features,
        "tfidf_ngram_range": list(
            engine.feature_extractor.tfidf_vectorizer.ngram_range
        ),
        "total_features": engine.feature_extractor.embedding_dim
        + engine.feature_extractor.tfidf_vectorizer.max_features,
        "cluster_sizes": cluster_info["cluster_sizes"],
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("✅ Saved metadata.json")

    # 4. Save pickle for local use (will be gitignored)
    pickle_file = output_dir / "cluster_engine.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(engine, f)

    pickle_size = pickle_file.stat().st_size / 1024 / 1024
    logger.info(
        f"✅ Saved cluster_engine.pkl ({pickle_size:.1f} MB) [local only, gitignored]"
    )

    total_json_size = (
        cluster_centers_size
        + tfidf_vocab_size
        + scaler_params_size
        + (metadata_file.stat().st_size / 1024)
    )
    logger.info(f"\n📊 Total JSON size: {total_json_size:.1f} KB (Git-tracked)")
    logger.info(f"📊 Pickle size: {pickle_size:.1f} MB (local only)")
    logger.info(
        f"📊 Size reduction for Git: {(1 - total_json_size / 1024 / pickle_size) * 100:.1f}%"
    )


def main():
    """Main clustering and sampling workflow."""
    parser = argparse.ArgumentParser(
        description="Cluster large dataset and sample representatives for profiling"
    )
    parser.add_argument(
        "--stream-size",
        type=int,
        default=3000,
        help="Number of questions to stream from HuggingFace (default: 3000)",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=5,
        help="Minimum K to test (default: 5)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=20,
        help="Maximum K to test (default: 20)",
    )
    parser.add_argument(
        "--samples-per-cluster",
        type=int,
        default=30,
        help="Number of samples per cluster for validation set (default: 30)",
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
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save, just show results",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("UniRouter: Cluster & Sample for Model Profiling")
    logger.info("=" * 80)
    logger.info(f"Stream size: {args.stream_size}")
    logger.info(f"K range: {args.k_min} to {args.k_max}")
    logger.info(f"Samples per cluster: {args.samples_per_cluster}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 80)

    try:
        # 1. Stream questions from HuggingFace
        questions = stream_questions_from_huggingface(max_questions=args.stream_size)

        if len(questions) == 0:
            logger.error("❌ No questions loaded!")
            sys.exit(1)

        # 2. Find optimal K value
        optimal_k, k_results = find_optimal_k(
            questions=questions,
            k_min=args.k_min,
            k_max=args.k_max,
            tfidf_dim=args.tfidf_dim,
            random_seed=args.random_seed,
        )

        # 3. Cluster with optimal K
        cluster_engine = cluster_with_k(
            questions=questions,
            k=optimal_k,
            tfidf_dim=args.tfidf_dim,
            random_seed=args.random_seed,
        )

        # 4. Stratified sampling from clusters
        sampled_questions = stratified_sample_from_clusters(
            questions=questions,
            cluster_assignments=cluster_engine.cluster_assignments,
            n_clusters=optimal_k,
            samples_per_cluster=args.samples_per_cluster,
            random_seed=args.random_seed,
        )

        # 5. Save (unless dry run)
        if args.dry_run:
            logger.info("\n🔍 DRY RUN - Nothing saved")
            logger.info(f"Would save {len(sampled_questions)} sampled questions")
            logger.info(f"Would save cluster engine with K={optimal_k}")
        else:
            save_validation_set(sampled_questions, VALIDATION_FILE)
            save_cluster_engine(cluster_engine, CLUSTERS_DIR)

            logger.info("\n✅ CLUSTERING & SAMPLING COMPLETE!")
            logger.info(f"\n{'='*80}")
            logger.info("SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"Streamed questions: {len(questions)}")
            logger.info(f"Optimal K: {optimal_k}")
            logger.info(f"Validation set size: {len(sampled_questions)}")
            logger.info(f"Cluster engine saved to: {CLUSTERS_DIR}/cluster_engine.pkl")
            logger.info(f"Validation set saved to: {VALIDATION_FILE}")
            logger.info(f"{'='*80}")

            logger.info("\n📌 NEXT STEPS:")
            logger.info("1. Backup old evaluation data:")
            logger.info("   mkdir -p adaptive_router/data/unirouter/predictions_backup")
            logger.info(
                "   cp -r adaptive_router/data/unirouter/predictions/* predictions_backup/"
            )
            logger.info("\n2. Clear old predictions and profiles:")
            logger.info("   rm adaptive_router/data/unirouter/predictions/*")
            logger.info(
                "   echo '{}' > adaptive_router/data/unirouter/clusters/llm_profiles.json"
            )
            logger.info("\n3. Test provider connections:")
            logger.info("   uv run python scripts/test_providers.py")
            logger.info("\n4. Evaluate models one-by-one:")
            logger.info(
                "   uv run python scripts/add_new_model/evaluate_model.py --provider <provider> --model <model>"
            )

    except Exception as e:
        logger.error(f"\n❌ Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
