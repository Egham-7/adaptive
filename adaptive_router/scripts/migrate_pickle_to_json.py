#!/usr/bin/env python3
"""Migrate cluster_engine.pkl to JSON files for Git-friendly storage.

This script converts the existing 89MB pickle file to lightweight JSON files:
- cluster_centers.json (~50KB)
- tfidf_vocabulary.json (~30KB)
- metadata.json (updated with config info)

Usage:
    python scripts/migrate_pickle_to_json.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

# Add adaptive_router to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_router.services.unirouter.cluster_engine import ClusterEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Custom unpickler for module path changes
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


def main():
    """Main migration workflow."""
    logger.info("=" * 80)
    logger.info("Migrating cluster_engine.pkl to JSON files")
    logger.info("=" * 80)

    # Paths
    adaptive_router_dir = Path(__file__).parent.parent / "adaptive_router"
    clusters_dir = adaptive_router_dir / "data" / "unirouter" / "clusters"
    pickle_file = clusters_dir / "cluster_engine.pkl"

    # Check if pickle exists
    if not pickle_file.exists():
        logger.error(f"Pickle file not found: {pickle_file}")
        logger.error("Nothing to migrate!")
        sys.exit(1)

    logger.info(f"Loading pickle file: {pickle_file}")
    pickle_size = pickle_file.stat().st_size / 1024 / 1024
    logger.info(f"Pickle size: {pickle_size:.1f} MB")

    # Load the pickle
    with open(pickle_file, "rb") as f:
        engine = ModulePathUnpickler(f).load()

    if not isinstance(engine, ClusterEngine):
        logger.error(f"Invalid cluster engine type: {type(engine)}")
        sys.exit(1)

    logger.info("✅ Successfully loaded ClusterEngine from pickle")

    # 1. Save cluster centers as JSON
    logger.info("\nSaving cluster centers to JSON...")
    cluster_centers_file = clusters_dir / "cluster_centers.json"
    cluster_data = {
        "cluster_centers": engine.kmeans.cluster_centers_.tolist(),
        "n_clusters": engine.n_clusters,
        "feature_dim": engine.kmeans.cluster_centers_.shape[1],
    }
    with open(cluster_centers_file, "w") as f:
        json.dump(cluster_data, f, indent=2)

    cluster_centers_size = cluster_centers_file.stat().st_size / 1024
    logger.info(f"✅ Saved cluster_centers.json ({cluster_centers_size:.1f} KB)")

    # 2. Save TF-IDF vocabulary as JSON
    logger.info("\nSaving TF-IDF vocabulary to JSON...")
    tfidf_vocab_file = clusters_dir / "tfidf_vocabulary.json"

    # Convert vocabulary to native Python types (numpy int64 -> int)
    vocabulary_native = {
        str(k): int(v) for k, v in engine.feature_extractor.tfidf_vectorizer.vocabulary_.items()
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

    # 3. Update metadata with enhanced config info
    logger.info("\nUpdating metadata.json...")
    metadata_file = clusters_dir / "metadata.json"

    cluster_info = engine.get_cluster_info()
    metadata = {
        "n_clusters": cluster_info["n_clusters"],
        "n_train_questions": cluster_info["n_questions"],
        "silhouette_score": cluster_info["silhouette_score"],
        "embedding_model": engine.feature_extractor.embedding_model_name,
        "embedding_dim": engine.feature_extractor.embedding_dim,
        "tfidf_max_features": engine.feature_extractor.tfidf_vectorizer.max_features,
        "tfidf_ngram_range": list(engine.feature_extractor.tfidf_vectorizer.ngram_range),
        "total_features": engine.feature_extractor.embedding_dim
        + engine.feature_extractor.tfidf_vectorizer.max_features,
        "cluster_sizes": cluster_info["cluster_sizes"],
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    metadata_size = metadata_file.stat().st_size / 1024
    logger.info(f"✅ Updated metadata.json ({metadata_size:.1f} KB)")

    # Summary
    total_json_size = cluster_centers_size + tfidf_vocab_size + metadata_size
    size_reduction = (1 - (total_json_size / 1024) / pickle_size) * 100

    logger.info("\n" + "=" * 80)
    logger.info("MIGRATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Pickle size: {pickle_size:.1f} MB (local only, gitignored)")
    logger.info(f"JSON size: {total_json_size:.1f} KB (Git-tracked)")
    logger.info(f"Size reduction for Git: {size_reduction:.1f}%")
    logger.info("\n📝 Next steps:")
    logger.info("1. Test that routing works with JSON files:")
    logger.info("   python -c 'from adaptive_router import ModelRouter; r = ModelRouter()'")
    logger.info("\n2. Add and commit JSON files:")
    logger.info("   git add adaptive_router/data/unirouter/clusters/cluster_centers.json")
    logger.info("   git add adaptive_router/data/unirouter/clusters/tfidf_vocabulary.json")
    logger.info("   git add adaptive_router/data/unirouter/clusters/metadata.json")
    logger.info("   git commit -m 'feat: add lightweight JSON cluster files'")
    logger.info("\n3. Push to remote (should work instantly now!):")
    logger.info("   git push origin feat/unirouter")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
