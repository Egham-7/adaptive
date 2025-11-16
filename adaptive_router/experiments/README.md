# Clustering Experiments

This directory contains Jupyter notebooks for analyzing and improving the clustering quality of the adaptive router.

## Goal

Improve silhouette score from **0.0353** to **0.4+** by:
1. Understanding the data distribution
2. Testing better embedding models
3. Finding optimal clustering parameters

## Notebooks

### 01_embedding_comparison.ipynb
**Status:** ✅ Complete
**Goal:** Compare different embedding models to find the best one for clustering

**Models tested:**
- all-MiniLM-L6-v2 (384D) - Current baseline
- all-mpnet-base-v2 (768D) - Better general embeddings
- microsoft/codebert-base (768D) - Code-specific embeddings
- BAAI/bge-large-en-v1.5 (1024D) - State-of-the-art

**Results:**
- **Winner:** CodeBERT at K=5 → 0.2517 silhouette (7x better than baseline!)
- CodeBERT significantly outperforms other models due to natural code vs academic split
- Other models peak at ~0.10-0.11 silhouette

**Hypothesis confirmed:** Code-specific embeddings work much better for mixed code/academic data.

---

### 02_codebert_optimization.ipynb
**Status:** 🔬 Active Experimentation
**Goal:** Optimize CodeBERT clustering to reach 0.4+ silhouette score

**Current best:** CodeBERT K=5 → 0.2517 (still 0.15 away from target)

**Experiments:**
1. **Lower K values** (K=2,3,4) - test if natural data split improves score
2. **Different algorithms** - HDBSCAN, Gaussian Mixture, Spectral Clustering, Agglomerative
3. **Hierarchical clustering** - 2-level approach:
   - Level 1: Split code vs academic (K=2)
   - Level 2: Sub-cluster each group separately
4. **Multiple metrics** - Silhouette, Calinski-Harabasz, Davies-Bouldin

**Strategy:** CodeBERT captures code vs academic split perfectly. Hierarchical approach should achieve much higher silhouette.

---

### 03_hierarchical_coding_router.ipynb
**Status:** 🚀 Ready to Run
**Goal:** Implement 4-level hierarchical clustering for real coding tasks (UniRoute-inspired)

**Inspired by:** [Universal Model Routing (arXiv:2502.08773)](https://arxiv.org/abs/2502.08773)

**Datasets:**
- **SWE-bench** (~2k real GitHub issues)
- **DS-1000** (1k data science tasks)
- **BigCodeBench** (~500 API tasks)
- **DebugBench** (~500 debugging tasks)

**Hierarchical Structure:**
1. **Level 1: Language** (Python, JavaScript, Java, etc.)
2. **Level 2: Domain** (Web, DataScience, Systems, Algorithms, API)
3. **Level 3: Task Type** (BugFix, Feature, Generation, Refactor, Test)
4. **Level 4: Complexity** (Simple, Medium, Complex)

**Key Features:**
- ✅ Real coding datasets (not academic physics/chemistry)
- ✅ Cluster-based routing (like UniRoute paper)
- ✅ Dynamic model profiling (add new models without re-clustering)
- ✅ Cost analysis and routing simulation
- ✅ Interpretable cluster paths (e.g., "Python-Web-BugFix-Complex")

**Expected Outcome:** Validate hierarchical clustering improves interpretability and enables efficient multi-dimensional routing for coding assistants.

## Setup

```bash
# Install Jupyter dependencies
uv add jupyter notebook matplotlib seaborn plotly scikit-learn umap-learn

# Start Jupyter
jupyter notebook experiments/
```

## Data

- **Training data:** ~3,500 questions from MMLU + Code MMLU
- **MMLU:** 16 academic subjects (CS, math, physics, chemistry, logic, etc.)
- **Code MMLU:** 9 coding subsets (API frameworks, code completion, DBMS/SQL, execution prediction, etc.)
- **Categories:** 25 different subjects total
