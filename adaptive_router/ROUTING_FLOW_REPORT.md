# Adaptive Router - Complete Routing Flow Documentation

**Version:** 1.0
**Date:** 2025-10-25
**Author:** System Analysis

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Complete Request Flow (A-Z)](#complete-request-flow-a-z)
4. [Detailed Component Analysis](#detailed-component-analysis)
5. [Data Structures](#data-structures)
6. [Algorithms](#algorithms)
7. [Performance Characteristics](#performance-characteristics)
8. [Error Handling](#error-handling)

---

## Overview

The Adaptive Router is an intelligent LLM model selection system that uses **cluster-based routing** with per-cluster error rates to select the optimal model for each request. The system implements the **UniRouter algorithm**, which assigns prompts to clusters based on semantic and lexical features, then selects models based on historical error rates and cost preferences.

**Key Features:**
- ✅ Cluster-based intelligent routing (K-means clustering)
- ✅ Hybrid feature extraction (semantic embeddings + TF-IDF)
- ✅ Cost-accuracy trade-off optimization
- ✅ Per-cluster model error rates
- ✅ Spherical K-means (cosine similarity)
- ✅ MinIO S3 storage integration

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Client Application                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP POST
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Application (app.py)                        │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  POST /select_model                                             │   │
│  │  - Validates ModelSelectionRequest                              │   │
│  │  - Logs request metadata                                        │   │
│  │  - Calls ModelRouter.select_model()                             │   │
│  │  - Returns ModelSelectionResponse                               │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   ModelRouter (model_router.py)                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  select_model(request: ModelSelectionRequest)                   │   │
│  │  1. Validate requested models (if provided)                     │   │
│  │  2. Map cost_bias → cost_preference                             │   │
│  │  3. Call _Router.route(prompt, cost_pref, allowed_models)       │   │
│  │  4. Parse model ID → provider + model_name                      │   │
│  │  5. Convert alternatives                                        │   │
│  │  6. Return ModelSelectionResponse                               │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     _Router (internal routing engine)                    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  route(question_text, cost_preference, allowed_models)          │   │
│  │  ────────────────────────────────────────────────────────────   │   │
│  │  STEP 1: Assign to cluster                                      │   │
│  │    → cluster_engine.assign_question(text)                       │   │
│  │    → Returns: (cluster_id, distance)                            │   │
│  │                                                                  │   │
│  │  STEP 2: Calculate lambda parameter                             │   │
│  │    → _calculate_lambda(cost_preference)                         │   │
│  │    → lambda = max - cost_pref * (max - min)                     │   │
│  │    → Inverted: high quality pref = low cost penalty             │   │
│  │                                                                  │   │
│  │  STEP 3: Filter models (if allowed_models specified)            │   │
│  │    → Create models_to_score dict                                │   │
│  │                                                                  │   │
│  │  STEP 4: Compute routing scores                                 │   │
│  │    FOR each model:                                              │   │
│  │      error_rate = model_features[model_id][cluster_id]          │   │
│  │      cost = model_features[model_id]["cost_per_1m_tokens"]      │   │
│  │      normalized_cost = _normalize_cost(cost)                    │   │
│  │      score = error_rate + lambda * normalized_cost              │   │
│  │                                                                  │   │
│  │  STEP 5: Select best model (lowest score)                       │   │
│  │    → best_model_id = argmin(scores)                             │   │
│  │                                                                  │   │
│  │  STEP 6: Generate reasoning & alternatives                      │   │
│  │    → _generate_reasoning()                                      │   │
│  │    → Sort alternatives by score                                 │   │
│  │                                                                  │   │
│  │  RETURN: RoutingDecision                                        │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  ClusterEngine (cluster_engine.py)                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  assign_question(question_text: str)                            │   │
│  │  ────────────────────────────────────────────────────────────   │   │
│  │  STEP 1: Create temporary CodeQuestion                          │   │
│  │    → question_id="temp"                                         │   │
│  │    → question=question_text                                     │   │
│  │    → choices=["A","B","C","D"], answer="A"                      │   │
│  │                                                                  │   │
│  │  STEP 2: Extract features                                       │   │
│  │    → feature_extractor.transform([temp_question])               │   │
│  │                                                                  │   │
│  │  STEP 3: Apply spherical normalization                          │   │
│  │    IF use_spherical = True:                                     │   │
│  │      features = normalize(features, norm="l2")                  │   │
│  │                                                                  │   │
│  │  STEP 4: Predict cluster                                        │   │
│  │    → cluster_id = kmeans.predict(features)[0]                   │   │
│  │                                                                  │   │
│  │  STEP 5: Compute distance to centroid                           │   │
│  │    → distances = kmeans.transform(features)[0]                  │   │
│  │    → distance = distances[cluster_id]                           │   │
│  │                                                                  │   │
│  │  RETURN: (cluster_id, distance)                                 │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                FeatureExtractor (feature_extractor.py)                   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  transform(questions: List[CodeQuestion])                       │   │
│  │  ────────────────────────────────────────────────────────────   │   │
│  │  STEP 1: Extract question texts                                 │   │
│  │    → texts = [q.question for q in questions]                    │   │
│  │                                                                  │   │
│  │  STEP 2: Generate semantic embeddings                           │   │
│  │    → embedding_model.encode(texts)                              │   │
│  │    → Model: all-MiniLM-L6-v2 (384 dimensions)                   │   │
│  │    → normalize_embeddings=True                                  │   │
│  │                                                                  │   │
│  │  STEP 3: Generate TF-IDF features                               │   │
│  │    → tfidf_vectorizer.transform(texts).toarray()                │   │
│  │    → Max features: 192-5000 (configurable)                      │   │
│  │    → N-gram range: (1, 2)                                       │   │
│  │                                                                  │   │
│  │  STEP 4: Normalize features                                     │   │
│  │    → embeddings_norm = embedding_scaler.transform(embeddings)   │   │
│  │    → tfidf_norm = tfidf_scaler.transform(tfidf_features)        │   │
│  │    → StandardScaler (mean=0, variance=1)                        │   │
│  │                                                                  │   │
│  │  STEP 5: Concatenate features                                   │   │
│  │    → hybrid = concat([embeddings_norm, tfidf_norm], axis=1)     │   │
│  │    → Total dimensions: 384 + 192 = 576                          │   │
│  │    → Implicit weighting: ~67% embeddings, ~33% TF-IDF           │   │
│  │                                                                  │   │
│  │  RETURN: hybrid_features (n_questions, 576)                     │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Request Flow (A-Z)

### Phase 1: Request Reception (FastAPI Layer)

**File:** `adaptive_router/app.py`

```python
# Line 90-143
@app.post("/select_model", response_model=ModelSelectionResponse)
async def select_model(request: ModelSelectionRequest, http_request: Request):
    """
    1. Receives HTTP POST request to /select_model
    2. Pydantic validates ModelSelectionRequest
    3. Starts performance timer
    4. Logs request metadata (prompt_length, cost_bias, client_ip)
    """
```

**Input:**
```json
{
  "prompt": "Write a Python function to calculate factorial",
  "cost_bias": 0.5,
  "models": [
    {"provider": "openai", "model_name": "gpt-4"},
    {"provider": "anthropic", "model_name": "claude-3-sonnet"}
  ]
}
```

**Actions:**
1. ✅ Validate request schema (Pydantic)
2. ✅ Get or create ModelRouter singleton instance
3. ✅ Log request metadata
4. ✅ Call `router.select_model(request)`
5. ✅ Handle exceptions (400 for validation, 500 for errors)
6. ✅ Log response metadata (elapsed time, selected model)

---

### Phase 2: Model Selection Orchestration (ModelRouter)

**File:** `adaptive_router/services/model_router.py`

#### 2.1 Initialization (Startup)

**Lines 48-81:**
```python
def __init__(self, config_file: Path | None = None):
    """
    Runs once at application startup.
    Loads all routing configuration from MinIO S3 storage.
    """
```

**Initialization Steps:**

1. **Load MinIO Settings** (lines 100-118)
   ```python
   minio_settings = MinIOSettings()  # From environment variables
   storage_loader = StorageProfileLoader.from_minio_settings(minio_settings)
   ```

   **Required Environment Variables:**
   - `S3_BUCKET_NAME` → Bucket name
   - `MINIO_PUBLIC_ENDPOINT` → MinIO endpoint URL
   - `MINIO_ROOT_USER` → Access key
   - `MINIO_ROOT_PASSWORD` → Secret key

2. **Load Profile from MinIO** (line 121)
   ```python
   profile_data = storage_loader.load_global_profile()
   ```

   **Profile Contains:**
   - `cluster_centers`: K-means centroids (n_clusters × 576 dimensions)
   - `llm_profiles`: Per-cluster error rates for each model
   - `tfidf_vocabulary`: TF-IDF vocabulary and IDF values
   - `scaler_parameters`: StandardScaler mean/scale for embeddings and TF-IDF
   - `metadata`: n_clusters, silhouette_score, embedding_model, etc.

3. **Build ClusterEngine** (lines 134-308)
   ```python
   cluster_engine = self._build_cluster_engine_from_data(
       cluster_centers_data, tfidf_data, scaler_data, metadata
   )
   ```

   **Reconstruction:**
   - Creates FeatureExtractor with pre-trained scalers
   - Restores K-means cluster centers
   - Sets K-means attributes (n_iter_, n_features_in_)
   - Marks ClusterEngine as fitted

4. **Load Model Configurations** (lines 146-172)
   ```python
   models_config = yaml.safe_load(config_file)
   models = [ModelConfig(**m) for m in models_config["gpt5_models"]]
   ```

   **For Each Model:**
   - Parse model config (provider, model_name, cost, capabilities)
   - Match with llm_profiles by ID or name
   - Store error_rates and cost_per_1m_tokens

5. **Initialize _Router** (lines 180-187)
   ```python
   router = _Router(
       cluster_engine=cluster_engine,
       model_features=model_features,  # error_rates + cost
       models=models,
       lambda_min=0.0,
       lambda_max=1.0,
       default_cost_preference=0.5
   )
   ```

#### 2.2 Request Processing

**Lines 310-404 (select_model method):**

**Step 1: Validate Requested Models** (lines 327-350)
```python
if request.models:
    supported = self.get_supported_models()
    # Build list of model IDs: "provider:model_name"
    requested = [f"{m.provider.lower()}:{m.model_name.lower()}"
                 for m in request.models if m.provider and m.model_name]
    # Validate all requested models are supported
    unsupported = [m for m in requested if m not in supported]
    if unsupported:
        raise ValueError(f"Models not supported: {unsupported}")
    allowed_model_ids = requested
```

**Step 2: Map cost_bias → cost_preference** (line 353)
```python
cost_preference = request.cost_bias if request.cost_bias is not None else None
```
- `cost_bias`: User's preference (0.0 = cheapest, 1.0 = most capable)
- Passed directly to Router

**Step 3: Call Internal Router** (lines 356-360)
```python
decision = self._router.route(
    question_text=request.prompt,
    cost_preference=cost_preference,
    allowed_models=allowed_model_ids
)
```
Returns `RoutingDecision` with:
- `selected_model_id`: "provider:model_name"
- `routing_score`: Combined error rate + cost score
- `predicted_accuracy`: 1.0 - error_rate
- `cluster_id`: Assigned cluster
- `alternatives`: Top 3 alternative models
- `reasoning`: Human-readable explanation

**Step 4: Parse Model ID** (lines 362-378)
```python
selected_model_parts = decision.selected_model_id.split(":", 1)
provider, model_name = selected_model_parts
```

**Step 5: Convert Alternatives** (lines 380-388)
```python
alternatives_list = []
for alt in decision.alternatives[:3]:  # Top 3
    alt_parts = alt["model_id"].split(":", 1)
    alternatives_list.append(Alternative(provider=alt_parts[0], model=alt_parts[1]))
```

**Step 6: Create Response** (lines 390-404)
```python
response = ModelSelectionResponse(
    provider=provider,
    model=model_name,
    alternatives=alternatives_list
)
logger.info(f"Selected: {provider}/{model_name} (cluster {cluster_id}, accuracy {accuracy:.2%})")
return response
```

---

### Phase 3: Routing Decision (_Router)

**File:** `adaptive_router/services/model_router.py` (lines 474-584)

**Core Algorithm:**

**Step 1: Assign to Cluster** (line 498)
```python
cluster_id, distance = self.cluster_engine.assign_question(question_text)
```
- Calls ClusterEngine to find nearest cluster centroid
- Returns cluster ID (0 to n_clusters-1) and Euclidean distance

**Step 2: Calculate Lambda Parameter** (lines 501, 586-600)
```python
lambda_param = self._calculate_lambda(cost_preference)

def _calculate_lambda(self, cost_preference: float) -> float:
    """
    Inverted mapping:
    - cost_preference = 0.0 (cheap) → lambda = 1.0 (high cost penalty)
    - cost_preference = 0.5 (balanced) → lambda = 0.5
    - cost_preference = 1.0 (quality) → lambda = 0.0 (no cost penalty)
    """
    return self.lambda_max - cost_preference * (self.lambda_max - self.lambda_min)
```

**Step 3: Filter Models** (lines 503-520)
```python
if allowed_models is not None:
    # Only score allowed models
    models_to_score = {
        model_id: features
        for model_id, features in self.model_features.items()
        if model_id in set(allowed_models)
    }
else:
    # Score all available models
    models_to_score = self.model_features
```

**Step 4: Compute Routing Scores** (lines 522-538)
```python
model_scores = {}
for model_id, features in models_to_score.items():
    error_rate = features["error_rates"][cluster_id]  # Per-cluster error rate
    cost = features["cost_per_1m_tokens"]

    normalized_cost = self._normalize_cost(cost)  # [0, 1] range
    score = error_rate + lambda_param * normalized_cost

    model_scores[model_id] = {
        "score": score,
        "error_rate": error_rate,
        "accuracy": 1.0 - error_rate,
        "cost": cost,
        "normalized_cost": normalized_cost
    }
```

**Scoring Formula:**
```
score = error_rate + λ × normalized_cost

Where:
- error_rate: Historical error rate for this cluster [0, 1]
- λ: Lambda parameter from cost_preference [0, 1]
- normalized_cost: (cost - min_cost) / (max_cost - min_cost) [0, 1]

Lower score = better model
```

**Step 5: Select Best Model** (lines 540-542)
```python
best_model_id = min(model_scores, key=lambda k: model_scores[k]["score"])
best_scores = model_scores[best_model_id]
```

**Step 6: Generate Reasoning** (lines 549-657)
```python
reasoning = self._generate_reasoning(
    cluster_id=cluster_id,
    cost_preference=cost_preference,
    lambda_param=lambda_param,
    selected_scores=best_scores
)

# Example output:
# "Question assigned to cluster 3; Balanced cost-accuracy routing (λ=0.50);
#  Strong predicted accuracy (85%)"
```

**Step 7: Prepare Alternatives** (lines 557-568)
```python
alternatives = [
    {
        "model_id": mid,
        "model_name": self.models[mid].name,
        "score": scores["score"],
        "accuracy": scores["accuracy"],
        "cost": scores["cost"]
    }
    for mid, scores in sorted(model_scores.items(), key=lambda x: x[1]["score"])
    if mid != best_model_id
]
# Sorted by score, excludes selected model
```

**Step 8: Return RoutingDecision** (lines 570-584)
```python
return RoutingDecision(
    selected_model_id=best_model_id,
    selected_model_name=model.name,
    routing_score=best_scores["score"],
    predicted_accuracy=best_scores["accuracy"],
    estimated_cost=best_scores["cost"] * 1000 / 1_000_000,  # Estimated for 1k tokens
    cluster_id=cluster_id,
    cluster_confidence=1.0 / (1.0 + distance),
    lambda_param=lambda_param,
    reasoning=reasoning,
    alternatives=alternatives,
    routing_time_ms=routing_time
)
```

---

### Phase 4: Cluster Assignment (ClusterEngine)

**File:** `adaptive_router/services/cluster_engine.py` (lines 156-185)

**Method:** `assign_question(question_text: str)`

**Step 1: Extract Features** (line 172)
```python
features = self.feature_extractor.transform([question_text])
```
- Passes raw text directly to FeatureExtractor (see Phase 5)
- No CodeQuestion wrapper needed (legacy from MMLU training removed)
- Returns hybrid features (576 dimensions by default)

**Step 2: Apply Spherical Normalization** (lines 174-178)
```python
if self.use_spherical:
    from sklearn.preprocessing import normalize
    features = normalize(features, norm="l2")
```
- **Critical for consistency!**
- K-means was trained on L2-normalized features
- Uses cosine similarity instead of Euclidean distance

**Step 3: Predict Cluster** (line 181)
```python
cluster_id = int(self.kmeans.predict(features)[0])
```
- scikit-learn K-means finds nearest centroid
- Returns cluster ID (0 to n_clusters-1)

**Step 4: Compute Distance** (lines 182-183)
```python
distances = self.kmeans.transform(features)[0]  # Distance to all centroids
distance = float(distances[cluster_id])  # Distance to assigned cluster
```
- `transform()` returns distances to all cluster centers
- Extract distance to assigned cluster for confidence

**Step 5: Return** (line 185)
```python
return cluster_id, distance
```

---

### Phase 5: Feature Extraction (FeatureExtractor)

**File:** `adaptive_router/services/feature_extractor.py` (lines 124-174)

**Method:** `transform(questions: Union[List[CodeQuestion], List[str]])`

**Step 1: Extract Question Texts** (lines 147-150)
```python
# Extract texts based on input type
if isinstance(questions[0], str):
    texts = questions  # Already raw text
else:
    texts = [q.question for q in questions]  # Extract from CodeQuestion
```
- **Flexible Input:** Accepts both raw text strings (production API) and CodeQuestion objects (training scripts)
- **Backwards Compatible:** Training scripts continue to work with structured CodeQuestion objects
- **No Wrapper Needed:** Production API passes text directly without dummy objects

**Step 2: Generate Semantic Embeddings** (lines 155-160)
```python
embeddings = self.embedding_model.encode(
    texts,
    show_progress_bar=False,
    batch_size=32,
    normalize_embeddings=True
)
```
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Output:** 384-dimensional dense vectors
- **Method:** Pre-trained transformer model
- **Captures:** Semantic meaning, context, similarity
- **normalize_embeddings=True:** Unit vectors (L2 norm = 1)

**Step 3: Generate TF-IDF Features** (line 153)
```python
tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
```
- **Vocabulary:** Pre-loaded from MinIO (5000 terms max by default)
- **N-grams:** Unigrams + bigrams (1,2)
- **Stop words:** English stop words removed
- **Output:** Sparse → dense array (192-5000 dimensions)
- **Captures:** Lexical patterns, keywords, terminology

**Step 4: Normalize Features** (lines 155-157)
```python
embeddings_normalized = self.embedding_scaler.transform(embeddings)
tfidf_normalized = self.tfidf_scaler.transform(tfidf_features)
```
- **Scaler Type:** StandardScaler (mean=0, variance=1)
- **Parameters:** Pre-loaded from MinIO (mean_, scale_)
- **Purpose:** Bring features to same scale for concatenation

**Step 5: Concatenate Features** (lines 159-162)
```python
hybrid_features = np.concatenate(
    [embeddings_normalized, tfidf_normalized],
    axis=1
)
```
- **Dimensions:** 384 (embeddings) + 192 (TF-IDF) = 576 total
- **Implicit Weighting:** 384/576 ≈ 67% embeddings, 192/576 ≈ 33% TF-IDF
- **Result:** Hybrid feature vector combining semantic + lexical information

**Step 6: Return** (line 164)
```python
return hybrid_features  # Shape: (n_questions, 576)
```

---

### Phase 6: Response Construction

**Back to FastAPI** (`adaptive_router/app.py` lines 143-157)

**Step 1: Receive Response from ModelRouter**
```python
response = router.select_model(request)
# Type: ModelSelectionResponse
```

**Step 2: Calculate Elapsed Time**
```python
elapsed = time.perf_counter() - start_time
```

**Step 3: Log Response**
```python
logger.info(
    "Model selection completed",
    extra={
        "elapsed_ms": round(elapsed * 1000, 2),
        "selected_provider": response.provider,
        "selected_model": response.model,
        "alternatives_count": len(response.alternatives)
    }
)
```

**Step 4: Return HTTP Response**
```python
return response  # FastAPI serializes to JSON
```

**Output:**
```json
{
  "provider": "openai",
  "model": "gpt-4",
  "alternatives": [
    {
      "provider": "anthropic",
      "model": "claude-3-sonnet"
    }
  ]
}
```

---

## Detailed Component Analysis

### 1. MinIO S3 Storage Integration

**Purpose:** Load pre-trained cluster profiles from Railway-hosted MinIO storage

**Stored Data:**
```
s3://adaptive-router-profiles/global/profile.json
├── cluster_centers/
│   ├── cluster_centers: [[...], [...], ...]  # n_clusters × 576 floats
│   ├── n_clusters: 20
│   └── feature_dim: 576
├── llm_profiles/
│   ├── "openai:gpt-4": [0.05, 0.03, ..., 0.04]  # n_clusters error rates
│   ├── "openai:gpt-5-mini": [0.08, 0.06, ..., 0.07]
│   └── ...
├── tfidf_vocabulary/
│   ├── vocabulary: {"word": index, ...}
│   ├── idf: [idf_value_1, idf_value_2, ...]
│   ├── max_features: 192
│   └── ngram_range: [1, 2]
├── scaler_parameters/
│   ├── embedding_scaler: {mean: [...], scale: [...]}
│   └── tfidf_scaler: {mean: [...], scale: [...]}
└── metadata/
    ├── n_clusters: 20
    ├── embedding_model: "all-MiniLM-L6-v2"
    ├── silhouette_score: 0.45
    └── ...
```

**Loading Process:**
1. Connect to MinIO using boto3
2. Download profile.json (gzipped)
3. Parse JSON
4. Reconstruct ClusterEngine components
5. Validate data integrity

---

### 2. Cluster-Based Routing Algorithm

**Algorithm:** UniRouter (Cluster-based routing with per-cluster error rates)

**Training Phase (Offline):**
```
1. Collect validation dataset (e.g., MMLU questions)
2. Extract hybrid features (embeddings + TF-IDF)
3. Perform K-means clustering (spherical, K=5-50)
4. For each cluster:
   a. Profile all models on cluster questions
   b. Calculate error rate per model
   c. Store: error_rates[model_id][cluster_id]
5. Save to MinIO
```

**Inference Phase (Online):**
```
1. Receive prompt
2. Extract features (same pipeline as training)
3. Assign to nearest cluster (cosine similarity)
4. Look up error_rates[model_id][cluster_id] for all models
5. Combine with cost and user preference
6. Select best model (lowest score)
```

**Why This Works:**
- ✅ Questions in same cluster have similar characteristics
- ✅ Per-cluster error rates are more accurate than global averages
- ✅ Semantic + lexical features capture task type
- ✅ Cost-accuracy trade-off respects user preferences

---

### 3. Spherical K-means

**Why Spherical:**
- Uses cosine similarity instead of Euclidean distance
- Better for high-dimensional sparse data (TF-IDF)
- Normalizes feature magnitudes

**Implementation:**
```python
# Training (fit)
features = normalize(features, norm="l2")  # Unit vectors
kmeans.fit(features)

# Inference (predict/transform)
features = normalize(features, norm="l2")  # MUST normalize!
cluster_id = kmeans.predict(features)
```

**Critical:** Both training and inference MUST apply L2 normalization

---

### 4. Hybrid Feature Extraction

**Embedding Features (384D):**
- Model: all-MiniLM-L6-v2 (sentence-transformers)
- Captures: Semantic meaning, context
- Advantages: Dense, continuous, semantic similarity
- Use case: Understanding intent, similar questions cluster together

**TF-IDF Features (192D):**
- Vocabulary: Top 5000 terms (configurable)
- N-grams: Unigrams + bigrams
- Captures: Keywords, terminology, domain-specific terms
- Advantages: Interpretable, exact matches
- Use case: Code questions, technical terms, specific topics

**Why Hybrid:**
- Embeddings: "Write a function to sort" ≈ "Implement sorting algorithm"
- TF-IDF: Distinguishes "Python" vs "JavaScript", "binary tree" vs "linked list"
- Combined: Best of both worlds

---

## Data Structures

### Request Models

```python
class ModelSelectionRequest(BaseModel):
    prompt: str  # Required: User's question/prompt
    cost_bias: float | None = 0.5  # 0.0=cheap, 1.0=quality
    models: List[ModelCapability] | None = None  # Optional: Restrict to these models

class ModelCapability(BaseModel):
    provider: str  # "openai", "anthropic", etc.
    model_name: str  # "gpt-4", "claude-3-sonnet", etc.
    # Other fields optional for request filtering
```

### Response Models

```python
class ModelSelectionResponse(BaseModel):
    provider: str  # Selected provider
    model: str  # Selected model name
    alternatives: List[Alternative]  # Top 3 alternatives

class Alternative(BaseModel):
    provider: str
    model: str
```

### Internal Models

```python
class RoutingDecision(BaseModel):
    selected_model_id: str  # "openai:gpt-4"
    selected_model_name: str  # "GPT-4"
    routing_score: float  # Combined error + cost score
    predicted_accuracy: float  # 1.0 - error_rate
    estimated_cost: float  # Estimated cost for 1k tokens
    cluster_id: int  # Assigned cluster (0 to n_clusters-1)
    cluster_confidence: float  # 1 / (1 + distance)
    lambda_param: float  # Calculated lambda
    reasoning: str  # Human-readable explanation
    alternatives: List[dict]  # All other models sorted by score
    routing_time_ms: float  # Processing time

class ModelConfig(BaseModel):
    id: str  # "openai:gpt-4"
    name: str  # "GPT-4"
    provider: str  # "openai"
    model_name: str  # "gpt-4"
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float
    cost_per_1m_tokens: float  # Average
    max_context_tokens: int
    supports_function_calling: bool
    supports_vision: bool
```

---

## Algorithms

### 1. Lambda Calculation

**Purpose:** Convert user's cost_bias to lambda parameter for scoring

```python
def _calculate_lambda(cost_preference: float) -> float:
    """
    Maps cost_preference [0, 1] to lambda [lambda_max, lambda_min]

    cost_preference = 0.0 (cheapest) → lambda = 1.0 (high cost penalty)
    cost_preference = 0.5 (balanced) → lambda = 0.5
    cost_preference = 1.0 (quality) → lambda = 0.0 (no cost penalty)
    """
    return lambda_max - cost_preference * (lambda_max - lambda_min)
```

**Default Values:**
- `lambda_min = 0.0`
- `lambda_max = 1.0`

**Effect:**
- High lambda → Cost matters more in scoring
- Low lambda → Accuracy matters more in scoring

---

### 2. Cost Normalization

**Purpose:** Normalize costs to [0, 1] range for fair comparison

```python
def _normalize_cost(cost: float) -> float:
    """
    Normalize cost to [0, 1] using min-max scaling
    """
    cost_range = max_cost - min_cost
    if cost_range < 1e-9:  # Avoid division by zero
        return 0.0
    return (cost - min_cost) / cost_range
```

**Example:**
```
Models:
  gpt-5-nano: $0.50/1M tokens
  gpt-4.1-nano: $1.00/1M tokens
  gpt-5-mini: $2.00/1M tokens
  gpt-5-codex: $4.00/1M tokens

Normalized:
  gpt-5-nano: (0.5 - 0.5) / (4.0 - 0.5) = 0.0
  gpt-4.1-nano: (1.0 - 0.5) / 3.5 = 0.14
  gpt-5-mini: (2.0 - 0.5) / 3.5 = 0.43
  gpt-5-codex: (4.0 - 0.5) / 3.5 = 1.0
```

---

### 3. Routing Score Calculation

**Formula:**
```
score = error_rate + λ × normalized_cost

Where:
- error_rate ∈ [0, 1]: Per-cluster historical error rate
- λ ∈ [0, 1]: Lambda parameter from cost_preference
- normalized_cost ∈ [0, 1]: Min-max normalized cost

Goal: Minimize score
```

**Example Calculation:**

```
Cluster 3 assignment:
  cost_preference = 0.5 → λ = 0.5

Model 1 (gpt-5-mini):
  error_rate[3] = 0.05
  cost = $2.00 → normalized_cost = 0.43
  score = 0.05 + 0.5 × 0.43 = 0.265

Model 2 (gpt-5-nano):
  error_rate[3] = 0.12
  cost = $0.50 → normalized_cost = 0.0
  score = 0.12 + 0.5 × 0.0 = 0.120  ← SELECTED (lowest score)

Model 3 (gpt-5-codex):
  error_rate[3] = 0.02
  cost = $4.00 → normalized_cost = 1.0
  score = 0.02 + 0.5 × 1.0 = 0.520
```

**Interpretation:**
- gpt-5-nano wins: Good balance of low cost + acceptable error rate
- gpt-5-codex has best accuracy but penalized by high cost
- gpt-5-mini middle ground

**With Different cost_preference:**

```
cost_preference = 1.0 (quality priority) → λ = 0.0:
  gpt-5-nano: 0.12 + 0 × 0.0 = 0.120
  gpt-5-codex: 0.02 + 0 × 1.0 = 0.020  ← SELECTED (best accuracy)

cost_preference = 0.0 (cost priority) → λ = 1.0:
  gpt-5-nano: 0.12 + 1.0 × 0.0 = 0.120  ← SELECTED (cheapest)
  gpt-5-codex: 0.02 + 1.0 × 1.0 = 1.020
```

---

### 4. Cluster Confidence Calculation

**Purpose:** Convert distance to confidence score

```python
cluster_confidence = 1.0 / (1.0 + distance)
```

**Properties:**
- distance = 0 → confidence = 1.0 (perfect match)
- distance = 1 → confidence = 0.5
- distance = ∞ → confidence → 0

**Usage:** Indicates how well the prompt fits the assigned cluster

---

## Performance Characteristics

### Latency Breakdown

**Typical Request (single prompt):**

```
Total: 30-100ms

├─ Feature Extraction: 20-50ms
│  ├─ Semantic Embeddings (sentence-transformers): 15-40ms
│  ├─ TF-IDF Vectorization: 2-5ms
│  ├─ Scaling: 1-2ms
│  └─ Concatenation: <1ms
│
├─ Cluster Assignment: 2-5ms
│  ├─ L2 Normalization: <1ms
│  └─ K-means Predict: 2-5ms
│
├─ Model Scoring: 2-5ms
│  ├─ Loop over models: 1-3ms
│  └─ Score calculation: 1-2ms
│
└─ Response Construction: 1-2ms
```

**First Request (Cold Start):**
- +2-5 seconds: Loading sentence-transformers model from HuggingFace cache
- +1-3 seconds: Loading MinIO profile data

**Optimization Opportunities:**
- ✅ Cache feature extraction for repeated prompts
- ✅ Batch multiple requests together
- ✅ Pre-load models at startup (already done)
- ✅ Use GPU for embeddings (if available)

### Throughput

**Estimated Capacity:**
- Single instance: 100-500 requests/second
- Bottleneck: Sentence transformer inference
- Scaling: Horizontal (multiple instances) + vertical (GPU)

### Memory Usage

**Baseline:**
- Sentence transformer model: 400-800 MB
- Cluster centers + profiles: 50-100 MB
- TF-IDF vocabulary: 10-20 MB
- Scalers: <1 MB

**Per Request:**
- Feature vectors: <1 KB
- Temporary objects: <10 KB

**Total:** ~500 MB - 1 GB per instance

---

## Error Handling

### Validation Errors (400)

**Triggers:**
```python
# Unsupported models
raise ValueError(f"Models not supported: {unsupported}")

# Invalid model ID format
raise ValueError(f"Invalid model ID: {model_id}")
```

**Response:**
```json
{
  "detail": "Models not supported by Router: ['unknown:model']. Supported models: [...]"
}
```

### Server Errors (500)

**Triggers:**
```python
# MinIO connection failure
raise ValueError("MinIO configuration error. Required env vars: ...")

# Profile loading failure
raise FileNotFoundError("Profile not found in MinIO")

# Unexpected exceptions
raise HTTPException(status_code=500, detail="Internal server error")
```

**Response:**
```json
{
  "detail": "Internal server error during model selection: <error message>"
}
```

### Graceful Degradation

**If No Models Match:**
```python
if not models_to_score:
    raise ValueError("No valid models found in allowed list")
```

**If Profile Missing for Model:**
```python
if not profile:
    logger.warning(f"Model {model_id} not in profiles, skipping")
    # Continue with other models
```

---

## Summary

### Key Takeaways

1. **Architecture:** Multi-layer (FastAPI → ModelRouter → _Router → ClusterEngine → FeatureExtractor)
2. **Algorithm:** Cluster-based routing with per-cluster error rates (UniRouter)
3. **Features:** Hybrid (semantic embeddings + TF-IDF)
4. **Scoring:** `score = error_rate + λ × normalized_cost`
5. **Storage:** MinIO S3 for pre-trained profiles
6. **Performance:** 30-100ms per request, 100-500 req/s throughput

### Critical Components

1. ✅ **Spherical normalization** (L2 norm) in both training and inference
2. ✅ **Per-cluster error rates** for accurate predictions
3. ✅ **Hybrid features** (384D embeddings + 192D TF-IDF)
4. ✅ **Lambda parameter** for cost-accuracy trade-off
5. ✅ **MinIO integration** for profile storage

### Flow Diagram (Simplified)

```
Request → FastAPI → ModelRouter → _Router
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
             ClusterEngine                      Score All Models
                    ↓                                   ↓
             FeatureExtractor            error_rate + λ × cost
                    ↓                                   ↓
          Embeddings + TF-IDF                  Select Best (min)
                    ↓                                   ↓
             Normalize (L2)                      Return Decision
                    ↓                                   ↓
             K-means Predict  ←───────────────── Response
```

---

**End of Report**
