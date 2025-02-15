from sentence_transformers import SentenceTransformer
import numpy as np

# Universal model with CPU optimizations
domain_model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")

# Precompute domain centroids (example values)
DOMAIN_CENTROIDS = {
    "coding": np.random.randn(384),  # Replace with your actual embeddings
    "medicine": np.random.randn(384),
    "science": np.random.randn(384)
}

def domain_score(query, domain):
    query_embed = domain_model.encode(query, convert_to_numpy=True)
    return np.dot(query_embed, DOMAIN_CENTROIDS[domain]) / (
        np.linalg.norm(query_embed) * np.linalg.norm(DOMAIN_CENTROIDS[domain]))