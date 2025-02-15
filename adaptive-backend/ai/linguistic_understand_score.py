import spacy
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])
nlp.add_pipe('sentencizer')
sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def linguistic_score(query):
    doc = nlp(query)
    
    # Improved scoring components
    length_score = min(1.0, len(query.split()) / 20)  # Reward appropriate length
    has_punctuation = any(char in query for char in '.!?')
    has_capitals = any(c.isupper() for c in query)
    structure_score = 0.3 * has_punctuation + 0.2 * has_capitals
    
    # Penalize very long sentences
    syntax_penalty = 0.5 if any(len(sent.text.split()) > 40 for sent in doc.sents) else 0
    
    # Get semantic richness through embeddings
    with torch.inference_mode():
        embeddings = sentence_model.encode(query, convert_to_tensor=True)
        semantic_score = float(torch.mean(torch.abs(embeddings))) / 2
    
    # Combine scores
    final_score = (length_score + structure_score + semantic_score) * (1 - syntax_penalty)
    return min(1.0, max(0.0, final_score))

# Test cases
test_sentences = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "This is a well-structured sentence with proper punctuation.",
    "this is poorly structured no punctuation or capitals",
    "This extremely long sentence that goes on and on with multiple clauses and unnecessary words just to make it exceed the forty word limit which will trigger our syntax error detection mechanism in the scoring function."
]

print("Enhanced Linguistic Score Test Results:")
print("-" * 50)
for sentence in test_sentences:
    score = linguistic_score(sentence)
    print(f"\nInput: {sentence}")
    print(f"Score: {score:.4f}")
