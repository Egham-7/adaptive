import textstat
from transformers import pipeline

# Tiny model for CPU efficiency
clarity_pipe = pipeline(
    "text-classification",
    model="finiteautomata/bertweet-base-sentiment-analysis",
    device=-1,  # Force CPU
    top_k=None
)

def pragmatic_clarity(query):
    try:
        # Readability score
        fk_score = textstat.flesch_kincaid_grade(query) / 20
        
        # Lightweight clarity prediction
        result = clarity_pipe(query)[0]
        clarity_prob = next(r['score'] for r in result if r['label'] == 'POS')
        
        return 0.7 * clarity_prob + 0.3 * (1 - fk_score)
    except:
        return 0.5  # Fallback value