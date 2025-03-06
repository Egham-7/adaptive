from llms import domain_parameters

def adjust_parameters(domain: str, prompt_scores: dict):
    # Define base values for each domain
 
    
    if domain not in domain_parameters:
        raise ValueError("Invalid domain. Choose from: " + ", ".join(domain_parameters.keys()))
    
    base = domain_parameters[domain]
    
    # Extract prompt scores
    creativity_scope = prompt_scores.get("creativity_scope", [0.5])[0]
    reasoning = prompt_scores.get("reasoning", [0.5])[0]
    contextual_knowledge = prompt_scores.get("contextual_knowledge", [0.5])[0]
    prompt_complexity_score = prompt_scores.get("prompt_complexity_score", [0.5])[0]
    domain_knowledge = prompt_scores.get("domain_knowledge", [0.5])[0]
    
    # Compute adjustments
    temperature = base["Temperature"] + (creativity_scope - 0.5) * 0.5
    top_p = base["TopP"] + (creativity_scope - 0.5) * 0.3
    presence_penalty = base["PresencePenalty"] + (domain_knowledge - 0.5) * 0.4
    frequency_penalty = base["FrequencyPenalty"] + (reasoning - 0.5) * 0.4
    max_tokens = base["MaxCompletionTokens"] + (contextual_knowledge - 0.5) * 500
    n = max(1, round(base["N"] + (prompt_complexity_score - 0.5) * 2))
    
    # Return final adjusted parameters
    return {
        "temperature": round(temperature, 2),
        "top_p": round(top_p, 2),
        "presence_penalty": round(presence_penalty, 2),
        "frequency_penalty": round(frequency_penalty, 2),
        "max_tokens": int(max_tokens),
        "n": n
    }

# Example usage
"""domain = "Science"
prompt_scores = {
    "creativity_scope": [0.2171],
    "reasoning": [0.1662],
    "contextual_knowledge": [0.6938],
    "prompt_complexity_score": [0.3017],
    "domain_knowledge": [0.3628]
}

params = adjust_parameters(domain, prompt_scores)
print(params)
"""