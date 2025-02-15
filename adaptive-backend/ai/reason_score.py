from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Small distilled model
model_name = "microsoft/deberta-v3-xsmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).cpu().eval()

def reasoning_score(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1)[0][1].item()  # Probability of logical reasoning