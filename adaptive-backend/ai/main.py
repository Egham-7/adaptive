from nvidiaModel.neom import model
from transformers import AutoTokenizer

prompt = ["Prompt: Write a Python script that uses a for loop."]

tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/prompt-task-and-complexity-classifier"
)
encoded_texts = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=True,
    max_length=512,
    padding="max_length",
    truncation=True,
)

result = model(encoded_texts)
print(result)


TASK_TYPE_MODELS = {
    "Open QA": "GPT-4",
    "Closed QA": "RoBERTa",
    "Summarization": "BART",
    "Text Generation": "GPT-3.5",
    "Code Generation": "CodeLlama",
    "Chatbot": "LLaMA 2",
    "Classification": "DistilBERT",
    "Rewrite": "BART",
    "Brainstorming": "Claude",
    "Extraction": "BERT",
    "Other": "GPT-4"
}

print(TASK_TYPE_MODELS[result["task_type_1"][0]])

