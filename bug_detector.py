import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load pre-trained CodeBERT model
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def classify_code(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "buggy" if probabilities[0][1] > probabilities[0][0] else "correct"

# Test example
if __name__ == "__main__":
    test_code = "def add(a, b): return a * b"  # Buggy function
    print(classify_code(test_code))
