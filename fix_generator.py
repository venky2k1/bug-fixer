from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load pre-trained CodeT5 model
fix_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

def generate_fix(buggy_code):
    inputs = tokenizer(buggy_code, return_tensors="pt", truncation=True, padding=True)
    outputs = fix_model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test example
if __name__ == "__main__":
    buggy_code = "def add(a, b): return a * b"
    print("Suggested Fix:\n", generate_fix(buggy_code))
