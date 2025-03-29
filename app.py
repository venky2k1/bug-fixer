from flask import Flask, request, jsonify
from flask_cors import CORS  # Fix CORS issue
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load CodeBERT model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base")

@app.route("/")
def home():
    return "Bug Detection and Fixing API is running!"

@app.route("/detect", methods=["POST"])
def detect_bug():
    try:
        data = request.get_json()
        code = data.get("code", "")

        if not code:
            return jsonify({"error": "No code provided"}), 400

        # Tokenize and classify
        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        prediction = torch.argmax(outputs.logits, dim=1).item()

        bug_status = "buggy" if prediction == 1 else "clean"

        return jsonify({"status": bug_status})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle errors properly

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Ensure compatibility with Docker
