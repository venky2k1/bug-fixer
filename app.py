from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)

# Load CodeBERT Model for Bug Detection
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
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        bug_status = "buggy" if prediction == 1 else "clean"
        return jsonify({"bug_status": bug_status})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/fix", methods=["POST"])
def fix_code():
    try:
        data = request.get_json()
        code = data.get("code", "")

        if not code:
            return jsonify({"error": "No code provided"}), 400

        # Simple rule-based fix (replace common typos)
        fixed_code = code.replace("prnt", "print").replace("imprt", "import")

        return jsonify({"suggested_fix": fixed_code})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
