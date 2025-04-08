from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# Load the trained model
model = joblib.load("best_model.pkl")

@app.route("/score", methods=["POST"])
def classify_text():
    """API endpoint to classify text."""
    data = request.json
    text = data.get("text", "")
    threshold = float(data.get("threshold", 0.5))

    prediction, propensity = score(text, model, threshold)

    return jsonify({"prediction": prediction, "propensity": propensity})

# Add host='0.0.0.0'
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  
