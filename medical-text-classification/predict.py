from flask import Flask, request, jsonify
import pickle

# ============================
# Load Model and Preprocessors
# ============================

# Load TF-IDF vectorizer
with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load trained model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ============================
# Initialize API
# ============================

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Ensure input contains "text"
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]

    # Transform text into vector
    X = tfidf.transform([text])

    # Predict
    pred_numeric = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_numeric])[0]

    return jsonify({"prediction": pred_label})

# ============================
# Run Server
# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
