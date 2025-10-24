from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    record = request.get_json()

    prob = model.predict_proba([record])[0, 1]

    return jsonify({"subscription_probability": float(prob)})


@app.route("/")
def home():
    return "Model is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
