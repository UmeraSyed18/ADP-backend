from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.earthquake_predict import predict_earthquake
from utils.wildfire_predict import predict_wildfire
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "🌍 AI Disaster Prediction API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must be JSON with 'city' and 'type'."}), 400

        city = data.get("city")
        prediction_type = data.get("type")

        if not city or not prediction_type:
            return jsonify({"error": "Missing 'city' or 'type' in request body."}), 400

        prediction_type = prediction_type.lower()
        if prediction_type == "earthquake":
            result = predict_earthquake(city)
        elif prediction_type == "wildfire":
            result = predict_wildfire(city)
        else:
            return jsonify({"error": "Invalid type. Must be 'earthquake' or 'wildfire'."}), 400

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
