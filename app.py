from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.earthquake_predict import predict_earthquake
from utils.wildfire_predict import predict_wildfire

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
            return jsonify({"error": "Request must be JSON with 'country' and 'disasterType'."}), 400

        country = data.get("country")
        disaster_type = data.get("disasterType")

        if not country or not disaster_type:
            return jsonify({"error": "Missing 'country' or 'disasterType' in request body."}), 400

        disaster_type = disaster_type.lower()
        if disaster_type == "earthquake":
            result = predict_earthquake(country)
        elif disaster_type == "wildfire":
            result = predict_wildfire(country)
        else:
            return jsonify({"error": "Invalid disasterType. Must be 'earthquake' or 'wildfire'."}), 400

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
