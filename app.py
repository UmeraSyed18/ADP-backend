# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from utils.earthquake_predict import predict_earthquake
# from utils.wildfire_predict import predict_wildfire

# app = Flask(__name__)
# CORS(app)

# @app.route("/")
# def index():
#     return "🌍 AI Disaster Prediction API is live!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "Request must be JSON with 'country' and 'disasterType'."}), 400

#         country = data.get("country")
#         disaster_type = data.get("disasterType")

#         if not country or not disaster_type:
#             return jsonify({"error": "Missing 'country' or 'disasterType' in request body."}), 400

#         disaster_type = disaster_type.lower()
#         if disaster_type == "earthquake":
#             result = predict_earthquake(country)
#         elif disaster_type == "wildfire":
#             result = predict_wildfire(country)
#         else:
#             return jsonify({"error": "Invalid disasterType. Must be 'earthquake' or 'wildfire'."}), 400

#         return jsonify({"status": "success", "data": result})

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.earthquake_predict import predict_earthquake
from utils.wildfire_predict import predict_wildfire
import os # Import os module

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
        # Log the full exception for debugging in Render logs
        print(f"An error occurred: {e}", flush=True) 
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Use Gunicorn for production deployment
    # Render will set the PORT environment variable
    # This block is usually not needed when using a start command like `gunicorn app:app`
    # but for local testing, it's useful to run with gunicorn if you don't use Flask's dev server.
    # For Render, you just need the Gunicorn command in the "Start Command" setting.
    
    # If you must run with app.run() for some reason (not recommended for prod):
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False) # Turn off debug mode for production