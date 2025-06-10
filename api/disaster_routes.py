import pandas as pd
from flask import Blueprint, request, jsonify
from ML_predictions.predict_earthquake import predict_quakes
from ML_predictions.predict_wildfire import predict_fires

disaster_bp = Blueprint("disasters", __name__)

# helper to load & filter CSV by a “location” text match
def load_and_filter(dtype, location):
    filename = f"data_cleaned/clean_{'earthquakes' if dtype=='earthquake' else 'wildfires_with_coords'}.csv"
    df = pd.read_csv(filename)
    # assume your CSV has a column named 'location' or 'place'
    return df[df["location"].str.contains(location, case=False, na=False)]

@disaster_bp.route("/events", methods=["GET"])
def get_past_events():
    """
    GET /api/disasters/events?type=<earthquake|wildfire>&location=<string>
    → returns array of past events at that location
    """
    dtype    = request.args.get("type", "").lower()
    location = request.args.get("location", "").strip()
    if dtype not in ("earthquake", "wildfire") or not location:
        return jsonify({"error": "type must be 'earthquake' or 'wildfire' and location is required"}), 400

    filtered = load_and_filter(dtype, location)
    return jsonify(filtered.to_dict(orient="records"))

@disaster_bp.route("/predictions", methods=["POST"])
def get_future_predictions():
    """
    POST /api/disasters/predictions
    Body JSON:
      {
        "type": "earthquake" | "wildfire",
        "location": "San Francisco, CA",
        "days": 30        # optional, defaults to 30
      }
    → returns your model’s prediction payload
    """
    payload  = request.get_json(force=True)
    dtype    = payload.get("type", "").lower()
    location = payload.get("location", "").strip()
    days     = int(payload.get("days", 30))

    if dtype not in ("earthquake", "wildfire") or not location:
        return jsonify({"error": "type must be 'earthquake' or 'wildfire' and location is required"}), 400

    if dtype == "earthquake":
        result = predict_quakes(location, days)
    else:
        result = predict_fires(location, days)

    return jsonify(result)
