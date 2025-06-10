import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Get lat/lon from location
def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="quake_predictor")
    location = geolocator.geocode(location_name)
    if not location:
        raise ValueError(" Location not found.")
    return round(location.latitude, 1), round(location.longitude, 1)

# Load models and data
models = {
    7: joblib.load("ML_predictions/earthquake_models/earthquake_7d.joblib"),
    15: joblib.load("ML_predictions/earthquake_models/earthquake_15d.joblib"),
    30: joblib.load("ML_predictions/earthquake_models/earthquake_30d.joblib"),
    60: joblib.load("ML_predictions/earthquake_models/earthquake_60d.joblib"),
}
kmeans = joblib.load("ML_predictions/earthquake_models/seismic_kmeans.joblib")

df = pd.read_csv("data_cleaned/clean_earthquakes.csv")
df = df.dropna(subset=['latitude', 'longitude', 'depth', 'mag'])
df['lat_bin'] = df['latitude'].round(1)
df['lon_bin'] = df['longitude'].round(1)

# Build input features
def build_features(lat_bin, lon_bin):
    # Search for nearby regional data within ±0.5°
    nearby = df[
        (df['lat_bin'].between(lat_bin - 0.5, lat_bin + 0.5)) &
        (df['lon_bin'].between(lon_bin - 0.5, lon_bin + 0.5))
    ]

    if nearby.empty:
        print("⚠ No local data found. Using global averages for prediction.")
        depth = df['depth'].mean()
        mag = df['mag'].mean()
        regional_density = df.shape[0]
    else:
        depth = nearby['depth'].mean()
        mag = nearby['mag'].mean()
        regional_density = nearby.shape[0]

    seismic_zone = kmeans.predict(pd.DataFrame([{
        'latitude': lat_bin,
        'longitude': lon_bin,
        'depth': depth,
        'mag': mag
    }]))[0]

    now = datetime.utcnow()
    return pd.DataFrame([{
        'lat_bin': lat_bin,
        'lon_bin': lon_bin,
        'depth': depth,
        'mag': mag,
        'regional_quake_density': regional_density,
        'seismic_zone': seismic_zone,
        'month': now.month,
        'day': now.day,
        'hour': now.hour,
        'dayofweek': now.weekday()
    }])

# Run prediction
try:
    location_name = input("Enter location (city or place name): ")
    lat_bin, lon_bin = get_coordinates(location_name)
    features = build_features(lat_bin, lon_bin)

    print(f"\n Predictions for: {location_name} (lat_bin: {lat_bin}, lon_bin: {lon_bin})")
    for days, model in models.items():
        prob = model.predict_proba(features)[0][1]
        print(f"➡ Earthquake in next {days} days: {round(prob * 100, 2)}% chance")

except Exception as e:
    print(f"Error: {str(e)}")


