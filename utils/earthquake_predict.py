import pandas as pd
import joblib
from datetime import datetime
from utils.geocode import get_coordinates

df = pd.read_csv("data_cleaned/clean_earthquakes.csv").dropna(subset=["latitude", "longitude", "depth", "mag"])
df["lat_bin"] = df["latitude"].round(1)
df["lon_bin"] = df["longitude"].round(1)

models = {
    7: joblib.load("ML_predictions/earthquake_models/earthquake_7d.joblib"),
    15: joblib.load("ML_predictions/earthquake_models/earthquake_15d.joblib"),
    30: joblib.load("ML_predictions/earthquake_models/earthquake_30d.joblib"),
    60: joblib.load("ML_predictions/earthquake_models/earthquake_60d.joblib"),
}
kmeans = joblib.load("ML_predictions/earthquake_models/seismic_kmeans.joblib")

def predict_earthquake(location):
    lat_bin, lon_bin = get_coordinates(location)

    nearby = df[
        (df["lat_bin"].between(lat_bin - 0.5, lat_bin + 0.5)) &
        (df["lon_bin"].between(lon_bin - 0.5, lon_bin + 0.5))
    ]

    if nearby.empty:
        depth = df["depth"].mean()
        mag = df["mag"].mean()
        regional_density = df.shape[0]
    else:
        depth = nearby["depth"].mean()
        mag = nearby["mag"].mean()
        regional_density = nearby.shape[0]

    seismic_zone = kmeans.predict(pd.DataFrame([{
        "latitude": lat_bin,
        "longitude": lon_bin,
        "depth": depth,
        "mag": mag
    }]))[0]

    now = datetime.utcnow()
    features = pd.DataFrame([{
        "lat_bin": lat_bin,
        "lon_bin": lon_bin,
        "depth": depth,
        "mag": mag,
        "regional_quake_density": regional_density,
        "seismic_zone": seismic_zone,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "dayofweek": now.weekday()
    }])

    result = {}
    for days, model in models.items():
        result[f"{days}_day_prob"] = round(model.predict_proba(features)[0][1] * 100, 2)

    return {
        "location": location,
        "lat_bin": lat_bin,
        "lon_bin": lon_bin,
        **result
    }
