# import pandas as pd
# import joblib
# from datetime import datetime
# from utils.geocode import get_coordinates

# df = pd.read_csv("data_cleaned/clean_earthquakes.csv").dropna(subset=["latitude", "longitude", "depth", "mag"])
# df["lat_bin"] = df["latitude"].round(1)
# df["lon_bin"] = df["longitude"].round(1)

# models = {
#     7: joblib.load("ML_predictions/earthquake_models/earthquake_7d.joblib"),
#     15: joblib.load("ML_predictions/earthquake_models/earthquake_15d.joblib"),
#     30: joblib.load("ML_predictions/earthquake_models/earthquake_30d.joblib"),
#     60: joblib.load("ML_predictions/earthquake_models/earthquake_60d.joblib"),
# }
# kmeans = joblib.load("ML_predictions/earthquake_models/seismic_kmeans.joblib")

# def predict_earthquake(location):
#     lat_bin, lon_bin = get_coordinates(location)

#     nearby = df[
#         (df["lat_bin"].between(lat_bin - 0.5, lat_bin + 0.5)) &
#         (df["lon_bin"].between(lon_bin - 0.5, lon_bin + 0.5))
#     ]

#     if nearby.empty:
#         depth = df["depth"].mean()
#         mag = df["mag"].mean()
#         regional_density = df.shape[0]
#     else:
#         depth = nearby["depth"].mean()
#         mag = nearby["mag"].mean()
#         regional_density = nearby.shape[0]

#     seismic_zone = kmeans.predict(pd.DataFrame([{
#         "latitude": lat_bin,
#         "longitude": lon_bin,
#         "depth": depth,
#         "mag": mag
#     }]))[0]

#     now = datetime.utcnow()
#     features = pd.DataFrame([{
#         "lat_bin": lat_bin,
#         "lon_bin": lon_bin,
#         "depth": depth,
#         "mag": mag,
#         "regional_quake_density": regional_density,
#         "seismic_zone": seismic_zone,
#         "month": now.month,
#         "day": now.day,
#         "hour": now.hour,
#         "dayofweek": now.weekday()
#     }])

#     result = {}
#     for days, model in models.items():
#         result[f"{days}_day_prob"] = round(model.predict_proba(features)[0][1] * 100, 2)

#     return {
#         "location": location,
#         "lat_bin": lat_bin,
#         "lon_bin": lon_bin,
#         **result
#     }


import pandas as pd
import joblib
from datetime import datetime
from utils.geocode import get_coordinates
import os # Import os module

# Initialize variables to store loaded data and models
# They will be loaded once when first accessed, then reused.
_earthquake_df = None
_earthquake_models = {}
_seismic_kmeans = None

def _load_earthquake_resources():
    """Loads earthquake data and models, ensuring they are loaded only once."""
    global _earthquake_df, _earthquake_models, _seismic_kmeans

    if _earthquake_df is None:
        # Construct absolute path for data file
        script_dir = os.path.dirname(__file__) # Directory of the current script
        data_path = os.path.join(script_dir, "..", "data_cleaned", "clean_earthquakes.csv")
        _earthquake_df = pd.read_csv(data_path).dropna(subset=["latitude", "longitude", "depth", "mag"])
        _earthquake_df["lat_bin"] = _earthquake_df["latitude"].round(1)
        _earthquake_df["lon_bin"] = _earthquake_df["longitude"].round(1)

    if not _earthquake_models: # Check if dictionary is empty
        model_base_path = os.path.join(script_dir, "..", "ML_predictions", "earthquake_models")
        _earthquake_models = {
            7: joblib.load(os.path.join(model_base_path, "earthquake_7d.joblib")),
            15: joblib.load(os.path.join(model_base_path, "earthquake_15d.joblib")),
            30: joblib.load(os.path.join(model_base_path, "earthquake_30d.joblib")),
            60: joblib.load(os.path.join(model_base_path, "earthquake_60d.joblib")),
        }
    
    if _seismic_kmeans is None:
        kmeans_path = os.path.join(model_base_path, "seismic_kmeans.joblib")
        _seismic_kmeans = joblib.load(kmeans_path)


def predict_earthquake(location):
    # Ensure resources are loaded before use
    _load_earthquake_resources()

    # Now use the global variables (they will be loaded after the first call)
    df = _earthquake_df
    models = _earthquake_models
    kmeans = _seismic_kmeans

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