# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime
# from utils.geocode import get_coordinates

# df = pd.read_csv("data_cleaned/clean_wildfires_with_coords.csv")
# df["lat_bin"] = df["Latitude"].round(2)
# df["lon_bin"] = df["Longitude"].round(2)
# df["country_code"] = df["country"].astype("category").cat.codes
# df["region_code"] = df["region"].astype("category").cat.codes

# unique_bins = (
#     df.drop_duplicates(["lat_bin", "lon_bin"])
#       .set_index(["lat_bin", "lon_bin"])[["country_code", "region_code"]]
# )

# model30 = joblib.load("ML_predictions/wildfire_models/wildfire_model_30d.joblib")
# model60 = joblib.load("ML_predictions/wildfire_models/wildfire_model_60d.joblib")

# def predict_wildfire(location):
#     lat, lon = get_coordinates(location)
#     lat_b = round(lat, 2)
#     lon_b = round(lon, 2)

#     if (lat_b, lon_b) not in unique_bins.index:
#         diffs = np.sqrt(
#             (unique_bins.index.get_level_values(0) - lat_b) ** 2 +
#             (unique_bins.index.get_level_values(1) - lon_b) ** 2
#         )
#         nearest = diffs.argmin()
#         lat_b = unique_bins.index.get_level_values(0)[nearest]
#         lon_b = unique_bins.index.get_level_values(1)[nearest]

#     cc, rc = unique_bins.loc[(lat_b, lon_b), ["country_code", "region_code"]]

#     now = datetime.utcnow()
#     feat = pd.DataFrame([{
#         "lat_bin": lat_b,
#         "lon_bin": lon_b,
#         "country_code": cc,
#         "region_code": rc,
#         "month": now.month,
#     }])

#     return {
#         "location": location,
#         "lat_bin": lat_b,
#         "lon_bin": lon_b,
#         "30_day_prob": round(model30.predict_proba(feat)[0][1] * 100, 2),
#         "60_day_prob": round(model60.predict_proba(feat)[0][1] * 100, 2),
#     }


import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from utils.geocode import get_coordinates
import os # Import os module

# Initialize variables to store loaded data and models
_wildfire_df = None
_unique_bins = None
_wildfire_model30 = None
_wildfire_model60 = None

def _load_wildfire_resources():
    """Loads wildfire data and models, ensuring they are loaded only once."""
    global _wildfire_df, _unique_bins, _wildfire_model30, _wildfire_model60

    if _wildfire_df is None:
        script_dir = os.path.dirname(__file__)
        data_path = os.path.join(script_dir, "..", "data_cleaned", "clean_wildfires_with_coords.csv")
        _wildfire_df = pd.read_csv(data_path)
        _wildfire_df["lat_bin"] = _wildfire_df["Latitude"].round(2)
        _wildfire_df["lon_bin"] = _wildfire_df["Longitude"].round(2)
        _wildfire_df["country_code"] = _wildfire_df["country"].astype("category").cat.codes
        _wildfire_df["region_code"] = _wildfire_df["region"].astype("category").cat.codes
        
        _unique_bins = (
            _wildfire_df.drop_duplicates(["lat_bin", "lon_bin"])
            .set_index(["lat_bin", "lon_bin"])[["country_code", "region_code"]]
        )

    if _wildfire_model30 is None:
        model_base_path = os.path.join(script_dir, "..", "ML_predictions", "wildfire_models")
        _wildfire_model30 = joblib.load(os.path.join(model_base_path, "wildfire_model_30d.joblib"))
        _wildfire_model60 = joblib.load(os.path.join(model_base_path, "wildfire_model_60d.joblib"))


def predict_wildfire(location):
    # Ensure resources are loaded before use
    _load_wildfire_resources()

    # Now use the global variables
    df = _wildfire_df
    unique_bins = _unique_bins
    model30 = _wildfire_model30
    model60 = _wildfire_model60

    lat, lon = get_coordinates(location)
    lat_b = round(lat, 2)
    lon_b = round(lon, 2)

    if (lat_b, lon_b) not in unique_bins.index:
        diffs = np.sqrt(
            (unique_bins.index.get_level_values(0) - lat_b) ** 2 +
            (unique_bins.index.get_level_values(1) - lon_b) ** 2
        )
        nearest = diffs.argmin()
        lat_b = unique_bins.index.get_level_values(0)[nearest]
        lon_b = unique_bins.index.get_level_values(1)[nearest]

    cc, rc = unique_bins.loc[(lat_b, lon_b), ["country_code", "region_code"]]

    now = datetime.utcnow()
    feat = pd.DataFrame([{
        "lat_bin": lat_b,
        "lon_bin": lon_b,
        "country_code": cc,
        "region_code": rc,
        "month": now.month,
    }])

    return {
        "location": location,
        "lat_bin": lat_b,
        "lon_bin": lon_b,
        "30_day_prob": round(model30.predict_proba(feat)[0][1] * 100, 2),
        "60_day_prob": round(model60.predict_proba(feat)[0][1] * 100, 2),
    }