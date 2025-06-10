import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from utils.geocode import get_coordinates

df = pd.read_csv("data_cleaned/clean_wildfires_with_coords.csv")
df["lat_bin"] = df["Latitude"].round(2)
df["lon_bin"] = df["Longitude"].round(2)
df["country_code"] = df["country"].astype("category").cat.codes
df["region_code"] = df["region"].astype("category").cat.codes

unique_bins = (
    df.drop_duplicates(["lat_bin", "lon_bin"])
      .set_index(["lat_bin", "lon_bin"])[["country_code", "region_code"]]
)

model30 = joblib.load("ML_predictions/wildfire_models/wildfire_model_30d.joblib")
model60 = joblib.load("ML_predictions/wildfire_models/wildfire_model_60d.joblib")

def predict_wildfire(location):
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
