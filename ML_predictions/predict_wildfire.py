import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from datetime import datetime
import sys
import warnings
warnings.filterwarnings("ignore")

# 1) Geocode user input
def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="wildfire_predictor")
    loc = geolocator.geocode(location_name)
    if loc is None:
        raise ValueError(f"Location '{location_name}' not found.")
    return loc.latitude, loc.longitude

# 2) Load your cleaned wildfire data (same file you trained on)
df = pd.read_csv("data_cleaned/clean_wildfires_with_coords.csv")

# Reconstruct the training‐time bins & codes (no noise here)
df['lat_bin']      = df['Latitude'].round(2)
df['lon_bin']      = df['Longitude'].round(2)
df['country_code'] = df['country'].astype("category").cat.codes
df['region_code']  = df['region'].astype("category").cat.codes

# Build a lookup of unique (lat_bin, lon_bin) → static features
unique_bins = (
    df
    .drop_duplicates(['lat_bin','lon_bin'])
    .set_index(['lat_bin','lon_bin'])
    [['country_code','region_code']]
)

# 3) Load trained models
model30 = joblib.load("ML_predictions/wildfire_models/wildfire_model_30d.joblib")
model60 = joblib.load("ML_predictions/wildfire_models/wildfire_model_60d.joblib")

# 4) Main prompt + feature‐builder
try:
    loc_name = input("Enter a location (city or place name): ").strip()
    lat, lon = get_coordinates(loc_name)
    print(f"Resolved to lat={lat:.4f}, lon={lon:.4f}")

    # Bin to match training
    lat_b = round(lat, 2)
    lon_b = round(lon, 2)

    # Find the nearest bin if exact not in our lookup
    if (lat_b, lon_b) not in unique_bins.index:
        # compute distance to all known bins and pick the closest
        diffs = np.sqrt(
            (unique_bins.index.get_level_values(0) - lat_b)**2 +
            (unique_bins.index.get_level_values(1) - lon_b)**2
        )
        nearest = diffs.argmin()
        lat_b = unique_bins.index.get_level_values(0)[nearest]
        lon_b = unique_bins.index.get_level_values(1)[nearest]
        print(f"No exact bin found; using nearest bin at lat_bin={lat_b}, lon_bin={lon_b}")

    cc, rc = unique_bins.loc[(lat_b, lon_b), ['country_code','region_code']]

    # Build the feature row
    now = datetime.utcnow()
    feat = pd.DataFrame([{
        'lat_bin':      lat_b,
        'lon_bin':      lon_b,
        'country_code': cc,
        'region_code':  rc,
        'month':        now.month
    }])

    # 5) Predict and display probabilities
    for days, mdl in [(30, model30), (60, model60)]:
        prob = mdl.predict_proba(feat)[0][1] * 100
        print(f"➡ Wildfire in next {days} days: {prob:.1f}% chance")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
