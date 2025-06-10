
# 1. Clean Earthquake Data

import pandas as pd
import os

# File paths
raw_path = 'data_raw'
clean_path = 'data_cleaned'

earthquake_files = [f for f in os.listdir(raw_path) if f.startswith('Earthquakes_') and f.endswith('.csv')]
eq_frames = [pd.read_csv(os.path.join(raw_path, f)) for f in earthquake_files]
earthquakes = pd.concat(eq_frames, ignore_index=True)

earthquakes['time'] = pd.to_datetime(earthquakes['time'], errors='coerce')
earthquakes = earthquakes[earthquakes['time'].dt.year >= 2013]
earthquakes = earthquakes[['time', 'latitude', 'longitude', 'depth', 'mag']].dropna()
earthquakes.to_csv(os.path.join(clean_path, 'clean_earthquakes.csv'), index=False)
print(f"\u2714 Earthquake data cleaned: {len(earthquakes)} rows saved.")


# 2. Clean Fire Data

from pathlib import Path
import os
import pandas as pd

# 1. Define input/output directories
data_raw     = Path('data_raw')
data_cleaned = Path('data_cleaned')

# 2. Load datasets with explicit dtypes
fires = pd.read_csv(
    data_raw / 'Fire_2002_to_2023.csv',
    dtype={'year': int, 'month': int, 'region': str, 'country': str, 
           'forest': float, 'savannah': float, 'grasslands': float}
)
cities = pd.read_csv(
    data_raw / 'worldcities.csv',
    dtype={'city_ascii': str, 'lat': float, 'lng': float}
)

# 3. Drop rows missing critical values
fires = fires.dropna(subset=['region', 'year', 'month'])

# 4. Filter to 2013–2023
fires = fires[(fires['year'] >= 2013) & (fires['year'] <= 2023)].copy()

# 5. Normalize keys for a case-insensitive merge
fires['region_lower']       = fires['region'].str.lower().str.strip()
cities['city_ascii_lower']  = cities['city_ascii'].str.lower().str.strip()

# 6. Merge to pull in latitude/longitude
merged = fires.merge(
    cities[['city_ascii_lower', 'lat', 'lng']],
    left_on='region_lower',
    right_on='city_ascii_lower',
    how='left'
)

# 7. Warn if any records failed to match
missing_geo = merged['lat'].isna().sum()
if missing_geo:
    print(f"Warning: {missing_geo} fire records have no matching coordinates.")

# 8. Drop unmatched records, rename, and include land-cover cols
fires_cleaned = (
    merged
    .dropna(subset=['lat', 'lng'])
    .rename(columns={
        'country_x': 'country',
        'lat':       'Latitude',
        'lng':       'Longitude'
    })
    # include your land-cover columns here alongside the basics
    [['year', 'month', 'country', 'region',
      'forest', 'savannas', 'shrublands_grasslands', 'croplands','other',
      'Latitude', 'Longitude']]
    .copy()
)

# 9. Ensure output directory exists, then save
data_cleaned.mkdir(parents=True, exist_ok=True)
fires_cleaned.to_csv(data_cleaned / 'clean_wildfires_with_coords.csv', index=False)

print(f"✔ Fire data cleaned and saved: {len(fires_cleaned)} rows.")

