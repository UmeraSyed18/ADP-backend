
# # 1. Clean Earthquake Data

# import pandas as pd
# import os

# # File paths
# raw_path = 'data_raw'
# clean_path = 'data_cleaned'

# earthquake_files = [f for f in os.listdir(raw_path) if f.startswith('Earthquakes_') and f.endswith('.csv')]
# eq_frames = [pd.read_csv(os.path.join(raw_path, f)) for f in earthquake_files]
# earthquakes = pd.concat(eq_frames, ignore_index=True)

# earthquakes['time'] = pd.to_datetime(earthquakes['time'], errors='coerce')
# earthquakes = earthquakes[earthquakes['time'].dt.year >= 2013]
# earthquakes = earthquakes[['time', 'latitude', 'longitude', 'depth', 'mag']].dropna()
# earthquakes.to_csv(os.path.join(clean_path, 'clean_earthquakes.csv'), index=False)
# print(f"\u2714 Earthquake data cleaned: {len(earthquakes)} rows saved.")


# 2. Clean Fire Data

# from pathlib import Path
# import os
# import pandas as pd

# # 1. Define input/output directories
# data_raw     = Path('data_raw')
# data_cleaned = Path('data_cleaned')

# # 2. Load datasets with explicit dtypes
# fires = pd.read_csv(
#     data_raw / 'Fire_2002_to_2023.csv',
#     dtype={'year': int, 'month': int, 'region': str, 'country': str, 
#            'forest': float, 'savannah': float, 'grasslands': float}
# )
# cities = pd.read_csv(
#     data_raw / 'worldcities.csv',
#     dtype={'city_ascii': str, 'lat': float, 'lng': float}
# )

# # 3. Drop rows missing critical values
# fires = fires.dropna(subset=['region', 'year', 'month'])

# # 4. Filter to 2013–2023
# fires = fires[(fires['year'] >= 2013) & (fires['year'] <= 2023)].copy()

# # 5. Normalize keys for a case-insensitive merge
# fires['region_lower']       = fires['region'].str.lower().str.strip()
# cities['city_ascii_lower']  = cities['city_ascii'].str.lower().str.strip()

# # 6. Merge to pull in latitude/longitude
# merged = fires.merge(
#     cities[['city_ascii_lower', 'lat', 'lng']],
#     left_on='region_lower',
#     right_on='city_ascii_lower',
#     how='left'
# )

# # 7. Warn if any records failed to match
# missing_geo = merged['lat'].isna().sum()
# if missing_geo:
#     print(f"Warning: {missing_geo} fire records have no matching coordinates.")

# # 8. Drop unmatched records, rename, and include land-cover cols
# fires_cleaned = (
#     merged
#     .dropna(subset=['lat', 'lng'])
#     .rename(columns={
#         'country_x': 'country',
#         'lat':       'Latitude',
#         'lng':       'Longitude'
#     })
#     # include your land-cover columns here alongside the basics
#     [['year', 'month', 'country', 'region',
#       'forest', 'savannas', 'shrublands_grasslands', 'croplands','other',
#       'Latitude', 'Longitude']]
#     .copy()
# )

# # 9. Ensure output directory exists, then save
# data_cleaned.mkdir(parents=True, exist_ok=True)
# fires_cleaned.to_csv(data_cleaned / 'clean_wildfires_with_coords.csv', index=False)

# print(f"✔ Fire data cleaned and saved: {len(fires_cleaned)} rows.")



# 3. Clean Flood Data

# import pandas as pd
# import os
# from rapidfuzz import process

# # Define paths
# data_raw = 'data_raw'
# data_cleaned = 'data_cleaned'

# # Load datasets
# floods = pd.read_csv(os.path.join(data_raw, 'Floods_2013_to_2023.csv'), encoding='ISO-8859-1')
# cities = pd.read_csv(os.path.join(data_raw, 'worldcities.csv'))

# # Drop rows with missing critical values
# floods = floods.dropna(subset=['Location', 'Country'])

# # Filter years between 2013 and 2023
# floods = floods[(floods['Start Year'] >= 2013) & (floods['Start Year'] <= 2023)]

# # Preprocess: lowercase for safe matching
# floods['Location_lower'] = floods['Location'].str.lower().str.strip()
# cities['city_ascii_lower'] = cities['city_ascii'].str.lower().str.strip()

# # Initialize lists to store lat/lng
# flood_lats = []
# flood_lngs = []

# # Build a list of cities and their coordinates
# city_coords = cities[['city_ascii_lower', 'lat', 'lng']].dropna(subset=['lat', 'lng'])

# # Iterate through flood locations
# for loc in floods['Location_lower']:
#     # Use rapidfuzz's process.extractOne to find the best match from city names
#     best_match = process.extractOne(loc, city_coords['city_ascii_lower'].values)
    
#     if best_match and best_match[1] > 80:  # Match threshold 
#         # If a match is found, get the latitude and longitude of the best match using loc
#         matched_city = city_coords.loc[city_coords['city_ascii_lower'] == best_match[0]]
#         flood_lats.append(matched_city['lat'].values[0])
#         flood_lngs.append(matched_city['lng'].values[0])
#     else:
#         # If no match or low score, append NaN
#         flood_lats.append(None)
#         flood_lngs.append(None)

# # Add Latitude and Longitude to floods dataframe
# floods['Latitude'] = flood_lats
# floods['Longitude'] = flood_lngs

# # Drop rows with missing coordinates 
# floods_cleaned = floods.dropna(subset=['Latitude', 'Longitude'])

# # Select relevant columns for training the model
# floods_cleaned = floods_cleaned[['Country', 'Region', 'Location', 'Disaster Type', 
#                                  'Latitude', 'Longitude',
#                                  'Start Year', 'Start Month', 'Start Day', 
#                                  'End Year', 'End Month', 'End Day',
#                                  'Total Deaths', 'No. Injured', 'No. Affected', 
#                                  'No. Homeless', 'Total Affected']]

# # Save cleaned flood data
# floods_cleaned.to_csv(os.path.join(data_cleaned, 'clean_floods_with_coords.csv'), index=False)
# print(f"✔ Flood data cleaned and saved: {len(floods_cleaned)} rows.")

