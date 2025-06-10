# check_memory.py
import pandas as pd
import os

# Adjust paths based on where you save this script relative to your data files
# For example, if check_memory.py is in your project root:
earthquake_data_path = "data_cleaned/clean_earthquakes.csv"
wildfire_data_path = "data_cleaned/clean_wildfires_with_coords.csv"

try:
    df_earthquake = pd.read_csv(earthquake_data_path)
    print("--- Earthquake DataFrame Info ---")
    df_earthquake.info(memory_usage='deep=True')
except FileNotFoundError:
    print(f"Error: {earthquake_data_path} not found. Check your path.")

try:
    df_wildfire = pd.read_csv(wildfire_data_path)
    print("\n--- Wildfire DataFrame Info ---")
    df_wildfire.info(memory_usage='deep=True')
except FileNotFoundError:
    print(f"Error: {wildfire_data_path} not found. Check your path.")