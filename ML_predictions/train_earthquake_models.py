import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Loading and preprocessing data...")

# Load and clean data
df = pd.read_csv("data_cleaned/clean_earthquakes.csv")
df = df.dropna(subset=['latitude', 'longitude', 'depth', 'mag', 'time'])
df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
df = df.dropna(subset=['time'])

# Add spatial bins
df['lat_bin'] = df['latitude'].round(1)
df['lon_bin'] = df['longitude'].round(1)

# Add temporal features
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour
df['dayofweek'] = df['time'].dt.dayofweek

# Regional density
bin_counts = df.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='regional_quake_density')
df = df.merge(bin_counts, on=['lat_bin', 'lon_bin'], how='left')

# Seismic zone clustering
print("Assigning seismic zones...")
kmeans = KMeans(n_clusters=10, random_state=42)
df['seismic_zone'] = kmeans.fit_predict(df[['latitude', 'longitude', 'depth', 'mag']])
joblib.dump(kmeans, "ML_predictions/earthquake_models/seismic_kmeans.joblib")

# Label function
def label_targets(df, days):
    df = df.sort_values('time')
    df['target'] = 0
    for (lat, lon), group in df.groupby(['lat_bin', 'lon_bin']):
        times = group['time'].reset_index(drop=True)
        for i in range(len(times)):
            future = times[(times > times[i]) & (times <= times[i] + pd.Timedelta(days=days))]
            if not future.empty:
                df.loc[group.index[i], 'target'] = 1
    return df

# Features used
features = [
    'lat_bin', 'lon_bin', 'depth', 'mag', 'regional_quake_density',
    'seismic_zone', 'month', 'day', 'hour', 'dayofweek'
]

results = {}

# Train model for each prediction window
for days in [7, 15, 30, 60]:
    print(f"Training model for next {days} days...")
    labeled = label_targets(df.copy(), days)
    labeled = labeled.dropna(subset=features + ['target'])

    X = labeled[features]
    y = labeled['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[days] = acc

    joblib.dump(model, f"ML_predictions/earthquake_models/earthquake_{days}d.joblib")

# Output results
print("\n✅ Training complete. Accuracy scores:")
for days, acc in results.items():
    print(f"{days}-day model: {acc:.2%}")

