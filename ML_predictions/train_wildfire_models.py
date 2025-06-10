import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("" \
"Loading fire data...")
df = pd.read_csv("data_cleaned/clean_wildfires_with_coords.csv")


# Add small random noise to prevent overfitting exact bins
df['lat_bin'] = (df['Latitude'] + np.random.normal(0, 0.05, size=len(df))).round(2)
df['lon_bin'] = (df['Longitude'] + np.random.normal(0, 0.05, size=len(df))).round(2)

# Time
df['year_month'] = pd.to_datetime(df['year'].astype(str) + "-" + df['month'].astype(str), format="%Y-%m")

# Encode country/region
df['country_code'] = df['country'].astype('category').cat.codes
df['region_code'] = df['region'].astype('category').cat.codes

# Label future fires
def label_targets(df, days):
    df = df.sort_values('year_month')
    df['target'] = 0
    for (lat, lon), group in df.groupby(['lat_bin', 'lon_bin']):
        times = group['year_month'].reset_index(drop=True)
        for i in range(len(times)):
            future = times[(times > times[i]) & (times <= times[i] + pd.DateOffset(days=days))]
            if not future.empty:
                df.loc[group.index[i], 'target'] = 1
    return df

# Features
features = ['lat_bin', 'lon_bin', 'country_code', 'region_code', 'month']

# Train for 30 and 60 days only
for days in [30, 60]:
    print(f"\n Training fire model for next {days} days...")
    labeled = label_targets(df.copy(), days)
    X = labeled[features]
    y = labeled['target']

    # Stratified split to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f" Accuracy for {days}-day model: {acc:.2%}")

    joblib.dump(model, f"ML_predictions/wildfire_models/wildfire_model_{days}d.joblib")

