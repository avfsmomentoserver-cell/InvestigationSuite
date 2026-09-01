import pandas as pd
from src.forecasting.sessionize import sessionize
from src.forecasting.features import compute_round_features
from src.forecasting.train import train_baseline

print('imports ok')

# Load sample data
df = pd.read_csv('examples/sample_recordings.csv')
print(df.head())

# Sessionize
s = sessionize(df, gap_minutes=30)
print(s.head())

# Features
f = compute_round_features(s)
print(f.head())

# Create synthetic target and train
f['target'] = f['event_count'] * 0.1
f.to_parquet('examples/round_features.parquet', index=False)
metrics = train_baseline('examples/round_features.parquet', 'target', 'models/baseline.joblib')
print(metrics)
