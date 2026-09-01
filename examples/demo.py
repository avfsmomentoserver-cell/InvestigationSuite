"""
Demo: run local ETL -> train -> serve -> SDK call
This is a minimal end-to-end script to show how components fit together.
"""
import os
from src.forecasting.etl import run_etl
from src.forecasting.train import train_baseline
import subprocess
import time
from api.sdk.python.forecast_sdk import ForecastSDK

# 1) create features from sample data
sample_csv = "examples/sample_recordings.csv"
features_parquet = "examples/round_features.parquet"
if not os.path.exists(features_parquet):
    run_etl(sample_csv, features_parquet, gap_minutes=30)

# 2) create a fake target for demo if missing
import pandas as pd
df = pd.read_parquet(features_parquet)
if 'target' not in df.columns:
    df['target'] = df['event_count'] * 0.1  # synthetic target
    df.to_parquet(features_parquet, index=False)

# 3) train model
os.makedirs('models', exist_ok=True)
metrics = train_baseline(features_parquet, 'target', 'models/baseline.joblib')
print("Training metrics:", metrics)

# 4) start server (in background)
proc = subprocess.Popen(["uvicorn", "src.forecasting.serve:app", "--port", "8000"])
time.sleep(2)

# 5) SDK request
sdk = ForecastSDK("http://localhost:8000")
sample_history = [{"ts": "2026-08-31T12:00:00Z", "features": df.iloc[0].to_dict()}]
resp = sdk.forecast("demo_entity", sample_history, horizon=1)
print("Forecast response:", resp)

# 6) cleanup
proc.terminate()
