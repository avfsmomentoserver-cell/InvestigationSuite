"""
Demo: run local ETL -> train -> serve -> SDK call
This is a minimal end-to-end script to show how components fit together.
"""
import os
import sys
from src.forecasting.etl import run_etl
from src.forecasting.train import train_baseline
import subprocess
import time
import requests
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

# 4) start server (in background) and wait until it is ready
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.forecasting.serve:app", "--port", "8000"])
print("Waiting for server on http://localhost:8000 ...")
for _ in range(20):
    try:
        r = requests.get("http://localhost:8000/docs", timeout=2)
        if r.status_code == 200:
            print("Server is ready.")
            break
    except Exception:
        pass
    time.sleep(0.5)
else:
    proc.terminate()
    sys.exit("ERROR: server did not become ready within 10 seconds")

# 5) SDK request
sdk = ForecastSDK("http://localhost:8000")
def _json_safe(obj):
    """Recursively convert pandas/numpy types to JSON-serializable Python primitives."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if hasattr(obj, 'isoformat'):  # pandas Timestamp / datetime
        return obj.isoformat()
    return obj


features_row = {k: _json_safe(v) for k, v in df.iloc[0].to_dict().items()}
sample_history = [{"ts": "2026-08-31T12:00:00Z", "features": features_row}]
resp = sdk.forecast("demo_entity", sample_history, horizon=1)
print("Forecast response:", resp)

# 6) cleanup
proc.terminate()
