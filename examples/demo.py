"""
Expanded Demo: ETL → generate scale data → create targets → train → serve → SDK
Generates ~1000 events across ~50 users, trains on 8-dimensional prediction schema.
"""
import os
import sys
import random
import pandas as pd
import numpy as np
from src.forecasting.etl import run_etl
from src.forecasting.train import train_baseline
import subprocess
import time
import requests
from api.sdk.python.forecast_sdk import ForecastSDK


# ── Step 0: Generate scale dataset ───────────────────────────────────────────
NUM_USERS = 50
EVENTS_PER_USER_MIN = 10
EVENTS_PER_USER_MAX = 40
GAP_HOURS_MIN = 1
GAP_HOURS_MAX = 48
USER_PREFIX = "user"
random.seed(42)
np.random.seed(42)

rows = []
base_ts = pd.Timestamp("2026-01-01T00:00:00")
for uid in range(NUM_USERS):
    u = f"{USER_PREFIX}_{uid+1}"
    current = base_ts + pd.Timedelta(hours=random.randint(0, 100))
    n_events = random.randint(EVENTS_PER_USER_MIN, EVENTS_PER_USER_MAX)
    event_types = ["click", "submit", "view", "scroll"]
    for _ in range(n_events):
        current += pd.Timedelta(minutes=random.randint(1, 60))
        rows.append({
            "user_id": u,
            "recorded_at": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_type": random.choice(event_types),
            "value": 1,
        })

scale_csv = "examples/scale_recordings.csv"
pd.DataFrame(rows).to_csv(scale_csv, index=False)
print(f"[SCALE] Generated {len(rows)} events across {NUM_USERS} users → {scale_csv}")

# ── Step 1: Run ETL on scale data ───────────────────────────────────────────
scale_parquet = "examples/scale_features.parquet"
print("[ETL] Running full pipeline...")
run_etl(scale_csv, scale_parquet, gap_minutes=30)

# ── Step 2: Create expanded target ───────────────────────────────────────────
df = pd.read_parquet(scale_parquet)
print(f"[FEATURES] Shape: {df.shape[0]} rounds, {df.shape[1]} columns")
print(f"[FEATURES] Columns: {list(df.columns)}")

if 'next_event_count' not in df.columns:
    # Predict next round's event count using a simple noisy transform
    # This simulates a real downstream target
    noise = np.random.normal(1.0, 0.3, size=len(df))
    df['next_event_count'] = np.maximum(1, (df['event_count'] * 0.9 * noise).astype(int))
    df.to_parquet(scale_parquet, index=False)
    print("[TARGET] Created 'next_event_count' target column")

# ── Step 3: Train model ──────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
print("[TRAIN] Starting training with expanded feature set...")
metrics = train_baseline(
    scale_parquet,
    'next_event_count',
    'models/baseline.joblib'
)
print(f"[TRAIN] Metrics: {metrics}")

# ── Step 4: Start server ─────────────────────────────────────────────────────
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.forecasting.serve:app", "--port", "8000"])
print("[SERVER] Waiting for server on http://localhost:8000 ...")
for _ in range(20):
    try:
        r = requests.get("http://localhost:8000/docs", timeout=2)
        if r.status_code == 200:
            print("[SERVER] Ready.")
            break
    except Exception:
        pass
    time.sleep(0.5)
else:
    proc.terminate()
    sys.exit("ERROR: server did not become ready within 10 seconds")

# ── Step 5: SDK request ──────────────────────────────────────────────────────
def _json_safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj

features_row = {k: _json_safe(v) for k, v in df.iloc[0].to_dict().items()}
sample_history = [{"ts": "2026-09-01T12:00:00Z", "features": features_row}]

sdk = ForecastSDK("http://localhost:8000")
resp = sdk.forecast("scale_entity", sample_history, horizon=1)
print("[PREDICTION]")
print(f"  Full response: {resp}")

# Pretty-print each prediction dimension
preds = resp.get('predictions', {})
for key, val in preds.items():
    flat_val = val[0] if isinstance(val, list) else val
    print(f"  {key}: {flat_val}")

meta = resp.get('metadata', {})
labels = meta.get('prediction_labels', {})
for key, val in preds.items():
    label = labels.get(key, '')
    flat_val = val[0] if isinstance(val, list) else val
    if label:
        print(f"    ↳ {label}: {flat_val}")

# ── Step 6: Cleanup ──────────────────────────────────────────────────────────
proc.terminate()
print("[DONE] Pipeline complete with expanded 8-dimension predictions")
