import pandas as pd
import numpy as np
from src.forecasting.sessionize import sessionize
from src.forecasting.features import compute_round_features

base = pd.Timestamp("2026-01-01T12:00:00")
df = pd.DataFrame({
    "user_id": ["u1", "u1", "u1"],
    "recorded_at": [base, base + pd.Timedelta(hours=1), base + pd.Timedelta(hours=3)],
})
sess = sessionize(df)
result = compute_round_features(sess)
print("round_ids:", result["round_id"].tolist())
print("gaps:", result["gap_since_prev_seconds"].tolist())
print("rolling_mean_gaps:", result["rolling_mean_gap"].tolist())
print("sorted rmg:", sorted(result["rolling_mean_gap"]))
