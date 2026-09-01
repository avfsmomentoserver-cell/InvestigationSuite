import pandas as pd
from src.forecasting.sessionize import sessionize

def test_sessionize_basic():
    data = pd.DataFrame({
        "user_id": ["u1","u1","u1","u2","u2"],
        "recorded_at": ["2026-08-28T09:00:00Z","2026-08-28T09:05:00Z","2026-08-28T10:00:00Z","2026-08-28T09:00:00Z","2026-08-28T11:00:00Z"]
    })
    df = sessionize(data, gap_minutes=30)
    assert 'round_id' in df.columns
    assert df['round_id'].nunique() >= 2
