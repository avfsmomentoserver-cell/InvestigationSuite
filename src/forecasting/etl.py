import pandas as pd
from src.forecasting.sessionize import sessionize
from src.forecasting.features import compute_round_features

def run_etl(source_csv, out_features_parquet, user_col='user_id', ts_col='recorded_at', gap_minutes=30):
    df = pd.read_csv(source_csv)
    df_session = sessionize(df, user_col=user_col, ts_col=ts_col, gap_minutes=gap_minutes)
    features = compute_round_features(df_session, user_col=user_col)
    features.to_parquet(out_features_parquet, index=False)
    return features
