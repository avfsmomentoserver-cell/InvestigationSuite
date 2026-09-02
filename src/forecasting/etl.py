import pandas as pd
from src.forecasting.sessionize import sessionize
from src.forecasting.features import compute_round_features

def run_etl(source_csv, out_features_parquet, user_col='user_id', ts_col='recorded_at', gap_minutes=30):
    df = pd.read_csv(source_csv)
    n_in = len(df)
    df = df.drop_duplicates(keep='first')
    n_dup = len(df)
    df = df.dropna(subset=[user_col, ts_col])
    n_valid = len(df)
    dropped = n_in - n_valid
    duplicates = n_in - n_dup
    nulls = n_dup - n_valid
    print(f"[DEDUP] input_rows={n_in} -> after_dedup={n_dup} -> after_na_filter={n_valid} -> dropped={dropped} ({duplicates} duplicates + {nulls} null-key rows)")
    df_session = sessionize(df, user_col=user_col, ts_col=ts_col, gap_minutes=gap_minutes)
    features = compute_round_features(df_session, user_col=user_col)
    features.to_parquet(out_features_parquet, index=False)
    return features
