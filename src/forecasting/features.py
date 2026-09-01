import pandas as pd
import numpy as np

def compute_round_features(sessionized_df, user_col='user_id', round_id_col='round_id', ts_col='recorded_at'):
    """
    Compute per-round aggregate features from sessionized events.
    Returns a DataFrame with one row per round.
    """
    df = sessionized_df.copy()
    agg = df.groupby([round_id_col, user_col]).agg(
        round_start=(ts_col, 'min'),
        round_end=(ts_col, 'max'),
        event_count=(ts_col, 'count'),
    ).reset_index()
    agg['duration_seconds'] = (agg['round_end'] - agg['round_start']).dt.total_seconds().fillna(0)
    # Example: counts per event_type (if present)
    if 'event_type' in df.columns:
        types = pd.crosstab(df[round_id_col], df['event_type'])
        types = types.reset_index().rename(columns={'index': round_id_col})
        agg = agg.merge(types, on=round_id_col, how='left')
    # Example: gap since previous round
    agg = agg.sort_values([user_col, 'round_start'])
    agg['prev_round_end'] = agg.groupby(user_col)['round_end'].shift(1)
    agg['gap_since_prev_seconds'] = (agg['round_start'] - agg['prev_round_end']).dt.total_seconds().fillna(-1)
    # Fill NaNs
    agg = agg.fillna(0)
    return agg
