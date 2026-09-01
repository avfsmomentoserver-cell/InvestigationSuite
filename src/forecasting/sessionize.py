import pandas as pd

def sessionize(df, user_col='user_id', ts_col='recorded_at', gap_minutes=30):
    """
    Sessionize recorded events into rounds based on time gaps.

    Inputs:
      df: DataFrame with at least [user_col, ts_col]
      user_col: grouping id (e.g., user or device)
      ts_col: timestamp column (string or datetime)
      gap_minutes: gap threshold in minutes to start a new round

    Returns:
      df with added columns: round_index (int), round_id (str), round_start, round_end
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([user_col, ts_col])
    gap = pd.Timedelta(minutes=gap_minutes)
    df['prev_ts'] = df.groupby(user_col)[ts_col].shift(1)
    df['is_new_round'] = df['prev_ts'].isna() | ((df[ts_col] - df['prev_ts']) > gap)
    df['round_index'] = df.groupby(user_col)['is_new_round'].cumsum().astype(int)
    df['round_id'] = df[user_col].astype(str) + '_' + df['round_index'].astype(str)
    # compute round start/end by aggregating
    round_bounds = df.groupby(['round_id', user_col]).agg(
        round_start=(ts_col, 'min'),
        round_end=(ts_col, 'max'),
        event_count=(ts_col, 'count')
    ).reset_index()
    df = df.merge(round_bounds[['round_id', 'round_start', 'round_end', 'event_count']], on='round_id', how='left')
    return df.drop(columns=['prev_ts', 'is_new_round'])
