import pandas as pd


def sessionize(df, user_col='user_id', ts_col='recorded_at', gap_minutes=30):
    """
    Sessionize recorded events into rounds based on time gaps.

    Fully deterministic: identical inputs always produce identical outputs,
    even when multiple events share the same timestamp for the same user.
    Tiebreaking uses the original row position (stable sort), making results
    equivalent to a SQL window-function implementation with ORDER BY.

    Inputs:
      df: DataFrame with at least [user_col, ts_col]
      user_col: grouping id (e.g., user or device)
      ts_col: timestamp column (string or datetime)
      gap_minutes: gap threshold in minutes to start a new round

    Returns:
      df with added columns: round_index (int), round_id (str), round_start, round_end
    """
    # Pre-filter: drop rows with null/NA in critical columns (defense-in-depth)
    df = df.dropna(subset=[user_col, ts_col]).reset_index(drop=True)

    if len(df) == 0:
        # Ensure proper dtypes even for empty result so downstream (features.py)
        # does not hit "operation 'sub' not supported for dtype 'str'" errors.
        df = df.copy()
        df[user_col] = df[user_col].astype(str)
        df[ts_col] = pd.to_datetime(df[ts_col])
        df['round_index'] = pd.Series(dtype='int64')
        df['round_id'] = pd.Series(dtype='object')
        df['round_start'] = pd.Series(dtype='datetime64[ns]')
        df['round_end'] = pd.Series(dtype='datetime64[ns]')
        df['event_count'] = pd.Series(dtype='int64')
        return df

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])

    # ── Deterministic sort: user → timestamp → original row index ──────────
    # Adding _seq as a tiebreaker guarantees identical timestamps for the
    # same user always receive the same round assignment, regardless of
    # how the input was ordered.  This mirrors a SQL ROW_NUMBER() OVER
    # (PARTITION BY user ORDER BY ts, rowid) pattern.
    df['_seq'] = range(len(df))
    df = df.sort_values([user_col, ts_col, '_seq'], ignore_index=True)

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

    # Drop helper columns; keep original column order + new columns
    df = df.drop(columns=['prev_ts', 'is_new_round', '_seq'])
    return df
