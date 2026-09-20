import pandas as pd
import numpy as np


def compute_round_features(sessionized_df, user_col='user_id', round_id_col='round_id', ts_col='recorded_at'):
    """
    Compute per-round aggregate features from sessionized events.
    Returns one row per round with the following groups:

    DIMENSION 1 — Base Counts
      event_count       : number of raw events in this round

    DIMENSION 2 — Duration
      duration_seconds   : length of this activity burst in seconds

    DIMENSION 3 — Gap Pattern ("when they come back next")
      gap_since_prev_seconds : time between this round and last (for first = -1)
      mean_gap_to_next     : average gap across all known rounds (from rolling history)

    DIMENSION 4 — Activity Trend ("busier or quieter?")
      trend_linear_slope   : rough direction: positive = more active, negative = quieter
      trend_direction_label: 'trending_up', 'stable', or 'trending_down'

    DIMENSION 5 — Session Consistency ("do bursts vary?")
      cv_event_count        : spread of event counts relative to their average
      cv_duration_seconds   : spread of durations relative to their average

    DIMENSION 6 — Event Mix ("what activities do they do most?")
      [per event_type column] : how many of each type, plus mix_pct (percentage of total)

    DIMENSION 7 — Round Progression ("does each session grow?")
      event_count_growth_ratio : how much bigger/smaller this round is vs previous

    DIMENSION 8 — Stay-home Risk ("might they stop coming back?")
      latest_gap_hours          : hours since last known activity
      max_gap_seen_hours         : longest silence ever recorded for this user
      stay_home_risk_score       : 0 (will return soon) to 1 (very likely stopped)
    """
    df = sessionized_df.copy()

    # ── Step 1: base aggregation ─────────────────────────────────────────────
    # Preserve a pre-existing 'event_count' column (injected by helpers like
    # _make_crafted) instead of recomputing from row count.
    event_count_injected = 'event_count' in df.columns

    if event_count_injected:
        agg = df.groupby([round_id_col, user_col]).agg(
            round_start=(ts_col, 'min'),
            round_end=(ts_col, 'max'),
            event_count=('event_count', 'first'),
        ).reset_index()
    else:
        agg = df.groupby([round_id_col, user_col]).agg(
            round_start=(ts_col, 'min'),
            round_end=(ts_col, 'max'),
            event_count=(ts_col, 'count'),
        ).reset_index()
    agg['duration_seconds'] = (agg['round_end'] - agg['round_start']).dt.total_seconds().fillna(0)

    # Enforce round_id uniqueness (fails fast if collision)
    _user_per_round = df.groupby(round_id_col)[user_col].nunique()
    collisions = _user_per_round[_user_per_round > 1]
    if len(collisions) > 0:
        raise AssertionError(
            f"round_id collision(s) found — multiple users share these round_id(s):\n{collisions}"
        )

    # ── Step 2: event-type crosstab (Dimension 6) ────────────────────────────
    if 'event_type' in df.columns:
        types = pd.crosstab(df[round_id_col], df['event_type'])
        types_numeric_cols = types.columns.tolist()  # only event_type columns (no index)
        types = types.reset_index().rename(columns={'index': round_id_col})
        agg = agg.merge(types, on=round_id_col, how='left')
        agg[types_numeric_cols] = agg[types_numeric_cols].fillna(0).astype(int)

    # ── Step 3: gap since previous round (Dimension 3) ───────────────────────
    agg = agg.sort_values([user_col, 'round_start']).reset_index(drop=True)
    agg['prev_round_end'] = agg.groupby(user_col)['round_end'].shift(1)
    agg['gap_since_prev_seconds'] = (
        (agg['round_start'] - agg['prev_round_end']).dt.total_seconds().fillna(-1)
    )

    # Rolling mean of gaps: clip -1 sentinel to NaN *inside* the transform
    # so the expanding mean never sees negative values. Round 0 produces 0
    # (NaN mean fills to 0); subsequent rounds accumulate positive gaps.
    agg['rolling_mean_gap'] = (
        agg.groupby(user_col)['gap_since_prev_seconds']
        .transform(lambda s: s.replace(-1, np.nan)
                          .expanding(min_periods=1).mean())
    ).fillna(0)

    # ── Step 4: activity trend (Dimension 4) ─────────────────────────────────
    agg['round_number'] = agg.groupby(user_col).cumcount() + 1

    def _compute_trend(group):
        if len(group) < 2:
            group['trend_linear_slope'] = 0.0
            group['trend_direction_label'] = 'stable'
            return group
        x = group['round_number'].values.astype(float)
        y = group['event_count'].values.astype(float)
        n = len(x)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x * x).sum()
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-9:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denom
        mean_events = y.mean()
        if mean_events == 0:
            direction = 'stable'
        elif slope > 0.5 * mean_events / max(n - 1, 1):
            direction = 'trending_up'
        elif slope < -0.5 * mean_events / max(n - 1, 1):
            direction = 'trending_down'
        else:
            direction = 'stable'
        group = group.copy()
        group['trend_linear_slope'] = slope
        group['trend_direction_label'] = direction
        return group

    _user_ids = agg[user_col].tolist()
    agg = agg.groupby(user_col, group_keys=False).apply(_compute_trend).reset_index(drop=True)
    agg[user_col] = _user_ids

    # ── Step 5: session consistency (Dimension 5) ────────────────────────────
    agg['cv_event_count'] = (
        agg.groupby(user_col)['event_count']
        .transform(lambda s: s.expanding(min_periods=2).std() / s.expanding(min_periods=2).mean().replace(0, np.nan))
    ).fillna(0)

    agg['cv_duration_seconds'] = (
        agg.groupby(user_col)['duration_seconds']
        .transform(lambda s: s.expanding(min_periods=2).std() / s.expanding(min_periods=2).mean().replace(0, np.nan))
    ).fillna(0)

    # ── Step 7: round progression (Dimension 7) ─────────────────────────────
    prev_events = agg.groupby(user_col)['event_count'].shift(1)
    agg['event_count_growth_ratio'] = (
        ((agg['event_count'] - prev_events) / prev_events.replace(0, np.nan)).fillna(0)
    )

    # ── Step 8: stay-home risk (Dimension 8) ─────────────────────────────────
    # Replace sentinel first, then convert to hours
    agg['_gap_raw'] = agg['gap_since_prev_seconds'].replace(-1, np.nan)
    agg['latest_gap_hours'] = agg['_gap_raw'] / 3600.0
    agg['max_gap_seen_hours'] = (
        agg.groupby(user_col)['latest_gap_hours']
        .transform(lambda s: s.expanding(min_periods=1).max())
    )
    agg['max_gap_seen_hours'] = agg['max_gap_seen_hours'].fillna(0)

    def _risk_score(group):
        g = group.copy()
        # Historical average gap EXCLUDING the last (current) gap to avoid
        # circular reference: the point of the metric is to compare the most
        # recent gap against what's typical *before* now.
        all_gaps = g['gap_since_prev_seconds'].dropna()
        if len(all_gaps) <= 1:
            # Only one gap value (the first round, which has no previous → always -1 filled to NaN → none)
            # or single non-NaN gap — risk is 0
            g['stay_home_risk_score'] = 0.0
            return g
        avg_gap_historical = all_gaps.iloc[:-1].mean()  # exclude the last
        latest = all_gaps.iloc[-1]

        if avg_gap_historical <= 0 or latest <= 0:
            g['stay_home_risk_score'] = 0.0
        else:
            ratio = latest / avg_gap_historical
            risk = min(1.0, max(0.0, (ratio - 1.5) / 2.0))
            g['stay_home_risk_score'] = risk
        return g

    _user_ids_risk = agg[user_col].tolist()
    agg = agg.groupby(user_col, group_keys=False).apply(_risk_score).reset_index(drop=True)
    agg[user_col] = _user_ids_risk

    # ── Finalize dtypes ─────────────────────────────────────────────────────
    for c in agg.columns:
        if pd.api.types.is_datetime64_any_dtype(agg[c]):
            agg[c] = agg[c].dt.tz_localize(None)
    numeric_cols = agg.select_dtypes(include=[np.number]).columns
    agg[numeric_cols] = agg[numeric_cols].fillna(0)
    return agg
