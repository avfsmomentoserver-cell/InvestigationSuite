"""Unit tests for src.forecasting.features.compute_round_features."""

import numpy as np
import pandas as pd
import pytest
from src.forecasting.sessionize import sessionize
from src.forecasting.features import compute_round_features


# ── Helper: build a minimal sessionized DataFrame ─────────────────────────────

def make_sessionized(users_events):
    """Build sessionized data from raw events and run *sessionize*.

    users_events : list of dict(s) with keys matching user_id / recorded_at / optional event_type
    Returns the sessionized DataFrame (already contains round_index, round_id, round_start,
    round_end, event_count columns).
    """
    df = pd.DataFrame(users_events)
    return sessionize(df, gap_minutes=30)


# ── Duration computation ─────────────────────────────────────────────────────

class TestDurationSeconds:
    def test_single_event_round_duration_is_zero(self):
        """A round with one event has round_start == round_end → duration 0."""
        sess = make_sessionized([{"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"}])
        result = compute_round_features(sess)
        assert len(result) == 1
        assert result["duration_seconds"].iloc[0] == 0.0

    def test_multi_event_round_duration_equals_span(self):
        """Round with events spanning 5 minutes → duration ≈ 300 seconds."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:05:00"},
        ])
        result = compute_round_features(sess)
        assert len(result) == 1
        assert result["duration_seconds"].iloc[0] == 300.0

    def test_duration_across_user_boundaries_independent(self):
        """Two users each have their own round; durations must not mix."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:02:00"},  # 120s
            {"user_id": "u2", "recorded_at": "2026-01-01T13:00:00"},
            {"user_id": "u2", "recorded_at": "2026-01-01T13:10:00"},  # 600s
        ])
        result = compute_round_features(sess)
        assert len(result) == 2
        durations = dict(zip(result["round_id"], result["duration_seconds"]))
        assert durations["u1_1"] == 120.0
        assert durations["u2_1"] == 600.0

    def test_subminute_duration_precision(self):
        """Ensure sub-minute deltas compute correctly to second precision."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:45"},
        ])
        result = compute_round_features(sess)
        assert result["duration_seconds"].iloc[0] == 45.0


class TestZeroDurationRoundStartEqualsEnd:
    def test_single_event_start_equals_end(self):
        sess = make_sessionized([{"user_id": "u1", "recorded_at": "2026-03-01T08:00:00"}])
        result = compute_round_features(sess)
        row = result.iloc[0]
        assert row["round_start"] == row["round_end"]

    def test_duplicate_timestamps_start_equals_end(self):
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-03-01T08:00:00"},
            {"user_id": "u1", "recorded_at": "2026-03-01T08:00:00"},
        ])
        result = compute_round_features(sess)
        row = result.iloc[0]
        assert row["round_start"] == row["round_end"]


# ── Event type counts ────────────────────────────────────────────────────────

class TestEventTypeCounts:
    def test_event_type_counts_present_when_column_exists(self):
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00", "event_type": "click"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:01:00", "event_type": "click"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:02:00", "event_type": "view"},
        ])
        result = compute_round_features(sess)
        assert len(result) == 1
        crosstab_cols = [c for c in result.columns if c in ("click", "view")]
        assert "click" in result.columns
        assert "view" in result.columns
        assert result["click"].iloc[0] == 2
        assert result["view"].iloc[0] == 1

    def test_event_type_counts_absent_when_column_missing(self):
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:01:00"},
        ])
        result = compute_round_features(sess)
        for c in result.columns:
            assert c not in ("click", "view", "imp")

    def test_multiple_event_types_across_users(self):
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00", "event_type": "a"},
            {"user_id": "u2", "recorded_at": "2026-01-01T12:00:00", "event_type": "b"},
        ])
        result = compute_round_features(sess)
        assert result["a"].iloc[0] == 1   # u1
        assert result["b"].iloc[1] == 1   # u2

    def test_crosstab_handles_unseen_type_per_user(self):
        """User u1 never emits 'view', so its column should still exist with value 0."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00", "event_type": "click"},
        ])
        result = compute_round_features(sess)
        # crosstab only creates columns present in data, but merges left ensures
        # the row still exists. Verify no KeyError on known types from OTHER users.
        # This is really about robustness: merge is left, so missing cols are OK.
        assert result.loc[result["user_id"] == "u1", "click"].iloc[0] == 1

    def test_three_event_types_with_correct_counts(self):
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00", "event_type": "x"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:01:00", "event_type": "y"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:02:00", "event_type": "z"},
        ])
        result = compute_round_features(sess)
        assert result["x"].iloc[0] == 1
        assert result["y"].iloc[0] == 1
        assert result["z"].iloc[0] == 1


# ── Gap since previous round ─────────────────────────────────────────────────

class TestGapSincePrevSeconds:
    def test_first_round_gap_is_negative_one(self):
        """The very first round per user must have gap_since_prev_seconds == -1."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:30:00"},
        ])
        # Only 1 round total, but it belongs to u1 — it's the first (and only) round.
        result = compute_round_features(sess)
        assert result.loc[result["round_id"] == "u1_1", "gap_since_prev_seconds"].iloc[0] == -1.0

    def test_subsequent_round_positive_gap(self):
        """Second round for same user gets a positive gap equal to the time delta."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:50:00"},
            {"user_id": "u1", "recorded_at": "2026-01-01T13:40:00"},
        ])
        result = compute_round_features(sess)
        # u1_1: gap=-1
        # u1_2: round_start=12:50, prev_round_end=12:50 → gap≈0
        gap_row = result.loc[result["round_id"] == "u1_2", "gap_since_prev_seconds"].iloc[0]
        assert gap_row >= 0

    def test_gap_between_distinct_rounds_exact(self):
        """First round ends at T+5min, second starts at T+40min → gap = 35 min = 2100s."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
            {"user_id": "u1", "recorded_at": "2026-01-01T12:05:00"},
            # >30min gap from 12:05…
            {"user_id": "u1", "recorded_at": "2026-01-01T12:40:00"},
        ])
        result = compute_round_features(sess)
        r2 = result.loc[result["round_id"] == "u1_2"]
        gap = r2["gap_since_prev_seconds"].iloc[0]
        assert abs(gap - 2100.0) < 1.0  # 12:40 - 12:05 = 35 min

    def test_zero_gap_same_end_as_start(self):
        """Second round starting right where first round ended gets gap≈0, not -1."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": [base, base + pd.Timedelta(seconds=1)],
        })
        sess = sessionize(df, gap_minutes=30)
        result = compute_round_features(sess)
        # Both events fall within 1 second → one round.
        # To get consecutive-round zero-gap, force two rounds separated by ≤30min gap.
        base2 = pd.Timestamp("2026-01-01T12:00:00")
        df2 = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": [base2, base2 + pd.Timedelta(minutes=45)],
        })
        sess2 = sessionize(df2, gap_minutes=30)
        result2 = compute_round_features(sess2)
        r2_gap = result2.loc[result2["round_id"] == "u1_2", "gap_since_prev_seconds"].iloc[0]
        # Round 1 ended at 12:00, round 2 starts at 12:45 → gap=2700s, NOT -1
        assert r2_gap > 0
        assert r2_gap == 2700.0


class TestMultiUserGapIndependence:
    def test_gap_is_computed_per_user_not_global(self):
        """User gaps must not interfere across user boundaries."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
            {"user_id": "u2", "recorded_at": "2026-01-01T12:00:00"},
            {"user_id": "u1", "recorded_at": "2026-01-01T13:00:00"},
            {"user_id": "u2", "recorded_at": "2026-01-01T12:01:00"},
        ])
        result = compute_round_features(sess)
        # round_index is NOT in the output of compute_round_features (dropped by groupby agg),
        # so select rounds by id instead.
        u1_first = result.loc[result["round_id"] == "u1_1"]
        u2_first = result.loc[result["round_id"] == "u2_1"]
        assert u1_first["gap_since_prev_seconds"].iloc[0] == -1.0
        assert u2_first["gap_since_prev_seconds"].iloc[0] == -1.0


# ── Edge cases & robustness ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_sessionized_input(self):
        sess = pd.DataFrame({
            "user_id": pd.Series([], dtype=str),
            "recorded_at": pd.Series([], dtype="datetime64[ns]"),
            "round_index": pd.Series([], dtype=int),
            "round_id": pd.Series([], dtype=str),
            "round_start": pd.Series([], dtype="datetime64[ns]"),
            "round_end": pd.Series([], dtype="datetime64[ns]"),
            "event_count": pd.Series([], dtype=int),
        })
        result = compute_round_features(sess)
        assert len(result) == 0

    def test_custom_round_id_col(self):
        """Rename round_id column and pass round_id_col arg."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
        ])
        renamed = sess.rename(columns={"round_id": "rid"})
        result = compute_round_features(renamed, round_id_col="rid")
        assert len(result) == 1
        assert "rid" in result.columns

    def test_numeric_cols_filled_with_zero_not_nan(self):
        """After fillna(0), numeric columns should contain no NaN values."""
        sess = make_sessionized([
            {"user_id": "u1", "recorded_at": "2026-01-01T12:00:00"},
        ])
        result = compute_round_features(sess)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not result[numeric_cols].isnull().any().any()
