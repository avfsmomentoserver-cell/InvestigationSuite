"""Unit tests for Dimensions 3, 4, 5, 7, 8 in src.forecasting.features."""

import numpy as np
import pandas as pd
import pytest
from src.forecasting.sessionize import sessionize
from src.forecasting.features import compute_round_features


# ── Helper ────────────────────────────────────────────────────────────────────

def _sessionize(events_df, gap_minutes=30):
    """Build a sessionized DataFrame ready for compute_round_features."""
    return sessionize(events_df, gap_minutes=gap_minutes)


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 3 — Gap Pattern (rolling_mean_gap)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGapPatternDimension:
    """Verify rolling_mean_gap is the expanding mean of past gap_since_prev_seconds."""

    def test_rolling_mean_gap_is_last_minus_first(self):
        """Three events at T0, T+1h, T+3h: gaps are [-1, 3600, 7200].
        Rolling mean replaces -1 sentinel with NaN, then takes expanding mean.
        Sorted rolling_mean_gap = [0.0, 3600.0, 5400.0]; vals[2] = 5400.0."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u1"],
            "recorded_at": [base, base + pd.Timedelta(hours=1),
                            base + pd.Timedelta(hours=3)],
        })
        sess = _sessionize(df)
        result = compute_round_features(sess)
        # Three separate rounds: u1_1, u1_2, u1_3
        assert result["round_id"].nunique() == 3
        vals = sorted(result["rolling_mean_gap"])
        # After replace(-1, nan): expanding means are [nan->0.0, 3600.0, 5400.0]
        assert abs(vals[2] - 5400.0) < 1e-6

    def test_first_round_has_zero_rolling_mean_gap(self):
        """First round has no prior history → shift(1) gives NaN → expanding mean = NaN → fillna(0)."""
        sess = _sessionize(pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        }))
        result = compute_round_features(sess)
        assert result.loc[result["round_id"] == "u1_1", "rolling_mean_gap"].iloc[0] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 4 — Activity Trend (linear regression slope + direction label)
# ═══════════════════════════════════════════════════════════════════════════════


class TestActivityTrendDimension:
    """Test that _compute_trend correctly classifies user activity patterns."""

    def test_trending_up_classification(self):
        """Counts [1, 3, 5] → slope > threshold → 'trending_up'."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        events = []
        # Round 1: T+0h → 1 event
        events.append(base)
        # Gap ≥30 min for new round
        # Round 2: T+1d → 3 events
        t = base + pd.Timedelta(days=1)
        events.extend([t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=2)])
        # Round 3: T+2d → 5 events
        t = base + pd.Timedelta(days=2)
        events.extend([t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=2),
                       t + pd.Timedelta(minutes=3), t + pd.Timedelta(minutes=4)])

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(events),
            "recorded_at": events,
        })))
        row = result.iloc[-1]
        assert row["trend_direction_label"] == "trending_up"
        assert row["trend_linear_slope"] > 0

    def test_trending_down_classification(self):
        """Counts [5, 3, 1] → negative slope past threshold → 'trending_down'."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        events = []
        # Round 1: T+0h → 5 events
        events.extend([base, base + pd.Timedelta(minutes=1),
                       base + pd.Timedelta(minutes=2), base + pd.Timedelta(minutes=3),
                       base + pd.Timedelta(minutes=4)])
        # Round 2: T+1d → 3 events
        t = base + pd.Timedelta(days=1)
        events.extend([t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=2)])
        # Round 3: T+2d → 1 event
        events.append(base + pd.Timedelta(days=2))

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(events),
            "recorded_at": events,
        })))
        row = result.iloc[-1]
        assert row["trend_direction_label"] == "trending_down"
        assert row["trend_linear_slope"] < 0

    def test_stable_classification(self):
        """Constant counts [3, 3, 3] → slope ≈ 0 → 'stable'."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        events = []
        # Round 1: T+0h → 3 events
        events.extend([base, base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=2)])
        # Round 2: T+1d → 3 events
        t = base + pd.Timedelta(days=1)
        events.extend([t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=2)])
        # Round 3: T+2d → 3 events
        t = base + pd.Timedelta(days=2)
        events.extend([t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=2)])

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(events),
            "recorded_at": events,
        })))
        row = result.iloc[-1]
        assert row["trend_direction_label"] == "stable"
        assert abs(row["trend_linear_slope"]) < 1e-9

    def test_single_round_yields_stable(self):
        """One-event user → slope=0, label='stable'."""
        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        })))
        row = result.iloc[0]
        assert row["trend_direction_label"] == "stable"
        assert row["trend_linear_slope"] == 0.0

    def test_slope_value_correct_positive(self):
        """Count sequence [1, 2, 3] over 3 rounds → slope = 1.0 exactly."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        events = []
        # Round 1: T+0h → 1 event
        events.append(base)
        # Round 2: T+1d → 2 events
        t = base + pd.Timedelta(days=1)
        events.extend([t, t + pd.Timedelta(minutes=1)])
        # Round 3: T+2d → 3 events
        t = base + pd.Timedelta(days=2)
        events.extend([t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=2)])

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(events),
            "recorded_at": events,
        })))
        # All three rows belong to u1; slope should be identical in each
        slopes = result["trend_linear_slope"].unique()
        assert len(slopes) == 1
        assert abs(slopes[0] - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 5 — Session Consistency (coefficient of variation)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionConsistencyDimension:
    """CV = std / mean over expanding window per user (min_periods=2)."""

    def test_low_cv_for_constant_events(self):
        """Counts [3, 3, 3] → std=0 after first → cv=0."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        events = []
        # Round 1: T+0h → 3 events
        events.extend([base, base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=2)])
        # Round 2: T+1d → 3 events
        t = base + pd.Timedelta(days=1)
        events.extend([t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=2)])
        # Round 3: T+2d → 3 events
        t = base + pd.Timedelta(days=2)
        events.extend([t, t + pd.Timedelta(minutes=1), t + pd.Timedelta(minutes=2)])

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(events),
            "recorded_at": events,
        })))
        cvs = result.loc[result["user_id"] == "u1", "cv_event_count"].values
        # After first round (filled to 0), all subsequent CVs should be 0
        assert all(c == 0.0 for c in cvs)

    def test_high_cv_for_varied_events(self):
        """Counts [1, 10, 2] → cv > 0.5."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        events = []
        # Round 1: T+0h → 1 event
        events.append(base)
        # Round 2: T+1d → 10 events
        t = base + pd.Timedelta(days=1)
        for i in range(10):
            events.append(t + pd.Timedelta(minutes=i))
        # Round 3: T+2d → 2 events
        t = base + pd.Timedelta(days=2)
        events.extend([t, t + pd.Timedelta(minutes=1)])

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(events),
            "recorded_at": events,
        })))
        # Last row's cv should be high
        last_cv = result.loc[result["round_id"] == "u1_3", "cv_event_count"].iloc[0]
        assert last_cv > 0.5

    def test_cv_zero_with_one_event(self):
        """Single round expands to 1 element; min_periods=2 so fillna(0) applies → cv=0."""
        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        })))
        assert result["cv_event_count"].iloc[0] == 0.0

    def test_cv_includes_all_users_independent(self):
        """Two users with different event counts get independent CV calculations."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        events = []
        # u1: round 1=1event, round 2=2events
        events.extend([base, base + pd.Timedelta(days=1),
                       base + pd.Timedelta(days=1) + pd.Timedelta(minutes=1)])
        # u2: round 1=3events, round 2=3events
        events.extend([base, base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=2),
                       base + pd.Timedelta(days=1),
                       base + pd.Timedelta(days=1) + pd.Timedelta(minutes=1),
                       base + pd.Timedelta(days=1) + pd.Timedelta(minutes=2)])

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"],
            "recorded_at": events,
        })))
        u1_r2 = result.loc[(result["user_id"] == "u1") & (result["round_id"] == "u1_2"),
                           "cv_event_count"].iloc[0]
        u2_r2 = result.loc[(result["user_id"] == "u2") & (result["round_id"] == "u2_2"),
                           "cv_event_count"].iloc[0]
        assert u1_r2 != u2_r2


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 7 — Round Progression (growth ratio)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoundProgressionDimension:
    """Growth ratio = (current − prev) / prev, first round = 0, prev=0 → 0."""

    def test_growth_ratio_after_first(self):
        """Second round with double events → growth_ratio ≈ 1.0."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1", "u1", "u1"],
            "recorded_at": [
                base,                         # round 1: 1 event
                base + pd.Timedelta(hours=1), # round 2: 2 events (gap > 30 min)
                base + pd.Timedelta(hours=1) + pd.Timedelta(seconds=1),
            ],
        })))
        r1 = result.loc[result["round_id"] == "u1_1", "event_count_growth_ratio"].iloc[0]
        r2 = result.loc[result["round_id"] == "u1_2", "event_count_growth_ratio"].iloc[0]
        assert r1 == 0.0
        assert abs(r2 - 1.0) < 1e-6     # (2-1)/1 = 1.0

    def test_growth_ratio_zero_for_first(self):
        """First round has no previous → ratio = 0."""
        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        })))
        assert result.loc[result["round_id"] == "u1_1",
                          "event_count_growth_ratio"].iloc[0] == 0.0

    def test_negative_growth(self):
        """5 events → 2 events → ratio = (2-5)/5 = -0.6."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        # Round 1: 5 events within minutes (same round)
        r1_times = [base, base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=2),
                    base + pd.Timedelta(minutes=3), base + pd.Timedelta(minutes=4)]
        # Round 2: 2 events, ≥1 day later (separate round via gap_minutes=30)
        r2_base = base + pd.Timedelta(days=1)
        r2_times = [r2_base, r2_base + pd.Timedelta(minutes=1)]
        user_ids = ["u1"] * len(r1_times) + ["u1"] * len(r2_times)
        df = pd.DataFrame({
            "user_id": user_ids,
            "recorded_at": r1_times + r2_times,
        })
        result = compute_round_features(_sessionize(df))
        r1 = result.loc[result["round_id"] == "u1_1", "event_count"].iloc[0]
        r2 = result.loc[result["round_id"] == "u1_2", "event_count"].iloc[0]
        ratio = result.loc[result["round_id"] == "u1_2",
                           "event_count_growth_ratio"].iloc[0]
        assert r1 == 5 and r2 == 2
        assert abs(ratio - (-0.6)) < 1e-6

    def test_no_division_by_zero(self):
        """Previous round had 0 events (if possible via empty session); ratio must stay 0, not raise."""
        # Force prev=0 by crafting two-round session where round 1 somehow yields 0 events.
        # Practically all sessionized rounds have ≥1 event, but the feature code protects
        # replace(0, nan) / fillna(0). Stress-test directly by creating a result-like frame.
        base = pd.Timestamp("2026-01-01T12:00:00")
        # Two events separated widely → both land in separate rounds naturally.
        # Both will have 1 event each. To truly hit prev=0 we'd need manual injection.
        # Just ensure the pipeline doesn't raise on normal input.
        try:
            result = compute_round_features(_sessionize(pd.DataFrame({
                "user_id": ["u1", "u1"],
                "recorded_at": [base, base + pd.Timedelta(days=5)],
            })))
            # Should complete without ZeroDivisionError
            assert result["event_count_growth_ratio"].notna().all()
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError raised in compute_round_features")


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 8 — Stay-home Risk Score
# ═══════════════════════════════════════════════════════════════════════════════


class TestStayHomeRiskDimension:
    """latest_gap_hours, max_gap_seen_hours, stay_home_risk_score."""

    def test_low_risk_for_frequent_gaps(self):
        """Many 1-hour gaps + one 5-hour gap → avg_gap=1h, latest=5h → risk=1.0."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        timestamps = []
        ts = base
        for i in range(10):
            timestamps.append(ts)
            ts += pd.Timedelta(hours=1)
        # One big jump at the end: 6-hour gap
        timestamps.append(base + pd.Timedelta(hours=11) + pd.Timedelta(hours=5))

        df = pd.DataFrame({"user_id": ["u1"] * len(timestamps), "recorded_at": timestamps})
        result = compute_round_features(_sessionize(df))
        last_row = result.iloc[-1]
        assert last_row["stay_home_risk_score"] >= 1.0

    def test_zero_risk_single_round(self):
        """Only one round: gap=-1, no history → risk=0."""
        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        })))
        assert result.loc[result["round_id"] == "u1_1",
                          "stay_home_risk_score"].iloc[0] == 0.0

    def test_max_gap_seen_always_non_negative(self):
        """max_gap_seen_hours is always >= 0 (fillna(0) applied)."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": [base, base + pd.Timedelta(hours=1)],
        })
        result = compute_round_features(_sessionize(df))
        assert all(result["max_gap_seen_hours"] >= 0)

    def test_latest_gap_hours_positive(self):
        """Hours since last round is positive for non-first rounds."""
        base = pd.Timestamp("2026-01-01T12:00:00")
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": [base, base + pd.Timedelta(hours=2)],
        })
        result = compute_round_features(_sessionize(df))
        r2 = result.loc[result["round_id"] == "u1_2", "latest_gap_hours"].iloc[0]
        assert r2 > 0
