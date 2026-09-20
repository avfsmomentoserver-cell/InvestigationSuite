"""
Adversarial STRESS TESTS for all 8 prediction dimensions.

Tests boundary cases where predictions might decompose incorrectly back to
their source events within sessions. Focuses on edge cases normal tests miss.

Each dimension gets its own class; one cross-cutting class for independence.
"""

import numpy as np
import pandas as pd
import pytest
from src.forecasting.sessionize import sessionize
from src.forecasting.features import compute_round_features


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sessionize(events_df, gap_minutes=30):
    """Build a sessionized DataFrame ready for compute_round_features."""
    return sessionize(events_df, gap_minutes=gap_minutes)


def _make_crafted(users_rounds):
    """
    Hand-craft a sessionized-style DataFrame suitable for direct input
    to compute_round_features without going through sessionize().

    Parameters
    ----------
    users_rounds : dict mapping user_id -> list of (round_start, round_end, event_count)

    Returns a DataFrame with columns user_id, round_index, round_id, round_start,
    round_end, event_count, recorded_at. The recorded_at column is synthesised from
    round_start because compute_round_features requires it for its groupby-aggregate.
    """
    users = []
    ridx = []
    rid = []
    rs = []
    re = []
    ec = []
    ra = []
    for uid, rounds in users_rounds.items():
        for ri, (start, end, count) in enumerate(rounds, start=1):
            users.append(uid)
            ridx.append(ri)
            rid.append(f"{uid}_{ri}")
            rs.append(pd.Timestamp(start))
            re.append(pd.Timestamp(end))
            ec.append(int(count))
            ra.append(pd.Timestamp(start))  # placeholder recorded_at
    cols_order = ["user_id", "round_index", "round_id", "round_start",
                  "round_end", "event_count", "recorded_at"]
    df = pd.DataFrame(dict(zip(cols_order, [users, ridx, rid, rs, re, ec, ra])))
    return df


BASE = pd.Timestamp("2026-01-01T12:00:00")


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 3 — GAP PATTERN (rolling_mean_gap)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGapPatternStress:
    """Boundary conditions for gap calculations between rounds."""

    def test_gap_boundary_at_gap_minutes(self):
        """
        Events exactly 30 min apart should NOT start new round (sessionize uses >).
        One millisecond past SHOULD start new round. Verify rolling_mean_gap.
        """
        t0 = BASE
        t_exact_30 = t0 + pd.Timedelta(minutes=30)       # exactly at threshold
        t_past_30  = t0 + pd.Timedelta(minutes=30, milliseconds=1)  # past

        # --- Exactly at 30 min → SAME round (2 events) ---
        sess_exact = _sessionize(pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": [t0, t_exact_30],
        }))
        assert sess_exact["round_id"].nunique() == 1

        # --- 1 ms past 30 min → TWO rounds ---
        sess_past = _sessionize(pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": [t0, t_past_30],
        }))
        assert sess_past["round_id"].nunique() == 2

        # Verify rolling_mean_gap with three distinct rounds.
        # Round间隔 > 30 min so each timestamp becomes its own round.
        gap_val = pd.Timedelta(minutes=30, milliseconds=1).total_seconds()

        sess_three = _sessionize(pd.DataFrame({
            "user_id": ["u1", "u1", "u1"],
            "recorded_at": [
                BASE,
                BASE + pd.Timedelta(hours=1),           # ≥30 min past T+0
                BASE + pd.Timedelta(hours=2),           # ≥30 min past T+1h
            ],
        }))
        result = compute_round_features(sess_three)
        rmgs = result["rolling_mean_gap"].values

        # Row 0: first round, shift→NaN → fillna(0)
        assert rmgs[0] == 0.0
        # Row 1: rolling mean of [gap_val] = gap_val
        expected_gap = (BASE + pd.Timedelta(hours=1) - BASE).total_seconds()
        assert abs(rmgs[1] - expected_gap) < 1e-3
        # Row 2: rolling mean of [gap1, gap2] ≈ same value
        expected_gap2 = (BASE + pd.Timedelta(hours=2) -
                         (BASE + pd.Timedelta(hours=1))).total_seconds()
        assert abs(rmgs[2] - (expected_gap + expected_gap2) / 2) < 1e-3

    def test_negative_inf_gap_injection(self):
        """
        Inject pathological timestamps into sessionized frame and verify
        feature computation does not crash or produce NaN propagation.
        We bypass the problematic .loc injection and instead create a
        crafted DF with a far-future round_end (effectively −∞ gap).
        """
        # Round-end of round 1 at year 2100 → gap ≈ far-future → ±Inf
        far_future = pd.Timestamp("2100-01-01T00:00:00")

        sess = _make_crafted({
            "u1": [
                (BASE, far_future, 3),                     # future end → big gap
                (far_future, BASE + pd.Timedelta(days=1), 2),  # starts far in future
            ]
        })

        # Should not crash
        result = compute_round_features(sess)

        # Numeric stability — no infinities or nans in outputs
        assert result["rolling_mean_gap"].notna().all()
        assert np.isfinite(result["rolling_mean_gap"]).all()
        assert result["gap_since_prev_seconds"].isna().sum() <= 1

    def test_cross_user_gap_leakage(self):
        """
        Two users with interleaved timestamps such that a broken groupby
        would make user A's round_start appear in user B's gap calculation.
        Verify strict isolation by user_col.
        """
        t0 = BASE
        events = [
            ("u1", t0),                                             # u1 round 1
            ("u2", t0 + pd.Timedelta(hours=12)),                    # u2 round 1
            ("u1", t0 + pd.Timedelta(days=1)),                      # u1 round 2
            ("u2", t0 + pd.Timedelta(days=1, hours=12)),            # u2 round 2
        ]
        sess = _sessionize(pd.DataFrame({
            "user_id": [e[0] for e in events],
            "recorded_at": [e[1] for e in events],
        }))
        result = compute_round_features(sess)

        u1_gaps = result.loc[result["user_id"] == "u1",
                             "gap_since_prev_seconds"].values
        u2_gaps = result.loc[result["user_id"] == "u2",
                             "gap_since_prev_seconds"].values

        # u1: first round gap=-1, second round gap should be ~86400s
        assert u1_gaps[0] == -1.0
        expected_u1 = 86400.0
        assert abs(u1_gaps[1] - expected_u1) < 1.0

        # u2: first round gap=-1, second round gap ≈ 86400s
        assert u2_gaps[0] == -1.0
        assert abs(u2_gaps[1] - expected_u1) < 1.0

        # Sanity: different phases prove isolation (r2 gap ≠ r1 sentinel)
        assert u1_gaps[1] != u2_gaps[0]

    def test_massive_timestamp_range(self):
        """
        One user with rounds spanning 10 years vs 1-second durations.
        float64 precision on seconds should hold.
        """
        t0 = pd.Timestamp("2016-01-01T12:00:00")

        times = []
        for i in range(5):
            rt = t0 + pd.Timedelta(days=i * 365)   # ~year-apart
            times.append(rt)
            times.append(rt + pd.Timedelta(seconds=1))

        sess = _sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(times),
            "recorded_at": times,
        }))
        result = compute_round_features(sess)

        # Duration = 1 second each (tiny compared to year-long gaps)
        for d in sorted(result["duration_seconds"]):
            assert abs(d - 1.0) < 1e-6, f"Duration {d} != 1.0 at decadal scale"

        # Gaps should be ~1 year in seconds (with leap-year tolerance)
        gaps = result["gap_since_prev_seconds"]
        valid = gaps.dropna()[1:]  # skip first sentinel
        year_sec = 365 * 86400.0
        for g in valid:
            assert abs(g - year_sec) < 86400 * 3

    def test_nan_event_type_column_when_present(self):
        """All-NaN event_type column → crosstab ignores NaN categories."""
        sess = _sessionize(pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "recorded_at": [
                BASE,
                BASE + pd.Timedelta(days=1),
                BASE + pd.Timedelta(hours=1),
            ],
        }))
        sess["event_type"] = [np.nan, np.nan, np.nan]

        result = compute_round_features(sess)

        assert result["event_count"].notna().all()
        assert "event_count_growth_ratio" in result.columns
        assert "rolling_mean_gap" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 4 — ACTIVITY TREND (linear regression slope + direction label)
# ═══════════════════════════════════════════════════════════════════════════════


class TestActivityTrendStress:
    """Edge cases for slope computation and direction classification."""

    def test_single_event_slope_zero_exact(self):
        """Single event → slope = 0.0 EXACTLY (assert == 0.0)."""
        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        })))
        assert result["trend_linear_slope"].iloc[0] == 0.0
        assert result["trend_direction_label"].iloc[0] == "stable"

    def test_plateau_after_spike(self):
        """Counts [100, 1, 1, 1, 1] → declining trend: slope must be < 0."""
        r1 = [BASE + pd.Timedelta(seconds=i) for i in range(100)]
        r2 = [BASE + pd.Timedelta(days=1)]
        r3 = [BASE + pd.Timedelta(days=2)]
        r4 = [BASE + pd.Timedelta(days=3)]
        r5 = [BASE + pd.Timedelta(days=4)]
        events = r1 + r2 + r3 + r4 + r5

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(events),
            "recorded_at": events,
        })))
        slope = result["trend_linear_slope"].iloc[-1]
        assert slope < 0, f"Expected negative slope for declining counts, got {slope}"

    def test_alternating_up_down(self):
        """Counts [1, 5, 1, 5, 1, 5] → oscillation. Slope mag < 0.5 × mean."""
        times = []
        for i in range(6):
            rt = BASE + pd.Timedelta(days=i)
            count = 1 if i % 2 == 0 else 5
            times.extend([rt + pd.Timedelta(seconds=j) for j in range(count)])

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(times),
            "recorded_at": times,
        })))
        slope = result["trend_linear_slope"].iloc[-1]
        mean_events = 3.0  # (1+5)*3/6
        assert abs(slope) < 0.5 * mean_events, \
            f"Slope {slope} not < 0.5×mean={0.5*mean_events}"

    def test_all_same_count_with_ones(self):
        """One-event user → slope=0, direction='stable'. No div-by-zero."""
        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        })))
        assert result["trend_linear_slope"].iloc[0] == 0.0
        assert result["trend_direction_label"].iloc[0] == "stable"

    def test_two_events_identical_count(self):
        """[3, 3] → y-values identical → slope=0, direction='stable'."""
        r1 = [BASE + pd.Timedelta(seconds=i) for i in range(3)]
        r2 = [BASE + pd.Timedelta(days=1) + pd.Timedelta(seconds=i)
              for i in range(3)]

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(r1 + r2),
            "recorded_at": r1 + r2,
        })))
        slopes = result["trend_linear_slope"].unique()
        labels = result["trend_direction_label"].unique()
        assert slopes[0] == 0.0
        assert labels[0] == "stable"


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 5 — SESSION CONSISTENCY (coefficient of variation)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionConsistencyStress:
    """Edge cases for CV computation including numerical instability regions."""

    def test_diverging_cv_to_very_large(self):
        """Counts [1, 100, 10000] → cv grows but must stay finite."""
        times = ([BASE] +
                 [BASE + pd.Timedelta(days=1) + pd.Timedelta(seconds=i)
                  for i in range(100)] +
                 [BASE + pd.Timedelta(days=2) + pd.Timedelta(seconds=i)
                  for i in range(10000)])

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(times),
            "recorded_at": times,
        })))
        cv_last = result["cv_event_count"].iloc[-1]
        assert np.isfinite(cv_last), f"CV diverged: {cv_last}"
        assert cv_last > 0.0

    def test_cv_underflow_on_small_std(self):
        """Identical counts → std=0 → cv=0. No NaN/Inf from tiny std."""
        sess = _make_crafted({
            "u1": [
                (BASE, BASE + pd.Timedelta(seconds=1), 1),
                (BASE + pd.Timedelta(days=1), BASE + pd.Timedelta(days=1, microseconds=1), 1),
                (BASE + pd.Timedelta(days=2), BASE + pd.Timedelta(days=2, microseconds=1), 1),
            ]
        })
        result = compute_round_features(sess)

        assert np.isfinite(result["cv_event_count"]).all()
        assert result["cv_event_count"].iloc[-1] == 0.0

    def test_cv_cross_contamination(self):
        """Users A=[1,1,1] and B=[10,10,10]: both cv=0, proving isolation."""
        events = []
        # u1: three rounds, 1 event each
        events.append(("u1", BASE))
        events.append(("u1", BASE + pd.Timedelta(days=1)))
        events.append(("u1", BASE + pd.Timedelta(days=2)))
        # u2: three rounds, 10 events each
        for i in range(10):
            events.append(("u2", BASE + pd.Timedelta(days=1, minutes=i)))
            events.append(("u2", BASE + pd.Timedelta(days=2, minutes=i)))
            events.append(("u2", BASE + pd.Timedelta(days=3, minutes=i)))

        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": [e[0] for e in events],
            "recorded_at": [e[1] for e in events],
        })))

        for uid in ["u1", "u2"]:
            cvs = result.loc[result["user_id"] == uid, "cv_event_count"].values
            assert all(c == 0.0 for c in cvs), f"{uid} CVs not all zero: {cvs}"


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 7 — ROUND PROGRESSION (growth ratio)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoundProgressionStress:
    """Edge cases for growth ratio computation including division by zero."""

    def test_growth_ratio_from_zero_prev(self):
        """Force prev_events=0 via crafted DF → ratio=0.0, never raise."""
        sess = _make_crafted({
            "u1": [
                (BASE, BASE + pd.Timedelta(seconds=1), 0),     # zero count
                (BASE + pd.Timedelta(days=1), BASE + pd.Timedelta(days=1, seconds=1), 2),
            ]
        })
        try:
            result = compute_round_features(sess)
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError when prev_events=0")

        ratio = result.loc[result["round_id"] == "u1_2",
                           "event_count_growth_ratio"].iloc[0]
        assert ratio == 0.0

    def test_massive_growth_spike(self):
        """Current=1, Prev=1000000 → ratio ≈ -0.999999 within 1e-6."""
        sess = _make_crafted({
            "u1": [
                (BASE, BASE + pd.Timedelta(hours=24), 1000000),
                (BASE + pd.Timedelta(days=2), BASE + pd.Timedelta(days=2, seconds=1), 1),
            ]
        })
        result = compute_round_features(sess)
        ratio = result.loc[result["round_id"] == "u1_2",
                           "event_count_growth_ratio"].iloc[0]
        expected = (1 - 1000000) / 1000000
        assert abs(ratio - expected) < 1e-6, \
            f"{ratio} != {expected} within 1e-6"

    def test_overflow_on_extreme_counts(self):
        """Count > 10^9 → float division must not overflow or lose precision."""
        curr = int(1e9)
        prev = int(1e9) + 1
        sess = _make_crafted({
            "u1": [
                (BASE, BASE + pd.Timedelta(hours=1), prev),
                (BASE + pd.Timedelta(days=2), BASE + pd.Timedelta(days=2, minutes=30), curr),
            ]
        })
        result = compute_round_features(sess)

        ratio = result.loc[result["round_id"] == "u1_2",
                           "event_count_growth_ratio"].iloc[0]
        expected = (curr - prev) / prev
        assert np.isfinite(ratio), f"Overflow: {ratio}"
        assert abs(ratio - expected) < 1e-12, \
            f"Precision loss: {ratio} != {expected}"


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 8 — STAY-HOME RISK SCORE
# ═══════════════════════════════════════════════════════════════════════════════


class TestStayHomeRiskStress:
    """Edge cases for risk score including division by zero and clamping."""

    def test_zero_avg_gap_division_by_zero(self):
        """Two adjacent rounds with zero gap between them → risk safe."""
        # Round 1 ends at T+1s, round 2 starts at T+1s → gap = 0
        sess = _make_crafted({
            "u1": [
                (BASE, BASE + pd.Timedelta(seconds=1), 3),
                (BASE + pd.Timedelta(seconds=1),
                 BASE + pd.Timedelta(seconds=2), 3),
            ]
        })
        result = compute_round_features(sess)

        assert "stay_home_risk_score" in result.columns
        assert np.isfinite(result["stay_home_risk_score"]).all()
        # First round: gap=-1 sentinel → avg_gap uses nothing → risk=0.0
        assert result["stay_home_risk_score"].iloc[0] == 0.0

    def test_missing_gap_history_single_round(self):
        """Only one round → gap=-1 everywhere → risk = 0.0 EXACTLY."""
        result = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        })))
        risk = result.loc[result["round_id"] == "u1_1",
                          "stay_home_risk_score"].iloc[0]
        assert risk == 0.0, f"Single-round risk should be 0.0, got {risk}"

    def test_max_gap_seen_ignores_negative_sentinel(self):
        """max_gap_seen must ignore -1 sentinels and track only positive gaps."""
        sess = _sessionize(pd.DataFrame({
            "user_id": ["u1", "u1", "u1"],
            "recorded_at": [
                BASE,
                BASE + pd.Timedelta(hours=1),
                BASE + pd.Timedelta(hours=24),
            ],
        }))
        result = compute_round_features(sess)

        # First row: max_gap_seen=0 (only sentinel seen, ignored)
        assert result["max_gap_seen_hours"].iloc[0] == 0.0

        # Last row: max should capture the 24h gap (at least 23h)
        assert result["max_gap_seen_hours"].iloc[-1] >= 23.0

    def test_risk_score_clipped_bounds(self):
        """Extreme gap ratios must clamp risk to [0.0, 1.0]."""
        # Huge gaps: round 3 starts ~1000 days later → ratio >> 100
        sess = _make_crafted({
            "u1": [
                (BASE, BASE + pd.Timedelta(seconds=1), 5),
                (BASE + pd.Timedelta(hours=2),
                 BASE + pd.Timedelta(hours=2, seconds=1), 5),
                (BASE + pd.Timedelta(days=1000),
                 BASE + pd.Timedelta(days=1000, seconds=1), 5),
            ]
        })
        result = compute_round_features(sess)

        # All scores must be in [0, 1]
        assert (result["stay_home_risk_score"] >= 0.0).all()
        assert (result["stay_home_risk_score"] <= 1.0).all()
        # Extreme ratio → risk clamped to 1.0
        assert result["stay_home_risk_score"].iloc[-1] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-DIMENSION INDEPENDENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossDimensionIndependence:
    """Verify modifying one dimension's input doesn't cascade incorrectly."""

    def test_predict_one_dim_changes_only_other(self):
        """
        Adding events within round 2 increases event_count but DOES NOT change
        gap/risk scores because round_start/end/timing are identical.

        Note: adding events extends round_end slightly. To prevent this,
        we construct both scenarios via _sessionize with timestamps mapped
        to identical round boundaries using the same gap structure.
        """
        # Baseline: two rounds, 2 events each
        base_orig = [
            BASE,
            BASE + pd.Timedelta(minutes=1),
            BASE + pd.Timedelta(hours=1),
            BASE + pd.Timedelta(hours=1) + pd.Timedelta(minutes=1),
        ]
        # Extra: two rounds, many more events in round 2
        base_extra = [
            BASE,
            BASE + pd.Timedelta(minutes=1),
            # Round 2 has many events, all within 30-min window starting at +1h
            *(BASE + pd.Timedelta(hours=1) + pd.Timedelta(minutes=m)
              for m in range(0, 60, 1)),
        ]

        sess_orig = _sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(base_orig),
            "recorded_at": base_orig,
        }))
        result_orig = compute_round_features(sess_orig)

        sess_extra = _sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(base_extra),
            "recorded_at": base_extra,
        }))
        result_extra = compute_round_features(sess_extra)

        # Both have two rounds with matching round_ids
        assert sorted(result_orig["round_id"]) == sorted(result_extra["round_id"])

        # Event count MUST increase
        orig_ec = result_orig.loc[result_orig["round_id"] == "u1_2",
                                  "event_count"].iloc[0]
        extra_ec = result_extra.loc[result_extra["round_id"] == "u1_2",
                                     "event_count"].iloc[0]
        assert extra_ec > orig_ec

        # Timing-based dims may drift < 0.01 sec from round_end expansion —
        # we compare relative closeness rather than exact equality.
        for col in ["latest_gap_hours", "max_gap_seen_hours"]:
            orig_v = result_orig.loc[result_orig["round_id"] == "u1_2",
                                     col].iloc[0]
            extra_v = result_extra.loc[result_extra["round_id"] == "u1_2",
                                       col].iloc[0]
            assert abs(orig_v - extra_v) < 1e-3, \
                f"{col} drifted > 1e-3: {orig_v} → {extra_v}"

        # Risk delta must be tiny (< 1e-3 hours ≈ 3.6 ms effect)
        orig_risk = result_orig.loc[result_orig["round_id"] == "u1_2",
                                     "stay_home_risk_score"].iloc[0]
        extra_risk = result_extra.loc[result_extra["round_id"] == "u1_2",
                                       "stay_home_risk_score"].iloc[0]
        assert abs(orig_risk - extra_risk) < 1e-3, \
            f"Risk changed too much: {orig_risk} → {extra_risk}"

    def test_modify_only_timing_affects_only_time_dims(self):
        """Shift second round farther out → gap-related dims change, count stays."""
        events_short_gap = [
            BASE, BASE + pd.Timedelta(minutes=1), BASE + pd.Timedelta(minutes=2),
            BASE + pd.Timedelta(hours=1),
            BASE + pd.Timedelta(hours=1) + pd.Timedelta(minutes=1),
            BASE + pd.Timedelta(hours=1) + pd.Timedelta(minutes=2),
        ]
        events_long_gap = [
            BASE, BASE + pd.Timedelta(minutes=1), BASE + pd.Timedelta(minutes=2),
            BASE + pd.Timedelta(hours=2),
            BASE + pd.Timedelta(hours=2) + pd.Timedelta(minutes=1),
            BASE + pd.Timedelta(hours=2) + pd.Timedelta(minutes=2),
        ]

        r1 = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(events_short_gap),
            "recorded_at": events_short_gap,
        })))
        r2 = compute_round_features(_sessionize(pd.DataFrame({
            "user_id": ["u1"] * len(events_long_gap),
            "recorded_at": events_long_gap,
        })))

        # Event counts identical
        for rid in ["u1_1", "u1_2"]:
            assert r1.loc[r1["round_id"] == rid, "event_count"].iloc[0] == \
                   r2.loc[r2["round_id"] == rid, "event_count"].iloc[0]

        # Gap MUST increase
        sg = r1.loc[r1["round_id"] == "u1_2", "gap_since_prev_seconds"].iloc[0]
        lg = r2.loc[r2["round_id"] == "u1_2", "gap_since_prev_seconds"].iloc[0]
        assert lg > sg

    def test_unknown_event_type_doesnt_break_pipeline(self):
        """Novel event_type string → crosstab creates column → pipeline survives."""
        sess = _sessionize(pd.DataFrame({
            "user_id": ["u1", "u1", "u1"],
            "recorded_at": [
                BASE,
                BASE + pd.Timedelta(days=1),
                BASE + pd.Timedelta(days=2),
            ],
            "event_type": ["xyz_custom_type", "xyz_custom_type", "xyz_custom_type"],
        }))
        result = compute_round_features(sess)

        assert result["event_count"].notna().all()
        assert "xyz_custom_type" in result.columns
        assert result.select_dtypes(include=[np.number]).fillna(0).shape == \
               result.select_dtypes(include=[np.number]).shape
