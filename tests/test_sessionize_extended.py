"""Extended unit tests for src.forecasting.sessionize.sessionize."""

import numpy as np
import pandas as pd
import pytest
from src.forecasting.sessionize import sessionize


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Two users, multiple events spread across different gaps."""
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u1", "u2", "u2"],
        "recorded_at": [
            "2026-08-28T09:00:00",
            "2026-08-28T09:10:00",
            "2026-08-28T09:20:00",
            "2026-08-28T10:00:00",  # >30min gap from 09:20 → new round
            "2026-08-28T09:00:00",
            "2026-08-28T09:25:00",
        ],
    })


# ── Basic round behaviour ────────────────────────────────────────────────────

class TestSingleEvent:
    def test_single_event_yields_one_round(self):
        df = pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        })
        result = sessionize(df)
        assert result["round_index"].unique().tolist() == [1]
        assert result["round_id"].nunique() == 1
        assert result["event_count"].iloc[0] == 1

    def test_two_users_each_have_own_round(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "recorded_at": ["2026-01-01T12:00:00", "2026-01-01T13:00:00"],
        })
        result = sessionize(df)
        ids = sorted(result["round_id"].unique())
        assert ids == ["u1_1", "u2_1"]


class TestGapBoundary:
    """Exactly-at-gap should NOT start a new round; just-above should."""

    def test_exactly_at_gap_no_new_round(self):
        """Events exactly gap_minutes apart stay in the SAME round."""
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": ["2026-01-01T12:00:00", "2026-01-01T12:30:00"],
        })
        result = sessionize(df, gap_minutes=30)
        assert result["round_index"].nunique() == 1

    def test_gap_one_millisecond_above_starts_new_round(self):
        # Build timestamps via arithmetic to avoid pandas string-parsing quirks
        base = pd.Timestamp("2026-01-01T12:00:00")
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": [base, base + pd.Timedelta(minutes=30) + pd.Timedelta(microseconds=1)],
        })
        result = sessionize(df, gap_minutes=30)
        assert result["round_index"].nunique() == 2
        # Should split into two distinct rounds
        assert result["round_id"].nunique() == 2

    def test_gap_just_below_threshold_same_round(self):
        base = pd.Timestamp("2026-01-01T12:00:00")
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": [base, base + pd.Timedelta(minutes=29, seconds=59, milliseconds=999)],
        })
        result = sessionize(df, gap_minutes=30)
        assert result["round_index"].nunique() == 1


class TestEmptyInput:
    def test_empty_dataframe_returns_empty_with_columns(self):
        df = pd.DataFrame({"user_id": [], "recorded_at": []})
        result = sessionize(df)
        assert len(result) == 0
        # The key added columns must still be present
        for col in ("round_index", "round_id", "round_start", "round_end", "event_count"):
            assert col in result.columns

    def test_empty_user_list(self):
        df = pd.DataFrame({"user_id": pd.Series([], dtype=str), "recorded_at": pd.Series([], dtype="datetime64[ns]")})
        result = sessionize(df)
        assert len(result) == 0


class TestMissingOrNullTimestamps:
    """Old behaviour: NA rows survived. New behaviour: pre-filter drops them."""

    def test_na_timestamps_filtered_out_before_sessionization(self):
        """Rows with null recorded_at are dropped before grouping."""
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "recorded_at": pd.to_datetime(["2026-01-01T12:00:00", None, "2026-01-01T13:00:00"]),
        })
        result = sessionize(df)
        # NA row dropped; only 2 valid rows produce 2 distinct users
        assert len(result) == 2
        assert result["round_id"].nunique() == 2

    def test_all_na_timestamps_yields_empty_result(self):
        """When every row has null recorded_at, sessionize returns empty DataFrame."""
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": pd.to_datetime([None, None]),
        })
        result = sessionize(df)
        assert len(result) == 0
        # Key columns must still be present
        for col in ("round_index", "round_id", "round_start", "round_end", "event_count"):
            assert col in result.columns


class TestDuplicateTimestamps:
    def test_same_user_same_time_same_round(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": ["2026-01-01T12:00:00", "2026-01-01T12:00:00"],
        })
        result = sessionize(df)
        assert result["round_index"].nunique() == 1
        assert result["event_count"].iloc[0] == 2

    def test_duplicates_across_users_separate_rounds(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "recorded_at": ["2026-01-01T12:00:00", "2026-01-01T12:00:00"],
        })
        result = sessionize(df)
        assert result["round_id"].nunique() == 2


class TestCustomGap:
    def test_larger_gap_creates_more_rounds(self):
        # Events at 0, 10, 20 minutes. With gap=30 → 1 round.
        # With gap=5 → three 1-minute steps → still 1 round
        # So use bigger gaps: 0 and 40 minutes, gap=30 → 2 rounds.
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "recorded_at": ["2026-01-01T12:00:00", "2026-01-01T12:40:00"],
        })
        tight = sessionize(df, gap_minutes=20)
        loose = sessionize(df, gap_minutes=60)
        assert tight["round_id"].nunique() == 2
        assert loose["round_id"].nunique() == 1

    def test_custom_gap_changes_round_count(self):
        # Events at T+0 and T+1 min. gap=60 → 1 round; gap=1 → still 1 (not >);
        # but using a wider spread so smaller gap splits more aggressively.
        base = pd.Timestamp("2026-01-01T12:00:00")
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u1"],
            "recorded_at": [base, base + pd.Timedelta(hours=1), base + pd.Timedelta(hours=2), base + pd.Timedelta(hours=3)],
        })
        r90 = sessionize(df, gap_minutes=90)   # 60 min gaps → all one round
        r30 = sessionize(df, gap_minutes=30)   # 60 min gaps → multiple rounds
        assert r90["round_id"].nunique() == 1
        assert r30["round_id"].nunique() == 4  # each 60-min step is >30min


class TestRoundBounds:
    def test_round_bounds_match_actual_min_max(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u2"],
            "recorded_at": [
                "2026-01-01T10:00:00",
                "2026-01-01T10:15:00",
                "2026-01-01T11:00:00",  # new round
                "2026-01-01T09:30:00",
            ],
        })
        result = sessionize(df)
        for _, row in result.iterrows():
            rid = row["round_id"]
            mask = result["round_id"] == rid
            actual_start = result.loc[mask, "recorded_at"].min()
            actual_end = result.loc[mask, "recorded_at"].max()
            assert row["round_start"] == actual_start
            assert row["round_end"] == actual_end

    def test_round_start_is_le(self):
        df = pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-06-15T08:30:00"],
        })
        result = sessionize(df)
        expected = pd.Timestamp("2026-06-15T08:30:00")
        assert result["round_start"].iloc[0] == expected
        assert result["round_end"].iloc[0] == expected

    def test_unsorted_input_still_groups_correctly(self):
        data = {
            "user_id": ["u2", "u1", "u1", "u2"],
            "recorded_at": [
                "2026-01-01T11:00:00",
                "2026-01-01T10:00:00",
                "2026-01-01T10:10:00",
                "2026-01-01T09:00:00",
            ],
        }
        result = sessionize(pd.DataFrame(data))
        u1_rows = result[result["user_id"] == "u1"]
        assert u1_rows["round_index"].nunique() == 1
        u2_rows = result[result["user_id"] == "u2"]
        assert u2_rows["round_index"].nunique() == 2  # 09:00 vs 11:00 → >30min gap


class TestOriginalColumnsPreserved:
    def test_all_input_columns_present_in_output(self):
        df = pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
            "extra_col": ["hello"],
        })
        result = sessionize(df)
        assert "extra_col" in result.columns
        assert result["extra_col"].iloc[0] == "hello"

    def test_round_id_format(self):
        df = pd.DataFrame({
            "user_id": ["u1"],
            "recorded_at": ["2026-01-01T12:00:00"],
        })
        result = sessionize(df)
        assert result["round_id"].iloc[0] == "u1_1"
