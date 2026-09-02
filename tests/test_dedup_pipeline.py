"""Integration tests for the full dedup pipeline (ETL → sessionize → features).

Covers the three dedup stories:
  Story 1 : Raw event dedup removes exact duplicates and null-key rows
  Story 2 : Sessionize defense-in-depth drops NA keys before processing
  Story 3 : Round-key collision fails fast via AssertionError
"""

import pytest
import pandas as pd
import hashlib

from src.forecasting.etl import run_etl
from src.forecasting.sessionize import sessionize
from src.forecasting.features import compute_round_features


def _hash(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


# ── Story 1: Dedup at ingestion ──────────────────────────────────────────────

class TestDedupAtIngestion:
    """Story 1: Raw event dedup removes exact duplicates and null-key rows."""

    def test_drop_exact_duplicate_rows(self, tmp_path):
        """Two identical rows collapse into one; the distinct row survives.

        Both events are within 30 min so they belong to the SAME round.
        Total unique events = 2 → sessionize → 1 round → 1 feature row.
        """
        csv = tmp_path / "in.csv"
        csv.write_text(
            "user_id,recorded_at,event_type,value\n"
            "u1,2026-01-01T12:00:00Z,click,1\n"
            "u1,2026-01-01T12:00:00Z,click,1\n"       # duplicate
            "u1,2026-01-01T12:15:00Z,submit,1\n"      # distinct, within 30 min
        )
        pq = str(tmp_path / "out.parquet")
        run_etl(str(csv), pq)
        df = pd.read_parquet(pq)
        assert len(df) == 1          # one round after dedup + sessionize

    def test_multiple_exactly_duplicate_sets(self, tmp_path):
        """Multiple groups of identical rows each collapse independently."""
        csv = tmp_path / "in.csv"
        csv.write_text(
            "user_id,recorded_at,event_type,value\n"
            "u1,2026-01-01T12:00:00Z,click,1\n"
            "u1,2026-01-01T12:00:00Z,click,1\n"       # dup A
            "u1,2026-01-01T12:00:00Z,click,1\n"       # dup A (third copy)
            "u2,2026-01-01T12:30:00Z,view,2\n"
            "u2,2026-01-01T12:30:00Z,view,2\n"         # dup B
        )
        pq = str(tmp_path / "out.parquet")
        run_etl(str(csv), pq)
        df = pd.read_parquet(pq)
        # u1_1 (12:00→12:00, same round) + u2_1 (12:30, different user)
        assert len(df) == 2

    def test_null_key_rows_dropped(self, tmp_path):
        """Rows missing user_id or recorded_at must be filtered out."""
        csv = tmp_path / "in.csv"
        csv.write_text(
            "user_id,recorded_at,event_type,value\n"
            ",2026-01-01T12:00:00Z,click,1\n"           # empty user_id
            "u1,2026-01-01T12:00:00Z,click,1\n"         # valid
            "u1,,submit,1\n"                              # empty timestamp
        )
        pq = str(tmp_path / "out.parquet")
        run_etl(str(csv), pq)
        df = pd.read_parquet(pq)
        assert len(df) == 1             # only the valid row survives

    def test_all_null_keys_yields_empty_output(self, tmp_path):
        """When every row has a null key, output parquet has zero rows."""
        csv = tmp_path / "in.csv"
        csv.write_text(
            "user_id,recorded_at,event_type,value\n"
            ",2026-01-01T12:00:00Z,click,1\n"
            "u1,,click,1\n"
        )
        pq = str(tmp_path / "out.parquet")
        result = run_etl(str(csv), pq)
        assert len(result) == 0

    def test_dedup_log_output_contains_counts(self, tmp_path, capsys):
        """The [DEDUP] log line must appear with input/output counts."""
        csv = tmp_path / "in.csv"
        csv.write_text(
            "user_id,recorded_at,event_type,value\n"
            "u1,2026-01-01T12:00:00Z,click,1\n"
            "u1,2026-01-01T12:00:00Z,click,1\n"
            "u1,2026-01-01T12:15:00Z,submit,1\n"
        )
        pq = str(tmp_path / "out.parquet")
        run_etl(str(csv), pq)
        captured = capsys.readouterr()
        assert "[DEDUP]" in captured.out

    def test_dedup_log_shows_correct_dropped_count(self, tmp_path, capsys):
        """Log shows 1 duplicate + 2 null-key rows dropped = 3 total."""
        csv = tmp_path / "in.csv"
        csv.write_text(
            "user_id,recorded_at,event_type,value\n"
            "u1,2026-01-01T12:00:00Z,click,1\n"
            "u1,2026-01-01T12:00:00Z,click,1\n"       # duplicate
            ",2026-01-01T12:00:00Z,view,2\n"           # null key
            "u2,,click,3\n"                             # null key
        )
        pq = str(tmp_path / "out.parquet")
        run_etl(str(csv), pq)
        captured = capsys.readouterr()
        assert "[DEDUP]" in captured.out
        # 4 in → after_dedup=3 (1 dup) → after_na_filter=1 (2 nulls) → dropped=3
        assert "dropped=3" in captured.out


# ── Story 2: Sessionize defence-in-depth ─────────────────────────────────────

class TestSessionizeNullFiltering:
    """Story 2: sessionize() pre-filters NA/null before session boundaries."""

    def test_sessionize_drops_null_user_id(self):
        """Rows with null user_id are removed before round assignment."""
        df = pd.DataFrame({
            "user_id": ["u1", None, "u2"],
            "recorded_at": pd.to_datetime([
                "2026-01-01T12:00:00",
                "2026-01-01T12:30:00",
                "2026-01-01T12:00:00",
            ]),
        })
        result = sessionize(df)
        assert len(result) == 2        # null row removed

    def test_sessionize_drops_null_timestamp(self):
        """Rows with null timestamp are removed before round assignment."""
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "recorded_at": pd.to_datetime([
                "2026-01-01T12:00:00",
                None,
                "2026-01-01T12:30:00",
            ]),
        })
        result = sessionize(df)
        assert len(result) == 2        # null ts row removed

    def test_sessionize_drops_both_null_fields(self):
        """A row null on both user_id and recorded_at is removed."""
        df = pd.DataFrame({
            "user_id": [None],
            "recorded_at": pd.to_datetime([None]),
        })
        result = sessionize(df)
        assert len(result) == 0

    def test_sessionize_preserves_valid_rows_after_filter(self):
        """Valid rows retain original columns after null filtering."""
        df = pd.DataFrame({
            "user_id": ["u1", None, "u2"],
            "recorded_at": pd.to_datetime(["2026-01-01T12:00:00", None, "2026-01-01T12:00:00"]),
            "event_type": ["click", "imp", "view"],
            "value": [1, 2, 3],
        })
        result = sessionize(df)
        assert "event_type" in result.columns
        assert "value" in result.columns
        assert len(result) == 2
        assert set(result["user_id"]) == {"u1", "u2"}


# ── Story 3: Round-key collision fails fast ──────────────────────────────────

class TestRoundIdUniquenessAssertion:
    """Story 3: Round-key collision raises AssertionError."""

    def test_collision_raises_assertion_error(self):
        """Two different users sharing a round_id triggers an error."""
        sessionized = pd.DataFrame({
            "round_id": ["r1", "r1"],
            "user_id": ["user_a", "user_b"],
            "recorded_at": pd.to_datetime([
                "2026-01-01T12:00:00",
                "2026-01-01T13:00:00",
            ]),
        })
        with pytest.raises(AssertionError, match="round_id collision"):
            compute_round_features(sessionized)

    def test_no_collision_passes(self):
        """Same user across two round_ids is perfectly fine."""
        sessionized = pd.DataFrame({
            "round_id": ["r1", "r2"],
            "user_id": ["user_a", "user_a"],
            "recorded_at": pd.to_datetime([
                "2026-01-01T12:00:00",
                "2026-01-01T13:00:00",
            ]),
        })
        result = compute_round_features(sessionized)
        assert len(result) == 2

    def test_single_user_one_round_passes(self):
        """Edge case: one user, one round — simplest valid input."""
        sessionized = pd.DataFrame({
            "round_id": ["r1"],
            "user_id": ["alice"],
            "recorded_at": pd.to_datetime(["2026-06-01T10:00:00"]),
        })
        result = compute_round_features(sessionized)
        assert len(result) == 1
        assert result["user_id"].iloc[0] == "alice"

    def test_many_users_no_cross_contamination(self):
        """Five users, each with one round — no collisions."""
        users = [f"user_{i}" for i in range(5)]
        rids = [f"r{i}" for i in range(5)]
        sessionized = pd.DataFrame({
            "round_id": rids,
            "user_id": users,
            "recorded_at": pd.to_datetime([f"2026-01-{i+1}T12:00:00" for i in range(5)]),
        })
        result = compute_round_features(sessionized)
        assert len(result) == 5


# ── End-to-end pipeline consistency ──────────────────────────────────────────

class TestPipelineConsistency:
    """Verify the full ETL pipeline produces coherent feature output."""

    def test_full_pipeline_produces_valid_parquet(self, tmp_path):
        """Writing and reading back the parquet should yield the same schema."""
        csv = tmp_path / "in.csv"
        csv.write_text(
            "user_id,recorded_at,event_type,value\n"
            "alice,2026-01-01T10:00:00Z,click,1\n"
            "alice,2026-01-01T10:10:00Z,view,2\n"
            "bob,2026-01-01T11:00:00Z,click,3\n"
            "alice,2026-01-01T12:00:00Z,submit,4\n"
        )
        pq = str(tmp_path / "out.parquet")
        features = run_etl(str(csv), pq)
        df_back = pd.read_parquet(pq)
        assert df_back.shape == features.shape
        # Parquet round-trip preserves dtypes
        for col in features.columns:
            assert features[col].dtype == df_back[col].dtype or \
                   str(features[col].dtype) == str(df_back[col].dtype)

    def test_duplicate_input_events_yield_same_features(self, tmp_path):
        """Adding duplicate raw rows should not change the final feature set."""
        csv_nodup = tmp_path / "nodup.csv"
        csv_dup = tmp_path / "dup.csv"
        base_data = (
            "user_id,recorded_at,event_type,value\n"
            "u1,2026-01-01T12:00:00Z,click,1\n"
            "u1,2026-01-01T12:10:00Z,view,2\n"
        )
        csv_nodup.write_text(base_data)
        csv_dup.write_text(base_data + "u1,2026-01-01T12:00:00Z,click,1\n")

        pq_nodup = str(tmp_path / "nodup.parquet")
        pq_dup = str(tmp_path / "dup.parquet")
        f1 = run_etl(str(csv_nodup), pq_nodup)
        f2 = run_etl(str(csv_dup), pq_dup)
        pd.testing.assert_frame_equal(f1.reset_index(drop=True), f2.reset_index(drop=True))
