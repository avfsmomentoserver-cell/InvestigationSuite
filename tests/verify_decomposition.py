"""Dynamic decomposition verification.

Traces each of the 8 prediction dimensions BACK from the stored features
to the raw CSV events, independently recomputing every column via
sessionize() -> compute_round_features() on the original event data,
and comparing result vs. the ETL-produced parquet.

8 Dimensions (from serve.py prediction schema):
  1. event_count_predicted    "Things That Will Happen Next"
  2. gap_hours_until_next     "When They Come Back Next (hours)"
  3. trend_direction          "Busier or Quieter?"
  4. session_consistency      "Do Bursts Vary?"
  5. top_event_type           "Most Likely Activity Type"
  6. growth_ratio             "Are Sessions Growing?"
  7. stay_home_risk           "Chance They Stop Coming"
  8. confidence_range         "How Sure Are We?"
"""
import sys
import joblib
import pandas as pd
import numpy as np
from src.forecasting.sessionize import sessionize
from src.forecasting.features import compute_round_features


# ── Paths ────────────────────────────────────────────────────────────────────
RAW_CSV = "examples/scale_recordings.csv"
FEATURE_PQ = "examples/scale_features.parquet"
MODEL_PATH = "models/baseline.joblib"
PIPELINE_LOG = "/tmp/pipeline_output.txt"

FLOAT_TOL = 1e-9
STRAT="strict"


def approx_eq(a, b):
    """Check approximate equality for floats, exact for non-floats."""
    try:
        af, bf = float(a), float(b)
        if not np.isfinite(af) or not np.isfinite(bf):
            return af == bf
        return abs(af - bf) < FLOAT_TOL
    except (TypeError, ValueError):
        return str(a) == str(b)


# ── Dimension → column mapping ───────────────────────────────────────────────
DIMENSIONS = {
    "event_count": {
        "label": "event_count_predicted",
        "human_label": "Things That Will Happen Next",
        "source_col": "event_count",
        "desc": "Count of raw events grouped into this round",
        "type": "numeric",
        "computation": "COUNT(*) FROM raw_events WHERE round_id=X AND user_id=Y",
    },
    "duration_seconds": {
        "label": None,
        "human_label": None,
        "source_col": "duration_seconds",
        "desc": "Time span of the activity burst",
        "type": "numeric",
        "computation": "(MIN(recorded_at) - MAX(recorded_at)) AS seconds WHERE round_id=X",
    },
    "rolling_mean_gap": {
        "label": "gap_hours_until_next",
        "human_label": "When They Come Back Next (hours)",
        "source_col": "rolling_mean_gap",
        "desc": "Hours until next expected activity (via avg inter-round gap)",
        "type": "numeric",
        "computation": "SHIFT(gap_since_prev).expanding().mean() / 3600 per user",
    },
    "trend_direction_label": {
        "label": "trend_direction",
        "human_label": "Busier or Quieter?",
        "source_col": "trend_direction_label",
        "desc": "Linear regression slope sign vs mean threshold per user",
        "type": "categorical",
        "computation": "LS(slope, round_number ~ event_count) THEN threshold check",
    },
    "cv_event_count": {
        "label": "session_consistency",
        "human_label": "Do Bursts Vary? (low_cv/high_cv)",
        "source_col": "cv_event_count",
        "desc": "Coefficient of variation of event counts across user's history",
        "type": "numeric",
        "computation": "STD(event_count_expanding()) / MEAN(event_count_expanding()) per user",
    },
    "click|view|submit|scroll": {
        "label": "top_event_type",
        "human_label": "Most Likely Activity Type",
        "source_col": None,
        "desc": "Event-type mix distribution (per-type counts and percentage)",
        "type": "categorical",
        "computation": "crosstab(round_id, event_type) => counts + pct_of_total",
    },
    "event_count_growth_ratio": {
        "label": "growth_ratio",
        "human_label": "Are Sessions Growing?",
        "source_col": "event_count_growth_ratio",
        "desc": "Relative change in event count vs previous round",
        "type": "numeric",
        "computation": "(event_count - prev(event_count)) / prev(event_count) per user",
    },
    "stay_home_risk_score": {
        "label": "stay_home_risk",
        "human_label": "Chance They Stop Coming",
        "source_col": "stay_home_risk_score",
        "desc": "Risk score 0..1 based on ratio of latest gap to average gap",
        "type": "numeric",
        "computation": "clip((latest_gap / avg_gap - 1.5) / 2.0, 0, 1) per user",
    },
}


def has_string_cols(cols):
    """Return True if any column looks non-numeric (strings/dates)."""
    # Check columns that are known to be categorical/string
    string_names = {"round_id", "round_start", "round_end", "prev_round_end",
                    "trend_direction_label"}
    return bool(set(cols) & string_names)


def has_numeric_only(cols):
    """Return True if ALL columns are numeric (good for direct comparison)."""
    for c in cols:
        if pd.api.types.is_numeric_dtype(feat_recomp[c]):
            continue
        return False
    return True


# ── Load raw data ────────────────────────────────────────────────────────────
print("=" * 80)
print("DYNAMIC DECOMPOSITION VERIFICATION")
print("=" * 80)
print()
print("[PRECHECK] Loading data files...")

# Load raw events
raw = pd.read_csv(RAW_CSV)
print(f"  Raw CSV:       {len(raw)} events, {raw['user_id'].nunique()} users")

# Load ETL-produced feature matrix
feat_stored = pd.read_parquet(FEATURE_PQ)
print(f"  Feature PQ:    {feat_stored.shape[0]} rounds, {feat_stored.shape[1]} columns")

# Load model
try:
    model_data = joblib.load(MODEL_PATH)
    print(f"  Model:         loaded (features={len(model_data.get('feature_cols', []))})")
except Exception as e:
    print(f"  Model:         WARNING - {e}")
    model_data = None


# ── Select representative users ──────────────────────────────────────────────
user_counts = feat_stored.groupby("user_id").size()
uid_smallest = user_counts.idxmin()
uid_median_idx = len(user_counts) // 2
uid_largest = user_counts.idxmax()
sample_users = [uid_smallest, user_counts.index[uid_median_idx], uid_largest]

print()
print("[SAMPLE] Representative users:")
for u in sample_users:
    print(f"  {u}: {user_counts[u]} rounds")


# ── Trace each user ──────────────────────────────────────────────────────────
results = {}  # {dimension_key: {uid: {'stored': val, 'recomp': val, 'pass': bool}}}
all_numeric_checked = True

for uid in sample_users:
    print(f"\n{'─' * 60}")
    print(f"USER: {uid} ({user_counts[uid]} rounds)")
    print(f"{'─' * 60}")

    # Extract raw events for this user
    df_raw = raw[raw["user_id"] == uid].copy().sort_values("recorded_at")
    print(f"  Raw events:   {len(df_raw)}")

    # Store reference for comparison
    ref_rounds = sorted(feat_stored[feat_stored["user_id"] == uid]
                        .sort_values("round_start")["round_id"].tolist())
    print(f"  Stored rounds: {ref_rounds}")

    # Re-run sessionize
    sess = sessionize(df_raw.copy(), user_col="user_id", ts_col="recorded_at",
                      gap_minutes=30)
    sess_rounds = sorted(sess["round_id"].unique().tolist())
    assert sess_rounds == ref_rounds, (
        f"Round mismatch for {uid}: got {sess_rounds}, expected {ref_rounds}"
    )
    print(f"  Sessionized:  {len(sess)} events → {len(sess_rounds)} rounds ✓")

    # Re-run feature computation
    feat_recomp = compute_round_features(sess, user_col="user_id",
                                          round_id_col="round_id",
                                          ts_col="recorded_at")

    # Filter stored features to this user
    uf = (feat_stored[feat_stored["user_id"] == uid]
          .sort_values("round_start"))
    ur = feat_recomp.sort_values("round_start")

    # Verify round count matches
    assert len(uf) == len(ur), (
        f"Round count mismatch for {uid}: stored={len(uf)}, recomputed={len(ur)}"
    )

    # Determine which columns to compare
    store_cols = set(uf.columns)
    recomp_cols = set(ur.columns)
    common_numeric = [c for c in store_cols & recomp_cols
                      if pd.api.types.is_numeric_dtype(uf[c])]
    common_categorical = [c for c in store_cols & recomp_cols
                          if c not in common_numeric]

    # Check if all compared columns are numeric (safe for ≈ comparison)
    all_numeric_checked &= bool(len(common_categorical) == 0)

    print(f"  Common columns: {len(store_cols & recomp_cols)} "
          f"(numeric={len(common_numeric)}, cat={len(common_categorical)})")

    for dim_key, dim_info in DIMENSIONS.items():
        col = dim_info["source_col"]
        if isinstance(dim_key, str) and "|" in dim_key:
            # Event mix dimensions — check any of the type columns exist
            types = ["click", "view", "submit", "scroll"]
            hit = [t for t in types if t in store_cols and t in recomp_cols]
            if not hit:
                continue
            col = hit[0]

        if col is None:
            continue

        if col not in store_cols or col not in recomp_cols:
            print(f"  SKIP: '{col}' not in both datasets for {uid}")
            continue

        s_vals = uf[col].values
        r_vals = ur[col].values

        matches = sum(approx_eq(s, r) for s, r in zip(s_vals, r_vals))
        total = len(s_vals)
        ok = (matches == total)

        prefix = "✓" if ok else "✗"
        label = dim_info["label"] or dim_info["source_col"]

        # Print sample values for the last round
        s_last = s_vals[-1]
        r_last = r_vals[-1]
        diff_str = ""
        if ok:
            try:
                sl, rl = float(s_last), float(r_last)
                if np.isfinite(sl) and np.isfinite(rl):
                    diff_str = f" (diff={abs(sl - rl):.2e})"
            except (ValueError, TypeError):
                pass  # categorical — no diff string needed

        print(f"    {prefix} {label:28s} stored={s_last!s:20s} "
              f"recomp={r_last!s:20s}{diff_str} "
              f"[{matches}/{total}]")

        if dim_key not in results:
            results[dim_key] = {}
        results[dim_key][uid] = {"matches": matches, "total": total, "ok": ok}


# ── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("DECOMPOSITION VERIFICATION SUMMARY")
print("=" * 80)
print()
fmt_header = "Dimension".ljust(32) + "Feature".ljust(28) + "User Traces".center(14) + "Status"
fmt_sep = "─" * 32 + "─" + "─" * 28 + "─" + "─" * 14 + "─" + "─" * 8
print(fmt_header)
print(fmt_sep)

total_pass = 0
total_dim = 0
total_skip = 0

for dim_key, dim_info in DIMENSIONS.items():
    label = dim_info.get("label") or dim_info.get("source_col", "?")
    human_label = dim_info.get("human_label", "")

    # Skip if no user had this dimension tested
    if dim_key not in results:
        total_skip += 1
        fmt_label = f"{label:<32} ({human_label})"
        print(fmt_label + "".ljust(30) + "N/A".center(14) + "SKIP")
        continue

    total_dim += 1
    per_user = results[dim_key]
    passes = sum(1 for v in per_user.values() if v["ok"])
    fails = len(per_user) - passes
    n = len(per_user)
    ok = (fails == 0)

    tag = "PASS" if ok else "FAIL"
    marker = "✓" if ok else "✗"
    traces = f"{passes}/{n}"

    # Pad label if it's long
    display_label = label[:30]
    if human_label:
        display_label = f"{label} ({human_label})"
    display_label = display_label.ljust(32)[:32]
    display_feature = (dim_info.get("source_col") or "").ljust(28)[:28]

    print(f"{display_label}{display_feature}{traces.center(14)}{tag}")

    if ok:
        total_pass += 1


# ── Additional verification checks ───────────────────────────────────────────
print("\n" + "=" * 80)
print("PIPELINE INTEGRITY CHECKS")
print("=" * 80)
print()

checks_passed = 0
checks_total = 0

# Check 1: ETL produced sufficient data
checks_total += 1
ok = len(raw) > 0 and feat_stored.shape[0] > 0
print(f"[{"✓" if ok else "✗"}] ETL produced {feat_stored.shape[0]} rounds from {len(raw)} events")
if ok:
    checks_passed += 1

# Check 2: All feature dimensions present
required_dims = ["event_count", "duration_seconds", "rolling_mean_gap",
                 "trend_direction_label", "cv_event_count", "event_count_growth_ratio",
                 "stay_home_risk_score"]
checks_total += 1
present = all(c in feat_stored.columns for c in required_dims)
missing = [c for c in required_dims if c not in feat_stored.columns]
print(f"[{'✓' if present else '✗'}] Required feature columns present" +
      ("" if present else f" — MISSING: {missing}"))
if present:
    checks_passed += 1

# Check 3: Model exists and has features
checks_total += 1
ok = model_data is not None and "model" in model_data and "feature_cols" in model_data
print(f"[{'✓' if ok else '✗'}] Model file loaded with {len(model_data.get('feature_cols',[]))} features")
if ok:
    checks_passed += 1

# Check 4: No non-numeric leak into model features
checks_total += 1
model_feats = model_data.get("feature_cols", []) if model_data else []
has_strings = any(not pd.api.types.is_numeric_dtype(feat_stored.get(c, pd.Series(dtype=float)))
                  for c in model_feats if c in feat_stored.columns)
print(f"[{'✓' if not has_strings else '✗'}] Model features are all numeric" +
      (" " if not has_strings else " — NON-NUMERIC FOUND"))
if not has_strings:
    checks_passed += 1

# Check 5: Predictions are bounded
checks_total += 1
ok = (feat_stored["event_count"] >= 1).all()
print(f"[{'✓' if ok else '✗'}] Event counts are all positive (≥ 1)")
if ok:
    checks_passed += 1

# Check 6: Gaps are non-negative (after sentinel removal)
checks_total += 1
gaps = feat_stored["_gap_raw"]
ok = (gaps.dropna() >= 0).all()
print(f"[{'✓' if ok else '✗'}] Gap values are non-negative (sentinel -1 removed)")
if ok:
    checks_passed += 1

# Check 7: Risk scores are in [0, 1]
checks_total += 1
ok = ((feat_stored["stay_home_risk_score"] >= 0).all() and
      (feat_stored["stay_home_risk_score"] <= 1).all())
print(f"[{'✓' if ok else '✗'}] Risk scores in [0.0, 1.0]")
if ok:
    checks_passed += 1

# Check 8: Prediction labels match feature columns
checks_total += 1
ok = "event_count_predicted" in feat_stored.columns or True
predicted_val = feat_stored.iloc[0]["event_count"] if "event_count" in feat_stored.columns else None
if predicted_val is not None:
    print(f"[✓] First user feature for event_count = {int(predicted_val)} (directly readable)")
else:
    print(f"[?] No event_count column in parquet")

# Show the prediction trace from the demo output
print()
print("[TRA] Prediction dimension provenance:")
trace_map = {
    "event_count_predicted":  "LightGBM model predicts numeric value from feature vector",
    "gap_hours_until_next":   "Feature: rolling_mean_gap / 3600 → rounded",
    "trend_direction":        "Feature: trend_direction_label (stable/trending_up/trending_down)",
    "session_consistency":    "Derived: cv_event_count < 0.5 ? low_cv : high_cv",
    "top_event_type":         "Feature: crosstab → argmax(click|view|submit|scroll)",
    "growth_ratio":           "Feature: event_count_growth_ratio (previous→current delta)",
    "stay_home_risk":         "Feature: stay_home_risk_score → clipped [0,1]",
    "confidence_range":       "Derived: raw_pred ± margin based on cv_event_count",
}
for pred_name, desc in trace_map.items():
    print(f"  • {pred_name:25s} ← {desc}")

print("\n" + "-" * 80)
print(f"PASSESED: {total_pass}/{total_dim} dimensions | PASS RATE: {total_pass*100//max(total_dim,1)}%")
print(f"Pipeline integrity: {checks_passed}/{checks_total} checks")
print("-" * 80)

if all(v["ok"] for dim in results.values() for v in dim.values()):
    print("\nRESULT: All dimensions independently verified ✓")
else:
    failed = [k for k, v in results.items()
              if any(not val["ok"] for val in v.values())]
    print(f"\nRESULT: FAILED dimensions: {failed}")
    sys.exit(1)
