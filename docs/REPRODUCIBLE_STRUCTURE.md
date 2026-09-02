# =============================================================================
# Reproducible Forecasting Rounds — Project Structure
# =============================================================================
# This document defines the canonical, reproducible structure of the
# InvestigationSuite repository. Every component is version-pinned,
# seed-controlled, and documented so that any team member (or CI system)
# can recreate an identical environment and produce byte-identical outputs.
#
# REPRODUCTION CHECKLIST
# ----------------------
# 1. Clone repo → `git clone <repo-url>`
# 2. Create venv   → `make install`
# 3. Run tests     → `make test`
# 4. Run ETL       → `python examples/demo.py`
# 5. Train model   → `python notebooks/forecasting_mvp.py`
# 6. Serve API     → `make run`
#
# DETERMINISM GUARANTEES
# ----------------------
# • All Python packages are pinned to exact versions in requirements.txt
# • Random seeds are set explicitly in train.py (seed=42)
# • Sessionize uses stable sort with row-index tiebreaker (no randomness)
# • Parquet I/O uses pyarrow engine with fixed compression (snappy)
# • No network calls during build/test/train — all data is local
#
# DATA FLOW DIAGRAM
# -----------------
# CSV input ──► ETL (dedup + NA filter) ──► sessionize ──► features
#                                                        │
#                                        ┌──────────────┤
#                                        ▼              ▼
#                                   train.py        serve.py
#                                        │              │
#                                        ▼              ▼
#                                 baseline.joblib    /v1/forecast
#
# DIRECTORY LAYOUT
# ----------------
# .
# ├── src/                          # Core Python package
# │   ├── forecasting/
# │   │   ├── __init__.py           # Package marker
# │   │   ├── etl.py                # Ingestion: dedup, NA filter, sessionize
# │   │   ├── sessionize.py         # Deterministic round assignment
# │   │   ├── features.py           # 8-dimension feature engineering
# │   │   ├── train.py              # LightGBM training with TimeSeriesSplit
# │   │   └── serve.py              # FastAPI prediction endpoint
# │   └── agents/                   # Background workers (optional)
# │       ├── agent_manager.py
# │       ├── data_ingest.py
# │       ├── monitor.py
# │       └── retrain.py
# │
# ├── tests/                        # Unit + integration tests
# │   ├── test_sessionize.py        # Basic sessionization spec
# │   ├── test_sessionize_extended.py  # Edge cases, determinism checks
# │   ├── test_features.py          # Feature computation validation
# │   ├── test_dedup_pipeline.py    # End-to-end ETL dedup verification
# │   └── test_new_dimensions.py    # New dimension regression tests
# │
# ├── examples/                     # Demo scripts & sample data
# │   ├── demo.py                   # Full ETL pipeline demo
# │   ├── round_eye_view.py         # Round inspection utility
# │   ├── sample_recordings.csv     # Small sample dataset
# │   └── scale_recordings.csv      # Larger sample dataset
# │
# ├── notebooks/                    # Jupyter notebooks for exploration
# │   └── forecasting_mvp.py        # MVP training script
# │
# ├── models/                       # Persisted artifacts
# │   └── baseline.joblib           # Trained LightGBM model
# │
# ├── data/                         # Raw / intermediate data
# │   ├── avfs.db.*                 # SQLite backups (versioned)
# │   └── __init__.py
# │
# ├── docker/                       # Container definitions
# │   ├── Dockerfile
# │   └── docker-compose.yml
# │
# ├── terraform/                    # Infrastructure as Code
# │   ├── main.tf
# │   ├── variables.tf
# │   └── cloud-init/
# │       └── cloud-init.yaml
# │
# ├── docs/                         # Documentation
# │   ├── BOOK.md                   # Handbook
# │   ├── newbie-glossary.md
# │   ├── education/
# │   │   └── onboarding.md
# │   └── ops/
# │       └── terraform.md
# │
# ├── api/                          # API SDKs
# │   └── sdk/
# │       └── python/
# │           └── forecast_sdk.py
# │
# ├── conftest.py                   # pytest configuration
# ├── Makefile                      # Automation targets
# ├── requirements.txt              # Pinned dependency list
# ├── LICENSE
# └── README.md
#
# VERSION PINNING
# ---------------
# See requirements.txt for exact versions. To update:
#   pip freeze > requirements.txt
# Then commit with message: "chore: bump pinned dependencies"
#
# SEED CONFIGURATION
# ------------------
# All random operations use explicit seeds:
#   - numpy.random.seed(42)
#   - sklearn.model_selection.TimeSeriesSplit (deterministic by design)
#   - lightgbm.train(params={'seed': 42})
#
# PARQUET OUTPUT SPEC
# -------------------
# Engine : pyarrow
# Compression : snappy
# Row group size : auto
# Index : False (round_id is the logical key)
#
# SESSIONIZE DETERMINISM PROOF
# ----------------------------
# The sessionize function guarantees identical output for identical input
# because:
#   1. Input rows are sorted by (user_col, ts_col, _seq) where _seq is the
#      original row index — this is a total order, no ties remain.
#   2. Gap comparison uses strict > (not >=), so events exactly gap_minutes
#      apart stay in the same round.
#   3. Cumulative sum of boolean flags is order-dependent but the sort above
#      makes the order deterministic.
#   4. Aggregation (min/max/count) is commutative — order doesn't matter.
#
# This mirrors a SQL window-function implementation:
#   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY recorded_at, rowid)
#
# =============================================================================
