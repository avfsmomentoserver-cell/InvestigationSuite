# Forecasting Rounds — Repo Skeleton

This repository is a starter kit for building a forecasting platform from recorded rounds (time-separated recording sessions).

Contents:
- src/: core python package (sessionize, ETL, feature engineering, training, serving)
- agents/: background agents for ingestion, retrain and monitoring
- examples/: sample data and demo script
- docs/: handbook, runbooks and onboarding
- terraform/: IaC to provision a Debian VM (example)
- docker/: Dockerfiles and compose for local infra
- .devcontainer/: VS Code devcontainer
- .github/: CI pipeline stubs
- tests/: unit tests

Quickstart (local prototype)
1. Create virtualenv and install:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
2. Run demo (example):
   python examples/demo.py
3. Run API server locally:
   uvicorn src.forecasting.serve:app --reload --port 8000

For production deployment and scaling guidance, see docs/BOOK.md and docs/ops/terraform.md

## Data Deduplication

Duplicate events silently inflate metrics if they survive ingestion. The pipeline applies four layers of dedup protection:

1. **Ingestion** (`src/forecasting/etl.py`): After reading CSV, all exact duplicate rows are dropped (keep-first policy). Rows with null `user_id` or `recorded_at` are also filtered out.

2. **Sessionization** (`src/forecasting/sessionize.py`): A pre-filter removes any remaining NA/null rows in critical columns before gap computation (defense-in-depth).

3. **Feature computation** (`src/forecasting/features.py`): Asserts each `round_id` maps to exactly one `user_id` — raises `AssertionError` if a multi-user collision is detected.

4. **Training** (`src/forecasting/train.py`): Validates no duplicate feature rows exist in the parquet before splitting into TimeSeriesSplit folds; logs fold-level sample counts.

**Log format:** The ingestion layer emits `[DEDUP] input_rows=N -> after_dedup=M -> after_na_filter=K -> dropped=D (X duplicates + Y null-key rows)`. Use these markers in logs to verify data quality at runtime. Duplicate detection uses a keep-first policy on full-row exact matches across all columns.
