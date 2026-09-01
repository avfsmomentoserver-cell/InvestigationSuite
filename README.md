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
