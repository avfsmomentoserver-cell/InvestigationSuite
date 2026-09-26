# AVFS Investigation Console

Forensic console and prediction pipeline built on the backup databases in `data/` of the `decomputation` branch.

## Contents
- `index.html`, `js/`, `styles.css`: static console (Command Center, evidence, analysis, forecast, 7 prediction pages, tools)
- `pipeline/decompose.py`: fingerprints the databases, quarantines injected, test and overwritten rows, writes `data/suite.json` and `data/series/*.json`
- `pipeline/predict.py`: 42 causal features, walk-forward models (empirical, L2 logistic, gradient boosting) vs the 0.97/m house curve, block bootstrap, Benjamini–Hochberg, calibration, backtest, controls; writes `data/predict.json`
- `pipeline/oracle_engine.py`: Python port of the browser forecast engine
- `data/`: precomputed outputs, so the console works without re-running anything

## Run
```
cd console
pip install numpy pandas scikit-learn
python pipeline/decompose.py --repo ..
python pipeline/predict.py --repo ..          # add --fast for a quicker run
python pipeline/predict.py --score rounds.csv  # score any history
python -m http.server 8000
```
Databases are opened read-only (`mode=ro&immutable=1`).

## Headline results
- 7,662 synthetic Aviator rows on Jul 3 and 257 overwritten backup rows are quarantined.
- 0 of 120 stream × target × model tests show an edge after FDR correction.
- The pipeline detects a planted +8 pp edge, and reproduces a fake z = +5.7 edge when the Jul 3 burst is included under shuffled cross-validation.
