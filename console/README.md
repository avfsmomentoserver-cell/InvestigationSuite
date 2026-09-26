# AVFS Investigation Console

Forensic console and prediction pipeline built on the backup databases in `data/` of the `decomputation` branch.

## Contents
- `index.html`, `js/`, `styles.css`: static console (Command Center, evidence, analysis, forecast, 7 prediction pages, tools)
- `pipeline/decompose.py`: fingerprints the databases, quarantines injected, test and overwritten rows, writes `data/suite.json` and `data/series/*.json`
- `pipeline/predict.py`: 42 causal features, walk-forward models (empirical, L2 logistic, gradient boosting) vs the 0.97/m house curve, block bootstrap, Benjamini–Hochberg, calibration, backtest, controls; writes `data/predict.json`
- `pipeline/platform_features.py`: features from the Momento Platform Book chapter 18 catalogue (tape integrity, CUSUM fingerprint, source comparator, signal strip, survival ETA, experiment registry); writes `data/platform.json`. With `--book <clone>` it also indexes the book into `data/book.json`, which is not committed here because the book repo is private
- `pipeline/oracle_engine.py`: Python port of the browser forecast engine
- `data/`: precomputed outputs, so the console works without re-running anything

## Run
```
cd console
pip install numpy pandas scikit-learn
python pipeline/decompose.py --repo ..
python pipeline/predict.py --repo ..          # add --fast for a quicker run
python pipeline/predict.py --score rounds.csv  # score any history
python pipeline/platform_features.py --repo .. --book ../../momento-platform-book
python -m http.server 8000
```
Databases are opened read-only (`mode=ro&immutable=1`).

## Headline results
- 7,662 synthetic Aviator rows on Jul 3 and 257 overwritten backup rows are quarantined.
- 0 of 120 stream × target × model tests show an edge after FDR correction.
- The pipeline detects a planted +8 pp edge, and reproduces a fake z = +5.7 edge when the Jul 3 burst is included under shuffled cross-validation.
- Platform book features: 18 of 38 built, 6 partial, 14 need the live platform. Tape integrity scores the Jul 3 burst 0.00 vs 0.997 for clean Aviator; every ETA threshold is memoryless; 0 of the 14 dashboard signals survive correction; the CUSUM fingerprint flags the spliced burst.
