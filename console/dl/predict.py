"""AVFS prediction pipeline.

Stages
  1 ingest     clean streams from the decomposition stage (quarantined rows excluded)
  2 features   47 strictly-causal features per round, built only from earlier rounds
  3 targets    y_m = 1[x >= m] for m in TARGETS
  4 models     house curve, empirical rate, L2 logistic, gradient boosting
  5 validate   expanding walk-forward, 5 folds, 200-round purge gap
  6 score      log-loss skill vs house, block-bootstrap z, Benjamini-Hochberg q
  7 calibrate  reliability bins and expected calibration error
  8 explain    feature/target correlation z, permutation importance
  9 backtest   EV-gated paper strategy vs a fair-game Monte Carlo fan
 10 controls   shuffled stream, planted edge + power curve, timestamp leak, burst leak
 11 deploy     final fit on all rounds, next-round probabilities, logistic weights for the browser

Usage
  python pipeline/predict.py --repo ../InvestigationSuite           # full run -> data/predict.json
  python pipeline/predict.py --repo ../InvestigationSuite --fast    # fewer bootstrap draws / controls
  python pipeline/predict.py --score rounds.csv                     # run the same tests on any history
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data"
TARGETS = [1.5, 2, 3, 5, 10, 20, 50, 100]
RTP = 0.97
WARMUP = 200
FOLDS = 5
PURGE = 200
INIT_TRAIN = 0.4
BLOCK = 250
MODELS = ["empirical", "logit", "hgb"]
MODEL_LABEL = {"house": "House curve", "empirical": "Empirical rate", "logit": "Logistic (L2)", "hgb": "Gradient boosting"}
STREAMS = ["aviator", "skyward", "skyward_deluxe", "aviator_jul29", "engine"]
EV_GATE = 0.02


def r(v, n=4):
    if v is None:
        return None
    v = float(v)
    return None if not math.isfinite(v) else round(v, n)


# ───────────────────────────── features ─────────────────────────────
def roll_mean(a: np.ndarray, w: int) -> np.ndarray:
    """mean of a[i-w .. i-1] (strictly before i); expanding before w rounds."""
    c = np.concatenate([[0.0], np.cumsum(a)])
    i = np.arange(len(a))
    lo = np.maximum(0, i - w)
    n = np.maximum(1, i - lo)
    return (c[i] - c[lo]) / n


def since_last(flag: np.ndarray) -> np.ndarray:
    """rounds since the last True strictly before i."""
    out = np.empty(len(flag))
    last = -1
    for i in range(len(flag)):
        out[i] = i - last if last >= 0 else i + 1
        if flag[i]:
            last = i
    return out


def run_below(x: np.ndarray, m: float) -> np.ndarray:
    out = np.empty(len(x))
    run = 0
    for i in range(len(x)):
        out[i] = run
        run = run + 1 if x[i] < m else 0
    return out


def build_features(x: np.ndarray, t: np.ndarray, leak: bool = False):
    """Feature matrix where row i uses only rounds < i (plus the start time of round i)."""
    n = len(x)
    lx = np.log(x)
    F: dict[str, np.ndarray] = {}
    mu = math.log(1 / RTP) + 0.0
    for k in range(1, 11):
        v = np.full(n, np.nan)
        v[k:] = lx[:-k]
        F[f"lag{k}_log"] = np.where(np.isnan(v), mu, v)
    for m in (1.5, 2, 10):
        v = np.zeros(n)
        v[1:] = (x[:-1] >= m)
        F[f"prev_ge{m:g}"] = v
    for k in (2, 3):
        v = np.zeros(n)
        v[k:] = (x[:-k] >= 2)
        F[f"lag{k}_ge2"] = v
    for m in (2, 10):
        h = (x >= m).astype(float)
        for w in (10, 50, 200):
            F[f"rate{m}_w{w}"] = roll_mean(h, w)
    for w in (20, 100):
        F[f"mean_log_w{w}"] = roll_mean(lx, w)
        F[f"std_log_w{w}"] = np.sqrt(np.maximum(0, roll_mean(lx * lx, w) - roll_mean(lx, w) ** 2))
    F["run_below2"] = run_below(x, 2)
    F["run_below1.5"] = run_below(x, 1.5)
    F["since_10x"] = np.log1p(since_last(x >= 10))
    F["since_100x"] = np.log1p(since_last(x >= 100))
    F["since_instant"] = np.log1p(since_last(x < 1.01))
    tp = np.empty(n)
    tp[0] = t[0]
    tp[1:] = t[:-1]
    hour = (tp % 86400) / 3600
    F["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    F["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = ((tp // 86400) + 4) % 7
    F["weekend"] = (dow >= 5).astype(float)
    g_prev = np.zeros(n)          # duration of round i-1 (t[i-1]-t[i-2]); known before round i
    g_prev[2:] = np.clip(t[1:-1] - t[:-2], 0, 3600)
    F["gap_prev_log"] = np.log1p(g_prev)
    new_sess = np.zeros(n, dtype=bool)
    new_sess[1:] = (t[1:] - t[:-1]) > 300
    pos = np.empty(n)
    p = 0
    for i in range(n):
        pos[i] = p
        p = 0 if (i + 1 < n and new_sess[i + 1]) else p + 1
    F["session_pos"] = np.log1p(pos)
    F["mean_log_all"] = roll_mean(lx, 10**9)
    F["rate2_all"] = roll_mean((x >= 2).astype(float), 10**9)
    F["max_log_w50"] = pd.Series(lx).shift(1).rolling(50, min_periods=1).max().fillna(0).to_numpy()
    F["min_log_w10"] = pd.Series(lx).shift(1).rolling(10, min_periods=1).min().fillna(0).to_numpy()
    F["ewm_log_a10"] = pd.Series(lx).shift(1).ewm(alpha=0.1).mean().fillna(mu).to_numpy()
    F["ewm_ge2_a05"] = pd.Series((x >= 2).astype(float)).shift(1).ewm(alpha=0.05).mean().fillna(0.485).to_numpy()
    F["color_run"] = run_below(x, 2) - run_below(-x, -2)  # signed run: + below 2x, - above
    if leak:
        g_now = np.zeros(n)
        g_now[1:] = np.clip(t[1:] - t[:-1], 0, 3600)
        F["LEAK_gap_now"] = np.log1p(g_now)
    names = list(F)
    X = np.column_stack([F[k] for k in names]).astype(float)
    return X, names


def house_p(m: float) -> float:
    return min(1 - 1e-4, RTP / m)


def ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ───────────────────────────── validation ─────────────────────────────
def folds(n: int):
    start = WARMUP + int((n - WARMUP) * INIT_TRAIN)
    edges = np.linspace(start, n, FOLDS + 1).astype(int)
    return [(WARMUP, max(WARMUP + 50, edges[k] - PURGE), edges[k], edges[k + 1]) for k in range(FOLDS)]


class Standard:
    def fit(self, X):
        self.m = X.mean(0)
        self.s = X.std(0)
        self.s[self.s < 1e-9] = 1
        return self

    def __call__(self, X):
        return (X - self.m) / self.s


def fit_predict(model: str, Xtr, ytr, Xte, seed=0):
    base = ytr.mean()
    if model == "empirical" or ytr.sum() < 5 or ytr.sum() > len(ytr) - 5:
        return np.full(len(Xte), np.clip(base, 1e-4, 1 - 1e-4)), None
    if model == "logit":
        sc = Standard().fit(Xtr)
        lr = LogisticRegression(C=0.05, max_iter=400)
        lr.fit(sc(Xtr), ytr)
        return lr.predict_proba(sc(Xte))[:, 1], (sc, lr)
    if model == "hgb":
        h = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                           n_iter_no_change=15, random_state=seed)
        h.fit(Xtr, ytr)
        return h.predict_proba(Xte)[:, 1], h
    raise ValueError(model)


def block_z(d: np.ndarray, B: int, rng) -> tuple[float, float]:
    n = len(d)
    if n < 2 * BLOCK:
        se = d.std(ddof=1) / math.sqrt(max(1, n))
        return (d.mean() / se if se > 0 else 0.0), se
    nb = n // BLOCK
    bm = d[: nb * BLOCK].reshape(nb, BLOCK).mean(1)
    idx = rng.integers(0, nb, size=(B, nb))
    boots = bm[idx].mean(1)
    se = boots.std(ddof=1)
    return (d.mean() / se if se > 0 else 0.0), se


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def auc(y, p):
    try:
        return roc_auc_score(y, p) if 0 < y.sum() < len(y) else None
    except Exception:
        return None


def reliability(p, y, bins=10):
    if len(p) == 0:
        return [], None
    qs = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return [{"p": r(p.mean(), 5), "o": r(y.mean(), 5), "n": int(len(p))}], r(abs(p.mean() - y.mean()), 5)
    b = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    out, ece = [], 0.0
    for k in range(len(qs) - 1):
        m = b == k
        if m.sum() == 0:
            continue
        pm, om = p[m].mean(), y[m].mean()
        se = math.sqrt(max(pm * (1 - pm), 1e-9) / m.sum())
        out.append({"p": r(pm, 5), "o": r(om, 5), "n": int(m.sum()), "se": r(se, 5)})
        ece += m.sum() / len(p) * abs(pm - om)
    return out, r(ece, 5)


def walk_forward(X, x, models=MODELS, targets=TARGETS, B=300, seed=0, keep_oof=True):
    n = len(x)
    F = folds(n)
    rng = np.random.default_rng(seed)
    test_idx = np.concatenate([np.arange(a, b) for _, _, a, b in F])
    res = {}
    oof = {}
    for m in targets:
        y = (x >= m).astype(int)
        yt = y[test_idx]
        ph = np.full(len(test_idx), house_p(m))
        llh = ll(ph, yt)
        res[m] = {"house": {"ll": r(llh.mean(), 5), "brier": r(((ph - yt) ** 2).mean(), 5), "auc": None, "skill": 0.0, "z": 0.0,
                            "rate_obs": r(yt.mean(), 5), "rate_house": r(house_p(m), 5)}}
        oof[m] = {"house": ph, "y": yt}
        for md in models:
            p = np.empty(len(test_idx))
            fz = []
            off = 0
            for tr_lo, tr_hi, te_lo, te_hi in F:
                pp, _ = fit_predict(md, X[tr_lo:tr_hi], y[tr_lo:tr_hi], X[te_lo:te_hi], seed)
                p[off: off + (te_hi - te_lo)] = pp
                dfold = llh[off: off + (te_hi - te_lo)] - ll(pp, y[te_lo:te_hi])
                fz.append(r(dfold.mean() * 1000, 3))
                off += te_hi - te_lo
            d = llh - ll(p, yt)
            z, se = block_z(d, B, rng)
            rel, ece = reliability(p, yt)
            res[m][md] = {"ll": r(ll(p, yt).mean(), 5), "brier": r(((p - yt) ** 2).mean(), 5), "auc": r(auc(yt, p), 4),
                          "skill": r(d.mean() * 1000, 4), "se": r(se * 1000, 4), "z": r(z, 3), "p": norm_sf(z),
                          "fold_skill": fz, "ece": ece, "rel": rel}
            if keep_oof:
                oof[m][md] = p
    return res, oof, test_idx, F


# ───────────────────────────── backtest ─────────────────────────────
def backtest(oof, x_test, rng, sims=500, points=300):
    out = {}
    T = np.array(TARGETS)
    n = len(x_test)
    stride = max(1, n // points)
    for md in ["house"] + MODELS:
        P = np.column_stack([oof[m][md] for m in TARGETS])
        ev = P * T[None, :] - 1
        k = ev.argmax(1)
        bet = ev[np.arange(n), k] > EV_GATE
        tg = T[k]
        win = x_test >= tg
        pnl = np.where(bet, np.where(win, tg - 1, -1.0), 0.0)
        eq = np.cumsum(pnl)
        nb = int(bet.sum())
        # fair-game null: same bet times and targets, wins drawn at RTP/m
        pw = np.where(bet, RTP / tg, 0)
        sim = np.zeros((sims, len(range(0, n, stride))))
        if nb:
            U = rng.random((sims, n))
            spnl = np.where(bet[None, :], np.where(U < pw[None, :], tg[None, :] - 1, -1.0), 0.0)
            sim = np.cumsum(spnl, 1)[:, ::stride]
        fan = np.percentile(sim, [5, 25, 50, 75, 95], axis=0) if nb else np.zeros((5, sim.shape[1]))
        final = eq[-1] if n else 0
        pct_null = float((sim[:, -1] < final).mean()) if nb else None
        peak = np.maximum.accumulate(np.concatenate([[0], eq]))[1:]
        dd = float((peak - eq).max()) if n else 0
        stakes = np.where(bet, 1.0, 0)
        roi = final / nb if nb else None
        tgt_mix = {f"{m:g}": int(((tg == m) & bet).sum()) for m in TARGETS if ((tg == m) & bet).sum()}
        out[md] = {"bets": nb, "pnl": r(final, 2), "roi": r(roi, 4), "max_dd": r(dd, 2), "pct_vs_null": r(pct_null, 3),
                   "hit": r(win[bet].mean(), 4) if nb else None, "targets": tgt_mix,
                   "eq": [r(v, 1) for v in eq[::stride]], "fan": [[r(v, 1) for v in row] for row in fan]}
    # always-2x baseline
    pnl2 = np.where(x_test >= 2, 1.0, -1.0)
    out["always2x"] = {"bets": int(n), "pnl": r(pnl2.sum(), 2), "roi": r(pnl2.mean(), 4), "eq": [r(v, 1) for v in np.cumsum(pnl2)[::stride]]}
    out["stride"] = stride
    return out


# ───────────────────────────── explain ─────────────────────────────
def feature_corr(X, names, x):
    Xw, xw = X[WARMUP:], x[WARMUP:]
    out = {}
    for m in (2, 10):
        y = (xw >= m).astype(float)
        yc = y - y.mean()
        Xc = Xw - Xw.mean(0)
        den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
        rr = np.where(den > 0, (Xc * yc[:, None]).sum(0) / np.where(den > 0, den, 1), 0)
        out[f"{m:g}"] = [r(v * math.sqrt(len(y)), 2) for v in rr]
    return out


def perm_importance(X, names, x, seed=0):
    n = len(x)
    lo, hi = WARMUP, WARMUP + int((n - WARMUP) * 0.8)
    y = (x >= 2).astype(int)
    h = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300,
                                       l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                       n_iter_no_change=15, random_state=seed).fit(X[lo:hi], y[lo:hi])
    pi = permutation_importance(h, X[hi:], y[hi:], scoring="neg_log_loss", n_repeats=5, random_state=seed)
    order = np.argsort(-pi.importances_mean)
    return [{"f": names[i], "imp": r(pi.importances_mean[i] * 1000, 4), "sd": r(pi.importances_std[i] * 1000, 4)} for i in order]


# ───────────────────────────── deploy ─────────────────────────────


def next_round(x, t, names_ref=None):
    x2 = np.concatenate([x, [1.0]])
    t2 = np.concatenate([t, [t[-1] + 10]])
    X2, names = build_features(x2, t2)
    Xtr, xnext = X2[WARMUP:-1], X2[-1:]
    xs = x[WARMUP:]
    out = {"features": {k: r(v, 5) for k, v in zip(names, xnext[0])}, "targets": []}
    weights = {}
    for m in TARGETS:
        y = (xs >= m).astype(int)
        row = {"m": m, "house": r(house_p(m), 5), "empirical": r(y.mean(), 5)}
        pl, obj = fit_predict("logit", Xtr, y, xnext)
        row["logit"] = r(pl[0], 5)
        if obj is not None:
            sc, lr = obj
            weights[f"{m:g}"] = {"mean": [r(v, 6) for v in sc.m], "scale": [r(v, 6) for v in sc.s],
                                 "coef": [r(v, 6) for v in lr.coef_[0]], "b": r(lr.intercept_[0], 6)}
        ph, _ = fit_predict("hgb", Xtr, y, xnext)
        row["hgb"] = r(ph[0], 5)
        best = max(("empirical", "logit", "hgb"), key=lambda k: row[k] or 0)
        row["ev_best"] = r(row[best] * m - 1, 4)
        row["ev_best_model"] = best
        out["targets"].append(row)
    return out, weights, names


# ───────────────────────────── controls ─────────────────────────────
def fair_stream(n, rng, planted=0.0):
    """Fair crash game (P[x>=m]=RTP/m on the 0.01 grid). With planted>0, after a round under 1.5x the next
    round reaches 2x with probability 0.485+planted (conditional shape kept)."""
    U = rng.random(n)
    x = np.floor(np.maximum(1.0, RTP / (1 - U)) * 100) / 100
    if planted:
        for i in range(1, n):
            if x[i - 1] < 1.5:
                want = rng.random() < house_p(2) + planted
                u = rng.random()
                if want:   # conditional on >=2: x = 2/(1-u')
                    x[i] = math.floor(max(2.0, 2.0 / (1 - u)) * 100) / 100
                else:      # conditional on <2: inverse cdf restricted to [1, 2)
                    q = u * (1 - house_p(2))
                    x[i] = math.floor(max(1.0, min(1.99, RTP / (1 - q))) * 100) / 100
    return x


def quick_skill(X, x, m=2, models=("logit", "hgb"), B=200, seed=0):
    res, _, _, _ = walk_forward(X, x, models=list(models), targets=[m], B=B, seed=seed, keep_oof=False)
    return {md: {"skill": res[m][md]["skill"], "z": res[m][md]["z"], "auc": res[m][md]["auc"]} for md in models}


def controls(streams, burst, fast, log):
    rng = np.random.default_rng(11)
    out = {}
    av = streams["aviator"]
    x, t = av["x"], av["t"]
    # negative control: shuffled order
    log("  control: shuffled")
    xs = rng.permutation(x)
    Xs, _ = build_features(xs, t)
    out["shuffled"] = {"n": int(len(xs)), "m2": quick_skill(Xs, xs, 2), "m10": quick_skill(Xs, xs, 10)}
    # timestamp leak: include the duration of the round being predicted
    log("  control: timestamp leak")
    Xl, _ = build_features(x, t, leak=True)
    out["timestamp_leak"] = {"n": int(len(x)), "m2": quick_skill(Xl, x, 2), "m10": quick_skill(Xl, x, 10),
                             "clean_m2": None}
    # burst leak: re-insert the Jul 3 rows and test on the window that contains them
    if burst is not None:
        log("  control: burst leak")
        bt, bx = burst["t"], burst["x"]
        mt = np.concatenate([t, bt])
        mx = np.concatenate([x, bx])
        o = np.argsort(mt, kind="stable")
        mt, mx = mt[o], mx[o]
        Xm, _ = build_features(mx, mt)
        Xc, _ = build_features(x, t)
        b0 = int(np.searchsorted(mt, bt.min()))
        c0 = int(np.searchsorted(t, bt.min()))
        win = 10000
        rows = {}
        for key, XX, xx, s0 in (("with_burst", Xm, mx, b0), ("clean", Xc, x, c0)):
            te_lo, te_hi = s0 - 500, min(len(xx), s0 - 500 + win)
            tr_hi = te_lo - PURGE
            y = (xx >= 2).astype(int)
            yt = y[te_lo:te_hi]
            llh = ll(np.full(len(yt), house_p(2)), yt)
            rr = {}
            for md in ("logit", "hgb"):
                p, _ = fit_predict(md, XX[WARMUP:tr_hi], y[WARMUP:tr_hi], XX[te_lo:te_hi])
                d = llh - ll(p, yt)
                z, _ = block_z(d, 200, rng)
                rr[md] = {"skill": r(d.mean() * 1000, 3), "z": r(z, 2), "auc": r(auc(yt, p), 4)}
                if md == "logit":
                    rr["trace"] = [r(v, 3) for v in pd.Series(p).rolling(100, min_periods=1).mean().to_numpy()[::50]]
                    rr["obs"] = [r(v, 3) for v in pd.Series(yt).rolling(100, min_periods=1).mean().to_numpy()[::50]]
            rows[key] = {"test_rows": int(te_hi - te_lo), "train_rows": int(tr_hi - WARMUP), **rr}
        # the common mistake: shuffled k-fold on the contaminated stream (burst rows land in train and test)
        kf = {}
        is_b = np.zeros(len(mx), dtype=bool)
        is_b[np.argsort(o)[len(x):]] = True
        for key, XX, xx, ib in (("with_burst", Xm, mx, is_b), ("clean", Xc, x, np.zeros(len(x), dtype=bool))):
            idx = np.arange(WARMUP, len(xx))
            perm = np.random.default_rng(4).permutation(idx)
            parts = np.array_split(perm, 5)
            y = (xx >= 2).astype(int)
            p = np.empty(len(xx))
            for k in range(5):
                te = parts[k]
                tr = np.concatenate([parts[j] for j in range(5) if j != k])
                p[te], _ = fit_predict("hgb", XX[tr], y[tr], XX[te])
            d = ll(np.full(len(idx), house_p(2)), y[idx]) - ll(p[idx], y[idx])
            z, _ = block_z(d, 200, rng)
            ibi = ib[idx]
            kf[key] = {"skill": r(d.mean() * 1000, 3), "z": r(z, 2), "auc": r(auc(y[idx], p[idx]), 4),
                       "skill_burst_rows": r(d[ibi].mean() * 1000, 2) if ibi.any() else None,
                       "skill_other_rows": r(d[~ibi].mean() * 1000, 3), "burst_rows": int(ibi.sum()),
                       "acc_burst_rows": r(((p[idx][ibi] >= 0.5) == y[idx][ibi]).mean(), 4) if ibi.any() else None}
        rows["random_kfold"] = kf
        out["burst_leak"] = rows
    # planted edge: positive control + power curve
    log("  control: planted edge / power")
    n = len(x)
    reps = 2 if fast else 3
    effects = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    power = []
    for e in effects:
        zs = []
        for k in range(reps):
            xp = fair_stream(n, np.random.default_rng(100 + k + int(e * 1000)), planted=e)
            Xp, _ = build_features(xp, t)
            q = quick_skill(Xp, xp, 2, models=("logit",), B=150, seed=k)
            zs.append(q["logit"]["z"])
        power.append({"effect": e, "z": [r(v, 2) for v in zs], "z_mean": r(float(np.mean(zs)), 2),
                      "detected": int(sum(1 for v in zs if v >= 3)), "reps": reps,
                      "frac_rounds": r(float((fair_stream(20000, np.random.default_rng(5)) < 1.5).mean()), 3)})
        log(f"    effect {e:.2f}: z {np.round(zs, 2).tolist()}")
    out["power"] = power
    return out


# ───────────────────────────── orchestration ─────────────────────────────
def load_streams(repo: Path):
    import decompose as D  # noqa: WPS433
    dec, series, _ = D.decompose(repo / "data")
    streams = {}
    for name in STREAMS:
        g = series[name]
        streams[name] = {"x": g.x.to_numpy(dtype=float), "t": D.epoch_s(g.t).astype(float)}
    b = D.EXTRA.get("burst_df")
    burst = {"x": b.x.to_numpy(dtype=float), "t": D.epoch_s(b.t).astype(float)} if b is not None else None
    return streams, burst, dec


def bh(pvals):
    p = np.array(pvals)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = o[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def run(repo: Path, fast: bool):
    T0 = time.time()
    stages = []
    B = 150 if fast else 400

    def log(msg):
        print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)

    def stage(name, t0, detail):
        stages.append({"stage": name, "sec": r(time.time() - t0, 2), "detail": detail})

    t0 = time.time()
    log("ingest")
    streams, burst, dec = load_streams(repo)
    h = hashlib.sha256()
    for s in STREAMS:
        h.update(streams[s]["x"].tobytes())
    data_hash = h.hexdigest()[:16]
    stage("ingest", t0, {s: int(len(streams[s]["x"])) for s in STREAMS})

    per, oofs, feats = {}, {}, None
    t_feat = t_val = t_bt = t_exp = t_dep = 0.0
    for s in STREAMS:
        x, t = streams[s]["x"], streams[s]["t"]
        a = time.time()
        X, names = build_features(x, t)
        feats = names
        t_feat += time.time() - a
        log(f"{s}: {len(x):,} rounds, {X.shape[1]} features")
        a = time.time()
        res, oof, test_idx, F = walk_forward(X, x, B=B, seed=1)
        t_val += time.time() - a
        a = time.time()
        bt = backtest(oof, x[test_idx], np.random.default_rng(3), sims=300 if fast else 600)
        t_bt += time.time() - a
        a = time.time()
        corr = feature_corr(X, names, x)
        imp = perm_importance(X, names, x) if (s in ("aviator", "skyward", "skyward_deluxe") and not fast) or s == "aviator" else None
        t_exp += time.time() - a
        a = time.time()
        nxt, weights, _ = next_round(x, t)
        t_dep += time.time() - a
        per[s] = {"n": int(len(x)), "folds": [{"train": [int(a_), int(b_)], "test": [int(c_), int(d_)]} for a_, b_, c_, d_ in F],
                  "test_rows": int(len(test_idx)), "scores": {f"{m:g}": v for m, v in res.items()}, "backtest": bt,
                  "corr": corr, "importance": imp, "next": nxt, "weights": weights,
                  "last": {"x": [r(v, 2) for v in x[-WARMUP - 5:]], "t": [int(v) for v in t[-WARMUP - 5:]]}}
        log(f"  best z: " + ", ".join(f"{m:g}x {max(res[m][md]['z'] for md in MODELS):+.2f}" for m in TARGETS))
    stage("features", time.time() - t_feat, {"count": len(feats), "warmup": WARMUP})
    stages[-1]["sec"] = r(t_feat, 2)
    stages.append({"stage": "validate", "sec": r(t_val, 2), "detail": {"folds": FOLDS, "purge": PURGE, "init_train": INIT_TRAIN, "bootstrap": B, "block": BLOCK}})
    # BH across every stream × target × model
    keys, pv = [], []
    for s in STREAMS:
        for m in TARGETS:
            for md in MODELS:
                keys.append((s, f"{m:g}", md))
                pv.append(per[s]["scores"][f"{m:g}"][md]["p"])
    q = bh(pv)
    board = []
    for (s, m, md), qq, pp in zip(keys, q, pv):
        sc = per[s]["scores"][m][md]
        sc["q"] = r(qq, 4)
        sc["p"] = r(pp, 5)
        verdict = "edge" if (qq < 0.05 and sc["skill"] > 0) else ("noise" if sc["skill"] > 0 else "worse")
        sc["verdict"] = verdict
        board.append({"stream": s, "m": m, "model": md, "skill": sc["skill"], "z": sc["z"], "q": sc["q"], "auc": sc["auc"],
                      "ece": sc["ece"], "verdict": verdict})
    board.sort(key=lambda d: -(d["z"] or -99))
    stages.append({"stage": "score", "sec": 0.01, "detail": {"tests": len(pv), "fdr": 0.05, "edges": int(sum(1 for b in board if b["verdict"] == "edge"))}})
    stages.append({"stage": "backtest", "sec": r(t_bt, 2), "detail": {"ev_gate": EV_GATE, "sims": 300 if fast else 600}})
    stages.append({"stage": "explain", "sec": r(t_exp, 2), "detail": {"perm_repeats": 5}})

    a = time.time()
    log("controls")
    ctl = controls(streams, burst, fast, log)
    stages.append({"stage": "controls", "sec": r(time.time() - a, 2), "detail": {k: True for k in ctl}})
    stages.append({"stage": "deploy", "sec": r(t_dep, 2), "detail": {"weights": "logistic per stream × target"}})

    bundle = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data_hash": data_hash,
              "config": {"targets": TARGETS, "rtp": RTP, "warmup": WARMUP, "folds": FOLDS, "purge": PURGE, "init_train": INIT_TRAIN,
                         "block": BLOCK, "bootstrap": B, "ev_gate": EV_GATE, "models": MODELS, "model_label": MODEL_LABEL,
                         "logit": {"C": 0.05}, "hgb": {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 300, "l2": 1.0},
                         "fast": fast},
              "features": feats, "streams": per, "leaderboard": board, "controls": ctl, "stages": stages,
              "runtime_sec": r(time.time() - T0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "predict.json", "w") as f:
        json.dump(bundle, f, separators=(",", ":"), default=float)
    log(f"wrote {OUT / 'predict.json'} {(OUT / 'predict.json').stat().st_size:,} bytes; edges {stages[3]['detail']['edges']}")


def score_file(path: Path):
    txt = path.read_text()
    vals = []
    for tok in txt.replace(",", "\n").replace(";", "\n").split():
        try:
            v = float(tok.strip().rstrip("xX"))
            if v >= 1:
                vals.append(v)
        except ValueError:
            pass
    x = np.array(vals)
    if len(x) < WARMUP + 1000:
        print(f"need at least {WARMUP + 1000} rounds, got {len(x)}")
        return
    t = np.arange(len(x)) * 20.0
    X, _ = build_features(x, t)
    res, _, _, _ = walk_forward(X, x, B=200, keep_oof=False)
    print(f"{len(x):,} rounds  (no timestamps: hour/gap features are constant)")
    print(f"{'target':>7} {'model':>10} {'skill mnat':>11} {'z':>7} {'auc':>7}")
    for m in TARGETS:
        for md in MODELS:
            s = res[m][md]
            print(f"{m:>6g}x {md:>10} {s['skill']:>11.3f} {s['z']:>7.2f} {s['auc'] if s['auc'] is not None else float('nan'):>7.3f}")
    print("edge needs z >= 3 after correcting for the 24 tests above")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="checkout of InvestigationSuite@decomputation")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--score", help="score any multiplier history (one value per line or CSV)")
    a = ap.parse_args()
    if a.score:
        score_file(Path(a.score))
    elif a.repo:
        run(Path(a.repo), a.fast)
    else:
        ap.error("--repo or --score required")


if __name__ == "__main__":
    main()
