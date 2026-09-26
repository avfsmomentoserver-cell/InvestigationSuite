#!/usr/bin/env python3
"""Momento Platform Book features, computed on the clean InvestigationSuite streams.

Implements the data side of these catalogue entries (momento-platform-book, chapter 18):
  F-01 Tape Integrity Score      F-03 Source fingerprinting (CUSUM)   F-10 Cross-source comparator
  F-11 Signal significance strip F-26 ETA Board + memorylessness test F-27 Hazard timeline
  F-34 Experiment Registry (entries for every family run by the suite)
and renders the book itself (chapters + appendices) for the in-app reader and search.

Usage:
  python pipeline/platform_features.py --repo ../InvestigationSuite [--book ../momento-platform-book]
Writes data/platform.json (and data/book.json when --book is given).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict as PR  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data"
RTP = PR.RTP
STREAMS = PR.STREAMS
SESSION_GAP = 300
r = PR.r


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def house_p(m):
    return min(1.0, RTP / m)


def bh(p):
    return PR.bh(p) if len(p) else np.array([])


# ───────────────────────── F-01 Tape Integrity ─────────────────────────
LOW = 1.2
P_LOW = 1 - RTP / LOW  # P(x < 1.2) under the house curve


def duration_model(x, t):
    """Round timestamps are written when a round ends, so the interval before round i grows with log x[i]
    (a 50x flight takes ~3x as long as a 1.5x one). Fit dt ~ a + b log x on normal-cadence intervals."""
    dt = np.diff(t)
    lx = np.log(x[1:])
    m = (dt > 0) & (dt < SESSION_GAP)
    if m.sum() < 50:
        return 0.0, 0.0
    A = np.column_stack([np.ones(m.sum()), lx[m]])
    a, b = np.linalg.lstsq(A, dt[m], rcond=None)[0]
    return float(a), float(b)


def integrity(x, t, label=None, model=None):
    a, b = model if model is not None else duration_model(x, t)
    dt = np.diff(t, prepend=t[0])
    sid = np.cumsum(dt > SESSION_GAP)
    rows = []
    for k in np.unique(sid):
        mk = np.where(sid == k)[0]
        xs, ts = x[mk], t[mk]
        n = len(xs)
        d = np.diff(ts)
        med = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
        gaps, miss = [], 0
        if med > 0:
            expect = a + b * np.log(xs[1:]) if b else np.full(len(d), med)
            excess = d - expect
            for j in np.where(excess > 1.5 * med)[0]:
                em = int(round(excess[j] / med))
                if em >= 1:
                    miss += em
                    gaps.append({"at": int(mk[j + 1]), "dt": r(float(d[j]), 1), "expect": r(float(expect[j]), 1), "est_missing": em})
        completeness = n / (n + miss)
        lo = int((xs < LOW).sum())
        share = lo / n
        z = (share - P_LOW) / math.sqrt(P_LOW * (1 - P_LOW) / n)
        # low-share component: 1 inside |z|<=2, decays beyond
        c_low = 1.0 if abs(z) <= 2 else math.exp(-(abs(z) - 2) / 2)
        comps = {"completeness": completeness, "low_share": c_low}
        w = {"completeness": 0.5, "low_share": 0.5}
        score = math.exp(sum(w[c] * math.log(max(1e-9, v)) for c, v in comps.items()) / sum(w.values()))
        ci = wilson(lo, n)
        rows.append({"id": int(k), "start": int(ts[0]), "end": int(ts[-1]), "i0": int(mk[0]), "n": n,
                     "median_dt": r(med, 2), "gaps": len(gaps), "est_missing": miss, "gap_list": gaps[:12],
                     "completeness": r(completeness, 4), "low_share": r(share, 4), "low_lo": r(ci[0], 4),
                     "low_hi": r(ci[1], 4), "low_z": r(z, 2), "integrity": r(score, 4),
                     "void": bool(miss >= 1), "label": label})
    nn = np.array([s["n"] for s in rows])
    ig = np.array([s["integrity"] for s in rows])
    return {"sessions": rows, "model": {"a": r(a, 3), "b": r(b, 3)}, "summary": {
        "sessions": len(rows), "rounds": int(nn.sum()), "est_missing": int(sum(s["est_missing"] for s in rows)),
        "weighted_integrity": r(float((ig * nn).sum() / nn.sum()), 4),
        "share_ge_095": r(float(nn[ig >= 0.95].sum() / nn.sum()), 4),
        "void_sessions": int(sum(s["void"] for s in rows)), "p_low_ref": r(P_LOW, 5)}}


# ───────────────────────── F-03 fingerprint / CUSUM ─────────────────────────
FP_BLOCK = 500
FP_STATS = [("rate2", 2.0, "Hit rate at 2x"), ("low12", None, "Share below 1.2x"), ("rate10", 10.0, "Hit rate at 10x")]


def block_stats(x, block=FP_BLOCK):
    nb = len(x) // block
    out = {}
    for key, m, _ in FP_STATS:
        p0 = P_LOW if key == "low12" else house_p(m)
        z = []
        for b in range(nb):
            xs = x[b * block:(b + 1) * block]
            v = (xs < LOW).mean() if key == "low12" else (xs >= m).mean()
            z.append((v - p0) / math.sqrt(p0 * (1 - p0) / block))
        out[key] = np.array(z)
    return out, nb


def cusum(z, k=0.5, h=5.0):
    sp = sm = 0.0
    up, dn, alarms = [], [], []
    for i, v in enumerate(z):
        sp = max(0.0, sp + v - k)
        sm = max(0.0, sm - v - k)
        up.append(round(sp, 3))
        dn.append(round(sm, 3))
        if sp > h or sm > h:
            alarms.append(i)
            sp = sm = 0.0
    return up, dn, alarms


def fingerprint(streams, burst, rng):
    out = {"block": FP_BLOCK, "k": 0.5, "h": 5.0, "streams": {}}
    for s in STREAMS:
        x = streams[s]["x"]
        zs, nb = block_stats(x)
        st = {"blocks": nb, "edge": r(1 - 2 * (x >= 2).mean(), 4), "stats": {}}
        for key, _, lab in FP_STATS:
            up, dn, al = cusum(zs[key])
            st["stats"][key] = {"label": lab, "z": [round(float(v), 3) for v in zs[key]], "up": up, "dn": dn, "alarms": al}
        out["streams"][s] = st
    # demo: aviator with the Jul 3 burst spliced back in at its true position
    if burst is not None:
        xa, ta = streams["aviator"]["x"], streams["aviator"]["t"]
        pos = int(np.searchsorted(ta, burst["t"][0]))
        xb = np.concatenate([xa[:pos], burst["x"], xa[pos:]])
        zs, nb = block_stats(xb)
        up, dn, al = cusum(zs["rate2"])
        out["burst_demo"] = {"insert_block": pos // FP_BLOCK, "burst_blocks": int(math.ceil(len(burst["x"]) / FP_BLOCK)),
                             "z": [round(float(v), 3) for v in zs["rate2"]], "up": up, "dn": dn, "alarms": al}
    # measurement: detection delay on an injected edge shift 3% -> 4%, false alarms on fair tapes
    delays, fa_blocks, fa = [], 0, 0
    for rep in range(40):
        n0, n1 = 60 * FP_BLOCK, 60 * FP_BLOCK
        a = PR.fair_stream(n0, rng)
        U = rng.random(n1)
        b = np.floor(np.maximum(1.0, 0.96 / (1 - U)) * 100) / 100
        zs, _ = block_stats(np.concatenate([a, b]))
        _, _, al = cusum(zs["rate2"])
        pre = [i for i in al if i < 60]
        post = [i for i in al if i >= 60]
        fa += len(pre)
        fa_blocks += 60
        delays.append((post[0] - 60 + 1) if post else None)
    got = [d for d in delays if d is not None]
    out["measurement"] = {"shift": "house edge 3% -> 4% at 2x", "reps": 40, "detected": len(got),
                          "median_delay_blocks": r(float(np.median(got)), 1) if got else None,
                          "median_delay_rounds": int(np.median(got) * FP_BLOCK) if got else None,
                          "false_alarms_per_1000_blocks": r(1000 * fa / fa_blocks, 2)}
    return out


# ───────────────────────── F-10 cross-source comparator ─────────────────────────
CMP_METRICS = [("rate2", "Hit rate at 2x", lambda x: x >= 2), ("rate10", "Hit rate at 10x", lambda x: x >= 10),
               ("rate100", "Hit rate at 100x", lambda x: x >= 100), ("low12", "Share below 1.2x", lambda x: x < LOW),
               ("instant", "Instant crash (1.00x)", lambda x: x < 1.01)]


def compare(streams):
    rows = []
    for key, lab, fn in CMP_METRICS:
        for i, a in enumerate(STREAMS):
            for b in STREAMS[i + 1:]:
                xa, xb = streams[a]["x"], streams[b]["x"]
                ka, kb, na, nb = int(fn(xa).sum()), int(fn(xb).sum()), len(xa), len(xb)
                pa, pb = ka / na, kb / nb
                pp = (ka + kb) / (na + nb)
                se0 = math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb)) or 1e-12
                se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb) or 1e-12
                z = (pa - pb) / se0
                rows.append({"metric": key, "a": a, "b": b, "pa": r(pa, 5), "pb": r(pb, 5), "na": na, "nb": nb,
                             "diff": r(pa - pb, 5), "lo": r(pa - pb - 1.96 * se, 5), "hi": r(pa - pb + 1.96 * se, 5),
                             "z": r(z, 2), "p": 2 * norm_sf(abs(z))})
    q = bh([x["p"] for x in rows])
    for x, qq in zip(rows, q):
        x["q"] = r(float(qq), 4)
        x["p"] = r(x["p"], 5)
    return {"metrics": [{"key": k, "label": lab} for k, lab, _ in CMP_METRICS], "rows": rows}


# ───────────────────────── F-11 signal significance strip ─────────────────────────
def signal_matrix(x, t):
    n = len(x)
    lx = np.log(x)
    X, names = PR.build_features(x, t)
    F = dict(zip(names, X.T))
    s10 = np.expm1(F["since_10x"])            # rounds since last 10x (n+1 if none)
    p10 = house_p(10)
    pressure = 1 - (1 - p10) ** s10           # KM/geometric percentile of the 10x drought
    run2 = F["run_below2"]
    sig = {}
    sig["pressure_ge70"] = ("Pressure ≥ 70 (10x drought percentile)", pressure >= 0.7)
    sig["drought_no_10x_40"] = ("No 10x in 40 rounds", s10 > 40)
    sig["cold_streak_10_sub2"] = ("Ten sub-2x rounds in a row", run2 >= 10)
    sig["three_lows"] = ("Last three rounds under 2x", run2 >= 3)
    sig["after_10x"] = ("Previous round ≥ 10x", F["prev_ge10"] > 0)
    inst = np.zeros(n, bool)
    inst[1:] = x[:-1] < 1.01
    sig["after_instant"] = ("Previous round crashed at 1.00x", inst)
    sig["after_100x_20"] = ("100x within the last 20 rounds", np.expm1(F["since_100x"]) <= 20)
    sig["hot_window"] = ("≥ 8 of last 10 reached 2x", F["rate2_w10"] >= 0.8)
    sig["cold_window"] = ("≤ 2 of last 10 reached 2x", F["rate2_w10"] <= 0.2)
    mx = np.exp(F["max_log_w50"])
    shelf = np.zeros(n, bool)
    rng10 = np.full(n, np.inf)
    for i in range(10, n):
        w = x[i - 10:i]
        rng10[i] = w.max() / w.min()
    shelf = rng10 < 2.2
    sig["variance_shelf"] = ("Last 10 rounds within a 2.2x range", shelf)
    rej = np.zeros(n, bool)
    band = ((x >= 7) & (x < 10)).astype(float)
    c = np.concatenate([[0], np.cumsum(band)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - 20)
    rej = ((c[idx] - c[lo]) >= 3) & (s10 > 20)
    sig["ceiling_rejection"] = ("3+ rejections in 7–10x, no 10x in 20", rej)
    med_a = np.full(n, np.nan)
    med_b = np.full(n, np.nan)
    for i in range(20, n):
        med_a[i] = np.median(x[i - 10:i])
        med_b[i] = np.median(x[i - 20:i - 10])
    sig["ascending_floor"] = ("Median rising by 0.5x, no 10x in 20", (med_a - med_b >= 0.5) & (s10 > 20))
    alt = np.zeros(n, bool)
    h = x >= 2
    alt[4:] = (h[3:-1] != h[2:-2]) & (h[2:-2] != h[1:-3]) & (h[1:-3] != h[:-4])
    sig["alternating"] = ("Last four alternate above/below 2x", alt)
    sig["session_start"] = ("First 20 rounds of a session", np.expm1(F["session_pos"]) < 20)
    _ = mx, lx
    return sig


def signal_strip(x, t, shuffle_rng=None):
    if shuffle_rng is not None:
        x = shuffle_rng.permutation(x)
    sig = signal_matrix(x, t)
    W = PR.WARMUP
    out = []
    for m in (2, 10):
        y = (x >= m)
        base = y[W:].mean()
        for key, (lab, mask) in sig.items():
            mk = mask.copy()
            mk[:W] = False
            n1 = int(mk.sum())
            k1 = int(y[mk].sum())
            rest = ~mk
            rest[:W] = False
            n0, k0 = int(rest.sum()), int(y[rest].sum())
            if n1 < 30 or n0 < 30:
                out.append({"key": key, "label": lab, "m": m, "n": n1, "hits": k1, "rate": None, "lift": None,
                            "lo": None, "hi": None, "p": 1.0, "base": r(base, 5)})
                continue
            p1, p0 = k1 / n1, k0 / n0
            pp = (k1 + k0) / (n1 + n0)
            se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0)) or 1e-12
            z = (p1 - p0) / se
            lo, hi = wilson(k1, n1)
            out.append({"key": key, "label": lab, "m": m, "n": n1, "hits": k1, "rate": r(p1, 5), "lift": r(p1 - base, 5),
                        "lo": r(lo - base, 5), "hi": r(hi - base, 5), "z": r(z, 2), "p": 2 * norm_sf(abs(z)),
                        "base": r(base, 5)})
    q = bh([s["p"] for s in out])
    for s, qq in zip(out, q):
        s["q"] = r(float(qq), 4)
        s["p"] = r(s["p"], 5)
    return out


def signal_stability(x, t):
    """F-11 measurement: of signals significant (p<0.05, raw) on the first half, how many stay so on the second."""
    h = len(x) // 2
    a = {(s["key"], s["m"]): s for s in signal_strip(x[:h], t[:h])}
    b = {(s["key"], s["m"]): s for s in signal_strip(x[h:], t[h:])}
    sig_a = [k for k, s in a.items() if s["p"] < 0.05 and s["lift"] is not None]
    keep = [k for k in sig_a if b[k]["p"] < 0.05 and b[k]["lift"] is not None and np.sign(b[k]["lift"]) == np.sign(a[k]["lift"])]
    return {"first_half_significant": len(sig_a), "replicated": len(keep), "tests": len(a)}


# ───────────────────────── F-26/27 ETA board + hazard ─────────────────────────
ETA_T = [2, 5, 10, 20, 50, 100]


def gaps_of(hit):
    """Completed waits (rounds until and including the hit) and the current open wait."""
    idx = np.where(hit)[0]
    if len(idx) == 0:
        return np.array([], int), len(hit)
    g = np.diff(np.concatenate([[-1], idx]))
    cur = len(hit) - 1 - idx[-1]
    return g, int(cur)


def km_survival(g, kmax):
    """S(k) = P(wait > k) for k = 0..kmax from completed waits (no censoring apart from the open wait)."""
    cnt = np.bincount(g, minlength=kmax + 2)[: kmax + 2]
    n = len(g)
    at_risk = n - np.concatenate([[0], np.cumsum(cnt)[:-1]])
    S = np.ones(kmax + 1)
    s = 1.0
    for k in range(1, kmax + 1):
        if at_risk[k] > 0:
            s *= 1 - cnt[k] / at_risk[k]
        S[k] = s
    return S


def cond_quantile(S, g, q):
    """Smallest k>=1 with 1 - S(g+k)/S(g) >= q."""
    if g >= len(S) or S[g] <= 0:
        return None
    for k in range(1, len(S) - g):
        if 1 - S[g + k] / S[g] >= q:
            return k
    return None


def hazard_fit(hit):
    """Discrete hazard: logit h(k) = b0 + b1 log k over every round at risk. Newton; returns b1, se."""
    k = np.empty(len(hit))
    c = 0
    for i, h in enumerate(hit):
        c += 1
        k[i] = c
        if h:
            c = 0
    X = np.column_stack([np.ones(len(k)), np.log(k)])
    y = hit.astype(float)
    b = np.array([math.log(y.mean() / (1 - y.mean())), 0.0])
    for _ in range(25):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    cov = np.linalg.inv(H)
    return float(b[1]), float(math.sqrt(cov[1, 1])), k


def eta_board(x):
    rows = []
    for T in ETA_T:
        hit = x >= T
        p = hit.mean()
        g, cur = gaps_of(hit)
        kmax = int(max(g.max() if len(g) else 1, cur) + 5 * (1 / max(p, 1e-6)))
        kmax = min(kmax, 200000)
        S = km_survival(g, kmax)
        med = cond_quantile(S, cur, 0.5)
        p90 = cond_quantile(S, cur, 0.9)
        pctl = 1 - S[min(cur, kmax)]
        geo_med = math.ceil(math.log(0.5) / math.log(1 - p)) if p > 0 else None
        geo_p90 = math.ceil(math.log(0.1) / math.log(1 - p)) if p > 0 else None
        b1, se, kk = hazard_fit(hit)
        z = b1 / se
        # hazard timeline: empirical hazard by wait k (binned)
        edges = sorted(set([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 70, 100, 150, 250, 400, 700, 1200, 2000]))
        edges = [e for e in edges if e <= max(10, 6 / max(p, 1e-6))] + [10**9]
        hz = []
        for a, b in zip(edges[:-1], edges[1:]):
            mk = (kk >= a) & (kk < b)
            n = int(mk.sum())
            if n < 30:
                continue
            kh = int(hit[mk].sum())
            lo, hi = wilson(kh, n)
            hz.append({"k0": a, "k1": b - 1 if b < 10**9 else None, "n": n, "h": r(kh / n, 5), "lo": r(lo, 5), "hi": r(hi, 5)})
        # median-ETA calibration: KM from first half, evaluated on the second half
        h2 = len(x) // 2
        g1, _ = gaps_of(hit[:h2])
        S1 = km_survival(g1, kmax)
        within, total, c = 0, 0, 0
        wait = 0
        stated = []
        for i in range(h2, len(x) - 1):
            m_i = cond_quantile(S1, min(wait, kmax - 1), 0.5)
            if m_i is not None and i + m_i < len(x):
                total += 1
                within += bool(hit[i:i + m_i].any())
                stated.append(1 - (1 - p) ** m_i)
            wait = 0 if hit[i] else wait + 1
            c += 1
            if c > 20000:
                break
        rows.append({"T": T, "p": r(p, 5), "p_house": r(house_p(T), 5), "waits": int(len(g)), "gap": cur,
                     "pctl": r(pctl, 4), "eta_med": med, "eta_p90": p90, "geo_med": geo_med, "geo_p90": geo_p90,
                     "next_p": r(1 - S[cur + 1] / S[cur] if cur + 1 <= kmax and S[cur] > 0 else p, 5),
                     "b1": r(b1, 4), "b1_lo": r(b1 - 1.96 * se, 4), "b1_hi": r(b1 + 1.96 * se, 4), "b1_z": r(z, 2),
                     "b1_p": 2 * norm_sf(abs(z)), "hazard": hz,
                     "median_calib": r(within / total, 4) if total else None, "median_calib_n": total,
                     "median_calib_geo": r(float(np.mean(stated)), 4) if stated else None,
                     "longest": int(g.max()) if len(g) else None,
                     "S": [r(float(S[k]), 5) for k in range(0, min(kmax, int(8 / max(p, 1e-6))) + 1, max(1, int(0.08 / max(p, 1e-6))))],
                     "S_step": max(1, int(0.08 / max(p, 1e-6)))})
    q = bh([rw["b1_p"] for rw in rows])
    for rw, qq in zip(rows, q):
        rw["b1_q"] = r(float(qq), 4)
        rw["b1_p"] = r(rw["b1_p"], 5)
    return rows


def memoryless_null(rng, reps=60, n=30000):
    cover, planted = 0, 0
    for _ in range(reps):
        x = PR.fair_stream(n, rng)
        b1, se, _ = hazard_fit(x >= 10)
        cover += abs(b1 / se) < 1.96
    # planted: hazard rising with the wait
    for _ in range(20):
        x = np.empty(n)
        w = 0
        for i in range(n):
            pk = min(0.5, 0.06 * (1 + 0.25 * math.log1p(w)))
            x[i] = 10.0 if rng.random() < pk else 1.5
            w = 0 if x[i] >= 10 else w + 1
        b1, se, _ = hazard_fit(x >= 10)
        planted += (b1 / se) > 1.96
    return {"null_reps": reps, "null_cover": r(cover / reps, 3), "planted_reps": 20, "planted_detected": planted}


# ───────────────────────── F-34 registry ─────────────────────────
def registry(pred, signals, cmp_, eta, fp, data_hash):
    ex = []

    def add(eid, family, hyp, spec, tests, best, verdict, power=None, page=None):
        spec_txt = "\n".join(f"{k}: {v}" for k, v in spec.items())
        ex.append({"id": eid, "family": family, "hypothesis": hyp, "spec": spec_txt, "tests": tests,
                   "best": best, "verdict": verdict, "power": power, "page": page,
                   "spec_hash": hashlib.sha256((spec_txt + data_hash).encode()).hexdigest()[:12]})

    lb = pred["leaderboard"]
    pw = next((p for p in pred["controls"]["power"] if p["detected"] == p["reps"]), None)
    add("EXP-001", "Next-round forecasting", "Some model beats the 0.97/m house curve on held-out rounds for some stream and target.",
        {"streams": ", ".join(STREAMS), "targets": "1.5,2,3,5,10,20,50,100", "models": "empirical, logit(L2 C=0.05), hgb",
         "validation": "walk-forward, 5 folds, purge 200", "metric": "log-loss skill vs house, block bootstrap 250",
         "family_correction": "Benjamini-Hochberg q<0.05"}, len(lb),
        f"z {lb[0]['z']:+.2f} ({lb[0]['stream']} {lb[0]['m']}x {lb[0]['model']})",
        "rejected" if not any(b["verdict"] == "edge" for b in lb) else "supported",
        f"+{pw['effect']*100:.0f} pp planted edge detected {pw['detected']}/{pw['reps']}" if pw else None, "leaderboard")
    sh = pred["controls"]["shuffled"]["m2"]
    add("EXP-002", "Control", "Shuffling Aviator destroys any skill (negative control).",
        {"stream": "aviator, permuted", "target": "2x", "models": "logit, hgb"}, 2,
        f"z {max(sh['logit']['z'], sh['hgb']['z']):+.2f}", "passed", None, "controls")
    tl = pred["controls"]["timestamp_leak"]["m2"]
    add("EXP-003", "Control", "Using the current round's duration leaks the outcome (leak detector must fire).",
        {"stream": "aviator", "feature": "t[i]-t[i-1] (forbidden)", "target": "2x"}, 2,
        f"z {tl['hgb']['z']:+.1f}, AUC {tl['hgb']['auc']:.3f}", "fired", None, "controls")
    kf = pred["controls"]["burst_leak"]["random_kfold"]["with_burst"]
    add("EXP-004", "Control", "The Jul 3 burst creates a fake edge under shuffled cross-validation.",
        {"stream": "aviator + 7,662 burst rows", "validation": "shuffled 5-fold (wrong) vs walk-forward", "model": "hgb", "target": "2x"}, 2,
        f"z {kf['z']:+.2f}; {kf['acc_burst_rows']*100:.1f}% right on burst rows", "fired", None, "controls")
    for s in STREAMS:
        rows = [x for x in signals[s]["strip"] if x["lift"] is not None]
        best = min(rows, key=lambda x: x["q"]) if rows else None
        add(f"EXP-01{STREAMS.index(s)}", "Signals (F-11)", f"At least one of the 14 dashboard signals changes the next-round hit rate on {s}.",
            {"stream": s, "signals": 14, "targets": "2x, 10x next round", "test": "two-proportion z vs non-signal rounds",
             "family_correction": "BH across 28"}, len(rows),
            f"q {best['q']:.3f} ({best['key']} @ {best['m']}x)" if best else "-",
            "supported" if best and best["q"] < 0.05 else "rejected", None, "signals")
    for s in STREAMS:
        e = eta[s]
        best = min(e, key=lambda x: x["b1_q"])
        add(f"EXP-02{STREAMS.index(s)}", "Memorylessness (F-26)", f"Waiting longer changes the chance of the next hit on {s} (\"overdue\").",
            {"stream": s, "thresholds": "2,5,10,20,50,100", "model": "logit h(k) = b0 + b1 log k", "null": "b1 = 0",
             "family_correction": "BH across 6"}, len(e), f"b1 {best['b1']:+.3f} (T={best['T']}x, q {best['b1_q']:.3f})",
            "supported" if best["b1_q"] < 0.05 else "rejected", None, "eta")
    sig_cmp = [x for x in cmp_["rows"] if x["q"] < 0.05]
    add("EXP-030", "Cross-source (F-10)", "Streams differ from each other in hit rates or low share.",
        {"pairs": 10, "metrics": ", ".join(m["key"] for m in cmp_["metrics"]), "test": "two-proportion z", "family_correction": "BH across 50"},
        len(cmp_["rows"]), f"{len(sig_cmp)} pairs with q<0.05", "supported" if sig_cmp else "rejected", None, "fingerprint")
    al = {s: sum(len(v["alarms"]) for v in fp["streams"][s]["stats"].values()) for s in STREAMS}
    m = fp["measurement"]
    add("EXP-031", "Fingerprint (F-03)", "A stream's statistical fingerprint shifts during the capture period.",
        {"block": FP_BLOCK, "statistics": "rate2, low12, rate10", "detector": "two-sided CUSUM k=0.5 h=5"},
        3 * len(STREAMS), ", ".join(f"{s}:{a}" for s, a in al.items()) + " alarms",
        "supported" if any(al.values()) else "rejected",
        f"3%->4% edge shift found {m['detected']}/{m['reps']}, median {m['median_delay_rounds']} rounds", "fingerprint")
    return ex


# ───────────────────────── book rendering ─────────────────────────
def build_book(book: Path):
    import markdown  # noqa: WPS433
    files = [("README.md", "index")] + [(f"chapters/{p.name}", p.stem) for p in sorted((book / "chapters").glob("*.md"))] + \
            [(f"appendices/{p.name}", p.stem) for p in sorted((book / "appendices").glob("*.md"))]
    link = {src: slug for src, slug in files}
    docs, chunks = [], []
    for src, slug in files:
        text = (book / src).read_text(encoding="utf-8")
        base = str(Path(src).parent)

        def fix(mm):
            tgt, anc = mm.group(2), mm.group(3) or ""
            norm = str(Path(base) / tgt) if base != "." else tgt
            norm = re.sub(r"[^/]+/\.\./", "", norm)
            return f"{mm.group(1)}(#/book?d={link[norm]}{anc.replace('#', '&h=')})" if norm in link else mm.group(0)

        text2 = re.sub(r"(\[[^\]]*\])\(((?!https?:)[^)#\s]+\.md)(#[^)]*)?\)", fix, text)
        md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"], extension_configs={"toc": {"permalink": False}})
        html = md.convert(text2)
        title = re.search(r"^#\s+(.+)$", text, re.M)
        title = title.group(1).strip() if title else slug
        toc = [{"id": t["id"], "name": t["name"], "level": t["level"]} for t in _flat(md.toc_tokens)]
        part = "Front" if slug == "index" else "Appendices" if src.startswith("appendices") else ("Part II" if slug[:2] in ("18", "19") else "Part I")
        docs.append({"slug": slug, "src": src, "title": title, "part": part, "html": html, "toc": toc,
                     "words": len(re.findall(r"\w+", text))})
        # search chunks by ## / ### section
        cur_id, cur_name, buf = None, title, []

        def flush():
            body = re.sub(r"[`*_>#|]", " ", "\n".join(buf))
            body = re.sub(r"\s+", " ", body).strip()
            if body:
                chunks.append({"d": slug, "h": cur_id, "t": cur_name, "doc": title, "x": body[:1800]})

        slugs = iter([t for t in _flat(md.toc_tokens)])
        heads = {t["name"]: t["id"] for t in _flat(md.toc_tokens)}
        for line in text.splitlines():
            mh = re.match(r"^(#{2,4})\s+(.+)$", line)
            if mh:
                flush()
                buf = []
                cur_name = mh.group(2).strip()
                cur_id = heads.get(re.sub(r"[`*]", "", cur_name).strip(), heads.get(cur_name))
            else:
                buf.append(line)
        flush()
        _ = slugs
    h = hashlib.sha256()
    for src, _ in files:
        h.update((book / src).read_bytes())
    return {"docs": docs, "chunks": chunks, "hash": h.hexdigest()[:16]}


def _flat(tokens):
    for t in tokens:
        yield t
        yield from _flat(t.get("children", []))


# ───────────────────────── catalogue status ─────────────────────────
CATALOGUE = {
    "F-01": ("built", "integrity"), "F-02": ("n/a", None), "F-03": ("built", "fingerprint"), "F-04": ("partial", "arena"),
    "F-05": ("n/a", None), "F-06": ("n/a", None), "F-07": ("built", "sequence"), "F-08": ("partial", "arena"),
    "F-09": ("partial", "leaderboard"), "F-10": ("built", "fingerprint"), "F-11": ("built", "signals"), "F-12": ("partial", "leaderboard"),
    "F-13": ("n/a", None), "F-14": ("n/a", None), "F-15": ("built", "arena"), "F-16": ("n/a", None),
    "F-17": ("built", "calibration"), "F-18": ("built", "arena"), "F-19": ("n/a", None), "F-20": ("n/a", None),
    "F-21": ("built", "arena"), "F-22": ("built", "arena"), "F-23": ("partial", "fingerprint"), "F-24": ("built", "arena"),
    "F-25": ("n/a", None), "F-26": ("built", "eta"), "F-27": ("built", "eta"), "F-28": ("n/a", None),
    "F-29": ("n/a", None), "F-30": ("built", "simulator"), "F-31": ("n/a", None), "F-32": ("built", "simulator"),
    "F-33": ("n/a", None), "F-34": ("built", "experiments"), "F-35": ("built", "nextround"), "F-36": ("partial", "fairness"),
    "F-37": ("partial", "book"), "F-38": ("n/a", None),
}
NA_WHY = {
    "F-02": "needs two live collectors on the same source", "F-05": "needs the live Durable Object backend",
    "F-06": "needs the V5 vocabulary tables", "F-13": "needs the live engine mixture and state machine",
    "F-14": "needs stored forecast versions", "F-16": "needs the engine ledger", "F-19": "needs product tiers",
    "F-20": "needs live engine weights", "F-25": "needs momentum anchors from the terminal", "F-28": "needs push notifications",
    "F-29": "needs in-round ticks", "F-31": "needs the V6 Auto-Tell signals", "F-33": "needs live sessions and the Guard engine",
    "F-38": "needs push notifications",
}


def catalogue(book: Path | None):
    rows = []
    text = (book / "chapters/18-feature-catalogue.md").read_text(encoding="utf-8") if book else ""
    for line in text.splitlines():
        m = re.match(r"^\|\s*(F-\d\d)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|", line)
        if m:
            fid = m.group(1)
            st, page = CATALOGUE.get(fid, ("n/a", None))
            anchor = None
            hm = re.search(rf"^### {fid} · .*$", text, re.M)
            if hm:
                anchor = hm.group(0)
            rows.append({"id": fid, "name": m.group(2).strip(), "area": m.group(3).strip(), "builds_on": m.group(4).strip(),
                         "effort": m.group(5).strip(), "priority": m.group(6).replace("*", "").strip(),
                         "status": st, "page": page, "why": NA_WHY.get(fid)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--book", type=Path, default=None)
    a = ap.parse_args()
    T0 = time.time()
    rng = np.random.default_rng(11)
    streams, burst, _ = PR.load_streams(a.repo)
    pred = json.loads((OUT / "predict.json").read_text())
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data_hash": pred["data_hash"]}
    print("integrity", flush=True)
    out["integrity"] = {s: integrity(streams[s]["x"], streams[s]["t"]) for s in STREAMS}
    if burst is not None:
        out["integrity_burst"] = integrity(burst["x"], burst["t"], "Jul 3 burst (quarantined)", model=duration_model(streams["aviator"]["x"], streams["aviator"]["t"]))["sessions"]
    print("fingerprint", flush=True)
    out["fingerprint"] = fingerprint(streams, burst, rng)
    out["compare"] = compare(streams)
    print("signals", flush=True)
    sigs = {}
    for s in STREAMS:
        x, t = streams[s]["x"], streams[s]["t"]
        strip = signal_strip(x, t)
        shuf = signal_strip(x, t, shuffle_rng=np.random.default_rng(5))
        sigs[s] = {"strip": strip, "shuffled_sig": sum(1 for z in shuf if z["q"] < 0.05), "shuffled_raw": sum(1 for z in shuf if z["p"] < 0.05),
                   "stability": signal_stability(x, t) if len(x) > 8000 else None}
    out["signals"] = sigs
    print("eta", flush=True)
    out["eta"] = {s: eta_board(streams[s]["x"]) for s in STREAMS}
    out["eta_checks"] = memoryless_null(rng)
    out["registry"] = registry(pred, sigs, out["compare"], out["eta"], out["fingerprint"], pred["data_hash"])
    out["catalogue"] = catalogue(a.book)
    out["runtime_sec"] = r(time.time() - T0, 1)
    (OUT / "platform.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
    if a.book:
        bk = build_book(a.book)
        (OUT / "book.json").write_text(json.dumps(bk, separators=(",", ":")))
        print("book", len(bk["docs"]), "docs", len(bk["chunks"]), "chunks")
    print(f"done {time.time() - T0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
