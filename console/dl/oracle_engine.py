"""Momento Oracle — Python port of engine.js (same math, same verdicts).

Usage:
    python oracle_engine.py path/to/momento.db [--method file|live-feed] [--json]

Walk-forward only: every round is forecast before it is revealed.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass, field

TARGETS = [1.5, 2, 3, 5, 10, 20, 50, 100]
MODELS = ["house", "empirical", "recent", "markov", "streak"]


def _clip(p: float) -> float:
    return min(1 - 1e-4, max(1e-4, p))


def _ll(p: float, y: int) -> float:
    return -math.log(p) if y else -math.log(1 - p)


@dataclass
class _T:
    m: float
    hit: int = 0
    tot: int = 0
    ring: list = field(default_factory=list)
    ring_sum: int = 0
    mk: list = field(default_factory=lambda: [[1, 1], [1, 1]])
    st: list = field(default_factory=lambda: [[1, 1] for _ in range(6)])
    run: int = 0
    prev: int = -1
    gap: int = 0
    cum: list = field(default_factory=lambda: [0.0] * len(MODELS))
    score: dict = field(default_factory=lambda: {n: [0.0, 0.0, 0.0] for n in MODELS + ["ensemble"]})
    n: int = 0


class Oracle:
    def __init__(self, rtp=0.97, eta=1.0, warm=50, window=200, prior=10, ev_threshold=0.0):
        self.rtp, self.eta, self.warm, self.window, self.prior, self.evt = rtp, eta, warm, window, prior, ev_threshold
        self.S = [_T(m) for m in TARGETS]
        self.xs: list[float] = []
        self.bets = 0
        self.pnl = 0.0
        self.pnl2 = 0.0

    def _models(self, s: _T):
        h = _clip(self.rtp / s.m)
        pr = self.prior
        emp = (s.hit + h * pr) / (s.tot + pr)
        rec = (s.ring_sum + h * pr) / (len(s.ring) + pr)
        mkv = h
        if s.prev >= 0:
            row = s.mk[s.prev]
            mkv = (row[1] - 1 + h * pr) / (row[0] + row[1] - 2 + pr)
        srow = s.st[min(s.run, 5)]
        stv = (srow[1] - 1 + h * pr) / (srow[0] + srow[1] - 2 + pr)
        return [_clip(v) for v in (h, emp, rec, mkv, stv)]

    def _forecast(self, s: _T):
        p = self._models(s)
        mn = min(s.cum)
        w = [math.exp(-self.eta * (c - mn)) for c in s.cum]
        z = sum(w)
        w = [v / z for v in w]
        return p, w, _clip(sum(a * b for a, b in zip(p, w)))

    def push(self, v: float):
        if not (v >= 1) or not math.isfinite(v):
            return
        t = len(self.xs)
        fs = [self._forecast(s) for s in self.S]
        if t >= self.warm:
            k = max(range(len(TARGETS)), key=lambda i: TARGETS[i] * fs[i][2] - 1)
            if TARGETS[k] * fs[k][2] - 1 > self.evt:
                r = TARGETS[k] - 1 if v >= TARGETS[k] else -1.0
                self.bets += 1
                self.pnl += r
                self.pnl2 += r * r
        for s, (p, _w, pe) in zip(self.S, fs):
            y = 1 if v >= s.m else 0
            if t >= self.warm:
                lh = _ll(p[0], y)
                for i, n in enumerate(MODELS):
                    l = _ll(p[i], y)
                    d = lh - l
                    sc = s.score[n]
                    sc[0] += l; sc[1] += d; sc[2] += d * d
                l = _ll(pe, y)
                d = lh - l
                sc = s.score["ensemble"]
                sc[0] += l; sc[1] += d; sc[2] += d * d
                s.n += 1
            for i, pi in enumerate(p):
                s.cum[i] += _ll(pi, y)
            s.hit += y
            s.tot += 1
            s.ring.append(y)
            s.ring_sum += y
            if len(s.ring) > self.window:
                s.ring_sum -= s.ring.pop(0)
            if s.prev >= 0:
                s.mk[s.prev][y] += 1
            s.st[min(s.run, 5)][y] += 1
            s.run = 0 if y else s.run + 1
            s.gap = 0 if y else s.gap + 1
            s.prev = y
        self.xs.append(v)

    @staticmethod
    def _stat(sc, n):
        if n < 2:
            return {"skill": 0.0, "z": 0.0}
        ll, d, d2 = sc
        mean = d / n
        var = max(d2 / n - mean * mean, 1e-18) * n / (n - 1)
        return {"skill": d / (ll + d) if ll + d > 0 else 0.0, "z": mean / math.sqrt(var / n)}

    def snapshot(self) -> dict:
        out = []
        for s in self.S:
            p, w, pe = self._forecast(s)
            ens = self._stat(s.score["ensemble"], s.n)
            z = ens["z"]
            verdict = "predictable" if z >= 3 else "weak" if z >= 2 else "overfit" if z <= -2 else "random"
            out.append({
                "m": s.m, "p_oracle": pe, "p_house": _clip(self.rtp / s.m), "ev": s.m * pe - 1,
                "eta_median": max(1, math.ceil(math.log(0.5) / math.log(1 - pe))),
                "eta_90": max(1, math.ceil(math.log(0.1) / math.log(1 - pe))),
                "gap": s.gap, "skill": ens["skill"], "z": z, "verdict": verdict,
                "weights": dict(zip(MODELS, w)),
                "models": {n: self._stat(s.score[n], s.n) for n in MODELS},
            })
        b = self.bets
        roi = self.pnl / b if b else 0.0
        sd = math.sqrt(max(self.pnl2 / b - roi * roi, 0) * b / (b - 1)) if b > 1 else 0.0
        se = sd / math.sqrt(b) if b else 0.0
        zs = roi / se if se else 0.0
        pred = [t["m"] for t in out if t["verdict"] == "predictable"]
        if len(self.xs) < self.warm + 30:
            head = "Collecting rounds"
        elif zs >= 3 and b >= 200:
            head = "Exploitable pattern (paper)"
        elif pred:
            head = "Signal found, not profitable"
        else:
            head = "No edge. House math holds."
        return {"n": len(self.xs), "headline": head, "predictable": pred, "targets": out,
                "strategy": {"bets": b, "roi": roi, "se": se, "z": zs}}


def load_rounds(db: str, method: str | None = None) -> list[float]:
    con = sqlite3.connect(db)
    q = "select multiplier from rounds"
    args: tuple = ()
    if method:
        q += " where ingest_method=?"
        args = (method,)
    q += " order by timestamp, id"
    return [r[0] for r in con.execute(q, args)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--method", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    o = Oracle()
    for v in load_rounds(a.db, a.method):
        o.push(v)
    s = o.snapshot()
    if a.json:
        print(json.dumps(s, indent=2))
        return
    print(f"{s['headline']}  ({s['n']} rounds)")
    for t in s["targets"]:
        print(f"  >= {t['m']:>5}x  P={t['p_oracle']:.4f}  house={t['p_house']:.4f}  skill={t['skill']*100:+.2f}%  z={t['z']:+.2f}  {t['verdict']}")
    st = s["strategy"]
    print(f"  paper strategy: {st['bets']} bets, ROI {st['roi']*100:+.1f}% +/- {st['se']*100:.1f}%, z={st['z']:.2f}")


if __name__ == "__main__":
    main()
