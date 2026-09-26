"""AVFS Investigation Suite — decomposition pipeline.

Reads every database on the InvestigationSuite `decomputation` branch
(live avfs.db, its three backups, the 7z mirror and momento.db), separates
real captures from injected / test / rewritten rows, then writes every
analysis the console renders as JSON under ../data/.

    python pipeline/decompose.py --repo /path/to/InvestigationSuite

Nothing here looks ahead: forecasts are walk-forward, one round at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from oracle_engine import Oracle, TARGETS  # noqa: E402

RTP = 0.97
SESSION_GAP = 300  # seconds, same as momento.db settings.analysis.session_gap_seconds
LADDER = [1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
MOON = [10, 50, 100, 500, 1000]
OUT = Path(__file__).resolve().parent.parent / "data"

SOURCE_META = {
    "burst": {"label": "Jul 3 synthetic burst", "db": "avfs.db", "note": "quarantined rows, kept only as a control"},
    "aviator": {"label": "Aviator", "kind": "capture", "db": "avfs.db"},
    "skyward": {"label": "Skyward", "kind": "capture", "db": "avfs.db"},
    "skyward_deluxe": {"label": "Skyward Deluxe", "kind": "capture", "db": "avfs.db"},
    "aviator_jul29": {"label": "Aviator (Jul 29-30)", "kind": "capture", "db": "momento.db"},
    "engine": {"label": "Momento engine (control)", "kind": "simulator", "db": "momento.db"},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r(v, n=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return round(float(v), n)


def offgrid(x: pd.Series) -> pd.Series:
    return (np.abs(x * 100 - np.round(x * 100)) > 1e-6)


# ───────────────────────────── evidence ledger ─────────────────────────────

def ledger(data: Path):
    files = []
    for p in sorted(data.iterdir()):
        if p.name.endswith(".py"):
            continue
        rec = {"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p) if p.stat().st_size else None}
        if p.stat().st_size and p.suffix not in (".7z",):
            try:
                c = sqlite3.connect(f"file:{p}?mode=ro&immutable=1", uri=True)
                tabs = [t for (t,) in c.execute("select name from sqlite_master where type='table'")]
                counts = {t: c.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tabs}
                rec["tables"] = {k: v for k, v in counts.items() if v}
                rec["empty_tables"] = sum(1 for v in counts.values() if not v)
                if "rounds" in tabs:
                    rec["rounds_by_source"] = dict(c.execute("select source,count(*) from rounds group by 1").fetchall())
                    rec["range"] = list(c.execute("select min(timestamp),max(timestamp) from rounds where timestamp>'2025'").fetchone())
            except sqlite3.DatabaseError:
                pass
        elif p.suffix == ".7z":
            try:
                import py7zr
                with py7zr.SevenZipFile(p) as z:
                    rec["archive"] = [{"name": f.filename, "bytes": f.uncompressed, "crc32": f.crc32} for f in z.list()]
            except Exception as e:  # pragma: no cover
                rec["archive_error"] = str(e)
        files.append(rec)

    # identical-content groups
    groups = defaultdict(list)
    for f in files:
        if f["sha256"]:
            groups[f["sha256"]].append(f["name"])
    dup_groups = [g for g in groups.values() if len(g) > 1]

    # 7z mirror check: extract crc32 vs on-disk crc32
    import zlib
    mirror = []
    for f in files:
        for a in f.get("archive", []):
            disk = data / a["name"]
            if disk.exists() and a["crc32"] is not None:
                crc = 0
                with open(disk, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 20), b""):
                        crc = zlib.crc32(b, crc)
                mirror.append({"name": a["name"], "match": crc == a["crc32"]})

    # backup -> live diff
    bak = next(data.glob("avfs.db.bak.*"))
    c = sqlite3.connect(f"file:{data / 'avfs.db'}?mode=ro&immutable=1", uri=True)
    c.execute(f"attach '{bak}' as b")
    added = c.execute("select m.source,count(*),min(m.id),max(m.id),min(m.timestamp),max(m.timestamp) from main.rounds m left join b.rounds b on b.id=m.id where b.id is null group by 1").fetchall()
    rewritten = c.execute("""select b.id,b.source,b.timestamp,b.multiplier,m.source,m.timestamp,m.multiplier
        from b.rounds b join main.rounds m on m.id=b.id
        where b.source<>m.source or b.multiplier<>m.multiplier or b.timestamp<>m.timestamp order by b.id""").fetchall()
    removed = c.execute("select count(*) from b.rounds b left join main.rounds m on m.id=b.id where m.id is null").fetchone()[0]
    rep_added = c.execute("select count(*) from main.replay_results m left join b.replay_results b on b.id=m.id where b.id is null").fetchone()[0]
    rw_src = Counter((row[1], row[4]) for row in rewritten)
    # were the overwritten backup rows preserved anywhere else in live?
    lost = 0
    for row in rewritten:
        hit = c.execute("select 1 from main.rounds where source=? and timestamp=? and multiplier=? limit 1", (row[1], row[2], row[3])).fetchone()
        lost += 0 if hit else 1
    diff = {
        "backup": bak.name,
        "added": [dict(zip(["source", "rows", "id_lo", "id_hi", "t_lo", "t_hi"], a)) for a in added],
        "rewritten": len(rewritten),
        "rewritten_id_range": [rewritten[0][0], rewritten[-1][0]] if rewritten else None,
        "rewritten_pairs": [{"was": k[0], "now": k[1], "rows": v} for k, v in rw_src.most_common()],
        "rewritten_lost": lost,
        "rewritten_sample": [dict(zip(["id", "was_source", "was_t", "was_x", "now_source", "now_t", "now_x"], row)) for row in rewritten[:12]],
        "removed": removed,
        "replay_added": rep_added,
    }
    return {"files": files, "duplicate_groups": dup_groups, "mirror": mirror, "diff": diff}


# ───────────────────────────── decomposition ─────────────────────────────

TEST_TOKENS = ("test", "_rt_", "inject", "broadcast", "unique", "_live.json", "LIVE", "target_")


def epoch_s(t) -> np.ndarray:
    """Seconds since epoch, independent of the pandas datetime resolution."""
    ts = pd.to_datetime(pd.Series(t), utc=True)
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).to_numpy(dtype="int64")


EXTRA: dict = {}


def decompose(data: Path):
    c = sqlite3.connect(f"file:{data / 'avfs.db'}?mode=ro&immutable=1", uri=True)
    df = pd.read_sql("select id,timestamp,multiplier,color,source_file,source from rounds", c)
    df["rule"] = ""
    df.loc[~df.source.isin(["aviator", "skyward", "skyward_deluxe"]), "rule"] = "unknown_source"
    df.loc[(df.rule == "") & df.timestamp.str.startswith("2024"), "rule"] = "seed_fixture"
    tmask = df.source_file.apply(lambda s: any(k in s for k in TEST_TOKENS))
    df.loc[(df.rule == "") & tmask, "rule"] = "test_fixture"
    # cross-source filename (skyward_deluxe file stored under skyward)
    xmask = (df.source == "skyward") & df.source_file.str.contains("skyward_deluxe")
    df.loc[(df.rule == "") & xmask, "rule"] = "mislabeled_source"
    df["t"] = pd.to_datetime(df.timestamp, utc=True, format="mixed")
    df["server_ts"] = df.timestamp.str.endswith("+00:00")
    df["off"] = offgrid(df.multiplier)
    # synthetic burst: aviator rows stamped server-side with microsecond clock, off the 0.01 grid
    burst = (df.rule == "") & (df.source == "aviator") & df.server_ts
    df.loc[burst, "rule"] = "synthetic_burst"
    df.loc[(df.rule == "") & df.off, "rule"] = "off_grid_value"
    df = df.sort_values(["source", "t", "id"])
    dmask = (df.rule == "") & df.duplicated(["source", "t"], keep="first")
    df.loc[dmask, "rule"] = "duplicate_timestamp"
    df.loc[(df.rule == "") & (df.multiplier < 1), "rule"] = "below_one"

    # burst profile for the chart (per minute counts + RTP@2)
    b = df[df.rule == "synthetic_burst"].copy()
    bprof = []
    if len(b):
        b["min"] = b.t.dt.floor("1min")
        for k, g in b.groupby("min"):
            bprof.append({"t": k.isoformat(), "n": int(len(g)), "rtp2": r(2 * (g.multiplier >= 2).mean(), 3), "med": r(g.multiplier.median(), 3)})
    bstats = None
    if len(b):
        dt = b.sort_values("t").t.diff().dt.total_seconds().dropna()
        bstats = {"rows": int(len(b)), "t_lo": b.t.min().isoformat(), "t_hi": b.t.max().isoformat(),
                  "off_grid": r(b.off.mean(), 3), "rtp2": r(2 * (b.multiplier >= 2).mean(), 3),
                  "rtp10": r(10 * (b.multiplier >= 10).mean(), 3), "median": r(b.multiplier.median(), 3),
                  "gap_median_s": r(dt.median(), 4), "p_below_1_01": r((b.multiplier < 1.01).mean(), 4),
                  "sample": b.multiplier.head(16).round(6).tolist()}
    kept = df[df.rule == ""]
    EXTRA["burst_df"] = pd.DataFrame({"t": b.t.values, "x": b.multiplier.values}) if len(b) else None
    genuine = {}
    for s, g in kept.groupby("source"):
        x = g.multiplier
        genuine[s] = {"rtp2": r(2 * (x >= 2).mean(), 3), "rtp10": r(10 * (x >= 10).mean(), 3), "median": r(x.median(), 3),
                      "p_below_1_01": r((x < 1.01).mean(), 4)}

    rules = []
    RULE_TEXT = {
        "unknown_source": "Source is not a tracked game (${src} template leak, JetX probes)",
        "seed_fixture": "2024-01-01 seed rows from momento_rounds_001.json",
        "test_fixture": "Written by test / probe / broadcast files",
        "mislabeled_source": "Skyward Deluxe file stored under Skyward",
        "synthetic_burst": "Aviator rows generated server-side on Jul 3, not captured",
        "off_grid_value": "Multiplier not on the 0.01 grid a real game shows",
        "duplicate_timestamp": "Same source + timestamp already kept",
        "below_one": "Multiplier below 1.00x",
    }
    for k, g in df[df.rule != ""].groupby("rule"):
        rules.append({"rule": k, "text": RULE_TEXT.get(k, k), "rows": int(len(g)), "by_source": g.source.value_counts().to_dict()})
    rules.sort(key=lambda d: -d["rows"])
    funnel = {"raw": int(len(df)), "kept": int(len(kept)), "quarantined": int((df.rule != "").sum())}
    by_source = {s: {"raw": int((df.source == s).sum()), "kept": int((kept.source == s).sum())} for s in ["aviator", "skyward", "skyward_deluxe"]}

    series = {}
    for s, g in kept.groupby("source"):
        g = g.sort_values(["t", "id"])
        series[s] = pd.DataFrame({"t": g.t.values, "x": g.multiplier.values, "color": g.color.values})

    # momento.db (later snapshot)
    m = sqlite3.connect(f"file:{data / 'momento.db'}?mode=ro&immutable=1", uri=True)
    md = pd.read_sql("select timestamp,multiplier,ingest_method,color from rounds order by timestamp,id", m)
    md["t"] = pd.to_datetime(md.timestamp, utc=True, format="mixed")
    for meth, name in (("file", "aviator_jul29"), ("live-feed", "engine")):
        g = md[md.ingest_method == meth]
        g = g[~g.duplicated(["t"])]
        series[name] = pd.DataFrame({"t": g.t.values, "x": g.multiplier.values, "color": g.color.values})
    settings = dict(m.execute("select key,value from settings").fetchall())
    return {"funnel": funnel, "by_source": by_source, "rules": rules, "burst": bstats, "burst_profile": bprof,
            "genuine": genuine, "settings_keys": list(settings)}, series, settings


# ───────────────────────────── analyses ─────────────────────────────

def audit(x: np.ndarray):
    n = len(x)
    rows = []
    for m in LADDER:
        p = RTP / m
        k = int((x >= m).sum())
        ph = k / n
        se = math.sqrt(p * (1 - p) / n)
        rows.append({"m": m, "hits": k, "expected": r(n * p, 1), "p_obs": r(ph, 5), "p_house": r(p, 5),
                     "rtp": r(m * ph, 4), "rtp_se": r(m * se, 4), "z": r((ph - p) / se, 2) if se else None})
    # instant crash
    inst = float((x < 1.01).mean())
    # survival curve (log grid)
    grid = np.unique(np.round(np.exp(np.linspace(0, math.log(max(x.max(), 2)), 80)), 2))
    surv = [{"m": float(g), "obs": r((x >= g).mean(), 6), "house": r(min(1, RTP / g), 6)} for g in grid if g >= 1]
    # KS-style max deviation on [1.01, 1000]
    dev = max(abs(d["obs"] - d["house"]) for d in surv if 1.01 <= d["m"] <= 1000)
    return {"n": n, "ladder": rows, "instant_crash": r(inst, 5), "instant_expected": r(1 - RTP / 1.01, 5),
            "survival": surv, "max_dev": r(dev, 5), "mean": r(x.mean(), 3), "median": r(np.median(x), 3), "max": r(x.max(), 2)}


def patterns(x: np.ndarray, t: pd.Series):
    n = len(x)
    lx = np.log(x)
    lx = lx - lx.mean()
    den = (lx * lx).sum()
    acf = [{"lag": k, "r": r((lx[:-k] * lx[k:]).sum() / den, 5)} for k in range(1, 21)]
    band = 2 / math.sqrt(n)
    # runs test on >=2x
    y = (x >= 2).astype(int)
    n1, n0 = y.sum(), n - y.sum()
    runs = 1 + int((y[1:] != y[:-1]).sum())
    mu = 2 * n1 * n0 / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1)
    runs_z = (runs - mu) / math.sqrt(var)
    # losing-streak (<2x) length distribution vs geometric
    streaks = []
    cur = 0
    for v in y:
        if v == 0:
            cur += 1
        else:
            if cur:
                streaks.append(cur)
            cur = 0
    sc = Counter(streaks)
    q = 1 - y.mean()
    tot = len(streaks)
    sdist = [{"len": L, "obs": sc.get(L, 0), "exp": r(tot * (q ** (L - 1)) * (1 - q), 1)} for L in range(1, 16)]
    longest = max(streaks) if streaks else 0
    # conditional next-round table
    buckets = [(1, 1.2, "<1.2x"), (1.2, 2, "1.2-2x"), (2, 5, "2-5x"), (5, 10, "5-10x"), (10, 1e12, "10x+")]
    cond = []
    prev, nxt = x[:-1], x[1:]
    base = (nxt >= 2).mean()
    for lo, hi, lab in buckets:
        mk = (prev >= lo) & (prev < hi)
        k = int(mk.sum())
        if not k:
            continue
        p = (nxt[mk] >= 2).mean()
        se = math.sqrt(base * (1 - base) / k)
        cond.append({"bucket": lab, "n": k, "p_next2": r(p, 4), "base": r(base, 4), "z": r((p - base) / se, 2)})
    # after-streak table: P(>=2 | current losing run length L)
    after = []
    run = 0
    hits = defaultdict(lambda: [0, 0])
    for v in y:
        hits[min(run, 8)][0] += 1
        hits[min(run, 8)][1] += v
        run = 0 if v else run + 1
    for L in range(0, 9):
        tot_, h = hits[L]
        if tot_:
            se = math.sqrt(base * (1 - base) / tot_)
            after.append({"run": f"{L}+" if L == 8 else str(L), "n": tot_, "p": r(h / tot_, 4), "z": r((h / tot_ - base) / se, 2)})
    # hour-of-day (UTC)
    hrs = pd.Series(pd.to_datetime(t).dt.hour.values)
    hour = []
    for h in range(24):
        mk = (hrs == h).values
        k = int(mk.sum())
        if k < 30:
            hour.append({"h": h, "n": k, "rtp2": None, "se": None})
            continue
        p = (x[mk] >= 2).mean()
        hour.append({"h": h, "n": k, "rtp2": r(2 * p, 4), "se": r(2 * math.sqrt(0.485 * 0.515 / k), 4)})
    hz = [((d["rtp2"] - RTP) / d["se"]) for d in hour if d["rtp2"] is not None]
    chi = float(sum(v * v for v in hz))
    return {"acf": acf, "band": r(band, 5), "acf_out": sum(1 for a in acf if abs(a["r"]) > band),
            "runs": {"runs": runs, "expected": r(mu, 1), "z": r(runs_z, 2)},
            "streaks": sdist, "longest_loss": longest, "streak_count": tot,
            "conditional": cond, "after_run": after, "hour": hour, "hour_chi": r(chi, 1), "hour_df": len(hz)}


def moonshots(x: np.ndarray, t: pd.Series):
    n = len(x)
    ts = pd.to_datetime(t)
    out = []
    for m in MOON:
        idx = np.flatnonzero(x >= m)
        p = RTP / m
        gaps = np.diff(idx) if len(idx) > 1 else np.array([])
        cur = int(n - 1 - idx[-1]) if len(idx) else n
        hist = []
        if len(gaps):
            edges = np.unique(np.round(np.quantile(gaps, np.linspace(0, 1, 13))).astype(int))
            if len(edges) > 1:
                cnt, e = np.histogram(gaps, bins=edges)
                for i in range(len(cnt)):
                    lo, hi = e[i], e[i + 1]
                    exp = len(gaps) * ((1 - p) ** (lo - 1) - (1 - p) ** (hi - 1))
                    hist.append({"lo": int(lo), "hi": int(hi), "obs": int(cnt[i]), "exp": r(exp, 1)})
        out.append({
            "m": m, "hits": int(len(idx)), "expected": r(n * p, 1), "rate_obs": r(len(idx) / n, 6), "rate_house": r(p, 6),
            "mean_gap": r(gaps.mean(), 1) if len(gaps) else None, "house_gap": r(1 / p, 1),
            "max_gap": int(gaps.max()) if len(gaps) else None, "current_gap": cur,
            "p_drought": r((1 - p) ** cur, 5),
            "eta50": max(1, math.ceil(math.log(0.5) / math.log(1 - p))), "eta90": max(1, math.ceil(math.log(0.1) / math.log(1 - p))),
            "last": ts.iloc[idx[-1]].isoformat() if len(idx) else None,
            "hist": hist,
        })
    top = np.argsort(-x)[:25]
    tops = [{"x": r(x[i], 2), "t": ts.iloc[i].isoformat(), "i": int(i)} for i in sorted(top, key=lambda i: -x[i])]
    return {"levels": out, "top": tops}


def sessions(x: np.ndarray, t: pd.Series):
    ts = pd.to_datetime(t).reset_index(drop=True)
    dt = ts.diff().dt.total_seconds().fillna(0).values
    sid = np.cumsum(dt > SESSION_GAP)
    out = []
    for k in np.unique(sid):
        mk = sid == k
        xs = x[mk]
        tt = ts[mk]
        out.append({"id": int(k), "start": tt.iloc[0].isoformat(), "end": tt.iloc[-1].isoformat(), "n": int(mk.sum()),
                    "mins": r((tt.iloc[-1] - tt.iloc[0]).total_seconds() / 60, 1), "max": r(xs.max(), 2),
                    "rtp2": r(2 * (xs >= 2).mean(), 3), "p100": int((xs >= 100).sum())})
    day = pd.DataFrame({"d": ts.dt.strftime("%Y-%m-%d"), "x": x})
    daily = [{"d": d, "n": int(len(g)), "rtp2": r(2 * (g.x >= 2).mean(), 3), "max": r(g.x.max(), 2)} for d, g in day.groupby("d")]
    lens = np.array([s["n"] for s in out])
    return {"count": len(out), "median_len": int(np.median(lens)), "longest": int(lens.max()), "list": out, "daily": daily,
            "cadence_s": r(float(np.median(dt[(dt > 0) & (dt <= SESSION_GAP)])), 2)}


class TrackedOracle(Oracle):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.curve = []

    def push(self, v):
        super().push(v)
        if len(self.xs) % 250 == 0:
            self.curve.append([len(self.xs), round(self.pnl, 2), self.bets])


def forecast(x: np.ndarray):
    o = TrackedOracle(rtp=RTP)
    for v in x:
        o.push(float(v))
    s = o.snapshot()
    s["curve"] = o.curve
    for tg in s["targets"]:
        for k in ("p_oracle", "p_house", "ev", "skill", "z"):
            tg[k] = r(tg[k], 5)
        tg["weights"] = {k: r(v, 4) for k, v in tg["weights"].items()}
        tg["models"] = {k: {"skill": r(v["skill"], 5), "z": r(v["z"], 2)} for k, v in tg["models"].items()}
    s["strategy"] = {k: r(v, 5) if isinstance(v, float) else v for k, v in s["strategy"].items()}
    return s


def replay_audit(data: Path, burst):
    c = sqlite3.connect(f"file:{data / 'avfs.db'}?mode=ro&immutable=1", uri=True)
    df = pd.read_sql("select id,feature,sessions_tested,scored_predictions,accuracy,status,created_at from replay_results", c)
    df["t"] = pd.to_datetime(df.created_at, utc=True, format="mixed")
    df["d"] = df.t.dt.strftime("%Y-%m-%d")
    daily = []
    for d, g in df.groupby("d"):
        w = g.scored_predictions.clip(lower=1)
        daily.append({"d": d, "runs": int(len(g)), "acc": r(g.accuracy.mean(), 4), "acc_w": r((g.accuracy * w).sum() / w.sum(), 4),
                      "preds": int(g.scored_predictions.sum()), "p90": r(g.accuracy.quantile(0.9), 4)})
    base = df[~df.d.isin(["2026-07-04", "2026-07-05", "2026-07-06"])]
    mu = base.accuracy.mean()
    sd = base.accuracy.std()
    for dd in daily:
        dd["flag"] = bool(dd["acc"] > mu + 0.15)
    hist, e = np.histogram(df.accuracy, bins=20, range=(0, 1))
    spike = df[df.d.isin(["2026-07-04", "2026-07-05", "2026-07-06"])]
    sample = df.sample(min(3000, len(df)), random_state=7)
    return {
        "runs": int(len(df)), "features": df.feature.value_counts().to_dict(), "status": df.status.value_counts().to_dict(),
        "acc_mean": r(df.accuracy.mean(), 4), "acc_base": r(mu, 4), "acc_base_sd": r(sd, 4),
        "spike": {"runs": int(len(spike)), "acc": r(spike.accuracy.mean(), 4), "share_over_80": r((spike.accuracy > 0.8).mean(), 3),
                  "base_share_over_80": r((base.accuracy > 0.8).mean(), 4), "preds": int(spike.scored_predictions.sum())},
        "sessions_tested": df.sessions_tested.value_counts().sort_index().to_dict(),
        "hist": [{"lo": r(e[i], 2), "n": int(hist[i])} for i in range(len(hist))],
        "daily": daily,
        "scatter": [[int(a), r(b, 3)] for a, b in zip(sample.scored_predictions, sample.accuracy)],
        "burst_before_spike": burst["t_hi"] if burst else None,
        "chance_2x": r(RTP / 2, 4),
    }


def write_series(series):
    (OUT / "series").mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, g in series.items():
        t = epoch_s(g.t)
        dt = np.diff(t, prepend=t[0])
        payload = {"t0": int(t[0]), "dt": dt.astype(int).tolist(), "x": [round(float(v), 2) for v in g.x.values]}
        with open(OUT / "series" / f"{name}.json", "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        meta[name] = {**SOURCE_META[name], "n": int(len(g)), "t_lo": pd.Timestamp(g.t.iloc[0]).isoformat(), "t_hi": pd.Timestamp(g.t.iloc[-1]).isoformat(),
                      "bytes": (OUT / "series" / f"{name}.json").stat().st_size}
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="checkout of InvestigationSuite@decomputation")
    a = ap.parse_args()
    data = Path(a.repo) / "data"
    OUT.mkdir(parents=True, exist_ok=True)

    print("ledger ...")
    led = ledger(data)
    print("decompose ...")
    dec, series, settings = decompose(data)
    b = EXTRA.get("burst_df")
    if b is not None:
        series["burst"] = b
    meta = write_series(series)
    series.pop("burst", None)
    dec["burst_series"] = meta.pop("burst", None)

    per = {}
    for name, g in series.items():
        print("analyse", name, len(g))
        x = g.x.values.astype(float)
        per[name] = {"meta": meta[name], "audit": audit(x), "patterns": patterns(x, g.t), "moon": moonshots(x, g.t),
                     "sessions": sessions(x, g.t), "forecast": forecast(x)}
    print("replay audit ...")
    rep = replay_audit(data, dec["burst"])

    safe_settings = {}
    for k, v in settings.items():
        try:
            safe_settings[k] = json.loads(v)
        except Exception:
            safe_settings[k] = v
    if "feed_chain" in safe_settings and isinstance(safe_settings["feed_chain"], dict):
        seed = safe_settings["feed_chain"].get("terminal_seed", "")
        safe_settings["feed_chain"]["terminal_seed"] = seed[:8] + "…" if seed else seed

    bundle = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "branch": "decomputation",
              "ledger": led, "decomposition": dec, "sources": per, "replay": rep, "momento_settings": safe_settings}
    with open(OUT / "suite.json", "w") as f:
        json.dump(bundle, f, separators=(",", ":"), default=str)
    print("wrote", OUT / "suite.json", (OUT / "suite.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
