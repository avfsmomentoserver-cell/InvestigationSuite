// Momento Oracle — end-product predictor engine.
// Walk-forward, online ensemble for crash-round outcomes.
// Every forecast is made BEFORE the round it scores is revealed; nothing is fit on the future.

export const TARGETS = [1.5, 2, 3, 5, 10, 20, 50, 100];
export const MODELS = ['house', 'empirical', 'recent', 'markov', 'streak'];
export const MODEL_LABEL = {
  house: 'House edge',
  empirical: 'Empirical',
  recent: 'Recent window',
  markov: 'Markov (prev round)',
  streak: 'Streak length',
  ensemble: 'Oracle ensemble',
};

const clip = (p) => Math.min(1 - 1e-4, Math.max(1e-4, p));
const logloss = (p, y) => -(y ? Math.log(p) : Math.log(1 - p));

export function bandOf(x) {
  if (x < 1.2) return 'dust';
  if (x < 1.5) return 'floor';
  if (x < 2) return 'low';
  if (x < 3) return 'base';
  if (x < 5) return 'mid';
  if (x < 10) return 'high';
  if (x < 20) return 'ignition';
  if (x < 50) return 'moonshot';
  if (x < 100) return 'mega';
  return 'cosmic';
}

export function createOracle(opts = {}) {
  const cfg = {
    rtp: 0.97,        // house return-to-player; P(X >= m) = rtp / m under a fair engine
    eta: 1.0,         // Hedge learning rate for ensemble weights
    warm: 50,         // rounds before scoring starts
    window: 200,      // recent-window length
    prior: 10,        // pseudo-counts pulling models toward the house curve
    evThreshold: 0.0, // paper strategy: only act when model EV exceeds this
    ...opts,
  };
  const K = TARGETS.length;
  const S = TARGETS.map((m) => ({
    m,
    hit: 0, tot: 0,
    ring: [], ringSum: 0,
    mk: [[1, 1], [1, 1]],                  // [prev][outcome]
    st: Array.from({ length: 6 }, () => [1, 1]), // [runBucket][outcome]
    run: 0, prev: -1, gap: 0,
    cumLL: MODELS.map(() => 0),
    // scoring accumulators: diff = ll_house - ll_model (positive = model better)
    score: [...MODELS, 'ensemble'].reduce((a, n) => ((a[n] = { ll: 0, d: 0, d2: 0 }), a), {}),
    n: 0,
    calib: [], // [p, y] for ensemble
  }));
  const xs = [];
  const strat = { bets: 0, pnl: 0, pnl2: 0, wins: 0, equity: [0], byTarget: {} };

  const house = (s) => clip(cfg.rtp / s.m);

  function models(s) {
    const h = house(s);
    const pr = cfg.prior;
    const emp = (s.hit + h * pr) / (s.tot + pr);
    const rec = (s.ringSum + h * pr) / (s.ring.length + pr);
    let mkv = h;
    if (s.prev >= 0) {
      const row = s.mk[s.prev];
      const n = row[0] + row[1] - 2;
      mkv = (row[1] - 1 + h * pr) / (n + pr);
    }
    const b = Math.min(s.run, 5);
    const srow = s.st[b];
    const sn = srow[0] + srow[1] - 2;
    const stv = (srow[1] - 1 + h * pr) / (sn + pr);
    return [h, emp, rec, mkv, stv].map(clip);
  }

  function weights(s) {
    const min = Math.min(...s.cumLL);
    const w = s.cumLL.map((l) => Math.exp(-cfg.eta * (l - min)));
    const z = w.reduce((a, b) => a + b, 0);
    return w.map((v) => v / z);
  }

  function forecast(s) {
    const p = models(s);
    const w = weights(s);
    const pe = clip(p.reduce((a, v, i) => a + v * w[i], 0));
    return { p, w, pe };
  }

  function push(v) {
    if (!(v >= 1) || !isFinite(v)) return;
    const t = xs.length;
    const fs = S.map(forecast);

    // paper strategy — decided before the reveal
    if (t >= cfg.warm) {
      let best = -1, bestEV = -Infinity;
      fs.forEach((f, k) => {
        const ev = TARGETS[k] * f.pe - 1;
        if (ev > bestEV) { bestEV = ev; best = k; }
      });
      if (bestEV > cfg.evThreshold) {
        const m = TARGETS[best];
        const r = v >= m ? m - 1 : -1;
        strat.bets++; strat.pnl += r; strat.pnl2 += r * r; if (r > 0) strat.wins++;
        const bt = (strat.byTarget[m] ||= { bets: 0, pnl: 0 });
        bt.bets++; bt.pnl += r;
      }
      strat.equity.push(strat.pnl);
    }

    S.forEach((s, k) => {
      const y = v >= s.m ? 1 : 0;
      const { p, pe } = fs[k];
      if (t >= cfg.warm) {
        const lh = logloss(p[0], y);
        MODELS.forEach((n, i) => {
          const l = logloss(p[i], y), d = lh - l, sc = s.score[n];
          sc.ll += l; sc.d += d; sc.d2 += d * d;
        });
        const le = logloss(pe, y), de = lh - le, se = s.score.ensemble;
        se.ll += le; se.d += de; se.d2 += de * de;
        s.n++;
        s.calib.push([pe, y]);
      }
      p.forEach((pi, i) => (s.cumLL[i] += logloss(pi, y)));
      s.hit += y; s.tot++;
      s.ring.push(y); s.ringSum += y;
      if (s.ring.length > cfg.window) s.ringSum -= s.ring.shift();
      if (s.prev >= 0) s.mk[s.prev][y]++;
      s.st[Math.min(s.run, 5)][y]++;
      s.run = y ? 0 : s.run + 1;
      s.gap = y ? 0 : s.gap + 1;
      s.prev = y;
    });
    xs.push(v);
  }

  function stat(sc, n) {
    if (n < 2) return { skill: 0, z: 0 };
    const llh = sc.ll + sc.d; // house total loss = model loss + diff
    const mean = sc.d / n;
    const varr = Math.max(sc.d2 / n - mean * mean, 1e-18) * (n / (n - 1));
    return { skill: llh > 0 ? sc.d / llh : 0, z: mean / Math.sqrt(varr / n) };
  }

  function verdictFor(z) {
    if (z >= 3) return 'predictable';
    if (z >= 2) return 'weak';
    if (z <= -2) return 'overfit';
    return 'random';
  }

  function snapshot() {
    const n = xs.length;
    const targets = S.map((s) => {
      const f = forecast(s);
      const perModel = {};
      [...MODELS, 'ensemble'].forEach((name) => (perModel[name] = stat(s.score[name], s.n)));
      const pe = f.pe;
      const eta50 = Math.max(1, Math.ceil(Math.log(0.5) / Math.log(1 - pe)));
      const eta90 = Math.max(1, Math.ceil(Math.log(0.1) / Math.log(1 - pe)));
      const empP = s.tot ? s.hit / s.tot : 0;
      const rtpHat = s.m * empP;
      const rtpSe = s.tot ? s.m * Math.sqrt(Math.max(empP * (1 - empP), 1e-9) / s.tot) : 0;
      return {
        m: s.m,
        house: house(s),
        models: Object.fromEntries(MODELS.map((nm, i) => [nm, f.p[i]])),
        weights: Object.fromEntries(MODELS.map((nm, i) => [nm, f.w[i]])),
        pe,
        edge: pe - house(s),
        ev: s.m * pe - 1,
        eta50, eta90,
        gap: s.gap,
        gapSurprise: Math.pow(1 - house(s), s.gap), // P(drought this long) under a fair engine
        run: s.run,
        scored: s.n,
        perModel,
        verdict: verdictFor(perModel.ensemble.z),
        rtpHat, rtpSe,
        rtpFlag: rtpHat - 2 * rtpSe > 1 ? 'overpays' : rtpHat + 2 * rtpSe < cfg.rtp * 0.9 ? 'underpays' : 'ok',
        calib: s.calib,
      };
    });
    const sb = strat.bets;
    const mean = sb ? strat.pnl / sb : 0;
    const sd = sb > 1 ? Math.sqrt(Math.max(strat.pnl2 / sb - mean * mean, 0) * (sb / (sb - 1))) : 0;
    const se = sb ? sd / Math.sqrt(sb) : 0;
    const strategy = {
      bets: sb, pnl: strat.pnl, roi: mean, se, z: se ? mean / se : 0,
      winRate: sb ? strat.wins / sb : 0, equity: strat.equity, byTarget: strat.byTarget,
    };
    const predictable = targets.filter((t) => t.verdict === 'predictable');
    const overpay = targets.filter((t) => t.rtpFlag === 'overpays');
    let headline, level;
    if (n < cfg.warm + 30) { headline = 'Collecting rounds'; level = 'pending'; }
    else if (strategy.z >= 3 && strategy.bets >= 200) { headline = 'Exploitable pattern (paper)'; level = 'edge'; }
    else if (predictable.length) { headline = 'Signal found, not profitable'; level = 'signal'; }
    else { headline = 'No edge. House math holds.'; level = 'none'; }
    return { n, cfg, targets, strategy, headline, level, predictable: predictable.map((t) => t.m), overpay: overpay.map((t) => t.m), last: xs.slice(-120) };
  }

  return { push, pushMany: (arr) => arr.forEach(push), snapshot, xs, cfg };
}

// Survival curve helper: empirical P(X >= m) on a log grid
export function survival(xs, points = 60) {
  const sorted = [...xs].sort((a, b) => a - b);
  const n = sorted.length;
  const out = [];
  const lo = Math.log(1.01), hi = Math.log(Math.max(2, sorted[n - 1] || 2));
  for (let i = 0; i < points; i++) {
    const m = Math.exp(lo + ((hi - lo) * i) / (points - 1));
    // count >= m via binary search
    let a = 0, b = n;
    while (a < b) { const mid = (a + b) >> 1; if (sorted[mid] < m) a = mid + 1; else b = mid; }
    const s = (n - a) / n;
    if (s > 0) out.push({ m, s });
  }
  return out;
}

export function calibrationBins(calib, bins = 10) {
  if (!calib.length) return [];
  const sorted = [...calib].sort((a, b) => a[0] - b[0]);
  const size = Math.max(1, Math.floor(sorted.length / bins));
  const out = [];
  for (let i = 0; i < sorted.length; i += size) {
    const chunk = sorted.slice(i, i + size);
    const p = chunk.reduce((a, c) => a + c[0], 0) / chunk.length;
    const y = chunk.reduce((a, c) => a + c[1], 0) / chunk.length;
    out.push({ p, y, n: chunk.length });
  }
  // fold a small remainder into the previous bin
  if (out.length > 1 && out[out.length - 1].n < size / 2) {
    const a = out[out.length - 2], b = out.pop(), n = a.n + b.n;
    out[out.length - 1] = { p: (a.p * a.n + b.p * b.n) / n, y: (a.y * a.n + b.y * b.n) / n, n };
  }
  return out;
}

export function parseRounds(text) {
  text = text.trim();
  if (!text) return [];
  try {
    const j = JSON.parse(text);
    const arr = Array.isArray(j) ? j : j.x || j.rounds || j.multipliers || [];
    return arr.map((r) => (typeof r === 'object' ? +(r.multiplier ?? r.x ?? r.value) : +r)).filter((v) => v >= 1);
  } catch { /* not JSON */ }
  const lines = text.split(/\r?\n/);
  const header = lines[0].toLowerCase();
  if (/[a-z]/.test(header) && header.includes(',')) {
    const cols = header.split(',').map((c) => c.trim());
    const idx = cols.findIndex((c) => /multiplier|crash|value|^x$|result/.test(c));
    if (idx >= 0) return lines.slice(1).map((l) => parseFloat(l.split(',')[idx])).filter((v) => v >= 1);
  }
  return text.split(/[\s,;]+/).map((s) => parseFloat(s.replace(/x$/i, ''))).filter((v) => v >= 1);
}
