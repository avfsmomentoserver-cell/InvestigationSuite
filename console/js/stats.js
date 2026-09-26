// Browser-side versions of the pipeline tests, used by Round Lab.
const RTP = 0.97;
export const LADDER = [1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];

export function auditLadder(xs, rtp = RTP) {
  const n = xs.length;
  return LADDER.map((m) => {
    const k = xs.reduce((a, x) => a + (x >= m ? 1 : 0), 0);
    const p = rtp / m;
    const se = Math.sqrt((p * (1 - p)) / n);
    return { m, hits: k, rtp: (m * k) / n, se: m * se, z: (k / n - p) / se };
  });
}

export function runsTest(xs, m = 2) {
  const y = xs.map((x) => (x >= m ? 1 : 0));
  const n = y.length;
  const n1 = y.reduce((a, b) => a + b, 0);
  const n0 = n - n1;
  let runs = 1;
  for (let i = 1; i < n; i++) if (y[i] !== y[i - 1]) runs++;
  const mu = (2 * n1 * n0) / n + 1;
  const v = ((mu - 1) * (mu - 2)) / (n - 1);
  return { runs, mu, z: (runs - mu) / Math.sqrt(v) };
}

export function acf(xs, lags = 10) {
  const l = xs.map(Math.log);
  const mean = l.reduce((a, b) => a + b, 0) / l.length;
  const c = l.map((v) => v - mean);
  const den = c.reduce((a, v) => a + v * v, 0);
  const out = [];
  for (let k = 1; k <= lags; k++) {
    let s = 0;
    for (let i = k; i < c.length; i++) s += c[i] * c[i - k];
    out.push(s / den);
  }
  return out;
}

export function moonLevels(xs, levels = [10, 50, 100, 500, 1000], rtp = RTP) {
  return levels.map((m) => {
    let last = -1;
    let hits = 0;
    xs.forEach((x, i) => { if (x >= m) { last = i; hits++; } });
    const p = rtp / m;
    const current = last < 0 ? xs.length : xs.length - 1 - last;
    return { m, hits, expected: xs.length * p, current, pDry: Math.pow(1 - p, current) };
  });
}
