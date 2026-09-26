// Prediction pipeline pages: Pipeline, Leaderboard, Features, Calibration, Backtest, Controls, Next Round.
// Data: data/predict.json, written by pipeline/predict.py.

const TG = [1.5, 2, 3, 5, 10, 20, 50, 100];
const MCOL = { house: '#5d616b', empirical: '#8fb4ff', logit: '#f0b34a', hgb: '#c49bff' };
let P = null;

export async function loadPredict() {
  if (P) return P;
  P = await (await fetch('data/predict.json')).json();
  return P;
}

// ─────────── feature builder (mirror of build_features in pipeline/predict.py, next-round row only) ───────────
export function nextFeatures(x, t) {
  const n = x.length;
  const lx = x.map(Math.log);
  const F = {};
  for (let k = 1; k <= 10; k++) F[`lag${k}_log`] = lx[n - k];
  F['prev_ge1.5'] = +(x[n - 1] >= 1.5);
  F['prev_ge2'] = +(x[n - 1] >= 2);
  F['prev_ge10'] = +(x[n - 1] >= 10);
  F['lag2_ge2'] = +(x[n - 2] >= 2);
  F['lag3_ge2'] = +(x[n - 3] >= 2);
  const meanLast = (arr, w) => { const lo = Math.max(0, n - w); let s = 0; for (let i = lo; i < n; i++) s += arr[i]; return s / Math.max(1, n - lo); };
  const h2 = x.map((v) => +(v >= 2));
  const h10 = x.map((v) => +(v >= 10));
  for (const w of [10, 50, 200]) F[`rate2_w${w}`] = meanLast(h2, w);
  for (const w of [10, 50, 200]) F[`rate10_w${w}`] = meanLast(h10, w);
  const lx2 = lx.map((v) => v * v);
  for (const w of [20, 100]) {
    const m = meanLast(lx, w);
    F[`mean_log_w${w}`] = m;
    F[`std_log_w${w}`] = Math.sqrt(Math.max(0, meanLast(lx2, w) - m * m));
  }
  const runWhile = (pred) => { let r = 0; for (let i = n - 1; i >= 0 && pred(x[i]); i--) r++; return r; };
  F['run_below2'] = runWhile((v) => v < 2);
  F['run_below1.5'] = runWhile((v) => v < 1.5);
  const since = (pred) => { for (let i = n - 1; i >= 0; i--) if (pred(x[i])) return n - i; return n + 1; };
  F['since_10x'] = Math.log1p(since((v) => v >= 10));
  F['since_100x'] = Math.log1p(since((v) => v >= 100));
  F['since_instant'] = Math.log1p(since((v) => v < 1.01));
  const tp = t[n - 1];
  const hour = (((tp % 86400) + 86400) % 86400) / 3600;
  F['hour_sin'] = Math.sin((2 * Math.PI * hour) / 24);
  F['hour_cos'] = Math.cos((2 * Math.PI * hour) / 24);
  F['weekend'] = +(((Math.floor(tp / 86400) + 4) % 7) >= 5);
  F['gap_prev_log'] = Math.log1p(Math.min(3600, Math.max(0, t[n - 1] - t[n - 2])));
  let last = 0;
  for (let k = n - 1; k >= 1; k--) if (t[k] - t[k - 1] > 300) { last = k; break; }
  F['session_pos'] = Math.log1p(n - last);
  F['mean_log_all'] = meanLast(lx, 1e9);
  F['rate2_all'] = meanLast(h2, 1e9);
  let mx = -Infinity; for (let i = Math.max(0, n - 50); i < n; i++) mx = Math.max(mx, lx[i]);
  let mn = Infinity; for (let i = Math.max(0, n - 10); i < n; i++) mn = Math.min(mn, lx[i]);
  F['max_log_w50'] = mx;
  F['min_log_w10'] = mn;
  const ewm = (arr, a) => { let num = 0, den = 0, w = 1; for (let i = n - 1; i >= 0; i--) { num += w * arr[i]; den += w; w *= 1 - a; if (w < 1e-12) break; } return num / den; };
  F['ewm_log_a10'] = ewm(lx, 0.1);
  F['ewm_ge2_a05'] = ewm(h2, 0.05);
  F['color_run'] = F['run_below2'] - runWhile((v) => v > 2);
  return F;
}

function logitP(W, vec) {
  let z = W.b;
  for (let i = 0; i < vec.length; i++) z += W.coef[i] * ((vec[i] - W.mean[i]) / W.scale[i]);
  return 1 / (1 + Math.exp(-z));
}

// ─────────── shared bits ───────────
function zColor(z) {
  if (z == null) return 'transparent';
  const a = Math.min(1, Math.abs(z) / 4);
  if (z >= 3) return `rgba(255,111,94,${0.35 + 0.4 * a})`;
  if (z > 0) return `rgba(240,179,74,${0.08 + 0.45 * a})`;
  return `rgba(143,180,255,${0.05 + 0.25 * a})`;
}
function verdict(v) {
  const m = { edge: 'bad', noise: 'acc', worse: 'ok' };
  const t = { edge: 'edge', noise: 'noise', worse: 'worse than house' };
  return `<span class="pill ${m[v] || ''}">${t[v] || v}</span>`;
}
function loading(v, C) { v.innerHTML = `<div class="skeleton-page"><div class="sk"></div><div class="sk"></div><div class="sk tall"></div></div>`; }
async function withData(v, C, fn) {
  const seq = C.state.seq;
  if (!P) loading(v, C);
  try { await loadPredict(); } catch (e) { v.innerHTML = `<div class="card err">Could not load data/predict.json: ${C.esc(e.message)}. Run <code>python pipeline/predict.py --repo …</code> first.</div>`; return; }
  if (C.state.seq !== seq) return;
  try { await fn(); } catch (e) { console.error(e); v.innerHTML = `<div class="card err">Render error: ${C.esc(e.message)}</div>`; }
}
const seg = (id, opts, cur) => `<div class="seg" id="${id}" role="tablist">${opts.map(([k, l]) => `<button role="tab" data-k="${k}" class="${String(k) === String(cur) ? 'on' : ''}">${l}</button>`).join('')}</div>`;
function bindSeg(id, cb) {
  const el = document.getElementById(id);
  el.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    el.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    cb(b.dataset.k);
  });
}

// ─────────── Pipeline ───────────
export function pagePipeline(v, C) {
  withData(v, C, () => {
    const { esc, n0, f, pct, sgn, card, kpi } = C;
    const c = P.config;
    const st = Object.fromEntries(P.stages.map((s) => [s.stage, s]));
    const edges = P.leaderboard.filter((b) => b.verdict === 'edge').length;
    const best = P.leaderboard[0];
    const pw = P.controls.power;
    const mde = pw.find((p) => p.detected === p.reps);
    const totalRounds = Object.values(P.streams).reduce((a, s) => a + s.n, 0);
    const nodes = [
      ['ingest', 'Ingest', `${n0(totalRounds)} clean rounds`, '5 streams from the decomposition; quarantined rows never enter.'],
      ['features', 'Features', `${P.features.length} causal features`, 'Lags, rolling hit rates, streaks, droughts, session and clock. Row i sees rounds < i only.'],
      ['targets', 'Targets', `${TG.length} thresholds`, 'y = 1 if the round reaches 1.5x … 100x.'],
      ['models', 'Models', `${c.models.length} + house`, 'Empirical rate, L2 logistic, gradient boosting, against the 0.97/m curve.'],
      ['validate', 'Walk-forward', `${c.folds} folds · purge ${c.purge}`, `Expanding train window from ${pct(c.init_train, 0)} of the data. Test rounds are always later than training rounds.`],
      ['score', 'Score', `${(st.score?.detail?.tests ?? 120)} tests · FDR 5%`, `Log-loss skill vs house, block bootstrap (${c.block}-round blocks, ${c.bootstrap} draws), Benjamini–Hochberg.`],
      ['calibrate', 'Calibrate', 'reliability + ECE', 'Do predicted probabilities match observed frequencies?'],
      ['explain', 'Explain', 'correlation + permutation', 'Which features move the needle, and by how much.'],
      ['backtest', 'Backtest', `EV gate ${c.ev_gate}`, 'Paper strategy on out-of-fold predictions against a fair-game Monte Carlo fan.'],
      ['controls', 'Controls', '4 checks', 'Shuffled stream, planted edge, timestamp leak, burst leak. Proves the tests can fire.'],
      ['deploy', 'Deploy', 'weights → browser', 'Final fit on all rounds; logistic weights run live on the Next Round page.'],
    ];
    v.innerHTML = `
    <p class="lede">A reproducible prediction pipeline over the clean backups. Every stream goes through the same eleven stages, every model is scored only on rounds it has never seen, and every claim of an edge has to survive a correction for the ${(st.score?.detail?.tests ?? 120)} tests being run at once.</p>
    <div class="grid g5">
      ${kpi('Hypotheses tested', n0((st.score?.detail?.tests ?? 120)), '5 streams × 8 targets × 3 models')}
      ${kpi('Edges after FDR', `${edges}`, edges ? 'see leaderboard' : 'q < 0.05 required', edges ? 'bad' : 'ok')}
      ${kpi('Best raw z', sgn(best.z), `${esc(C.label(best.stream))} · ${best.m}x · ${esc(c.model_label[best.model])}`, best.z >= 3 ? 'bad' : '')}
      ${kpi('Min detectable edge', mde ? `+${f(mde.effect * 100, 0)} pp` : '–', 'planted in a fair stream, same n as Aviator', 'info')}
      ${kpi('Run time', `${f(P.runtime_sec, 0)} s`, `data ${esc(P.data_hash)}`)}
    </div>
    ${card('Pipeline', 'Click a stage to jump to its evidence. Timings are from the last run.', `
      <ol class="dag">${nodes.map(([k, name, metric, text], i) => {
        const s = st[k];
        const link = { validate: 'leaderboard', score: 'leaderboard', calibrate: 'calibration', explain: 'features', backtest: 'backtest', controls: 'controls', deploy: 'nextround', features: 'features' }[k];
        const inner = `<span class="dag-n">${String(i + 1).padStart(2, '0')}</span><span class="dag-t">${name}</span><span class="dag-m mono">${metric}</span><span class="dag-d">${text}</span>${s ? `<span class="dag-s mono">${f(s.sec, 1)} s</span>` : ''}`;
        return `<li class="dag-node ${k === 'controls' ? 'ctl' : ''}">${link ? `<a href="#/${link}">${inner}</a>` : `<div>${inner}</div>`}</li>`;
      }).join('')}</ol>`, 'mt')}
    <div class="grid g2 mt">
      ${card('Model cards', 'What each model sees and how it is constrained.', `
        <div class="tbl-wrap"><table><thead><tr><th>Model</th><th>Inputs</th><th>Constraints</th></tr></thead><tbody>
          <tr><td><span style="color:${MCOL.house}">●</span> House curve</td><td>nothing</td><td class="small wrap">P(x ≥ m) = ${c.rtp}/m. The null every model must beat.</td></tr>
          <tr><td><span style="color:${MCOL.empirical}">●</span> Empirical rate</td><td>training labels</td><td class="small wrap">Hit rate of the training window. Tests whether the stream's RTP differs from 97%.</td></tr>
          <tr><td><span style="color:${MCOL.logit}">●</span> Logistic (L2)</td><td>${P.features.length} features</td><td class="small wrap">Standardised, C = ${c.logit.C}. Linear effects only; weights exported to the browser.</td></tr>
          <tr><td><span style="color:${MCOL.hgb}">●</span> Gradient boosting</td><td>${P.features.length} features</td><td class="small wrap">${c.hgb.max_iter} trees max, ${c.hgb.max_leaf_nodes} leaves, ≥ ${c.hgb.min_samples_leaf} rounds per leaf, early stopping. Catches interactions.</td></tr>
        </tbody></table></div>`)}
      ${card('Top of the leaderboard', 'Highest z out of 120. With this many tests, a few z near 2 are expected from noise.', `
        <div class="tbl-wrap"><table><thead><tr><th>Stream</th><th class="num">Target</th><th>Model</th><th class="num">Skill</th><th class="num">z</th><th class="num">q</th><th>Verdict</th></tr></thead><tbody>
        ${P.leaderboard.slice(0, 6).map((b) => `<tr><td>${esc(C.label(b.stream))}</td><td class="num">${b.m}x</td><td>${esc(c.model_label[b.model])}</td><td class="num">${sgn(b.skill, 2)}</td><td class="num">${sgn(b.z)}</td><td class="num">${f(b.q, 3)}</td><td>${verdict(b.verdict)}</td></tr>`).join('')}
        </tbody></table></div><p class="tiny faint mt">Skill is in milli-nats per round of log-loss saved against the house curve. q is the Benjamini–Hochberg adjusted p-value.</p>`)}
    </div>
    ${card('Run it yourself', 'Everything on these pages comes from one command. Scripts are on the Method & Downloads page.', `
      <pre class="code">git clone -b investigation-console https://github.com/avfsmomentoserver-cell/InvestigationSuite
cd InvestigationSuite/console
pip install numpy pandas scikit-learn
python pipeline/decompose.py --repo ..     # clean streams  -> data/suite.json
python pipeline/predict.py   --repo ..     # this pipeline  -> data/predict.json
python pipeline/predict.py   --score my_rounds.csv   # same tests on any history
python -m http.server 8000                 # open http://localhost:8000</pre>`, 'mt')}`;
  });
}

// ─────────── Leaderboard ───────────
export function pageLeaderboard(v, C) {
  withData(v, C, () => {
    const { esc, n0, f, sgn, card, kpi, chart } = C;
    const c = P.config;
    const streams = Object.keys(P.streams);
    let metric = 'z';
    const all = P.leaderboard;
    const pos = all.filter((b) => b.skill > 0).length;
    const z2 = all.filter((b) => b.z >= 2).length;
    const worst = [...all].sort((a, b) => a.z - b.z)[0];
    v.innerHTML = `
    <p class="lede">Every model on every stream at every target, scored walk-forward against the house curve. A real edge would show up as a red cell that survives the false-discovery correction. Blue means the model did worse than simply assuming ${c.rtp}/m.</p>
    <div class="grid g4">
      ${kpi('Cells', n0(all.length), 'stream × model × target')}
      ${kpi('Beat the house at all', `${pos}`, `${f((pos / all.length) * 100, 0)}% of cells, by any margin`)}
      ${kpi('z ≥ 2', `${z2}`, `~${f(all.length * 0.023, 1)} expected by chance`, z2 > all.length * 0.05 ? 'acc' : '')}
      ${kpi('Edges (q < 0.05)', `${all.filter((b) => b.verdict === 'edge').length}`, 'after Benjamini–Hochberg', 'ok')}
    </div>
    ${card('Skill heatmap', 'Rows are stream and model, columns are targets.', `
      <div class="row" style="margin-bottom:12px">${seg('lb-metric', [['z', 'z-score'], ['skill', 'skill (mnat)'], ['auc', 'AUC'], ['q', 'q-value']], metric)}</div>
      <div class="tbl-wrap"><table class="hm" id="lb-hm"></table></div>
      <div class="legend-row tiny faint mt"><span><i style="background:rgba(143,180,255,.3)"></i>worse than house</span><span><i style="background:rgba(240,179,74,.35)"></i>better, not significant</span><span><i style="background:rgba(255,111,94,.7)"></i>z ≥ 3</span></div>`, 'mt')}
    <div class="grid g2 mt">
      ${card('Distribution of z', 'Most cells sit below zero: every fitted model pays a variance cost against the exact house curve. A real edge would show up as a tail past z = 3; there is none.', `<div class="chart short"><canvas id="c-zhist"></canvas></div>`)}
      ${card('Fold stability', 'Skill per walk-forward fold at 2x. A real signal stays on one side of zero across folds.', `
        <div class="row" style="margin-bottom:8px">${seg('lb-src', streams.map((s) => [s, C.label(s)]), 'aviator')}</div>
        <div class="chart short"><canvas id="c-fold"></canvas></div>`)}
    </div>
    ${card('Most negative cell', '', `<p class="small muted">${esc(C.label(worst.stream))} · ${worst.m}x · ${esc(c.model_label[worst.model])}: z ${sgn(worst.z)}. Flexible models pay for every parameter they fit; on a memoryless game that cost shows up as negative skill, not as an edge.</p>`, 'mt')}`;
    const draw = () => {
      const fmt = { z: (s) => sgn(s.z, 1), skill: (s) => sgn(s.skill, 2), auc: (s) => f(s.auc, 3), q: (s) => f(s.q, 2) }[metric];
      const col = (s) => (metric === 'auc' ? zColor(((s.auc || 0.5) - 0.5) * 200) : metric === 'q' ? (s.q < 0.05 ? zColor(4) : s.q < 0.5 ? zColor(1) : 'transparent') : zColor(s.z));
      document.getElementById('lb-hm').innerHTML = `<thead><tr><th>Stream</th><th>Model</th>${TG.map((m) => `<th class="num">${m}x</th>`).join('')}</tr></thead><tbody>
        ${streams.map((s) => c.models.map((md, i) => `<tr class="${i === 0 ? 'grp-top' : ''}"><td>${i === 0 ? `<b>${esc(C.label(s))}</b>` : ''}</td><td class="small"><span style="color:${MCOL[md]}">●</span> ${esc(c.model_label[md])}</td>${TG.map((m) => { const sc = P.streams[s].scores[m][md]; return `<td class="num hm-c" style="background:${col(sc)}" title="skill ${sgn(sc.skill, 3)} mnat · z ${sgn(sc.z)} · q ${f(sc.q, 3)} · AUC ${f(sc.auc, 3)}">${fmt(sc)}</td>`; }).join('')}</tr>`).join('')).join('')}</tbody>`;
    };
    draw();
    bindSeg('lb-metric', (k) => { metric = k; draw(); });
    const bins = [];
    for (let b = -5; b < 4; b += 0.5) bins.push(b);
    const counts = bins.map((b) => all.filter((x) => x.z >= b && x.z < b + 0.5).length);
    chart('c-zhist', {
      type: 'bar',
      data: { labels: bins.map((b) => b.toFixed(1)), datasets: [{ data: counts, backgroundColor: bins.map((b) => (b >= 3 ? '#ff6f5e' : b >= 2 ? '#f0b34a' : b >= 0 ? '#6b5a3a' : '#3a4560')), borderRadius: 2 }] },
      options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'z' } }, y: { title: { display: true, text: 'cells' } } } },
    });
    let fc = null;
    const drawFold = (s) => {
      if (fc) { fc.destroy(); C.state.charts.splice(C.state.charts.indexOf(fc), 1); }
      const sc = P.streams[s].scores['2'];
      fc = chart('c-fold', {
        type: 'line',
        data: { labels: sc.logit.fold_skill.map((_, i) => `fold ${i + 1}`), datasets: c.models.map((md) => ({ label: c.model_label[md], data: sc[md].fold_skill, borderColor: MCOL[md], backgroundColor: MCOL[md], pointRadius: 3, borderWidth: 1.6 })) },
        options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { title: { display: true, text: 'mnat / round' }, grid: { color: (x) => (x.tick.value === 0 ? '#4a4f5c' : '#1f2229') } } } },
      });
    };
    drawFold('aviator');
    bindSeg('lb-src', drawFold);
  });
}

// ─────────── Features ───────────
export function pageFeatures(v, C) {
  withData(v, C, () => {
    const { esc, f, sgn, card, kpi, chart, state } = C;
    const s = state.source;
    const S = P.streams[s];
    const streams = Object.keys(P.streams);
    let tgt = '2';
    const zs = streams.flatMap((k) => P.streams[k].corr['2']);
    const over3 = zs.filter((z) => Math.abs(z) >= 3).length;
    const imp = S.importance;
    const W = S.weights['2'];
    v.innerHTML = `
    <p class="lede">What the models had to work with. Each cell is the correlation between a feature and the next round reaching the target, expressed as a z-score (r·√n). On a memoryless game every cell is noise; with ${zs.length} cells per target, about ${f(zs.length * 0.0027, 1)} should pass |z| ≥ 3 by luck.</p>
    <div class="grid g4">
      ${kpi('Features', P.features.length, 'all strictly causal')}
      ${kpi('|z| ≥ 3 at 2x', over3, `of ${zs.length}; ~${f(zs.length * 0.0027, 1)} expected`, over3 > 3 ? 'acc' : 'ok')}
      ${kpi('Top permutation gain', imp ? `${f(imp[0].imp, 2)} mnat` : '–', imp ? `${esc(imp[0].f)} on ${esc(C.label(s))}` : 'run without --fast for this stream')}
      ${kpi('Largest logistic weight', W ? f(Math.max(...W.coef.map(Math.abs)), 3) : '–', 'standardised, 2x target')}
    </div>
    ${card('Feature × stream correlation', 'z-score of the correlation with the next-round label. Hover a cell for the raw value.', `
      <div class="row" style="margin-bottom:12px">${seg('ft-t', [['2', 'next ≥ 2x'], ['10', 'next ≥ 10x']], tgt)}</div>
      <div class="tbl-wrap" style="max-height:560px"><table class="hm" id="ft-hm"></table></div>`, 'mt')}
    <div class="grid g2 mt">
      ${card(`Permutation importance · ${esc(C.label(s))}`, 'Log-loss lost on held-out rounds when a feature is shuffled (gradient boosting, 2x). Bars within their error are nothing.', imp ? `<div class="chart tall"><canvas id="c-imp"></canvas></div>` : `<p class="small muted">Computed for Aviator, Skyward and Skyward Deluxe.</p>`)}
      ${card(`Logistic weights · ${esc(C.label(s))}`, 'Standardised coefficients of the final 2x model. The L2 penalty pulls noise towards zero.', W ? `<div class="chart tall"><canvas id="c-coef"></canvas></div>` : '')}
    </div>`;
    const draw = () => {
      document.getElementById('ft-hm').innerHTML = `<thead><tr><th>Feature</th>${streams.map((k) => `<th class="num">${esc(C.label(k))}</th>`).join('')}</tr></thead><tbody>
        ${P.features.map((fn, i) => `<tr><td class="mono small">${esc(fn)}</td>${streams.map((k) => { const z = P.streams[k].corr[tgt][i]; return `<td class="num hm-c" style="background:${zColor(Math.abs(z))}" title="${esc(fn)} · ${esc(C.label(k))} · z ${sgn(z)}">${sgn(z, 1)}</td>`; }).join('')}</tr>`).join('')}</tbody>`;
    };
    draw();
    bindSeg('ft-t', (k) => { tgt = k; draw(); });
    if (imp) {
      const top = imp.slice(0, 16);
      chart('c-imp', {
        type: 'bar',
        data: { labels: top.map((d) => d.f), datasets: [
          { label: 'gain', data: top.map((d) => d.imp), backgroundColor: top.map((d) => (d.imp > 2 * d.sd ? '#f0b34a' : '#3a3f4b')), borderRadius: 2 },
          { label: '±1 sd', data: top.map((d) => [d.imp - d.sd, d.imp + d.sd]), backgroundColor: 'rgba(143,180,255,.25)', barPercentage: 0.25, grouped: false },
        ] },
        options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'mnat / round' } }, y: { ticks: { font: { size: 10 } } } } },
      });
    }
    if (W) {
      const idx = P.features.map((_, i) => i).sort((a, b) => Math.abs(W.coef[b]) - Math.abs(W.coef[a])).slice(0, 16);
      chart('c-coef', {
        type: 'bar',
        data: { labels: idx.map((i) => P.features[i]), datasets: [{ data: idx.map((i) => W.coef[i]), backgroundColor: idx.map((i) => (W.coef[i] > 0 ? '#f0b34a' : '#8fb4ff')), borderRadius: 2 }] },
        options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { y: { ticks: { font: { size: 10 } } } } },
      });
    }
  });
}

// ─────────── Calibration ───────────
export function pageCalibration(v, C) {
  withData(v, C, () => {
    const { esc, f, pct, card, kpi, chart, state } = C;
    const s = state.source;
    const S = P.streams[s];
    const models = P.config.models;
    let tgt = '2';
    v.innerHTML = `
    <p class="lede">A forecast is only useful if its probabilities mean what they say. Out-of-fold predictions for ${esc(C.label(s))} are grouped into ten bins; each point compares the average prediction with how often the round actually reached the target.</p>
    <div class="grid g4" id="cal-kpi"></div>
    ${card('Reliability diagram', 'On the diagonal is perfect calibration. Error bars are ±2 standard errors of the observed rate.', `
      <div class="row" style="margin-bottom:12px">${seg('cal-t', TG.map((m) => [String(m), m + 'x']), tgt)}</div>
      <div class="chart tall"><canvas id="c-rel"></canvas></div>`, 'mt')}
    ${card('Scores by target', 'Log-loss and Brier are lower-is-better; the house row is the bar to clear. AUC 0.5 means no ranking ability.', `<div class="tbl-wrap"><table id="cal-tbl"></table></div>`, 'mt')}`;
    let rc = null;
    const draw = () => {
      const sc = S.scores[tgt];
      document.getElementById('cal-kpi').innerHTML = [
        kpi('Observed rate', pct(sc.house.rate_obs, 2), `house ${pct(sc.house.rate_house, 2)}`),
        ...models.map((md) => kpi(`${P.config.model_label[md]} ECE`, pct(sc[md].ece, 2), `AUC ${f(sc[md].auc, 3)}`)),
      ].join('');
      if (rc) { rc.destroy(); state.charts.splice(state.charts.indexOf(rc), 1); }
      const pts = models.flatMap((md) => sc[md].rel.map((b) => b.p));
      const lo = Math.max(0, Math.min(...pts, sc.house.rate_house) * 0.9);
      const hi = Math.min(1, Math.max(...pts, sc.house.rate_house) * 1.1);
      rc = chart('c-rel', {
        type: 'scatter',
        data: { datasets: [
          { label: 'perfect', data: [{ x: lo, y: lo }, { x: hi, y: hi }], type: 'line', borderColor: '#4a4f5c', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
          ...models.map((md) => ({ label: P.config.model_label[md], data: sc[md].rel.map((b) => ({ x: b.p, y: b.o, n: b.n, se: b.se })), borderColor: MCOL[md], backgroundColor: MCOL[md], showLine: true, pointRadius: 3.5, borderWidth: 1.4 })),
          ...models.filter((md) => md !== 'empirical').map((md) => ({ label: '', data: sc[md].rel.flatMap((b) => [{ x: b.p, y: b.o - 2 * b.se }, { x: b.p, y: b.o + 2 * b.se }, { x: null, y: null }]), type: 'line', borderColor: MCOL[md] + '55', pointRadius: 0, borderWidth: 1, spanGaps: false })),
        ] },
        options: { plugins: { legend: { position: 'bottom', labels: { filter: (i) => i.text } }, tooltip: { callbacks: { label: (c) => (c.raw.n ? `pred ${pct(c.raw.x, 2)} · obs ${pct(c.raw.y, 2)} · n ${c.raw.n}` : '') } } },
          scales: { x: { type: 'linear', min: lo, max: hi, title: { display: true, text: 'predicted' }, ticks: { callback: (v) => pct(v, 1) } }, y: { min: Math.max(0, lo - 0.05), max: Math.min(1, hi + 0.05), title: { display: true, text: 'observed' }, ticks: { callback: (v) => pct(v, 1) } } } },
      });
      document.getElementById('cal-tbl').innerHTML = `<thead><tr><th class="num">Target</th><th>Model</th><th class="num">Log-loss</th><th class="num">Brier</th><th class="num">AUC</th><th class="num">ECE</th><th class="num">Skill</th></tr></thead><tbody>
        ${TG.map((m) => ['house', ...models].map((md, i) => { const x = S.scores[m][md]; return `<tr class="${i === 0 ? 'grp-top' : ''} ${String(m) === tgt ? 'sel' : ''}"><td class="num">${i === 0 ? m + 'x' : ''}</td><td class="small"><span style="color:${MCOL[md]}">●</span> ${esc(P.config.model_label[md])}</td><td class="num">${f(x.ll, 5)}</td><td class="num">${f(x.brier, 5)}</td><td class="num">${x.auc == null ? '–' : f(x.auc, 3)}</td><td class="num">${x.ece == null ? '–' : pct(x.ece, 2)}</td><td class="num ${x.skill > 0 ? 'acc' : 'faint'}">${md === 'house' ? '0' : C.sgn(x.skill, 3)}</td></tr>`; }).join('')).join('')}</tbody>`;
    };
    draw();
    bindSeg('cal-t', (k) => { tgt = k; draw(); });
  });
}

// ─────────── Backtest ───────────
export function pageBacktest(v, C) {
  withData(v, C, () => {
    const { esc, n0, f, pct, sgn, card, kpi, chart, state } = C;
    const s = state.source;
    const B = P.streams[s].backtest;
    let md = 'logit';
    const models = ['empirical', 'logit', 'hgb'];
    v.innerHTML = `
    <p class="lede">What would have happened with real stakes. On every out-of-fold round the model picks the target with the best expected value and bets 1 unit if that EV beats ${P.config.ev_gate}. The shaded fan is the same bet sequence replayed 600 times in a fair 97% game: if the model had an edge, its line would climb out of the top of the fan.</p>
    <div class="grid g4" id="bt-kpi"></div>
    ${card('Equity vs fair-game fan', `${esc(C.label(s))}, ${n0(P.streams[s].test_rows)} out-of-fold rounds.`, `
      <div class="row" style="margin-bottom:12px">${seg('bt-m', models.map((m) => [m, P.config.model_label[m]]), md)}</div>
      <div class="chart tall"><canvas id="c-bt"></canvas></div>`, 'mt')}
    ${card('Strategies', 'Percentile vs null: where the final P&L lands among fair-game replays of the same bets. 50% is exactly what luck gives.', `<div class="tbl-wrap"><table id="bt-tbl"></table></div>`, 'mt')}`;
    let bc = null;
    const draw = () => {
      const b = B[md];
      document.getElementById('bt-kpi').innerHTML = [
        kpi('Bets', n0(b.bets), `of ${n0(P.streams[s].test_rows)} rounds`),
        kpi('P&L', `${sgn(b.pnl, 0)} u`, b.roi == null ? 'no bets placed' : `ROI ${sgn(b.roi * 100, 1)}%`, b.pnl < 0 ? 'bad' : 'ok'),
        kpi('Max drawdown', `${f(b.max_dd, 0)} u`, ''),
        kpi('Percentile vs null', b.pct_vs_null == null ? '–' : pct(b.pct_vs_null, 0), 'an edge would sit above 95%', b.pct_vs_null > 0.95 ? 'bad' : ''),
      ].join('');
      if (bc) { bc.destroy(); state.charts.splice(state.charts.indexOf(bc), 1); }
      const xs = b.eq.map((_, i) => i * B.stride);
      const pt = (arr) => arr.map((y, i) => ({ x: xs[i], y }));
      bc = chart('c-bt', {
        type: 'line',
        data: { datasets: [
          { label: '5–95%', data: pt(b.fan[4]), borderWidth: 0, pointRadius: 0, backgroundColor: 'rgba(143,180,255,.08)', fill: '+1' },
          { label: '', data: pt(b.fan[0]), borderWidth: 0, pointRadius: 0 },
          { label: '25–75%', data: pt(b.fan[3]), borderWidth: 0, pointRadius: 0, backgroundColor: 'rgba(143,180,255,.16)', fill: '+1' },
          { label: '', data: pt(b.fan[1]), borderWidth: 0, pointRadius: 0 },
          { label: 'fair-game median', data: pt(b.fan[2]), borderColor: '#5d616b', borderDash: [4, 4], borderWidth: 1, pointRadius: 0 },
          { label: 'always bet 2x', data: pt(B.always2x.eq), borderColor: '#3f4450', borderWidth: 1.2, pointRadius: 0 },
          { label: P.config.model_label[md], data: pt(b.eq), borderColor: MCOL[md], borderWidth: 2, pointRadius: 0 },
        ] },
        options: { parsing: false, interaction: { mode: 'nearest', intersect: false }, plugins: { legend: { position: 'bottom', labels: { filter: (i) => i.text } } }, scales: { x: { type: 'linear', title: { display: true, text: 'out-of-fold round' } }, y: { title: { display: true, text: 'units' } } } },
      });
    };
    draw();
    bindSeg('bt-m', (k) => { md = k; draw(); });
    document.getElementById('bt-tbl').innerHTML = `<thead><tr><th>Strategy</th><th class="num">Bets</th><th class="num">Hit rate</th><th class="num">P&L</th><th class="num">ROI</th><th class="num">Max DD</th><th class="num">vs null</th><th>Target mix</th></tr></thead><tbody>
      ${[...models.map((m) => [P.config.model_label[m], B[m]]), ['Always bet 2x', B.always2x]].map(([name, b]) => `<tr><td>${esc(name)}</td><td class="num">${n0(b.bets)}</td><td class="num">${b.hit == null ? '–' : pct(b.hit, 1)}</td><td class="num ${b.pnl < 0 ? 'bad' : ''}">${sgn(b.pnl, 0)}</td><td class="num">${b.roi == null ? '–' : sgn(b.roi * 100, 1) + '%'}</td><td class="num">${b.max_dd == null ? '–' : f(b.max_dd, 0)}</td><td class="num">${b.pct_vs_null == null ? '–' : pct(b.pct_vs_null, 0)}</td><td class="small faint">${b.targets ? Object.entries(b.targets).sort((a, c) => c[1] - a[1]).slice(0, 4).map(([k, n]) => `${k}x ${n0(n)}`).join(' · ') || '–' : '2x every round'}</td></tr>`).join('')}</tbody>`;
  });
}

// ─────────── Controls ───────────
export function pageControls(v, C) {
  withData(v, C, () => {
    const { esc, n0, f, pct, sgn, card, kpi, chart } = C;
    const K = P.controls;
    const pw = K.power;
    const mde = pw.find((p) => p.detected === p.reps);
    const bl = K.burst_leak;
    const kf = bl.random_kfold;
    const tl = K.timestamp_leak;
    const sh = K.shuffled;
    v.innerHTML = `
    <p class="lede">A pipeline that never finds anything is only convincing if it can find something when it is there, and if it fails loudly when the data is broken. These four checks run on every build.</p>
    <div class="grid g4">
      ${kpi('Shuffled Aviator', `z ${sgn(Math.max(sh.m2.logit.z, sh.m2.hgb.z))}`, 'order destroyed; must stay below 3', 'ok')}
      ${kpi('Planted edge found at', mde ? `+${f(mde.effect * 100, 0)} pp` : '–', `${mde ? mde.reps + '/' + mde.reps : ''} runs, z ${mde ? f(mde.z_mean, 1) : ''}`, 'info')}
      ${kpi('Timestamp leak', `z ${f(tl.m2.hgb.z, 0)}`, `AUC ${f(tl.m2.hgb.auc, 3)} if the round's own end time slips in`, 'bad')}
      ${kpi('Burst leak (shuffled CV)', `z ${f(kf.with_burst.z, 1)}`, `${pct(kf.with_burst.acc_burst_rows, 0)} right on the ${n0(kf.with_burst.burst_rows)} burst rows`, 'bad')}
    </div>
    <div class="grid g2 mt">
      ${card('Power curve', 'A fair stream the size of Aviator with a planted rule: after a round under 1.5x, the next round reaches 2x more often. Each dot is one full walk-forward run of the logistic model.', `<div class="chart"><canvas id="c-pow"></canvas></div>
        <p class="small muted mt">The pipeline reliably flags a planted edge of ${mde ? '+' + f(mde.effect * 100, 0) + ' percentage points' : 'large size'} on about a third of rounds. Smaller edges can hide in data this size, but they are also too small to beat a 3% house margin after the bets they would need. At zero effect the logistic model scores below the house, which is the price of fitting ${P.features.length} weights to noise.</p>`)}
      ${card('Negative control: shuffled order', 'Same values, random order. Anything the models find here is overfitting.', `
        <div class="tbl-wrap"><table><thead><tr><th>Target</th><th>Model</th><th class="num">Skill</th><th class="num">z</th><th class="num">AUC</th></tr></thead><tbody>
        ${[['2x', sh.m2], ['10x', sh.m10]].flatMap(([t, o]) => ['logit', 'hgb'].map((m) => `<tr><td>${t}</td><td>${esc(P.config.model_label[m])}</td><td class="num">${sgn(o[m].skill, 3)}</td><td class="num">${sgn(o[m].z)}</td><td class="num">${f(o[m].auc, 3)}</td></tr>`)).join('')}
        </tbody></table></div>
        <p class="small muted mt">Shuffled and real Aviator score the same, which is what a memoryless game predicts.</p>`)}
    </div>
    <div class="grid g2 mt">
      ${card('Timestamp leak', 'The capture time of a round is written when it crashes, so the gap since the previous round contains its flight time.', `
        <div class="tbl-wrap"><table><thead><tr><th>Target</th><th>Model</th><th class="num">Skill</th><th class="num">z</th><th class="num">AUC</th></tr></thead><tbody>
        ${[['2x', tl.m2], ['10x', tl.m10]].flatMap(([t, o]) => ['logit', 'hgb'].map((m) => `<tr><td>${t}</td><td>${esc(P.config.model_label[m])}</td><td class="num bad">${sgn(o[m].skill, 1)}</td><td class="num bad">${sgn(o[m].z, 1)}</td><td class="num bad">${f(o[m].auc, 3)}</td></tr>`)).join('')}
        </tbody></table></div>
        <div class="callout bad mt"><b>Never use t[i] − t[i−1] as a feature.</b> It predicts the round almost perfectly because it is the round. The pipeline only uses the previous round's duration, which is known before the next round starts.</div>`)}
      ${card('Burst leak', `Aviator with the ${n0(kf.with_burst.burst_rows)} quarantined Jul 3 rows put back, gradient boosting at 2x.`, `
        <div class="tbl-wrap"><table><thead><tr><th>Validation</th><th>Data</th><th class="num">Skill</th><th class="num">z</th><th class="num">AUC</th></tr></thead><tbody>
          <tr><td>Shuffled 5-fold</td><td>with burst</td><td class="num bad">${sgn(kf.with_burst.skill, 1)}</td><td class="num bad">${sgn(kf.with_burst.z, 1)}</td><td class="num">${f(kf.with_burst.auc, 3)}</td></tr>
          <tr><td>Shuffled 5-fold</td><td>clean</td><td class="num">${sgn(kf.clean.skill, 2)}</td><td class="num">${sgn(kf.clean.z, 1)}</td><td class="num">${f(kf.clean.auc, 3)}</td></tr>
          <tr><td>Walk-forward window</td><td>with burst</td><td class="num">${sgn(bl.with_burst.hgb.skill, 1)}</td><td class="num">${sgn(bl.with_burst.hgb.z, 1)}</td><td class="num">${f(bl.with_burst.hgb.auc, 3)}</td></tr>
          <tr><td>Walk-forward window</td><td>clean</td><td class="num">${sgn(bl.clean.hgb.skill, 2)}</td><td class="num">${sgn(bl.clean.hgb.z, 1)}</td><td class="num">${f(bl.clean.hgb.auc, 3)}</td></tr>
        </tbody></table></div>
        <p class="small muted mt">With shuffled folds the model sees burst rows in training and recognises the rest in testing: it calls ${pct(kf.with_burst.acc_burst_rows, 1)} of them correctly, the same territory as the ${'77.5%'} replay spike on Jul 4–6. On all other rows its skill is ${sgn(kf.with_burst.skill_other_rows, 2)} mnat, i.e. nothing. Evaluated walk-forward, where the burst is unseen, the same models fall apart instead.</p>`)}
    </div>
    ${card('Walk-forward through the burst', 'Rolling 100-round mean: the logistic prediction against what actually happened, on the test window that contains the Jul 3 rows.', `<div class="chart short"><canvas id="c-bw"></canvas></div>`, 'mt')}`;
    chart('c-pow', {
      type: 'scatter',
      data: { datasets: [
        { label: 'runs', data: pw.flatMap((p) => p.z.map((z) => ({ x: p.effect * 100, y: z }))), backgroundColor: '#8fb4ff', pointRadius: 3.5 },
        { label: 'mean', type: 'line', data: pw.map((p) => ({ x: p.effect * 100, y: p.z_mean })), borderColor: '#f0b34a', pointRadius: 0, borderWidth: 1.6 },
        { label: 'z = 3', type: 'line', data: [{ x: 0, y: 3 }, { x: pw[pw.length - 1].effect * 100, y: 3 }], borderColor: '#ff6f5e', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
      ] },
      options: { plugins: { legend: { position: 'bottom' } }, scales: { x: { type: 'linear', title: { display: true, text: 'planted edge (percentage points at 2x)' } }, y: { title: { display: true, text: 'walk-forward z' } } } },
    });
    const wb = bl.with_burst;
    chart('c-bw', {
      type: 'line',
      data: { labels: wb.trace.map((_, i) => i * 50), datasets: [
        { label: 'observed ≥2x rate', data: wb.obs, borderColor: '#ff6f5e', pointRadius: 0, borderWidth: 1.4 },
        { label: 'logistic prediction', data: wb.trace, borderColor: '#f0b34a', pointRadius: 0, borderWidth: 1.6 },
        { label: 'house', data: wb.trace.map(() => 0.485), borderColor: '#5d616b', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
      ] },
      options: { plugins: { legend: { position: 'bottom' } }, scales: { x: { title: { display: true, text: 'round in test window' }, ticks: { maxTicksLimit: 10 } }, y: { min: 0, max: 1, ticks: { callback: (v) => v * 100 + '%' } } } },
    });
  });
}

// ─────────── Next Round ───────────
export function pageNextRound(v, C) {
  withData(v, C, async () => {
    const { esc, n0, f, pct, sgn, card, kpi, chart, state, loadSeries } = C;
    const s = state.source;
    const S = P.streams[s];
    const ser = await loadSeries(s);
    const base = { x: ser.x.slice(), t: ser.t.slice() };
    let cur = { x: base.x.slice(), t: base.t.slice() };
    const names = P.features;
    const py = S.next;
    const fv = nextFeatures(cur.x, cur.t);
    const diffs = names.map((k) => Math.abs(fv[k] - py.features[k]));
    const maxDiff = Math.max(...diffs);
    const parityOk = maxDiff < 1e-3;
    const pLogitBase = TG.map((m) => logitP(S.weights[String(m)], names.map((k) => fv[k])));
    const pyLogit = py.targets.map((t) => t.logit);
    const pDiff = Math.max(...pLogitBase.map((p, i) => Math.abs(p - pyLogit[i])));
    v.innerHTML = `
    <p class="lede">The deployed models for ${esc(C.label(s))}, trained on all ${n0(S.n)} clean rounds. The logistic model runs here in your browser from its exported weights; the gradient boosting numbers come from the last pipeline run. Add rounds to see how little any history moves the forecast.</p>
    <div class="grid g4" id="nr-kpi"></div>
    <div class="grid g2 mt">
      ${card('Next-round probabilities', 'Chance the next round reaches each target. EV is per 1 unit staked at that target.', `<div class="tbl-wrap"><table id="nr-tbl"></table></div>`)}
      ${card('Model ÷ house', 'How far each model leans away from the house curve. 1.00 is no opinion.', `<div class="chart"><canvas id="c-nr"></canvas></div>`)}
    </div>
    ${card('What if the next rounds were…', 'Append hypothetical rounds to the history and re-run the browser model. Timestamps advance 20 s per round.', `
      <div class="row">
        ${[1, 1.5, 2.5, 10, 100].map((x) => `<button class="btn ghost nr-add" data-x="${x}">+ ${x.toFixed(2)}x</button>`).join('')}
        <button class="btn ghost" id="nr-rand">+ 10 fair rounds</button>
        <button class="btn ghost" id="nr-reset">Reset</button>
      </div>
      <div class="chips mt" id="nr-tail"></div>`, 'mt')}
    ${card('Paste your own history', `At least ${205} rounds, newest last. Uses the ${esc(C.label(s))} weights; hour and session features are neutralised.`, `
      <textarea id="nr-in" class="ta" placeholder="1.42&#10;3.10&#10;…"></textarea>
      <div class="row mt"><button class="btn" id="nr-run">Predict next round</button><span class="small muted" id="nr-msg"></span></div>`, 'mt')}
    ${card('Parity check', 'The browser rebuilds all features from the raw series and must match the Python pipeline exactly.', `
      <div class="callout ${parityOk && pDiff < 1e-4 ? 'ok' : 'bad'}"><b>${parityOk && pDiff < 1e-4 ? 'Browser and pipeline agree.' : 'Mismatch between browser and pipeline.'}</b> · max feature difference ${maxDiff.toExponential(1)}, max probability difference ${pDiff.toExponential(1)} across ${names.length} features and ${TG.length} targets.</div>
      <details class="mt"><summary class="small muted">Feature values for the next round</summary><div class="tbl-wrap"><table><thead><tr><th>Feature</th><th class="num">Browser</th><th class="num">Pipeline</th></tr></thead><tbody>${names.map((k, i) => `<tr><td class="mono small">${esc(k)}</td><td class="num">${f(fv[k], 5)}</td><td class="num faint">${f(py.features[k], 5)}</td></tr>`).join('')}</tbody></table></div></details>`, 'mt')}`;
    let nc = null;
    const render = (x, t, note) => {
      const F = nextFeatures(x, t);
      const vec = names.map((k) => F[k]);
      const rows = TG.map((m, i) => {
        const pl = logitP(S.weights[String(m)], vec);
        const t0 = py.targets[i];
        const house = t0.house;
        const cand = [['logit', pl], ['hgb', t0.hgb], ['empirical', t0.empirical]];
        const bestE = cand.reduce((a, b) => (b[1] * m - 1 > a[1] * m - 1 ? b : a));
        return { m, house, emp: t0.empirical, pl, hgb: t0.hgb, ev: bestE[1] * m - 1, evm: bestE[0], sc: S.scores[String(m)] };
      });
      const top = rows.reduce((a, b) => (b.ev > a.ev ? b : a));
      const anyEdge = rows.some((r) => ['logit', 'hgb', 'empirical'].some((md) => r.sc[md].verdict === 'edge'));
      document.getElementById('nr-kpi').innerHTML = [
        kpi('Action', top.ev > P.config.ev_gate && anyEdge ? 'Bet' : 'No bet', anyEdge ? 'a validated edge exists' : 'no model passed validation', top.ev > P.config.ev_gate && anyEdge ? 'bad' : 'ok'),
        kpi('Largest model EV', `${sgn(top.ev * 100, 1)}%`, `${top.m}x via ${P.config.model_label[top.evm]}, not validated: out of sample it scored ${sgn(top.sc[top.evm].skill, 2)} mnat vs house`, 'faint'),
        kpi('P(next ≥ 2x)', pct(rows[1].pl, 2), `house ${pct(rows[1].house, 2)}`),
        kpi('History', n0(x.length), note || `last round ${f(x[x.length - 1])}x`),
      ].join('');
      document.getElementById('nr-tbl').innerHTML = `<thead><tr><th class="num">Target</th><th class="num">House</th><th class="num">Empirical</th><th class="num">Logistic</th><th class="num">Boosting</th><th class="num">Best EV</th><th>Verdict</th></tr></thead><tbody>
        ${rows.map((r) => `<tr><td class="num">${r.m}x</td><td class="num faint">${pct(r.house, 2)}</td><td class="num">${pct(r.emp, 2)}</td><td class="num acc">${pct(r.pl, 2)}</td><td class="num">${pct(r.hgb, 2)}</td><td class="num ${r.ev > 0 ? 'acc' : 'faint'}">${sgn(r.ev * 100, 1)}%</td><td>${['logit', 'hgb', 'empirical'].some((md) => r.sc[md].verdict === 'edge') ? verdict('edge') : '<span class="pill">no edge</span>'}</td></tr>`).join('')}</tbody>`;
      if (nc) { nc.destroy(); state.charts.splice(state.charts.indexOf(nc), 1); }
      nc = chart('c-nr', {
        type: 'bar',
        data: { labels: TG.map((m) => m + 'x'), datasets: [
          { label: 'Logistic (browser)', data: rows.map((r) => r.pl / r.house), backgroundColor: MCOL.logit, borderRadius: 2 },
          { label: 'Gradient boosting', data: rows.map((r) => r.hgb / r.house), backgroundColor: MCOL.hgb, borderRadius: 2 },
          { label: 'Empirical', data: rows.map((r) => r.emp / r.house), backgroundColor: MCOL.empirical, borderRadius: 2 },
        ] },
        options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: false, suggestedMin: 0.8, suggestedMax: 1.2, grid: { color: (c) => (c.tick.value === 1 ? '#4a4f5c' : '#1f2229') } } } },
      });
      document.getElementById('nr-tail').innerHTML = x.slice(-24).map((v) => C.chip(v)).join('');
    };
    render(cur.x, cur.t);
    const push = (vals) => { for (const x of vals) { cur.x.push(x); cur.t.push(cur.t[cur.t.length - 1] + 20); } render(cur.x, cur.t, `${cur.x.length - base.x.length} hypothetical`); };
    v.querySelectorAll('.nr-add').forEach((b) => b.addEventListener('click', () => push([+b.dataset.x])));
    document.getElementById('nr-rand').addEventListener('click', () => push(Array.from({ length: 10 }, () => Math.floor(Math.max(1, 0.97 / (1 - Math.random())) * 100) / 100)));
    document.getElementById('nr-reset').addEventListener('click', () => { cur = { x: base.x.slice(), t: base.t.slice() }; render(cur.x, cur.t); });
    document.getElementById('nr-run').addEventListener('click', () => {
      const xs = (document.getElementById('nr-in').value.match(/\d+(?:\.\d+)?/g) || []).map(Number).filter((x) => x >= 1);
      const msg = document.getElementById('nr-msg');
      if (xs.length < 205) { msg.textContent = `Found ${xs.length} rounds; need at least 205.`; return; }
      msg.textContent = `Using ${n0(xs.length)} pasted rounds.`;
      const t = xs.map((_, i) => 1.7e9 + i * 20);
      cur = { x: xs, t };
      render(xs, t, 'pasted history');
    });
  });
}
