import { createOracle, parseRounds, TARGETS, MODELS, MODEL_LABEL } from './engine.js';
import { auditLadder, runsTest, acf, moonLevels } from './stats.js';
import * as PR from './predict.js';
import * as PF from './platform.js';

const ORDER = ['aviator', 'skyward', 'skyward_deluxe', 'aviator_jul29', 'engine'];
const COLORS = { aviator: '#f0b34a', skyward: '#8fb4ff', skyward_deluxe: '#c49bff', aviator_jul29: '#62d2a2', engine: '#8d9099' };
const state = { data: null, source: 'aviator', charts: [], series: {}, seq: 0 };

const ICON = {
  command: 'M3 12h4l3-8 4 16 3-8h4',
  ledger: 'M5 4h11l3 3v13H5zM9 10h7M9 14h7M9 18h4',
  decomposition: 'M4 5h16l-6 7v6l-4 2v-8z',
  fairness: 'M12 3v18M5 7h14M7 7l-3 7h6zM17 7l-3 7h6z',
  patterns: 'M4 18l4-6 4 3 4-8 4 5',
  moonshots: 'M12 3c3 3 4 7 3 11l-3 3-3-3c-1-4 0-8 3-11zM9 17l-2 4M15 17l2 4',
  sessions: 'M4 6h16M4 12h10M4 18h13',
  forecast: 'M4 19V5M4 19h16M8 15l3-4 3 2 5-6',
  replay: 'M4 12a8 8 0 1 0 3-6.2M4 4v4h4',
  lab: 'M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3',
  method: 'M6 4h12v16H6zM9 8h6M9 12h6M9 16h3',
  pipeline: 'M4 6h4v4H4zM16 6h4v4h-4zM10 14h4v4h-4zM8 8h8M6 10l6 4M18 10l-6 4',
  leaderboard: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
  features: 'M4 4h4v4H4zM10 4h4v4h-4zM16 4h4v4h-4zM4 10h4v4H4zM10 10h4v4h-4zM4 16h4v4H4z',
  calibration: 'M4 20L20 4M4 20h16M4 20V4M8 15h.01M12 12h.01M15 8h.01',
  backtest: 'M3 17l5-5 4 3 8-9M16 6h4v4',
  controls: 'M12 3l8 4v5c0 5-3.5 8-8 9-4.5-1-8-4-8-9V7zM9 12l2 2 4-4',
  nextround: 'M12 3a9 9 0 1 0 9 9M12 7v5l3 3M17 3h4v4M21 3l-6 6',
  catalogue: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM15 15l2 2 4-4',
  eta: 'M12 21a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM12 9v4l3 2M10 2h4',
  signals: 'M3 12h3M18 12h3M12 3v3M12 18v3M7 7l1.5 1.5M15.5 15.5L17 17M7 17l1.5-1.5M15.5 8.5L17 7M9 12a3 3 0 1 0 6 0 3 3 0 0 0-6 0',
  integrity: 'M4 12l5 5L20 6M4 4h4M4 20h16',
  fingerprint: 'M12 4a7 7 0 0 0-7 7v3M19 11a7 7 0 0 0-3-5.7M9 20c1-2 1.5-5 1.5-8a1.5 1.5 0 1 1 3 0c0 3-.5 6-1.5 9M16 20c.5-1.5 1-4 1-7',
  sequence: 'M4 7h4v4H4zM10 7h4v4h-4zM16 7h4v4h-4zM6 15v3M12 15v3M18 15v3',
  arena: 'M6 3h12v5a6 6 0 0 1-12 0zM9 20h6M12 14v6M6 5H3v2a3 3 0 0 0 3 3M18 5h3v2a3 3 0 0 1-3 3',
  simulator: 'M3 20h18M5 16l4-5 3 3 4-6 3 4M17 4h4v4',
  experiments: 'M5 3h14M8 3v5l-4 9a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3l-4-9V3M7 14h10',
  book: 'M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM4 19V5M8 7h7',
};
const ROUTES = [
  { id: 'command', group: 'Overview', title: 'Command Center', perSource: false, render: pageCommand },
  { id: 'catalogue', group: 'Overview', title: 'Platform Coverage', perSource: false, render: (v) => PF.pageCatalogue(v, CTX) },
  { id: 'eta', group: 'Now', title: 'ETA Board', perSource: true, render: (v) => PF.pageEta(v, CTX) },
  { id: 'signals', group: 'Now', title: 'Signal Strip', perSource: true, render: (v) => PF.pageSignals(v, CTX) },
  { id: 'ledger', group: 'Evidence', title: 'Backup Ledger', perSource: false, render: pageLedger },
  { id: 'decomposition', group: 'Evidence', title: 'Decomposition', perSource: false, render: pageDecomp },
  { id: 'integrity', group: 'Evidence', title: 'Tape Integrity', perSource: true, render: (v) => PF.pageIntegrity(v, CTX) },
  { id: 'fingerprint', group: 'Evidence', title: 'Fingerprint & Compare', perSource: true, render: (v) => PF.pageFingerprint(v, CTX) },
  { id: 'fairness', group: 'Analysis', title: 'Fairness Audit', perSource: true, render: pageFairness },
  { id: 'patterns', group: 'Analysis', title: 'Pattern DNA', perSource: true, render: pagePatterns },
  { id: 'moonshots', group: 'Analysis', title: 'Moonshot Pressure', perSource: true, render: pageMoon },
  { id: 'sessions', group: 'Analysis', title: 'Sessions', perSource: true, render: pageSessions },
  { id: 'forecast', group: 'Forecast', title: 'Forecast Studio', perSource: true, render: pageForecast },
  { id: 'replay', group: 'Forecast', title: 'Replay Audit', perSource: false, render: pageReplay },
  { id: 'pipeline', group: 'Prediction', title: 'Pipeline', perSource: false, render: (v) => PR.pagePipeline(v, CTX) },
  { id: 'leaderboard', group: 'Prediction', title: 'Model Leaderboard', perSource: false, render: (v) => PR.pageLeaderboard(v, CTX) },
  { id: 'features', group: 'Prediction', title: 'Feature Analysis', perSource: true, render: (v) => PR.pageFeatures(v, CTX) },
  { id: 'calibration', group: 'Prediction', title: 'Calibration', perSource: true, render: (v) => PR.pageCalibration(v, CTX) },
  { id: 'backtest', group: 'Prediction', title: 'Backtest', perSource: true, render: (v) => PR.pageBacktest(v, CTX) },
  { id: 'controls', group: 'Prediction', title: 'Controls & Power', perSource: false, render: (v) => PR.pageControls(v, CTX) },
  { id: 'nextround', group: 'Prediction', title: 'Next Round', perSource: true, render: (v) => PR.pageNextRound(v, CTX) },
  { id: 'arena', group: 'Lab', title: 'Predictor Arena', perSource: true, render: (v) => PF.pageArena(v, CTX) },
  { id: 'sequence', group: 'Lab', title: 'Sequence Search', perSource: true, render: (v) => PF.pageSequence(v, CTX) },
  { id: 'experiments', group: 'Lab', title: 'Experiment Registry', perSource: false, render: (v) => PF.pageExperiments(v, CTX) },
  { id: 'simulator', group: 'Lab', title: 'Bankroll & Ledger', perSource: true, render: (v) => PF.pageSimulator(v, CTX) },
  { id: 'book', group: 'Knowledge', title: 'Platform Book', perSource: false, render: (v) => PF.pageBook(v, CTX) },
  { id: 'lab', group: 'Tools', title: 'Round Lab', perSource: false, render: pageLab },
  { id: 'method', group: 'Tools', title: 'Method & Downloads', perSource: false, render: pageMethod },
];

// ─────────── utils ───────────
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const n0 = (v) => (v == null ? '–' : Math.round(v).toLocaleString('en-US'));
const f = (v, d = 2) => (v == null || !isFinite(v) ? '–' : (+v).toFixed(d));
const pct = (v, d = 1) => (v == null ? '–' : (v * 100).toFixed(d) + '%');
const sgn = (v, d = 2) => (v == null ? '–' : (v >= 0 ? '+' : '') + (+v).toFixed(d));
const mb = (b) => (b / 1048576).toFixed(1) + ' MB';
const dt = (s) => (s ? s.replace('T', ' ').slice(0, 16) : '–');
const day = (s) => (s ? s.slice(0, 10) : '–');
const label = (s) => state.data.sources[s].meta.label;
const zPill = (z, hi = 3, lo = 2) => {
  if (z == null) return '<span class="pill">–</span>';
  const a = Math.abs(z);
  const cls = a >= hi ? 'bad' : a >= lo ? 'acc' : 'ok';
  return `<span class="pill ${cls}">z ${sgn(z)}</span>`;
};
const verdictPill = (v) => {
  const m = { predictable: 'bad', weak: 'acc', random: 'ok', overfit: 'info' };
  return `<span class="pill ${m[v] || ''}">${esc(v)}</span>`;
};
const chip = (x) => `<span class="chip ${x < 2 ? 'lo' : x < 10 ? 'mid' : x < 100 ? 'hi' : 'moon'}">${f(x)}x</span>`;

function card(title, sub, body, extra = '') {
  return `<section class="card ${extra}"><div class="card-head"><div><h2>${title}</h2>${sub ? `<p class="sub">${sub}</p>` : ''}</div></div>${body}</section>`;
}
function kpi(lab, val, note = '', cls = '') {
  return `<div class="card kpi"><div class="lab">${lab}</div><div class="val ${cls}">${val}</div>${note ? `<div class="note">${note}</div>` : ''}</div>`;
}

Chart.defaults.color = '#8d9099';
Chart.defaults.font.family = "'Geist Mono', monospace";
Chart.defaults.font.size = 11;
Chart.defaults.borderColor = '#262a33';
Chart.defaults.plugins.legend.labels.boxWidth = 10;
Chart.defaults.plugins.legend.labels.boxHeight = 10;
Chart.defaults.animation = { duration: 350 };
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.tooltip.backgroundColor = '#1b1e25';
Chart.defaults.plugins.tooltip.borderColor = '#323744';
Chart.defaults.plugins.tooltip.borderWidth = 1;

function chart(id, cfg) {
  const el = document.getElementById(id);
  if (!el) return null;
  const c = new Chart(el, cfg);
  state.charts.push(c);
  return c;
}
const CTX = { state, $, esc, n0, f, pct, sgn, card, kpi, chart, chip, loadSeries, explain: PF.explainHTML, get label() { return label; } };
function killCharts() { state.charts.forEach((c) => c.destroy()); state.charts = []; }

async function loadSeries(name) {
  if (state.series[name]) return state.series[name];
  const j = await (await fetch(`data/series/${name}.json`)).json();
  const t = new Array(j.x.length);
  let acc = j.t0;
  for (let i = 0; i < j.dt.length; i++) { acc = i === 0 ? j.t0 : acc + j.dt[i]; t[i] = acc; }
  state.series[name] = { x: j.x, t };
  return state.series[name];
}

// ─────────── boot ───────────
async function boot() {
  try {
    state.data = await (await fetch('data/suite.json')).json();
  } catch (e) {
    $('#view').innerHTML = `<div class="card err">Could not load data/suite.json: ${esc(e.message)}</div>`;
    return;
  }
  const sel = $('#source');
  sel.innerHTML = ORDER.map((s) => `<option value="${s}">${esc(label(s))}</option>`).join('');
  sel.addEventListener('change', () => { state.source = sel.value; route(); });
  let grp = '';
  $('#nav').innerHTML = ROUTES.map((r) => {
    const g = r.group !== grp ? `<div class="grp">${r.group}</div>` : '';
    grp = r.group;
    return `${g}<a href="#/${r.id}" data-id="${r.id}"><svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="${ICON[r.id]}"/></svg>${r.title}</a>`;
  }).join('');
  const d = state.data;
  $('#build-meta').innerHTML = `branch <span class="acc">${esc(d.branch)}</span><br>built ${esc(dt(d.generated))} UTC<br>${n0(d.decomposition.funnel.raw)} raw → ${n0(d.decomposition.funnel.kept)} kept`;
  $('#menu').addEventListener('click', () => $('#side').classList.toggle('open'));
  $('#nav').addEventListener('click', () => $('#side').classList.remove('open'));
  window.addEventListener('hashchange', route);
  route();
}

function route() {
  const [, id, q] = (location.hash.match(/^#\/([a-z]+)(?:\?(.*))?/) || [null, 'command']);
  const r = ROUTES.find((x) => x.id === id) || ROUTES[0];
  if (q) {
    const src = new URLSearchParams(q).get('src');
    if (src && ORDER.includes(src)) { state.source = src; $('#source').value = src; }
  }
  document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('on', a.dataset.id === r.id));
  $('#crumb-group').textContent = r.group;
  $('#crumb-page').textContent = r.title;
  $('#src-pick').classList.toggle('hide', !r.perSource);
  document.title = `${r.title} · AVFS Investigation Suite`;
  killCharts();
  state.seq++;
  const v = $('#view');
  v.innerHTML = '';
  try { r.render(v); } catch (e) { console.error(e); v.innerHTML = `<div class="card err">Render error: ${esc(e.message)}</div>`; }
  window.scrollTo(0, 0);
}

// ─────────── Command Center ───────────
function pageCommand(v) {
  const d = state.data;
  const dec = d.decomposition;
  const led = d.ledger;
  const S = d.sources;
  const rep = d.replay;
  const real = ['aviator', 'skyward', 'skyward_deluxe', 'aviator_jul29'];
  const realN = real.reduce((a, s) => a + S[s].meta.n, 0);
  const anyEdge = ORDER.some((s) => S[s].forecast.predictable.length);
  const b = dec.burst;
  const ga = dec.genuine.aviator;
  v.innerHTML = `
  <p class="lede">Every database on the <b>decomputation</b> branch was opened, fingerprinted and taken apart row by row. Real captures were separated from injected, test and overwritten rows, then every game was audited and forecast walk-forward. The results below come only from the clean data.</p>
  <div class="grid g5">
    ${kpi('Raw rows', n0(dec.funnel.raw), `${led.files.length} files on the branch`)}
    ${kpi('Kept', n0(dec.funnel.kept), `${pct(dec.funnel.kept / dec.funnel.raw)} of raw`, 'ok')}
    ${kpi('Quarantined', n0(dec.funnel.quarantined), `${dec.rules.length} rules fired`, 'acc')}
    ${kpi('Real rounds analysed', n0(realN), '4 capture streams')}
    ${kpi('Predictable targets', anyEdge ? 'some' : '0 / 40', '8 targets × 5 streams', anyEdge ? 'bad' : 'ok')}
  </div>
  <div class="grid g2 mt">
    ${card('Case findings', 'Ranked by how much they change what you can trust in this data.', `
      ${finding(1, 'high', `${n0(b.rows)} Aviator rows were generated, not captured`, `On Jul 3 between ${b.t_lo.slice(11, 16)} and ${b.t_hi.slice(11, 16)} UTC, rows arrived ${f(b.gap_median_s * 1000, 1)} ms apart with ${pct(b.off_grid, 0)} of values off the 0.01 grid. Those rows pay ${f(b.rtp2)}x at 2x against ${f(ga.rtp2)}x for real Aviator captures.`, 'decomposition')}
      ${finding(2, 'high', `Replay accuracy jumped from ${pct(rep.acc_base, 0)} to ${pct(rep.spike.acc, 0)} right after that burst`, `${n0(rep.spike.runs)} replay runs on Jul 4 to 6 scored above 80% in ${pct(rep.spike.share_over_80, 0)} of cases, against ${pct(rep.spike.base_share_over_80, 2)} on every other day. The most likely cause is replays running on the injected rows.`, 'replay')}
      ${finding(3, 'mid', `${n0(led.diff.rewritten)} backup rows were overwritten in place`, `Row ids ${led.diff.rewritten_id_range.join('–')} hold different rounds in the live database than in the Jul 14 backup, and ${n0(led.diff.rewritten_lost)} of the originals exist nowhere else in live. The three backups are byte-identical, so they protect only one moment.`, 'ledger')}
      ${finding(4, 'ok', 'Clean captures match the house curve', `Aviator, Skyward and Skyward Deluxe pay ${f(S.aviator.audit.ladder[1].rtp, 3)}, ${f(S.skyward.audit.ladder[1].rtp, 3)} and ${f(S.skyward_deluxe.audit.ladder[1].rtp, 3)} at 2x against 0.970 expected. No next-round model beats the house baseline at any target.`, 'fairness')}
      ${finding(5, 'ok', 'Moonshots arrive on schedule', `100x rounds: ${S.aviator.moon.levels[2].hits} vs ${f(S.aviator.moon.levels[2].expected, 0)} expected on Aviator, ${S.skyward.moon.levels[2].hits} vs ${f(S.skyward.moon.levels[2].expected, 0)} on Skyward, ${S.skyward_deluxe.moon.levels[2].hits} vs ${f(S.skyward_deluxe.moon.levels[2].expected, 0)} on Deluxe. Droughts follow the geometric wait of a memoryless game.`, 'moonshots')}
      ${finding(6, 'ok', 'The prediction pipeline finds 0 edges in 120 tests', 'Empirical, logistic and gradient-boosting models on 42 causal features, scored walk-forward with block bootstrap and FDR control. The same pipeline detects a planted +8 pp edge every time, and turns the Jul 3 burst into a fake z = +5.7 edge under shuffled cross-validation, a likely mechanism behind the Jul 4–6 replay spike.', 'pipeline')}
      ${finding(7, 'ok', 'The platform book’s own tests agree: nothing to exploit, but the tape can be policed', 'Of the 38 features in chapter 18, 18 run here in full and 6 in part. Tape integrity scores the Jul 3 burst at 0.00 and clean Aviator at 0.997. Every ETA threshold is memoryless, none of the 14 dashboard signals survives correction, and a CUSUM fingerprint flags the spliced burst within one block.', 'catalogue')}
    `)}
    <div class="grid">
      ${card('Streams at a glance', 'Clean rows only. Payout at 2x should be 0.970 under a 97% RTP.', `
        <div class="tbl-wrap"><table><thead><tr><th>Stream</th><th class="num">Rounds</th><th class="num">Pays @2x</th><th class="num">Paper ROI</th><th>Verdict</th></tr></thead><tbody>
        ${ORDER.map((s) => {
          const x = S[s];
          const st = x.forecast.strategy;
          return `<tr><td><a href="#/forecast?src=${s}" style="text-decoration:none"><span style="color:${COLORS[s]}">●</span> ${esc(x.meta.label)}</a></td><td class="num">${n0(x.meta.n)}</td><td class="num">${f(x.audit.ladder[1].rtp, 3)}</td><td class="num ${st.roi < 0 ? 'bad' : ''}">${sgn(st.roi * 100, 1)}%</td><td>${x.forecast.predictable.length ? '<span class="pill bad">signal</span>' : '<span class="pill ok">no edge</span>'}</td></tr>`;
        }).join('')}
        </tbody></table></div>`)}
      ${card('Deviation from the house curve', 'z-score of observed hit rate per target. Inside ±2 is noise.', `<div class="chart"><canvas id="c-ladder"></canvas></div>`)}
    </div>
  </div>
  <div class="grid g2 mt">
    ${card('Where the raw rows went', '', `<div id="funnel"></div>`)}
    ${card('Replay accuracy by day', 'Mean accuracy of session replay runs. The Jul 4–6 spike follows the Jul 3 burst.', `<div class="chart short"><canvas id="c-rep"></canvas></div>`)}
  </div>`;

  const maxRule = Math.max(...dec.rules.map((r) => r.rows));
  $('#funnel').innerHTML = `
    <div class="row" style="justify-content:space-between;margin-bottom:8px"><span class="small">Kept <span class="mono ok">${n0(dec.funnel.kept)}</span> <span class="faint">(${pct(dec.funnel.kept / dec.funnel.raw)})</span></span><span class="small">Quarantined <span class="mono acc">${n0(dec.funnel.quarantined)}</span></span></div>
    <div class="bar" style="height:14px;display:flex"><i style="width:${(dec.funnel.kept / dec.funnel.raw) * 100}%;background:var(--ok);border-radius:3px 0 0 3px"></i><i style="width:${(dec.funnel.quarantined / dec.funnel.raw) * 100}%;background:var(--bad);border-radius:0"></i></div>
    <h3 class="mt">Quarantine by rule <span class="faint" style="text-transform:none;letter-spacing:0">(bars relative to the largest rule)</span></h3>
    ${dec.rules.map((r) => `<div class="bar-row"><span title="${esc(r.text)}">${esc(r.rule.replace(/_/g, ' '))}</span><div class="bar ${r.rule === 'synthetic_burst' ? 'bad' : ''}"><i style="width:${Math.max(0.6, (r.rows / maxRule) * 100)}%"></i></div><span class="mono num">${n0(r.rows)}</span></div>`).join('')}`;

  const levels = S.aviator.audit.ladder.map((l) => l.m);
  chart('c-ladder', {
    type: 'line',
    data: { labels: levels.map((m) => m + 'x'), datasets: ORDER.map((s) => ({ label: label(s), data: S[s].audit.ladder.map((l) => l.z), borderColor: COLORS[s], backgroundColor: COLORS[s], pointRadius: 2.5, tension: 0.25, borderWidth: 1.6 })) },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { suggestedMin: -3, suggestedMax: 3, grid: { color: (c) => (Math.abs(c.tick.value) === 2 ? '#4a3a1c' : '#1f2229') } } } },
  });
  chart('c-rep', {
    type: 'bar',
    data: { labels: rep.daily.map((x) => x.d.slice(5)), datasets: [{ data: rep.daily.map((x) => x.acc), backgroundColor: rep.daily.map((x) => (x.flag ? '#ff6f5e' : '#3a3f4b')), borderRadius: 2 }] },
    options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `accuracy ${pct(c.raw)}` } } }, scales: { y: { min: 0, max: 1, ticks: { callback: (v) => v * 100 + '%' } }, x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 10 } } } },
  });
}
function finding(n, sev, title, text, link) {
  return `<div class="finding sev-${sev}"><div class="n">${n}</div><div><h4>${title}</h4><p>${text}</p><a href="#/${link}">Open evidence →</a></div></div>`;
}

// ─────────── Backup Ledger ───────────
function pageLedger(v) {
  const L = state.data.ledger;
  const D = L.diff;
  const byName = Object.fromEntries(L.files.map((x) => [x.name, x]));
  const live = byName['avfs.db'];
  const bak = byName[D.backup];
  const mom = byName['momento.db'];
  const arc = byName['avfs.db.7z'];
  v.innerHTML = `
  <p class="lede">Every file in <code>data/</code> on the <b>decomputation</b> branch, fingerprinted by SHA-256. Identical files are grouped, the 7z archive is checked against the files on disk, and the live database is diffed row by row against the backup.</p>
  <div class="grid g4">
    ${kpi('Files', L.files.length, `${mb(L.files.reduce((a, x) => a + x.bytes, 0))} total`)}
    ${kpi('Distinct backups', '1', `${L.duplicate_groups[0]?.length || 0} files, same bytes`, 'acc')}
    ${kpi('Rows added since backup', n0(D.added.reduce((a, x) => a + x.rows, 0)), D.added.map((x) => x.source).join(', '))}
    ${kpi('Rows overwritten', n0(D.rewritten), `${n0(D.rewritten_lost)} originals unrecoverable from live`, 'bad')}
  </div>
  <div class="grid g2 mt">
    ${card('Snapshot timeline', 'Reconstructed from row timestamps and archive metadata.', `<div class="timeline">
      <div class="tl-item muted"><div class="when">2026-06-09</div><div class="dot"></div><div><h4>First capture</h4><p>Aviator collection starts (aviator_rounds_*.json).</p></div></div>
      <div class="tl-item"><div class="when">2026-06-20</div><div class="dot"></div><div><h4>Three games in parallel</h4><p>Skyward and Skyward Deluxe join. Replay runs start being logged.</p></div></div>
      <div class="tl-item bad"><div class="when">2026-07-03 05:02</div><div class="dot"></div><div><h4>Synthetic burst</h4><p>${n0(state.data.decomposition.burst.rows)} Aviator rows written in 7 minutes by a server process.</p></div></div>
      <div class="tl-item bad"><div class="when">2026-07-04 → 06</div><div class="dot"></div><div><h4>Replay accuracy spike</h4><p>${n0(state.data.replay.spike.runs)} runs averaging ${pct(state.data.replay.spike.acc, 0)}.</p></div></div>
      <div class="tl-item"><div class="when">2026-07-14 11:32</div><div class="dot"></div><div><h4>Backup taken ×3</h4><p>${esc(L.duplicate_groups[0]?.join(', '))}. Byte-identical.</p></div></div>
      <div class="tl-item bad"><div class="when">2026-07-14 11:42</div><div class="dot"></div><div><h4>Ids ${D.rewritten_id_range.join('–')} reused</h4><p>${n0(D.rewritten)} rows overwritten with new rounds after the backup.</p></div></div>
      <div class="tl-item"><div class="when">${day(live.range[1])}</div><div class="dot"></div><div><h4>Last live write</h4><p>avfs.db reaches ${n0(Object.values(live.rounds_by_source).reduce((a, b) => a + b, 0))} rows.</p></div></div>
      <div class="tl-item"><div class="when">${day(mom.range[1])}</div><div class="dot"></div><div><h4>momento.db</h4><p>A separate Momento platform snapshot: ${n0(mom.tables.rounds)} rounds, ${mom.empty_tables} empty tables.</p></div></div>
      <div class="tl-item muted"><div class="when">2026-08-31</div><div class="dot"></div><div><h4>7z mirror</h4><p>${arc.archive.length} entries; all ${L.mirror.filter((m) => m.match).length} data files match disk CRC32.</p></div></div>
    </div>`)}
    <div class="grid">
      ${card('Backup → live diff', `${esc(D.backup)} vs avfs.db`, `
        <div class="cmp">
          <div class="h">Change</div><div class="h num">Rows</div><div class="h">Detail</div>
          ${D.added.map((a) => `<div>Added (${esc(a.source)})</div><div class="mono num">${n0(a.rows)}</div><div class="small muted">ids ${a.id_lo}–${a.id_hi}</div>`).join('')}
          <div>Overwritten in place</div><div class="mono num bad">${n0(D.rewritten)}</div><div class="small muted">ids ${D.rewritten_id_range.join('–')}</div>
          <div>Deleted</div><div class="mono num">${n0(D.removed)}</div><div class="small muted">no ids missing</div>
          <div>Replay runs added</div><div class="mono num">${n0(D.replay_added)}</div><div class="small muted">replay_results</div>
        </div>
        <h3 class="mt">What the overwritten ids used to hold</h3>
        <div class="chips">${D.rewritten_pairs.map((p) => `<span class="chip">${esc(p.was)} → ${esc(p.now)} · ${p.rows}</span>`).join('')}</div>`)}
      ${card('Overwritten rows, first 12', 'Backup value on the left, live value on the right.', `<div class="tbl-wrap"><table><thead><tr><th>id</th><th>was</th><th class="num">x</th><th>now</th><th class="num">x</th></tr></thead><tbody>
        ${D.rewritten_sample.map((r) => `<tr><td class="mono">${r.id}</td><td>${esc(r.was_source)}<br><span class="faint tiny mono">${esc(dt(r.was_t).slice(5))}</span></td><td class="num">${f(r.was_x)}</td><td>${esc(r.now_source)}<br><span class="faint tiny mono">${esc(dt(r.now_t).slice(5))}</span></td><td class="num">${f(r.now_x)}</td></tr>`).join('')}
      </tbody></table></div>`)}
    </div>
  </div>
  ${card('File fingerprints', '', `<div class="tbl-wrap"><table><thead><tr><th>File</th><th class="num">Size</th><th>SHA-256</th><th>Contents</th></tr></thead><tbody>
    ${L.files.map((x) => {
      const dup = L.duplicate_groups.find((g) => g.includes(x.name));
      const contents = x.rounds_by_source ? Object.entries(x.rounds_by_source).map(([k, n]) => `${esc(k)} ${n0(n)}`).join(' · ') : x.archive ? `${x.archive.length} files, mirror of data/` : x.bytes ? '' : 'empty';
      return `<tr><td class="mono">${esc(x.name)} ${dup ? '<span class="pill acc">identical ×' + dup.length + '</span>' : ''}</td><td class="num">${mb(x.bytes)}</td><td class="mono faint">${x.sha256 ? x.sha256.slice(0, 16) + '…' : '–'}</td><td class="wrap small muted">${contents}</td></tr>`;
    }).join('')}
  </tbody></table></div>`, 'mt')}
  <div class="callout mt"><b>What to do:</b> keep backups at different times, not three copies of one moment, and never reuse primary keys after a restore. Ids ${D.rewritten_id_range.join('–')} are proof that a restore or re-import reset the sequence.</div>`;
}

// ─────────── Decomposition ───────────
function pageDecomp(v) {
  const D = state.data.decomposition;
  const b = D.burst;
  const g = D.genuine.aviator;
  v.innerHTML = `
  <p class="lede">Each raw row runs through eight rules in order and is quarantined by the first one it breaks. Nothing is deleted from the source files; the console simply ignores quarantined rows.</p>
  <div class="grid g4">
    ${D.by_source ? Object.entries(D.by_source).map(([s, x]) => kpi(esc(label(s)), n0(x.kept), `${n0(x.raw - x.kept)} quarantined of ${n0(x.raw)}`)).join('') : ''}
    ${kpi('Other sources', n0(D.rules.find((r) => r.rule === 'unknown_source')?.rows || 0), 'JetX probes, ${src} template leak', 'acc')}
  </div>
  <div class="grid g2 mt">
    ${card('Quarantine rules', 'Applied top to bottom; a row is counted once.', `<div class="tbl-wrap"><table><thead><tr><th>Rule</th><th class="num">Rows</th><th>By source</th></tr></thead><tbody>
      ${D.rules.map((r) => `<tr><td class="wrap"><b>${esc(r.rule.replace(/_/g, ' '))}</b><br><span class="small muted">${esc(r.text)}</span></td><td class="num">${n0(r.rows)}</td><td class="small muted wrap">${Object.entries(r.by_source).map(([k, n]) => `${esc(k)} ${n0(n)}`).join(' · ')}</td></tr>`).join('')}
    </tbody></table></div>`)}
    ${card('The Jul 3 burst vs real Aviator', 'The same game, the same source label, two completely different processes.', `
      <div class="cmp">
        <div class="h">Measure</div><div class="h num">Burst</div><div class="h num">Real</div>
        <div>Rows</div><div class="mono num">${n0(b.rows)}</div><div class="mono num">${n0(state.data.sources.aviator.meta.n)}</div>
        <div>Median gap between rows</div><div class="mono num bad">${f(b.gap_median_s * 1000, 1)} ms</div><div class="mono num">${f(state.data.sources.aviator.sessions.cadence_s, 1)} s</div>
        <div>Values off the 0.01 grid</div><div class="mono num bad">${pct(b.off_grid, 1)}</div><div class="mono num">0%</div>
        <div>Pays at 2x</div><div class="mono num bad">${f(b.rtp2, 3)}</div><div class="mono num">${f(g.rtp2, 3)}</div>
        <div>Pays at 10x</div><div class="mono num bad">${f(b.rtp10, 3)}</div><div class="mono num">${f(g.rtp10, 3)}</div>
        <div>Median multiplier</div><div class="mono num">${f(b.median, 2)}</div><div class="mono num">${f(g.median, 2)}</div>
        <div>Instant crashes (&lt;1.01x)</div><div class="mono num bad">${pct(b.p_below_1_01, 2)}</div><div class="mono num">${pct(g.p_below_1_01, 2)}</div>
      </div>
      <h3 class="mt">First values in the burst</h3>
      <div class="chips">${b.sample.map((x) => `<span class="chip">${x}</span>`).join('')}</div>
      <p class="small muted mt">Values like 2.1375 and 1.92125 have 4–6 decimals, which a game display never shows. They look like smoothed or averaged values, not crash points. ${pct(b.rtp2 / 2, 0)} of them reach 2x but only ${pct(b.rtp10 / 10, 2)} reach 10x, which is why the burst pays almost double at 2x and almost nothing at 10x.</p>`)}
  </div>
  ${card('Burst profile', 'Rows per minute on Jul 3 (UTC). The whole burst lasted seven minutes.', `<div class="chart short"><canvas id="c-burst"></canvas></div>`, 'mt')}
  <div class="callout bad mt"><b>Keep this out of the model.</b> If a feature pipeline, replay or backtest read Aviator rows from Jul 3 05:02–05:09 UTC, its scores are inflated. The Replay Audit page shows what that looked like.</div>`;
  chart('c-burst', {
    type: 'bar',
    data: { labels: D.burst_profile.map((x) => x.t.slice(11, 16)), datasets: [
      { label: 'rows', data: D.burst_profile.map((x) => x.n), backgroundColor: '#ff6f5e', borderRadius: 2, yAxisID: 'y' },
      { label: 'pays @2x', type: 'line', data: D.burst_profile.map((x) => x.rtp2), borderColor: '#f0b34a', pointRadius: 3, yAxisID: 'y2' },
    ] },
    options: { scales: { y: { title: { display: true, text: 'rows' } }, y2: { position: 'right', min: 0, max: 2.2, grid: { display: false } } } },
  });
}

// ─────────── Fairness ───────────
function pageFairness(v) {
  const s = state.source;
  const A = state.data.sources[s].audit;
  const worst = A.ladder.reduce((a, b) => (Math.abs(b.z) > Math.abs(a.z) ? b : a));
  const flagged = A.ladder.filter((l) => Math.abs(l.z) >= 2).length;
  v.innerHTML = `
  <p class="lede">For a fair crash game with 97% RTP, the share of rounds reaching m is <b>0.97 / m</b> at every m. This page checks ${esc(label(s))} against that curve at 11 targets, with ±2σ bands that shrink as rounds accumulate.</p>
  <div class="grid g4">
    ${kpi('Rounds', n0(A.n), esc(label(s)))}
    ${kpi('Pays @2x', f(A.ladder[1].rtp, 3), `±${f(2 * A.ladder[1].rtp_se, 3)} (2σ)`)}
    ${kpi('Instant crashes', pct(A.instant_crash, 2), `${pct(A.instant_expected, 2)} if the edge were all at 1.00x`)}
    ${kpi('Largest deviation', `${worst.m}x`, `z ${sgn(worst.z)} · ${flagged} of 11 beyond ±2`, Math.abs(worst.z) >= 3 ? 'bad' : Math.abs(worst.z) >= 2 ? 'acc' : 'ok')}
  </div>
  <div class="grid g2 mt">
    ${card('Payout at each target', 'Observed payout rate (m × hit rate) with a 2σ band. The line is 0.97.', `<div class="chart"><canvas id="c-rtp"></canvas></div>`)}
    ${card('Survival curve', 'Share of rounds reaching m, log–log. Dashed is the house curve.', `<div class="chart"><canvas id="c-surv"></canvas></div>`)}
  </div>
  ${card('Ladder', `With 11 targets, about one |z| above 2 is expected by chance alone. Treat |z| ≥ 3 as the bar for a real deviation.`, `<div class="tbl-wrap"><table><thead><tr><th class="num">Target</th><th class="num">Hits</th><th class="num">Expected</th><th class="num">Hit rate</th><th class="num">House</th><th class="num">Pays</th><th class="num">±2σ</th><th>Deviation</th></tr></thead><tbody>
    ${A.ladder.map((l) => `<tr><td class="num">${l.m}x</td><td class="num">${n0(l.hits)}</td><td class="num">${n0(l.expected)}</td><td class="num">${pct(l.p_obs, 3)}</td><td class="num faint">${pct(l.p_house, 3)}</td><td class="num">${f(l.rtp, 3)}</td><td class="num faint">${f(2 * l.rtp_se, 3)}</td><td>${zPill(l.z)}</td></tr>`).join('')}
  </tbody></table></div>`, 'mt')}`;
  chart('c-rtp', {
    type: 'bar',
    data: { labels: A.ladder.map((l) => l.m + 'x'), datasets: [
      { label: '2σ band', data: A.ladder.map((l) => [l.rtp - 2 * l.rtp_se, l.rtp + 2 * l.rtp_se]), backgroundColor: 'rgba(240,179,74,.18)', borderColor: 'rgba(240,179,74,.5)', borderWidth: 1, barPercentage: 0.5 },
      { label: 'observed', type: 'line', data: A.ladder.map((l) => l.rtp), borderColor: '#f0b34a', backgroundColor: '#f0b34a', showLine: false, pointRadius: 4 },
      { label: 'house 0.97', type: 'line', data: A.ladder.map(() => 0.97), borderColor: '#62d2a2', borderDash: [4, 4], pointRadius: 0, borderWidth: 1.2 },
    ] },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: false, suggestedMin: 0.8, suggestedMax: 1.2 } } },
  });
  chart('c-surv', {
    type: 'line',
    data: { datasets: [
      { label: 'observed', data: A.survival.filter((p) => p.obs > 0).map((p) => ({ x: p.m, y: p.obs })), borderColor: COLORS[s], pointRadius: 0, borderWidth: 2 },
      { label: 'house 0.97/m', data: A.survival.map((p) => ({ x: p.m, y: p.house })), borderColor: '#8d9099', borderDash: [5, 4], pointRadius: 0, borderWidth: 1.2 },
    ] },
    options: { parsing: false, plugins: { legend: { position: 'bottom' } }, scales: { x: { type: 'logarithmic', title: { display: true, text: 'multiplier' } }, y: { type: 'logarithmic' } } },
  });
}

// ─────────── Patterns ───────────
function pagePatterns(v) {
  const s = state.source;
  const P = state.data.sources[s].patterns;
  v.innerHTML = `
  <p class="lede">If rounds are independent, yesterday's crash tells you nothing about the next one. These tests look for memory in ${esc(label(s))}: correlation between rounds, streak lengths, what follows a low round, and time of day.</p>
  <div class="grid g4">
    ${kpi('Runs test (≥2x)', `z ${sgn(P.runs.z)}`, `${n0(P.runs.runs)} runs vs ${n0(P.runs.expected)} expected`, Math.abs(P.runs.z) >= 3 ? 'bad' : Math.abs(P.runs.z) >= 2 ? 'acc' : 'ok')}
    ${kpi('Autocorrelation', `${P.acf_out} / 20`, `lags outside ±${f(P.band, 4)}; ~1 expected`, P.acf_out > 3 ? 'bad' : 'ok')}
    ${kpi('Longest run under 2x', P.longest_loss, `${n0(P.streak_count)} losing streaks`)}
    ${kpi('Hour of day', `χ² ${f(P.hour_chi, 1)}`, `${P.hour_df} hours; ~${P.hour_df} expected if flat`, P.hour_chi > P.hour_df * 2 ? 'bad' : 'ok')}
  </div>
  <div class="grid g2 mt">
    ${card('Autocorrelation of log multiplier', 'Bars outside the band would mean rounds remember each other.', `<div class="chart short"><canvas id="c-acf"></canvas></div>`)}
    ${card('Losing streak lengths', 'Runs of rounds under 2x, observed vs a memoryless game.', `<div class="chart short"><canvas id="c-streak"></canvas></div>`)}
  </div>
  <div class="grid g2 mt">
    ${card('What follows what', 'Chance the next round reaches 2x, by the previous round.', `<div class="tbl-wrap"><table><thead><tr><th>Previous round</th><th class="num">Cases</th><th class="num">Next ≥2x</th><th class="num">Baseline</th><th>Deviation</th></tr></thead><tbody>
      ${P.conditional.map((c) => `<tr><td>${esc(c.bucket)}</td><td class="num">${n0(c.n)}</td><td class="num">${pct(c.p_next2)}</td><td class="num faint">${pct(c.base)}</td><td>${zPill(c.z)}</td></tr>`).join('')}
    </tbody></table></div>
    <h3 class="mt">After a losing streak of length L</h3>
    <p class="tiny faint" style="margin:-4px 0 6px">Nine rows per stream across five streams: one |z| near 3 somewhere is expected by chance. A real pattern repeats across streams.</p>
    <div class="tbl-wrap"><table><thead><tr><th>L</th><th class="num">Cases</th><th class="num">Next ≥2x</th><th>Deviation</th></tr></thead><tbody>
      ${P.after_run.map((c) => `<tr><td class="mono">${c.run}</td><td class="num">${n0(c.n)}</td><td class="num">${pct(c.p)}</td><td>${zPill(c.z)}</td></tr>`).join('')}
    </tbody></table></div>`)}
    ${card('Hour of day (UTC)', 'Payout at 2x by capture hour with a 2σ band. Low-volume hours are hidden.', `<div class="chart tall"><canvas id="c-hour"></canvas></div>`)}
  </div>`;
  chart('c-acf', {
    type: 'bar',
    data: { labels: P.acf.map((a) => a.lag), datasets: [
      { data: P.acf.map((a) => a.r), backgroundColor: P.acf.map((a) => (Math.abs(a.r) > P.band ? '#ff6f5e' : '#8fb4ff')), borderRadius: 2 },
      { type: 'line', data: P.acf.map(() => P.band), borderColor: '#f0b34a', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
      { type: 'line', data: P.acf.map(() => -P.band), borderColor: '#f0b34a', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
    ] },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'lag' } }, y: { suggestedMin: -P.band * 3, suggestedMax: P.band * 3 } } },
  });
  chart('c-streak', {
    type: 'bar',
    data: { labels: P.streaks.map((x) => x.len), datasets: [
      { label: 'observed', data: P.streaks.map((x) => x.obs), backgroundColor: '#8fb4ff', borderRadius: 2 },
      { label: 'memoryless', type: 'line', data: P.streaks.map((x) => x.exp), borderColor: '#f0b34a', pointRadius: 2 },
    ] },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { type: 'logarithmic' }, x: { title: { display: true, text: 'streak length' } } } },
  });
  const H = P.hour.filter((h) => h.rtp2 != null);
  chart('c-hour', {
    type: 'bar',
    data: { labels: H.map((h) => String(h.h).padStart(2, '0')), datasets: [
      { label: '2σ band', data: H.map((h) => [h.rtp2 - 2 * h.se, h.rtp2 + 2 * h.se]), backgroundColor: 'rgba(143,180,255,.16)', borderColor: 'rgba(143,180,255,.4)', borderWidth: 1 },
      { label: 'pays @2x', type: 'line', data: H.map((h) => h.rtp2), borderColor: COLORS[s], backgroundColor: COLORS[s], showLine: false, pointRadius: 3.5 },
      { label: 'house', type: 'line', data: H.map(() => 0.97), borderColor: '#62d2a2', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
    ] },
    options: { plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { afterLabel: (c) => `${n0(H[c.dataIndex].n)} rounds` } } }, scales: { y: { beginAtZero: false, suggestedMin: 0.85, suggestedMax: 1.12 } } },
  });
}

// ─────────── Moonshots ───────────
function pageMoon(v) {
  const s = state.source;
  const M = state.data.sources[s].moon;
  let sel = 2;
  v.innerHTML = `
  <p class="lede">Big multipliers in ${esc(label(s))}: how often they land, how long the current wait is, and how surprising that wait would be in a fair game. A drought never makes the next round more likely to go big.</p>
  <div class="grid g5" id="moon-cards">
    ${M.levels.map((l, i) => `<button class="card kpi" data-i="${i}" style="text-align:left;cursor:pointer;${i === sel ? 'border-color:var(--accent)' : ''}">
      <div class="lab">${l.m}x+</div>
      <div class="val">${n0(l.hits)} <span class="small muted">/ ${n0(l.expected)}</span></div>
      <div class="note">wait now ${n0(l.current_gap)} · P(this dry) ${pct(l.p_drought, 1)}</div>
      <div class="note">next: 50% within ${n0(l.eta50)}, 90% within ${n0(l.eta90)}</div>
    </button>`).join('')}
  </div>
  <div class="grid g2 mt">
    ${card('Gaps between hits', '<span id="gap-title"></span>', `<div class="chart"><canvas id="c-gap"></canvas></div>`)}
    ${card('Largest rounds', 'Top 25 clean rounds.', `<div class="tbl-wrap" style="max-height:300px;overflow-y:auto"><table><thead><tr><th>#</th><th class="num">Multiplier</th><th>When (UTC)</th><th class="num">Round index</th></tr></thead><tbody>
      ${M.top.map((t, i) => `<tr><td class="faint">${i + 1}</td><td class="num">${chip(t.x)}</td><td class="mono small">${esc(dt(t.t))}</td><td class="num faint">${n0(t.i)}</td></tr>`).join('')}
    </tbody></table></div>`)}
  </div>
  ${card('Level summary', '', `<div class="tbl-wrap"><table><thead><tr><th class="num">Level</th><th class="num">Hits</th><th class="num">Expected</th><th class="num">Mean gap</th><th class="num">House gap</th><th class="num">Max gap</th><th class="num">Current wait</th><th class="num">P(this dry)</th><th>Last hit</th></tr></thead><tbody>
    ${M.levels.map((l) => `<tr><td class="num">${l.m}x</td><td class="num">${n0(l.hits)}</td><td class="num faint">${n0(l.expected)}</td><td class="num">${n0(l.mean_gap)}</td><td class="num faint">${n0(l.house_gap)}</td><td class="num">${n0(l.max_gap)}</td><td class="num">${n0(l.current_gap)}</td><td class="num ${l.p_drought < 0.01 ? 'bad' : ''}">${pct(l.p_drought, 2)}</td><td class="mono small">${esc(dt(l.last))}</td></tr>`).join('')}
  </tbody></table></div>`, 'mt')}`;
  const draw = () => {
    state.charts.filter((c) => c.canvas.id === 'c-gap').forEach((c) => c.destroy());
    state.charts = state.charts.filter((c) => c.canvas.id !== 'c-gap');
    const l = M.levels[sel];
    $('#gap-title').textContent = `${l.m}x+ gaps, observed vs the geometric wait of a fair game.`;
    chart('c-gap', {
      type: 'bar',
      data: { labels: l.hist.map((h) => `${h.lo}–${h.hi}`), datasets: [
        { label: 'observed', data: l.hist.map((h) => h.obs), backgroundColor: COLORS[s], borderRadius: 2 },
        { label: 'fair game', type: 'line', data: l.hist.map((h) => h.exp), borderColor: '#e9e6df', borderDash: [4, 3], pointRadius: 2 },
      ] },
      options: { plugins: { legend: { position: 'bottom' } }, scales: { x: { title: { display: true, text: 'rounds between hits' }, ticks: { maxRotation: 45 } } } },
    });
  };
  draw();
  $('#moon-cards').addEventListener('click', (e) => {
    const b = e.target.closest('[data-i]');
    if (!b) return;
    sel = +b.dataset.i;
    document.querySelectorAll('#moon-cards [data-i]').forEach((x) => (x.style.borderColor = +x.dataset.i === sel ? 'var(--accent)' : ''));
    draw();
  });
}

// ─────────── Sessions ───────────
function pageSessions(v) {
  const s = state.source;
  const S = state.data.sources[s].sessions;
  const list = [...S.list].sort((a, b) => b.n - a.n);
  v.innerHTML = `
  <p class="lede">Rounds are grouped into collection sessions wherever the recorder stopped for more than 5 minutes (the same 300-second gap used in the Momento settings).</p>
  <div class="grid g4">
    ${kpi('Sessions', n0(S.count))}
    ${kpi('Median length', n0(S.median_len), 'rounds')}
    ${kpi('Longest', n0(S.longest), 'rounds')}
    ${kpi('Round cadence', `${f(S.cadence_s, 1)} s`, 'median gap inside a session')}
  </div>
  ${card('Daily volume and payout', 'Bars: rounds captured. Line: payout at 2x that day.', `<div class="chart"><canvas id="c-daily"></canvas></div>`, 'mt')}
  ${card('Sessions', 'Longest first.', `<div class="tbl-wrap" style="max-height:440px;overflow-y:auto"><table><thead><tr><th>#</th><th>Start (UTC)</th><th>End</th><th class="num">Rounds</th><th class="num">Minutes</th><th class="num">Pays @2x</th><th class="num">100x+</th><th class="num">Peak</th></tr></thead><tbody>
    ${list.map((x) => `<tr><td class="faint">${x.id + 1}</td><td class="mono small">${esc(dt(x.start))}</td><td class="mono small faint">${esc(dt(x.end))}</td><td class="num">${n0(x.n)}</td><td class="num">${n0(x.mins)}</td><td class="num">${f(x.rtp2, 3)}</td><td class="num">${x.p100}</td><td class="num">${chip(x.max)}</td></tr>`).join('')}
  </tbody></table></div>`, 'mt')}`;
  chart('c-daily', {
    type: 'bar',
    data: { labels: S.daily.map((d) => d.d.slice(5)), datasets: [
      { label: 'rounds', data: S.daily.map((d) => d.n), backgroundColor: 'rgba(143,180,255,.45)', borderRadius: 2, yAxisID: 'y' },
      { label: 'pays @2x', type: 'line', data: S.daily.map((d) => (d.n >= 100 ? d.rtp2 : null)), borderColor: '#f0b34a', pointRadius: 2, yAxisID: 'y2', spanGaps: true },
    ] },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { y2: { position: 'right', suggestedMin: 0.8, suggestedMax: 1.15, grid: { display: false } } } },
  });
}

// ─────────── Forecast ───────────
function pageForecast(v) {
  const s = state.source;
  const F = state.data.sources[s].forecast;
  const lvl = F.predictable.length ? 'bad' : 'ok';
  v.innerHTML = `
  <p class="lede">Five next-round models (house, long-run rate, recent window, previous round, streak length) are blended by a Hedge ensemble that trusts whichever has predicted best so far. Every round is forecast <b>before</b> it is revealed. Skill is measured against the house curve 0.97/m.</p>
  <div class="callout ${lvl}"><b>${esc(F.headline)}</b> · ${esc(label(s))}, ${n0(F.n)} rounds. ${F.predictable.length ? `Targets with signal: ${F.predictable.join(', ')}x.` : 'No target shows a skill z of 3 or more.'} Paper strategy: ${n0(F.strategy.bets)} bets, ROI ${sgn(F.strategy.roi * 100, 1)}% ± ${f(F.strategy.se * 100, 1)}% (z ${sgn(F.strategy.z)}).</div>
  <div class="grid g2 mt">
    ${card('Next round', 'Ensemble probability of reaching each target, against the house curve.', `<div class="tbl-wrap"><table><thead><tr><th class="num">Target</th><th class="num">Model</th><th class="num">House</th><th class="num">EV</th><th class="num">Wait 50/90%</th><th class="num">Skill</th><th>Verdict</th></tr></thead><tbody>
      ${F.targets.map((t) => `<tr><td class="num">${t.m}x</td><td class="num">${pct(t.p_oracle, 2)}</td><td class="num faint">${pct(t.p_house, 2)}</td><td class="num ${t.verdict === 'predictable' ? '' : 'faint'}">${sgn(t.ev * 100, 1)}%</td><td class="num">${t.eta_median} / ${t.eta_90}</td><td class="num">${zPill(t.z)}</td><td>${verdictPill(t.verdict)}</td></tr>`).join('')}
    </tbody></table></div><p class="tiny faint mt">EV is faded unless the target has proven skill. A negative EV means the model expects to lose money betting that target.</p>`)}
    ${card('Skill by target', 'Walk-forward z of log-loss improvement over the house curve. Signal needs z ≥ 3.', `<div class="chart"><canvas id="c-skill"></canvas></div>`)}
  </div>
  <div class="grid g2 mt">
    ${card('Paper strategy equity', 'Cumulative units if you bet 1 unit whenever the model saw positive EV.', `<div class="chart short"><canvas id="c-eq"></canvas></div>`)}
    ${card('Which model the ensemble trusts', 'Current weights per target.', `<div class="chart short"><canvas id="c-w"></canvas></div>`)}
  </div>
  ${card('Re-run in the browser', 'Load the clean series and replay the whole walk-forward with your own settings. Nothing is stored.', `
    <div class="controls">
      <label>House RTP <span class="mono acc" id="o-rtp-v">0.97</span><input type="range" id="o-rtp" min="0.9" max="1" step="0.005" value="0.97"></label>
      <label>Recent window <span class="mono acc" id="o-win-v">200</span><input type="range" id="o-win" min="50" max="1000" step="50" value="200"></label>
      <label>EV gate <span class="mono acc" id="o-ev-v">0.00</span><input type="range" id="o-ev" min="0" max="0.2" step="0.01" value="0"></label>
      <div class="row"><button class="btn" id="o-run">Run ${n0(F.n)} rounds</button></div>
    </div>
    <div class="progress"><i id="o-prog"></i></div>
    <div id="o-out" class="mt small muted">Precomputed results above use RTP 0.97, window 200, EV gate 0.</div>`, 'mt')}`;
  drawForecastCharts(F, s);
  const bind = (id, fmt) => { const el = $('#' + id); el.addEventListener('input', () => ($('#' + id + '-v').textContent = fmt(+el.value))); };
  bind('o-rtp', (x) => x.toFixed(3));
  bind('o-win', (x) => x);
  bind('o-ev', (x) => x.toFixed(2));
  $('#o-run').addEventListener('click', async () => {
    const btn = $('#o-run');
    btn.disabled = true;
    const ser = await loadSeries(s);
    const o = createOracle({ rtp: +$('#o-rtp').value, window: +$('#o-win').value, evThreshold: +$('#o-ev').value });
    const xs = ser.x;
    const chunk = 2500;
    for (let i = 0; i < xs.length; i += chunk) {
      for (let j = i; j < Math.min(i + chunk, xs.length); j++) o.push(xs[j]);
      $('#o-prog').style.width = `${Math.min(100, ((i + chunk) / xs.length) * 100)}%`;
      await new Promise((r) => setTimeout(r, 0));
    }
    const snap = o.snapshot();
    const st = snap.strategy;
    $('#o-out').innerHTML = `<div class="callout ${snap.predictable.length ? 'bad' : 'ok'}"><b>${esc(snap.headline)}</b> · ${n0(snap.n)} rounds · paper ${n0(st.bets)} bets, ROI ${sgn(st.roi * 100, 1)}% ± ${f(st.se * 100, 1)}% (z ${sgn(st.z)})</div>
      <div class="chips mt">${snap.targets.map((t) => `<span class="chip">${t.m}x ${pct(t.pe, 1)} · z ${sgn(t.perModel.ensemble.z, 1)}</span>`).join('')}</div>`;
    btn.disabled = false;
  });
}
function drawForecastCharts(F, s) {
  chart('c-skill', {
    type: 'bar',
    data: { labels: F.targets.map((t) => t.m + 'x'), datasets: [
      { label: 'ensemble', data: F.targets.map((t) => t.z), backgroundColor: F.targets.map((t) => (t.z >= 3 ? '#ff6f5e' : t.z >= 2 ? '#f0b34a' : '#3a3f4b')), borderRadius: 3 },
      ...['markov', 'streak'].map((m, i) => ({ label: MODEL_LABEL[m] || m, type: 'line', data: F.targets.map((t) => t.models[m].z), borderColor: i ? '#c49bff' : '#8fb4ff', pointRadius: 2.5, borderWidth: 1.2, showLine: false })),
    ] },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { suggestedMin: -3, suggestedMax: 4, grid: { color: (c) => (c.tick.value === 3 ? '#5a2a24' : '#1f2229') } } } },
  });
  chart('c-eq', {
    type: 'line',
    data: { datasets: [{ data: F.curve.map((c) => ({ x: c[0], y: c[1] })), borderColor: COLORS[s], pointRadius: 0, borderWidth: 1.6, fill: { target: 'origin', above: 'rgba(98,210,162,.08)', below: 'rgba(255,111,94,.08)' } }] },
    options: { parsing: false, plugins: { legend: { display: false } }, scales: { x: { type: 'linear', title: { display: true, text: 'round' } }, y: { title: { display: true, text: 'units' } } } },
  });
  const pal = ['#5d616b', '#8fb4ff', '#62d2a2', '#f0b34a', '#c49bff'];
  chart('c-w', {
    type: 'bar',
    data: { labels: F.targets.map((t) => t.m + 'x'), datasets: MODELS.map((m, i) => ({ label: MODEL_LABEL[m] || m, data: F.targets.map((t) => t.weights[m]), backgroundColor: pal[i] })) },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { x: { stacked: true }, y: { stacked: true, max: 1, ticks: { callback: (v) => v * 100 + '%' } } } },
  });
}

// ─────────── Replay ───────────
function pageReplay(v) {
  const R = state.data.replay;
  v.innerHTML = `
  <p class="lede">The live database logged <b>${n0(R.runs)}</b> <code>session_replay_test</code> runs between Jun 20 and Jul 19. Each run replays captured sessions and records how often its prediction matched. A fair game gives about <b>${pct(R.chance_2x, 1)}</b> for any fixed call at 2x, so steady accuracy near ${pct(R.acc_base, 0)} is what a no-edge predictor looks like.</p>
  <div class="grid g4">
    ${kpi('Replay runs', n0(R.runs), Object.entries(R.status).map(([k, n]) => `${k} ${n0(n)}`).join(' · '))}
    ${kpi('Normal-day accuracy', pct(R.acc_base, 1), `σ ${pct(R.acc_base_sd, 1)} per run`)}
    ${kpi('Jul 4–6 accuracy', pct(R.spike.acc, 1), `${n0(R.spike.runs)} runs`, 'bad')}
    ${kpi('Runs above 80%', pct(R.spike.share_over_80, 0), `vs ${pct(R.spike.base_share_over_80, 2)} on other days`, 'bad')}
  </div>
  <div class="grid g2 mt">
    ${card('Accuracy by day', 'Red days are more than 15 points above normal.', `<div class="chart"><canvas id="c-rd"></canvas></div>`)}
    ${card('Accuracy distribution', 'All runs.', `<div class="chart"><canvas id="c-rh"></canvas></div>`)}
  </div>
  <div class="grid g2 mt">
    ${card('Accuracy vs predictions scored', '3,000-run sample. High accuracy at large sample sizes is not luck.', `<div class="chart"><canvas id="c-rs"></canvas></div>`)}
    ${card('Reading this', '', `<div class="prose">
      <p>Normal replay days sit at ${pct(R.acc_base, 0)}, which is chance level. On Jul 4, 5 and 6, accuracy jumped to ${pct(R.spike.acc, 0)} across ${n0(R.spike.preds)} scored predictions. That is far too many to be a lucky streak.</p>
      <p>The jump starts the day after ${n0(state.data.decomposition.burst.rows)} synthetic Aviator rows were written, and over 90% of those rows reach 2x. A predictor calling 2x on them would look brilliant. The timing points strongly at that burst, but <code>replay_results</code> has no session ids, so the link is an inference, not proof.</p>
      <p>Accuracy dropped back to chance on Jul 7 and stayed there. Treat any model, threshold or feature tuned on Jul 4–6 replays as unvalidated.</p>
    </div>`)}
  </div>`;
  chart('c-rd', {
    type: 'bar',
    data: { labels: R.daily.map((d) => d.d.slice(5)), datasets: [
      { label: 'mean accuracy', data: R.daily.map((d) => d.acc), backgroundColor: R.daily.map((d) => (d.flag ? '#ff6f5e' : '#8fb4ff')), borderRadius: 2 },
      { label: 'chance', type: 'line', data: R.daily.map(() => R.acc_base), borderColor: '#f0b34a', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
    ] },
    options: { plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { afterLabel: (c) => `${n0(R.daily[c.dataIndex].runs)} runs` } } }, scales: { y: { min: 0, max: 1, ticks: { callback: (x) => x * 100 + '%' } } } },
  });
  chart('c-rh', {
    type: 'bar',
    data: { labels: R.hist.map((h) => pct(h.lo, 0)), datasets: [{ data: R.hist.map((h) => h.n), backgroundColor: R.hist.map((h) => (h.lo >= 0.5 ? '#ff6f5e' : '#8fb4ff')), borderRadius: 2 }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { type: 'logarithmic', title: { display: true, text: 'runs' } } } },
  });
  chart('c-rs', {
    type: 'scatter',
    data: { datasets: [{ data: R.scatter.map((p) => ({ x: Math.max(1, p[0]), y: p[1] })), backgroundColor: R.scatter.map((p) => (p[1] > 0.5 ? 'rgba(255,111,94,.55)' : 'rgba(143,180,255,.35)')), pointRadius: 2 }] },
    options: { parsing: false, plugins: { legend: { display: false } }, scales: { x: { type: 'logarithmic', title: { display: true, text: 'predictions scored' } }, y: { min: 0, max: 1, ticks: { callback: (x) => x * 100 + '%' } } } },
  });
}

// ─────────── Lab ───────────
function pageLab(v) {
  v.innerHTML = `
  <p class="lede">Paste or upload any crash-game history and it gets the same treatment as the backups: grid check, fairness ladder, runs test, autocorrelation, moonshot waits and the walk-forward forecast. It runs in your browser and nothing is uploaded.</p>
  <div class="grid g2">
    ${card('Input', 'One multiplier per line, comma lists, CSV with a multiplier column, or JSON arrays.', `
      <div class="drop"><textarea id="lab-in" placeholder="1.42&#10;3.10&#10;1.00&#10;27.55&#10;…"></textarea>
      <div class="row mt"><button class="btn" id="lab-run">Analyse</button><label class="btn ghost" style="cursor:pointer">Upload file<input type="file" id="lab-file" accept=".csv,.json,.txt" hidden></label>
      <span class="small muted">or load:</span>
      <select id="lab-src"><option value="">a clean stream…</option>${ORDER.map((s) => `<option value="${s}">${esc(label(s))}</option>`).join('')}</select></div></div>`)}
    ${card('Result', '', `<div id="lab-out" class="small muted">Waiting for rounds.</div>`)}
  </div>
  <div class="grid g2 mt" id="lab-charts" hidden>
    ${card('Payout ladder', '', `<div class="chart short"><canvas id="c-lab-l"></canvas></div>`)}
    ${card('Forecast skill', '', `<div class="chart short"><canvas id="c-lab-s"></canvas></div>`)}
  </div>`;
  $('#lab-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) $('#lab-in').value = await file.text();
  });
  $('#lab-src').addEventListener('change', async (e) => {
    if (!e.target.value) return;
    const ser = await loadSeries(e.target.value);
    $('#lab-in').value = ser.x.slice(-5000).join('\n');
  });
  $('#lab-run').addEventListener('click', runLab);
}
async function runLab() {
  const xs = parseRounds($('#lab-in').value);
  const out = $('#lab-out');
  if (xs.length < 100) { out.innerHTML = `<span class="err">Found ${xs.length} rounds. Paste at least 100 for meaningful tests.</span>`; return; }
  out.innerHTML = 'Running…';
  await new Promise((r) => setTimeout(r, 20));
  const off = xs.filter((x) => Math.abs(x * 100 - Math.round(x * 100)) > 1e-6).length;
  const L = auditLadder(xs);
  const R = runsTest(xs);
  const A = acf(xs, 10);
  const band = 2 / Math.sqrt(xs.length);
  const M = moonLevels(xs);
  const o = createOracle();
  xs.forEach((x) => o.push(x));
  const snap = o.snapshot();
  const worst = L.reduce((a, b) => (Math.abs(b.z) > Math.abs(a.z) ? b : a));
  out.innerHTML = `
    <div class="callout ${snap.predictable.length ? 'bad' : 'ok'}"><b>${esc(snap.headline)}</b> · ${n0(xs.length)} rounds</div>
    <div class="cmp mt">
      <div class="h">Check</div><div class="h num">Value</div><div class="h">Read</div>
      <div>Off 0.01 grid</div><div class="mono num">${n0(off)}</div><div>${off / xs.length > 0.01 ? '<span class="pill bad">suspect</span>' : '<span class="pill ok">clean</span>'}</div>
      <div>Pays @2x</div><div class="mono num">${f(L[1].rtp, 3)}</div><div>${zPill(L[1].z)}</div>
      <div>Worst target</div><div class="mono num">${worst.m}x</div><div>${zPill(worst.z)}</div>
      <div>Runs test</div><div class="mono num">${n0(R.runs)}</div><div>${zPill(R.z)}</div>
      <div>ACF lags outside band</div><div class="mono num">${A.filter((a) => Math.abs(a) > band).length} / 10</div><div class="small muted">±${f(band, 3)}</div>
      <div>100x wait now</div><div class="mono num">${n0(M[2].current)}</div><div class="small muted">P(this dry) ${pct(M[2].pDry, 1)}</div>
      <div>Paper strategy</div><div class="mono num">${sgn(snap.strategy.roi * 100, 1)}%</div><div>${zPill(snap.strategy.z)}</div>
    </div>`;
  $('#lab-charts').hidden = false;
  state.charts.filter((c) => c.canvas.id.startsWith('c-lab')).forEach((c) => c.destroy());
  state.charts = state.charts.filter((c) => !c.canvas.id.startsWith('c-lab'));
  chart('c-lab-l', {
    type: 'bar',
    data: { labels: L.map((l) => l.m + 'x'), datasets: [
      { label: '2σ', data: L.map((l) => [l.rtp - 2 * l.se, l.rtp + 2 * l.se]), backgroundColor: 'rgba(240,179,74,.18)', borderColor: 'rgba(240,179,74,.5)', borderWidth: 1 },
      { label: 'pays', type: 'line', data: L.map((l) => l.rtp), borderColor: '#f0b34a', showLine: false, pointRadius: 4 },
      { label: 'house', type: 'line', data: L.map(() => 0.97), borderColor: '#62d2a2', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
    ] },
    options: { plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: false } } },
  });
  chart('c-lab-s', {
    type: 'bar',
    data: { labels: snap.targets.map((t) => t.m + 'x'), datasets: [{ data: snap.targets.map((t) => t.perModel.ensemble.z), backgroundColor: snap.targets.map((t) => (t.perModel.ensemble.z >= 3 ? '#ff6f5e' : '#3a3f4b')), borderRadius: 3 }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { suggestedMin: -3, suggestedMax: 4 } } },
  });
}

// ─────────── Method ───────────
function pageMethod(v) {
  const d = state.data;
  v.innerHTML = `
  <div class="grid g2">
    ${card('How this was built', '', `<div class="prose">
      <h3>1. Ledger</h3><p>Every file in <code>data/</code> on <code>InvestigationSuite@decomputation</code> is hashed. The 7z archive is opened and each entry's CRC32 is checked against the file on disk. The live <code>avfs.db</code> is attached next to the Jul 14 backup and diffed by primary key.</p>
      <h3>2. Decomposition</h3><p>Rows are tested in order: unknown source, 2024 seed, test or probe file, mislabeled source, server-generated Aviator burst, value off the 0.01 grid, duplicate timestamp, below 1.00x. The first rule a row breaks is the one it is quarantined under.</p>
      <h3>3. Analysis</h3><p>Fairness ladder against 0.97/m with binomial σ. Wald–Wolfowitz runs test at 2x, autocorrelation of log multipliers to lag 20, streak lengths against a geometric distribution, conditional next-round tables, and hour-of-day χ².</p>
      <h3>4. Forecast</h3><p>The Momento Oracle engine: five models shrunk toward the house curve, a Hedge ensemble, walk-forward log-loss skill with a paired z per target. A target counts as predictable only at z ≥ 3. The paper strategy counts as an edge only at z ≥ 3 over 200+ bets.</p>
      <h3>5. Replay audit</h3><p>Daily means of <code>replay_results.accuracy</code>, with days more than 15 points above the normal-day mean flagged.</p>
    </div>`)}
    <div class="grid">
      ${card('Downloads', '', `<div class="tbl-wrap"><table><tbody>
        <tr><td><a class="acc" href="data/suite.json" download>suite.json</a></td><td class="small muted wrap">Every analysis on this site in one file</td></tr>
        ${ORDER.map((s) => `<tr><td><a class="acc" href="data/series/${s}.json" download>series/${s}.json</a></td><td class="small muted wrap">${esc(label(s))}, ${n0(d.sources[s].meta.n)} clean rounds (t0 + deltas, x)</td></tr>`).join('')}
        <tr><td><a class="acc" href="dl/decompose.py" download>decompose.py</a></td><td class="small muted wrap">The pipeline that produced everything here</td></tr>
        <tr><td><a class="acc" href="dl/oracle_engine.py" download>oracle_engine.py</a></td><td class="small muted wrap">Forecast engine, Python port</td></tr>
        <tr><td><a class="acc" href="dl/predict.py" download>predict.py</a></td><td class="small muted wrap">Prediction pipeline: features, walk-forward models, calibration, backtest, controls. <span class="mono">python predict.py --repo ../InvestigationSuite [--fast]</span> or <span class="mono">--score rounds.csv</span></td></tr>
        <tr><td><a class="acc" href="data/predict.json" download>predict.json</a></td><td class="small muted wrap">Full pipeline output behind the Prediction pages</td></tr>
        <tr><td><a class="acc" href="dl/platform_features.py" download>platform_features.py</a></td><td class="small muted wrap">Platform-book features (chapter 18): integrity, fingerprint, signals, ETA, compare, registry, book index. <span class="mono">python platform_features.py --repo ../InvestigationSuite --book ../momento-platform-book</span></td></tr>
        <tr><td><a class="acc" href="data/platform.json" download>platform.json</a></td><td class="small muted wrap">Output behind the Now, Evidence and Lab pages</td></tr>
      </tbody></table></div>`)}
      ${card('Rebuild from the repo', '', `<pre>git clone -b decomputation \\
  https://github.com/avfsmomentoserver-cell/InvestigationSuite IS
pip install pandas numpy py7zr
python pipeline/decompose.py --repo IS
python pipeline/predict.py --repo IS
python pipeline/platform_features.py --repo IS --book mpb
python -m http.server 8080</pre>`)}
      ${card('Limits', '', `<div class="prose"><p>Captures are what the recorder saw. Missed rounds inside a session cannot be detected from multipliers alone. The replay link to the burst is inferred from timing, and the momento.db engine stream is a simulator kept as a control, not a real game.</p></div>`)}
    </div>
  </div>`;
}

boot();
