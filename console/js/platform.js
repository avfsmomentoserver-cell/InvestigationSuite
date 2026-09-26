// Platform pages built from the Momento Platform Book (momento-platform-book, chapter 18 feature catalogue).
// Data: data/platform.json and data/book.json (pipeline/platform_features.py), data/predict.json, data/series/*.json.
// State lives in memory only; nothing is written to browser storage.

import { loadPredict, nextFeatures } from './predict.js';

let PL = null;
let BK = null;
const REG = []; // session experiments (sequence searches + custom runs)
const LEDGER = []; // arena track record (hash chain)

export async function loadPlatform() {
  if (!PL) PL = await (await fetch('data/platform.json')).json();
  return PL;
}
async function loadBook() {
  if (!BK) BK = await (await fetch('data/book.json')).json();
  return BK;
}

// ─────────── stats helpers ───────────
const normSf = (z) => 0.5 * erfc(z / Math.SQRT2);
function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
function wilson(k, n, z = 1.96) {
  if (!n) return [null, null];
  const p = k / n, d = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [c - h, c + h];
}
function bh(ps) {
  const n = ps.length, o = ps.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
  const q = new Array(n);
  let prev = 1;
  for (let r = n; r >= 1; r--) { const [p, i] = o[r - 1]; prev = Math.min(prev, (p * n) / r); q[i] = prev; }
  return q;
}
function twoProp(k1, n1, k0, n0) {
  const p1 = k1 / n1, p0 = k0 / n0, pp = (k1 + k0) / (n1 + n0);
  const se = Math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0)) || 1e-12;
  const z = (p1 - p0) / se;
  return { p1, p0, z, p: 2 * normSf(Math.abs(z)) };
}
function rng(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const RTP = 0.97;
const houseP = (m) => Math.min(1, RTP / m);
const fairRound = (u) => Math.floor(Math.max(1, RTP / (1 - u)) * 100) / 100;
function shuffle(a, r) { const b = a.slice(); for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; }
async function sha256(s) {
  const b = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s));
  return [...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, '0')).join('');
}
const BANDS = [1, 1.2, 1.5, 2, 3, 5, 10, 20, 50, 100];
const BAND_NAMES = ['dust', 'floor', 'low', 'base', 'mid', 'high', 'ignition', 'moonshot', 'mega', 'cosmic'];
const bandOf = (x) => { let b = 0; for (let i = 0; i < BANDS.length; i++) if (x >= BANDS[i]) b = i; return b; };
const qPill = (q, esc) => (q == null ? '<span class="pill">–</span>' : `<span class="pill ${q < 0.05 ? 'bad' : q < 0.2 ? 'acc' : 'ok'}">q ${q < 0.001 ? '<0.001' : q.toFixed(3)}</span>`);
const STREAMS = ['aviator', 'skyward', 'skyward_deluxe', 'aviator_jul29', 'engine'];

function loading(v) { v.innerHTML = '<div class="skeleton-page"><div class="sk" style="height:18px;width:52%"></div><div class="grid g4 mt">' + '<div class="sk" style="height:92px"></div>'.repeat(4) + '</div><div class="sk mt" style="height:320px"></div></div>'; }
async function withPL(v, C, fn, needBook = false) {
  const seq = C.state.seq;
  if (!PL || (needBook && !BK)) loading(v);
  try { await loadPlatform(); if (needBook) await loadBook(); } catch (e) { v.innerHTML = `<div class="card err">Could not load platform data: ${C.esc(e.message)}. Run <code>python pipeline/platform_features.py --repo … --book …</code> first.</div>`; return; }
  if (C.state.seq !== seq) return;
  try { await fn(seq); } catch (e) { console.error(e); v.innerHTML = `<div class="card err">Render error: ${C.esc(e.message)}</div>`; }
}
const seg = (id, opts, cur) => `<div class="seg" id="${id}" role="tablist">${opts.map(([k, l]) => `<button role="tab" data-k="${k}" class="${String(k) === String(cur) ? 'on' : ''}">${l}</button>`).join('')}</div>`;
function bindSeg(id, cb) {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    el.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    cb(b.dataset.k);
  });
}
function killChart(C, c) { if (c) { c.destroy(); const i = C.state.charts.indexOf(c); if (i >= 0) C.state.charts.splice(i, 1); } return null; }
const bookLink = (d, h, txt) => `<a class="acc" href="#/book?d=${d}${h ? `&h=${h}` : ''}">${txt}</a>`;
const fid = (id) => `<span class="pill info">${id}</span>`;
function ciBar(lo, hi, pt, span) {
  // horizontal CI glyph centred on 0, span = max |value| shown
  const x = (v) => 50 + (Math.max(-span, Math.min(span, v)) / span) * 50;
  if (lo == null) return '<div class="cibar"></div>';
  const sig = lo > 0 || hi < 0;
  return `<div class="cibar"><i class="zero"></i><b class="${sig ? 'sig' : ''}" style="left:${x(lo)}%;width:${Math.max(0.8, x(hi) - x(lo))}%"></b><em style="left:${x(pt)}%"></em></div>`;
}

// ═══════════ Book reader + Ask the book (F-37, partial) ═══════════
export function pageBook(v, C) {
  withPL(v, C, async () => {
    const { esc } = C;
    const qs = new URLSearchParams((location.hash.split('?')[1]) || '');
    const slug = qs.get('d') || 'index';
    const doc = BK.docs.find((d) => d.slug === slug) || BK.docs[0];
    const parts = ['Front', 'Part I', 'Part II', 'Appendices'];
    const words = BK.docs.reduce((a, d) => a + d.words, 0);
    v.innerHTML = `
    <p class="lede">The Momento Platform Book, read inside the suite. It documents the 13 repositories, their engines and the 38-feature catalogue this console now implements. Every feature page links back to its specification here. ${esc(BK.docs.length)} documents, ${C.n0(words)} words, content hash <span class="mono">${esc(BK.hash)}</span>.</p>
    ${C.card('Ask the book', 'Ranked full-text search over every section (BM25). Answers are the best-matching passages, each cited to its chapter and section. There is no language model here: it quotes, it does not paraphrase.', `
      <form id="bk-form" class="row"><input id="bk-q" type="text" placeholder="e.g. memorylessness test, house edge CUSUM, hash chain" style="flex:1;min-width:220px" value="${esc(qs.get('q') || '')}"><button class="btn" type="submit">Search</button></form>
      <div class="chips mt">${['Is overdue real?', 'tape integrity score', 'Benjamini-Hochberg', 'bankroll ruin', 'provably fair verification', 'Kaplan-Meier ETA'].map((s) => `<button class="btn ghost bk-ex" type="button" data-q="${esc(s)}">${esc(s)}</button>`).join('')}</div>
      <div id="bk-res" class="mt"></div>`)}
    <div class="book mt">
      <nav class="book-toc card" aria-label="Book contents">${parts.map((p) => `<div class="grp">${p}</div>${BK.docs.filter((d) => d.part === p).map((d) => `<a href="#/book?d=${d.slug}" class="${d.slug === doc.slug ? 'on' : ''}">${esc(d.title.replace(/^Appendix /, 'App. '))}</a>`).join('')}`).join('')}</nav>
      <article class="card book-doc">
        <div class="row" style="justify-content:space-between"><span class="tiny faint mono">${esc(doc.src)} · ${C.n0(doc.words)} words</span>${doc.toc.filter((t) => t.level === 2).length > 2 ? `<select id="bk-sec" aria-label="Jump to section"><option value="">Jump to section…</option>${doc.toc.filter((t) => t.level <= 3).map((t) => `<option value="${esc(t.id)}">${t.level === 3 ? '  ' : ''}${esc(t.name.replace(/[`*]/g, ''))}</option>`).join('')}</select>` : ''}</div>
        <div class="md" id="bk-md">${doc.html}</div>
      </article>
    </div>`;
    const md = document.getElementById('bk-md');
    md.querySelectorAll('a[href^="http"]').forEach((a) => { a.target = '_blank'; a.rel = 'noopener'; });
    md.querySelectorAll('table').forEach((t) => { const w = document.createElement('div'); w.className = 'tbl-wrap'; t.parentNode.insertBefore(w, t); w.appendChild(t); });
    // Link feature IDs to their built page
    const cat = Object.fromEntries(PL.catalogue.map((c) => [c.id, c]));
    md.querySelectorAll('h3').forEach((h) => {
      const m = h.textContent.match(/^(F-\d\d)/);
      if (m && cat[m[1]] && cat[m[1]].page) {
        const a = document.createElement('a');
        a.href = `#/${cat[m[1]].page}`; a.className = 'pill ' + (cat[m[1]].status === 'built' ? 'ok' : 'acc'); a.style.marginLeft = '10px'; a.style.textDecoration = 'none';
        a.textContent = cat[m[1]].status === 'built' ? 'Open in suite' : 'Partial in suite';
        h.appendChild(a);
      }
    });
    const jump = (id) => { const el = id && document.getElementById(id); if (el) el.scrollIntoView({ block: 'start' }); };
    const sec = document.getElementById('bk-sec');
    if (sec) sec.addEventListener('change', () => jump(sec.value));
    if (qs.get('h')) setTimeout(() => jump(qs.get('h')), 30);
    // search
    const tok = (s) => s.toLowerCase().normalize('NFKD').replace(/[^\p{L}\p{N}\s-]/gu, ' ').split(/[\s-]+/).filter((w) => w.length > 1 && !STOP.has(w));
    if (!BK.idx) {
      const docs = BK.chunks.map((c) => { const t = tok(c.t + ' ' + c.t + ' ' + c.x); const tf = {}; t.forEach((w) => (tf[w] = (tf[w] || 0) + 1)); return { tf, len: t.length }; });
      const df = {};
      docs.forEach((d) => Object.keys(d.tf).forEach((w) => (df[w] = (df[w] || 0) + 1)));
      BK.idx = { docs, df, avg: docs.reduce((a, d) => a + d.len, 0) / docs.length };
    }
    const search = (q) => {
      const terms = [...new Set(tok(q))];
      const { docs, df, avg } = BK.idx;
      const N = docs.length;
      const sc = docs.map((d, i) => {
        let s = 0;
        for (const w of terms) {
          const keys = d.tf[w] ? [w] : Object.keys(d.tf).filter((k) => w.length > 4 && k.startsWith(w.slice(0, Math.max(5, w.length - 2))));
          for (const k of keys.slice(0, 3)) {
            const idf = Math.log(1 + (N - df[k] + 0.5) / (df[k] + 0.5));
            const f = d.tf[k];
            s += (idf * f * 2.2) / (f + 1.2 * (0.25 + 0.75 * (d.len / avg)));
          }
        }
        return [s, i];
      }).filter((x) => x[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, 6);
      const out = document.getElementById('bk-res');
      if (!terms.length) { out.innerHTML = ''; return; }
      if (!sc.length) { out.innerHTML = '<div class="callout">No section of the book mentions those words. Try a shorter query.</div>'; return; }
      const hl = (s) => { let t = esc(s); terms.forEach((w) => { t = t.replace(new RegExp(`\\b(${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\w*)`, 'gi'), '<mark>$1</mark>'); }); return t; };
      const snip = (x) => {
        const lx = x.toLowerCase();
        let at = terms.map((w) => lx.indexOf(w)).filter((i) => i >= 0).sort((a, b) => a - b)[0] ?? 0;
        at = Math.max(0, at - 120);
        return (at ? '…' : '') + x.slice(at, at + 380) + (x.length > at + 380 ? '…' : '');
      };
      const top = sc[0][0];
      out.innerHTML = sc.map(([s, i], k) => {
        const c = BK.chunks[i];
        return `<div class="hit ${k === 0 ? 'best' : ''}"><div class="row" style="justify-content:space-between"><a class="acc" href="#/book?d=${c.d}${c.h ? `&h=${c.h}` : ''}"><b>${esc(c.t.replace(/[`*]/g, ''))}</b></a><span class="tiny faint mono">${esc(c.doc)} · relevance ${(s / top * 100).toFixed(0)}</span></div><p class="small muted">${hl(snip(c.x))}</p></div>`;
      }).join('');
    };
    document.getElementById('bk-form').addEventListener('submit', (e) => { e.preventDefault(); search(document.getElementById('bk-q').value); });
    v.querySelectorAll('.bk-ex').forEach((b) => b.addEventListener('click', () => { document.getElementById('bk-q').value = b.dataset.q; search(b.dataset.q); }));
    if (qs.get('q')) search(qs.get('q'));
  }, true);
}
const STOP = new Set('the a an and or of to in on for is are be by with as at it this that from was were not no its into than then so if can each per all any our we you your their there which who what when how does do has have had but also more most such these those only one two over under use used via'.split(' '));

// ═══════════ Feature catalogue coverage ═══════════
export function pageCatalogue(v, C) {
  withPL(v, C, async () => {
    const { esc, kpi, card } = C;
    const cat = PL.catalogue;
    const cnt = (s, pr) => cat.filter((c) => c.status === s && (!pr || c.priority === pr)).length;
    const p1 = cat.filter((c) => c.priority === 'P1');
    v.innerHTML = `
    <p class="lede">Chapter 18 of the book invents 38 features. This is how many run here, on the offline clean data from the backups. "Built" means a working version you can use in this console. "Partial" means part of the feature. "Needs the live platform" means it depends on something the backups do not contain, such as live collectors or push alerts.</p>
    <div class="grid g4">
      ${kpi('Built here', `${cnt('built')}/38`, 'working on the clean data', 'ok')}
      ${kpi('Partial', cnt('partial'), 'part of the specification')}
      ${kpi('Needs live platform', cnt('n/a'), 'collectors, alerts, engine ledger', 'faint')}
      ${kpi('P1 coverage', `${p1.filter((c) => c.status !== 'n/a').length}/${p1.length}`, `${p1.filter((c) => c.status === 'built').length} built, ${p1.filter((c) => c.status === 'partial').length} partial`, 'acc')}
    </div>
    ${card('Feature catalogue', `Specifications: ${bookLink('18-feature-catalogue', null, 'chapter 18')}. Rows link to the page that implements each feature.`, `
      <div class="row">${seg('cat-pr', [['all', 'All'], ['P1', 'P1'], ['P2', 'P2'], ['P3', 'P3']], 'all')}${seg('cat-st', [['all', 'Any status'], ['built', 'Built'], ['partial', 'Partial'], ['n/a', 'Needs live']], 'all')}</div>
      <div class="tbl-wrap mt"><table id="cat-tbl"></table></div>`, 'mt')}`;
    let pr = 'all', st = 'all';
    const draw = () => {
      const rows = cat.filter((c) => (pr === 'all' || c.priority === pr) && (st === 'all' || c.status === st));
      document.getElementById('cat-tbl').innerHTML = `<thead><tr><th>ID</th><th>Feature</th><th>Area</th><th>Priority</th><th>Effort</th><th>Status</th><th>Where</th></tr></thead><tbody>${rows.map((c) => `<tr>
        <td class="mono">${c.id}</td><td class="wrap">${esc(c.name.replace(/`/g, ''))}</td><td class="small muted">${esc(c.area)}</td><td><span class="pill ${c.priority === 'P1' ? 'acc' : ''}">${c.priority}</span></td><td class="mono small">${esc(c.effort)}</td>
        <td>${c.status === 'built' ? '<span class="pill ok">built</span>' : c.status === 'partial' ? '<span class="pill acc">partial</span>' : '<span class="pill">needs live</span>'}</td>
        <td class="small">${c.page ? `<a class="acc" href="#/${c.page}">${esc(PAGE_NAME[c.page] || c.page)}</a>` : `<span class="faint">${esc(c.why || '')}</span>`}</td></tr>`).join('')}</tbody>`;
    };
    bindSeg('cat-pr', (k) => { pr = k; draw(); });
    bindSeg('cat-st', (k) => { st = k; draw(); });
    draw();
  });
}
const PAGE_NAME = { integrity: 'Tape Integrity', fingerprint: 'Fingerprint & Compare', sequence: 'Sequence Search', arena: 'Predictor Arena', leaderboard: 'Model Leaderboard', signals: 'Signal Strip', calibration: 'Calibration', eta: 'ETA Board', simulator: 'Bankroll & Ledger', experiments: 'Experiment Registry', nextround: 'Next Round (explain)', fairness: 'Fairness Audit', book: 'Platform Book' };

// ═══════════ F-01 Tape Integrity ═══════════
export function pageIntegrity(v, C) {
  withPL(v, C, async () => {
    const { esc, n0, f, pct, kpi, card, chart } = C;
    const s = C.state.source;
    const I = PL.integrity[s];
    const S = I.summary;
    const B = PL.integrity_burst && PL.integrity_burst[0];
    const col = (g) => (g >= 0.95 ? 'var(--ok)' : g >= 0.8 ? 'var(--accent)' : 'var(--bad)');
    v.innerHTML = `
    <p class="lede">${fid('F-01')} Every session gets a score between 0 and 1 from two components: completeness, from gaps in the timestamps that are too long to be explained by the rounds themselves, and the share of rounds below 1.2x, compared with the ${pct(S.p_low_ref, 2)} a fair 97% curve predicts. The score is their geometric mean. Spec: ${bookLink('18-feature-catalogue', 'f-01-tape-integrity-score-p1-s', 'F-01')}.</p>
    <div class="grid g4">
      ${kpi('Weighted integrity', f(S.weighted_integrity, 3), `${S.sessions} sessions, ${n0(S.rounds)} rounds`, S.weighted_integrity >= 0.95 ? 'ok' : 'acc')}
      ${kpi('Rounds in sessions ≥ 0.95', pct(S.share_ge_095), 'book target: ≥ 95% after a cleanup')}
      ${kpi('Estimated missing rounds', n0(S.est_missing), `${pct(S.est_missing / (S.rounds + S.est_missing), 2)} of the tape`)}
      ${kpi('Sessions with a gap', `${S.void_sessions}/${S.sessions}`, 'void rule: at least one missing round', S.void_sessions ? 'acc' : 'ok')}
    </div>
    ${card('Integrity heat-strip', 'One tile per session, width proportional to rounds. Hover a tile for its numbers.', `<div class="strip" id="ti-strip">${I.sessions.map((x) => `<i style="flex:${x.n};background:${col(x.integrity)}" title="Session ${x.id} · ${n0(x.n)} rounds · integrity ${f(x.integrity, 3)} · completeness ${pct(x.completeness)} · low share ${pct(x.low_share)} (z ${x.low_z})"></i>`).join('')}</div>
      <div class="legend-row mt small muted"><span><i style="background:var(--ok)"></i>≥ 0.95</span><span><i style="background:var(--accent)"></i>0.80–0.95</span><span><i style="background:var(--bad)"></i>&lt; 0.80</span></div>`, 'mt')}
    <div class="grid g2 mt">
      ${card('Completeness vs low-share', 'Each dot is a session. The shaded band is |z| ≤ 2 for the low share.', '<div class="chart"><canvas id="c-ti"></canvas></div>')}
      ${card('Does the detector catch the injected burst?', 'The same score applied to the 7,662 rows quarantined on Jul 3.', B ? `
        <div class="grid g2">${kpi('Burst integrity', f(B.integrity, 3), 'score for the quarantined block', 'bad')}${kpi('Share below 1.2x', pct(B.low_share, 2), `fair curve: ${pct(S.p_low_ref, 1)} · z ${B.low_z}`, 'bad')}</div>
        <div class="callout bad mt"><b>Caught.</b> A fair tape has about ${pct(S.p_low_ref, 0)} of rounds below 1.2x. The burst has ${pct(B.low_share, 2)}, which gives it a score of ${f(B.integrity, 2)}. Scoring integrity on every ingest would have flagged it automatically instead of it being found by hand.</div>
        <p class="small muted mt">Duration model used for gaps on this stream: interval ≈ ${f(I.model.a, 1)} s + ${f(I.model.b, 1)} s × ln(x). A 50x flight takes several times longer than a 1.2x one, so a long interval after a big round is expected rather than a sign of missing data. Only the excess counts.</p>` : '<p class="muted">No burst in this build.</p>')}
    </div>
    ${card('Sessions', 'Worst integrity first.', `<div class="tbl-wrap"><table><thead><tr><th>#</th><th>Start (UTC)</th><th class="num">Rounds</th><th class="num">Median Δt</th><th class="num">Gaps</th><th class="num">Est. missing</th><th class="num">Completeness</th><th class="num">Low share [95% CI]</th><th class="num">z</th><th class="num">Integrity</th></tr></thead><tbody>
      ${I.sessions.slice().sort((a, b) => a.integrity - b.integrity).map((x) => `<tr><td class="mono">${x.id}</td><td class="mono small">${new Date(x.start * 1000).toISOString().slice(0, 16).replace('T', ' ')}</td><td class="num">${n0(x.n)}</td><td class="num">${f(x.median_dt, 1)} s</td><td class="num">${x.gaps}</td><td class="num">${n0(x.est_missing)}</td><td class="num">${pct(x.completeness, 2)}</td><td class="num">${pct(x.low_share)} <span class="faint">[${pct(x.low_lo)}, ${pct(x.low_hi)}]</span></td><td class="num ${Math.abs(x.low_z) > 2 ? 'acc' : ''}">${x.low_z}</td><td class="num" style="color:${col(x.integrity)}">${f(x.integrity, 3)}</td></tr>`).join('')}</tbody></table></div>`, 'mt')}`;
    chart('c-ti', {
      type: 'scatter',
      data: { datasets: [{ label: 'sessions', data: I.sessions.map((x) => ({ x: x.low_z, y: x.completeness * 100, r: Math.max(3, Math.sqrt(x.n) / 6) })), backgroundColor: I.sessions.map((x) => (x.integrity >= 0.95 ? '#62d2a288' : x.integrity >= 0.8 ? '#f0b34a88' : '#ff6f5e88')), pointRadius: I.sessions.map((x) => Math.max(3, Math.sqrt(x.n) / 12)) }] },
      options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'low-share z (below 1.2x)' }, suggestedMin: -4, suggestedMax: 4 }, y: { title: { display: true, text: 'completeness %' }, suggestedMax: 100 } } },
      plugins: [{ id: 'band', beforeDraw(c) { const { ctx, chartArea: a, scales: { x } } = c; ctx.save(); ctx.fillStyle = 'rgba(98,210,162,0.07)'; ctx.fillRect(x.getPixelForValue(-2), a.top, x.getPixelForValue(2) - x.getPixelForValue(-2), a.bottom - a.top); ctx.restore(); } }],
    });
  });
}

// ═══════════ F-03 Fingerprint + F-10 Compare ═══════════
export function pageFingerprint(v, C) {
  withPL(v, C, async () => {
    const { esc, f, pct, kpi, card, chart, n0 } = C;
    const s = C.state.source;
    const FP = PL.fingerprint;
    const st = FP.streams[s];
    const M = FP.measurement;
    const BD = FP.burst_demo;
    const alarms = Object.values(st.stats).reduce((a, x) => a + x.alarms.length, 0);
    v.innerHTML = `
    <p class="lede">${fid('F-03')} ${fid('F-10')} A stream's fingerprint is a set of rates that a fair 97% game pins down exactly: the hit rate at 2x and at 10x, and the share below 1.2x. Every block of ${FP.block} rounds is turned into a z-score against that curve and fed to a two-sided CUSUM (k = ${FP.k}, h = ${FP.h}). An alarm means the stream has drifted. The comparator below tests whether the streams differ from each other.</p>
    <div class="grid g4">
      ${kpi('Blocks', n0(st.blocks), `${n0(st.blocks * FP.block)} rounds`)}
      ${kpi('CUSUM alarms', alarms, 'across three statistics', alarms ? 'acc' : 'ok')}
      ${kpi('Edge at 2x', pct(st.edge, 2), '1 − 2 × P(≥ 2x); fair 3.00%')}
      ${kpi('Detection power', `${M.detected}/${M.reps}`, `3% → 4% edge shift in ${n0(M.median_delay_rounds)} rounds (median); ${M.false_alarms_per_1000_blocks} false alarms / 1,000 blocks`)}
    </div>
    ${card('CUSUM by statistic', 'Upper and lower sums. They reset after each alarm.', `${seg('fp-st', Object.entries(st.stats).map(([k, x]) => [k, x.label]), 'rate2')}<div class="chart tall mt"><canvas id="c-fp"></canvas></div>`, 'mt')}
    ${BD ? card('Replay: the Jul 3 burst spliced back into Aviator', `The burst is put back at its real position (block ${BD.insert_block}). The CUSUM fires in block ${BD.alarms[0] ?? '–'} and keeps firing for ${BD.alarms.length} blocks. On the clean stream it never fires.`, '<div class="chart"><canvas id="c-fpb"></canvas></div>', 'mt') : ''}
    ${card('Cross-source comparator', 'Two-proportion tests between every pair of streams. Benjamini–Hochberg across all 50 comparisons. A significant difference in instant crashes points to a different settlement rule or a different game build rather than to a forecastable pattern.', `${seg('cmp-m', PL.compare.metrics.map((m) => [m.key, m.label]), 'instant')}<div class="tbl-wrap mt"><table id="cmp-tbl"></table></div>`, 'mt')}`;
    let c1 = null;
    const draw = (k) => {
      c1 = killChart(C, c1);
      const x = st.stats[k];
      c1 = chart('c-fp', { type: 'line', data: { labels: x.up.map((_, i) => i), datasets: [
        { label: 'S+ (above fair)', data: x.up, borderColor: '#f0b34a', pointRadius: 0, borderWidth: 1.5 },
        { label: 'S− (below fair)', data: x.dn, borderColor: '#8fb4ff', pointRadius: 0, borderWidth: 1.5 },
        { label: `alarm h = ${FP.h}`, data: x.up.map(() => FP.h), borderColor: '#ff6f5e', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
      ] }, options: { interaction: { mode: 'index', intersect: false }, scales: { x: { title: { display: true, text: `block (${FP.block} rounds)` }, ticks: { maxTicksLimit: 12 } }, y: { min: 0 } } } });
    };
    bindSeg('fp-st', draw);
    draw('rate2');
    if (BD) chart('c-fpb', { type: 'line', data: { labels: BD.up.map((_, i) => i), datasets: [
      { label: 'S+ hit rate at 2x', data: BD.up, borderColor: '#ff6f5e', pointRadius: 0, borderWidth: 1.5 },
      { label: 'S−', data: BD.dn, borderColor: '#8fb4ff', pointRadius: 0, borderWidth: 1.5 },
      { label: 'alarm', data: BD.up.map(() => FP.h), borderColor: '#5d616b', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
    ] }, options: { interaction: { mode: 'index', intersect: false }, scales: { x: { ticks: { maxTicksLimit: 12 } }, y: { type: 'logarithmic', min: 0.1 } } } });
    const drawCmp = (m) => {
      const L = (k) => C.label(k);
      const rows = PL.compare.rows.filter((r) => r.metric === m);
      const span = Math.max(...rows.map((r) => Math.max(Math.abs(r.lo), Math.abs(r.hi)))) * 1.1;
      document.getElementById('cmp-tbl').innerHTML = `<thead><tr><th>Stream A</th><th>Stream B</th><th class="num">A</th><th class="num">B</th><th class="num">A − B [95% CI]</th><th style="min-width:160px">CI</th><th class="num">z</th><th>BH</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${esc(L(r.a))}</td><td>${esc(L(r.b))}</td><td class="num">${pct(r.pa, 2)}</td><td class="num">${pct(r.pb, 2)}</td><td class="num">${(r.diff >= 0 ? '+' : '') + (r.diff * 100).toFixed(2)} pp <span class="faint">[${(r.lo * 100).toFixed(2)}, ${(r.hi * 100).toFixed(2)}]</span></td><td>${ciBar(r.lo, r.hi, r.diff, span)}</td><td class="num">${f(r.z, 2)}</td><td>${qPill(r.q)}</td></tr>`).join('')}</tbody>`;
    };
    bindSeg('cmp-m', drawCmp);
    drawCmp('instant');
  });
}

// ═══════════ F-11 Signal significance strip ═══════════
export function pageSignals(v, C) {
  withPL(v, C, async () => {
    const { esc, n0, f, pct, kpi, card } = C;
    const s = C.state.source;
    const G = PL.signals[s];
    const valid = G.strip.filter((x) => x.lift != null);
    const sig = valid.filter((x) => x.q < 0.05).length;
    const raw = valid.filter((x) => x.p < 0.05).length;
    v.innerHTML = `
    <p class="lede">${fid('F-11')} The 14 "pressure" and pattern signals Momento dashboards show, from ${bookLink('12-research-suite', null, 'chapter 12')}. Each is computed only from rounds before the one it predicts. Each tile shows the change in the next-round hit rate when the signal is on, its 95% interval, and a Benjamini–Hochberg q-value over everything on the screen.</p>
    <div class="grid g4">
      ${kpi('Signals with q < 0.05', `${sig}/${valid.length}`, 'after correcting for 28 tests', sig ? 'bad' : 'ok')}
      ${kpi('Raw p < 0.05', raw, `about ${f(valid.length * 0.05, 1)} expected by chance alone`)}
      ${kpi('Same test, shuffled tape', `${G.shuffled_sig} with q < 0.05`, `${G.shuffled_raw} with raw p < 0.05: the false-positive floor`)}
      ${kpi('Replicated on the second half', G.stability ? `${G.stability.replicated}/${G.stability.first_half_significant}` : '–', G.stability ? 'first-half hits that held up with the same sign' : 'stream too short to split')}
    </div>
    ${card('Signal strip', 'Lift = hit rate when the signal is on minus the base rate. A bar that crosses zero means no measurable effect.', `${seg('sg-m', [['2', 'Next round ≥ 2x'], ['10', 'Next round ≥ 10x']], '2')}<div class="tiles mt" id="sg-tiles"></div>`, 'mt')}
    <div class="callout mt"><b>How to read this.</b> Pressure ≥ 70 fires on a large share of rounds, and its lift interval straddles zero on every stream. The book records the same result for the live platform (+3.53%, not significant). Showing pressure as a percentile of the drought is honest. Implying that it moves the next round is not.</div>`;
    const draw = (m) => {
      const rows = G.strip.filter((x) => String(x.m) === m);
      const span = Math.max(0.02, ...rows.filter((r) => r.lo != null).map((r) => Math.max(Math.abs(r.lo), Math.abs(r.hi)))) * 1.05;
      document.getElementById('sg-tiles').innerHTML = rows.map((r) => `<div class="tile ${r.q != null && r.q < 0.05 ? 'hot' : ''}">
        <div class="row" style="justify-content:space-between;gap:6px"><span class="small">${esc(r.label)}</span>${r.lift == null ? '<span class="pill">too rare</span>' : qPill(r.q)}</div>
        <div class="mono mt" style="font-size:1.05rem">${r.lift == null ? '–' : `${r.lift >= 0 ? '+' : ''}${(r.lift * 100).toFixed(2)} pp`}</div>
        ${ciBar(r.lo, r.hi, r.lift ?? 0, span)}
        <div class="tiny faint">${n0(r.n)} rounds on · hit ${r.rate == null ? '–' : pct(r.rate, 1)} vs base ${pct(r.base, 1)}${r.lo != null ? ` · CI [${(r.lo * 100).toFixed(1)}, ${(r.hi * 100).toFixed(1)}]` : ''}</div></div>`).join('');
    };
    bindSeg('sg-m', draw);
    draw('2');
  });
}

// ═══════════ F-26 ETA Board + F-27 hazard timeline ═══════════
export function pageEta(v, C) {
  withPL(v, C, async () => {
    const { esc, n0, f, pct, kpi, card, chart } = C;
    const s = C.state.source;
    const E = PL.eta[s];
    const K = PL.eta_checks;
    const anyMem = E.filter((e) => e.b1_q < 0.05).length;
    v.innerHTML = `
    <p class="lede">${fid('F-26')} ${fid('F-27')} For each threshold: how long since the last hit, how unusual that drought is (its Kaplan–Meier percentile, which is what "pressure" should mean), and the conditional median and p90 wait from S(g + k)/S(g). The memorylessness test asks the real question behind "overdue": does a longer wait raise the next-round chance? Spec: ${bookLink('10-survival-eta', null, 'chapter 10')}.</p>
    <div class="grid g4">
      ${kpi('Thresholds with memory', `${anyMem}/${E.length}`, 'β₁ ≠ 0 at q < 0.05', anyMem ? 'bad' : 'ok')}
      ${kpi('Null check', pct(K.null_cover, 1), `β₁ CI covers 0 on ${K.null_reps} fair tapes (target 95%)`)}
      ${kpi('Planted hazard', `${K.planted_detected}/${K.planted_reps}`, 'tapes with a rising hazard detected', 'ok')}
      ${kpi('Median-ETA calibration', E[2].median_calib == null ? '–' : pct(E[2].median_calib, 1), `10x: events before the stated median, second half (geometric expects ${pct(E[2].median_calib_geo, 1)})`)}
    </div>
    ${card('ETA Board', 'State at the end of the clean stream. "Next round" is the KM hazard at the current gap; the house column is 0.97/T.', `<div class="tbl-wrap"><table><thead><tr><th class="num">Threshold</th><th class="num">Gap</th><th class="num">KM percentile</th><th class="num">Next round</th><th class="num">House</th><th class="num">Median ETA</th><th class="num">p90 ETA</th><th class="num">Geometric med / p90</th><th class="num">β₁ [95% CI]</th><th>Memoryless?</th><th class="num">Median calib.</th></tr></thead><tbody>
      ${E.map((e) => `<tr><td class="num">${e.T}x</td><td class="num">${n0(e.gap)}</td><td class="num ${e.pctl >= 0.9 ? 'acc' : ''}">${pct(e.pctl, 0)}</td><td class="num">${pct(e.next_p, 2)}</td><td class="num faint">${pct(e.p_house, 2)}</td><td class="num">${e.eta_med ?? '–'}</td><td class="num">${e.eta_p90 ?? '–'}</td><td class="num faint">${e.geo_med} / ${e.geo_p90}</td><td class="num">${(e.b1 >= 0 ? '+' : '') + f(e.b1, 3)} <span class="faint">[${f(e.b1_lo, 3)}, ${f(e.b1_hi, 3)}]</span></td><td>${e.b1_q < 0.05 ? '<span class="pill bad">memory, q ' + f(e.b1_q, 3) + '</span>' : '<span class="pill ok">memoryless</span>'}</td><td class="num">${e.median_calib == null ? '–' : pct(e.median_calib, 1)} <span class="faint">/ ${pct(e.median_calib_geo, 1)}</span></td></tr>`).join('')}</tbody></table></div>
      <p class="small muted mt">What the UI should say, following the book: "${E[2].T}x gap is at the ${pct(E[2].pctl, 0)} percentile; next-round chance ${pct(E[2].next_p, 1)} (${E[2].b1_q < 0.05 ? 'the gap matters on this stream' : 'unchanged by the gap'})." Median calibration compares the share of hits that arrived before the stated median with what a geometric wait implies (the median of a discrete wait covers a little over 50%).</p>`, 'mt')}
    <div class="grid g2 mt">
      ${card('Hazard timeline', 'Empirical chance of a hit on the next round, by how long the wait has already lasted. Flat means memoryless.', `${seg('eta-t', E.map((e) => [e.T, `${e.T}x`]), 10)}<div class="chart mt"><canvas id="c-hz"></canvas></div>`)}
      ${card('Survival: Kaplan–Meier vs geometric', 'P(wait > k). The two curves lie on top of each other when the tape has no memory.', '<div class="chart mt" style="margin-top:44px"><canvas id="c-km"></canvas></div>')}
    </div>`;
    let c1 = null, c2 = null;
    const draw = (T) => {
      const e = E.find((x) => String(x.T) === String(T));
      c1 = killChart(C, c1); c2 = killChart(C, c2);
      const labs = e.hazard.map((h) => (h.k1 == null ? `${h.k0}+` : h.k0 === h.k1 ? `${h.k0}` : `${h.k0}–${h.k1}`));
      c1 = chart('c-hz', { type: 'line', data: { labels: labs, datasets: [
        { label: '95% CI', data: e.hazard.map((h) => h.hi * 100), borderColor: 'transparent', backgroundColor: 'rgba(240,179,74,0.15)', fill: '+1', pointRadius: 0 },
        { label: 'ci-lo', data: e.hazard.map((h) => h.lo * 100), borderColor: 'transparent', pointRadius: 0, fill: false },
        { label: 'hazard h(k)', data: e.hazard.map((h) => h.h * 100), borderColor: '#f0b34a', pointRadius: 2, borderWidth: 1.5 },
        { label: `overall rate ${pct(e.p, 2)}`, data: e.hazard.map(() => e.p * 100), borderColor: '#8fb4ff', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
      ] }, options: { plugins: { legend: { labels: { filter: (i) => i.text !== 'ci-lo' } } }, scales: { x: { title: { display: true, text: 'rounds waited so far (k)' } }, y: { title: { display: true, text: '%' }, min: 0 } } } });
      const ks = e.S.map((_, i) => i * e.S_step);
      c2 = chart('c-km', { type: 'line', data: { labels: ks, datasets: [
        { label: 'Kaplan–Meier', data: e.S, borderColor: '#f0b34a', pointRadius: 0, borderWidth: 1.8 },
        { label: 'geometric', data: ks.map((k) => Math.pow(1 - e.p, k)), borderColor: '#8fb4ff', borderDash: [4, 4], pointRadius: 0, borderWidth: 1.2 },
      ] }, options: { interaction: { mode: 'index', intersect: false }, scales: { x: { title: { display: true, text: 'k rounds' }, ticks: { maxTicksLimit: 10 } }, y: { min: 0, max: 1 } } } });
    };
    bindSeg('eta-t', draw);
    draw(10);
  });
}

// ═══════════ F-07 Sequence search ═══════════
const TOKENS = [
  ...BAND_NAMES.map((nm, i) => ({ k: `b${i}`, label: nm, lo: i, hi: i, tip: `${BANDS[i]}x–${BANDS[i + 1] ? BANDS[i + 1] + 'x' : '∞'}` })),
  { k: 'u2', label: '<2x', lo: 0, hi: 2, tip: 'any round under 2x' },
  { k: 'o2', label: '≥2x', lo: 3, hi: 9, tip: 'any round at or above 2x' },
  { k: 'o10', label: '≥10x', lo: 6, hi: 9, tip: 'any round at or above 10x' },
  { k: 'any', label: 'any', lo: 0, hi: 9, tip: 'wildcard' },
];
export function pageSequence(v, C) {
  withPL(v, C, async (seq) => {
    const { esc, n0, f, pct, kpi, card, chart, chip } = C;
    const s = C.state.source;
    const ser = await C.loadSeries(s);
    if (C.state.seq !== seq) return;
    const x = ser.x;
    const bands = Uint8Array.from(x, bandOf);
    let pat = ['u2', 'u2', 'u2'];
    let tgt = 2;
    v.innerHTML = `
    <p class="lede">${fid('F-07')} The tape as a string over the book's 10-band alphabet (${bookLink('05-linguistics', null, 'chapter 5')}). Build a pattern, and the search finds every place it occurred and what came next, compared with the base rate, the house curve and 20 shuffled copies of the same tape. Each search is logged in the session's ${'<a class="acc" href="#/experiments">Experiment Registry</a>'} and counted in its multiple-testing correction.</p>
    ${card('Pattern', 'Click tokens to append. The pattern matches the most recent rounds first (oldest on the left).', `
      <div class="chips" id="sq-pal">${TOKENS.map((t) => `<button class="btn ghost sq-tok" data-k="${t.k}" title="${esc(t.tip)}">${esc(t.label)}</button>`).join('')}</div>
      <div class="row mt"><div class="chips" id="sq-pat" style="flex:1;min-height:34px;align-items:center"></div><button class="btn ghost" id="sq-back">Remove last</button><button class="btn ghost" id="sq-clr">Clear</button></div>
      <div class="row mt">${seg('sq-t', [1.5, 2, 5, 10].map((m) => [m, `next ≥ ${m}x`]), 2)}<button class="btn" id="sq-run">Search</button><span class="small muted" id="sq-msg"></span></div>
      <div class="chips mt"><span class="tiny faint">Presets:</span>${[['three lows', ['u2', 'u2', 'u2']], ['ten sub-2x', Array(10).fill('u2')], ['after a 10x', ['o10']], ['dust, dust', ['b0', 'b0']], ['high then dust', ['b5', 'b0']], ['alternating', ['u2', 'o2', 'u2', 'o2']]].map(([n, p]) => `<button class="btn ghost sq-pre" data-p="${p.join(',')}">${n}</button>`).join('')}</div>`)}
    <div id="sq-out" class="mt"></div>`;
    const drawPat = () => { document.getElementById('sq-pat').innerHTML = pat.length ? pat.map((k) => `<span class="pill acc">${esc(TOKENS.find((t) => t.k === k).label)}</span>`).join('') : '<span class="small faint">empty pattern: add tokens</span>'; };
    drawPat();
    v.querySelectorAll('.sq-tok').forEach((b) => b.addEventListener('click', () => { if (pat.length < 12) pat.push(b.dataset.k); drawPat(); }));
    v.querySelectorAll('.sq-pre').forEach((b) => b.addEventListener('click', () => { pat = b.dataset.p.split(','); drawPat(); run(); }));
    document.getElementById('sq-back').addEventListener('click', () => { pat.pop(); drawPat(); });
    document.getElementById('sq-clr').addEventListener('click', () => { pat = []; drawPat(); });
    bindSeg('sq-t', (k) => { tgt = +k; });
    let ch = null;
    const matchAll = (bb) => {
      const L = pat.length, toks = pat.map((k) => TOKENS.find((t) => t.k === k));
      const ends = [];
      for (let i = L; i < bb.length; i++) {
        let ok = true;
        for (let j = 0; j < L; j++) { const b = bb[i - L + j]; if (b < toks[j].lo || b > toks[j].hi) { ok = false; break; } }
        if (ok) ends.push(i); // i = index of the round that follows the pattern
      }
      return ends;
    };
    const run = async () => {
      if (!pat.length) return;
      const ends = matchAll(bands);
      const n1 = ends.length;
      const k1 = ends.reduce((a, i) => a + (x[i] >= tgt), 0);
      const nAll = x.length, kAll = x.reduce((a, v2) => a + (v2 >= tgt), 0);
      const out = document.getElementById('sq-out');
      if (n1 < 20) { out.innerHTML = `<div class="callout">Only ${n1} matches: too rare to test. Shorten the pattern or use wider tokens.</div>`; return; }
      const tp = twoProp(k1, n1, kAll - k1, nAll - n1);
      const [lo, hi] = wilson(k1, n1);
      const r = rng(97 + pat.length * 13 + tgt);
      const sh = [];
      for (let b = 0; b < 20; b++) {
        const bb = Uint8Array.from(shuffle(Array.from(bands.keys()), r), (i) => bands[i]);
        const xs = Array.from(bb, (bi, i2) => bi); // band only; outcome from band vs target
        const e2 = matchAll(bb);
        const thr = bandOf(tgt);
        const kk = e2.reduce((a, i) => a + (xs[i] >= thr && (BANDS[thr] === tgt)), 0);
        sh.push(e2.length ? kk / e2.length : null);
      }
      const shV = sh.filter((z) => z != null);
      const base = kAll / nAll;
      const moreExtreme = shV.filter((z) => Math.abs(z - base) >= Math.abs(k1 / n1 - base)).length;
      const spec = `pattern: [${pat.map((k) => TOKENS.find((t) => t.k === k).label).join(', ')}]\nstream: ${s}\noutcome: next round >= ${tgt}x\ntest: two-proportion z vs all other rounds`;
      const hash = (await sha256(spec + PL.data_hash)).slice(0, 12);
      REG.push({ id: `S-${String(REG.length + 1).padStart(3, '0')}`, family: 'Sequence search (F-07)', hypothesis: `Pattern [${pat.map((k) => TOKENS.find((t) => t.k === k).label).join(' ')}] changes the next-round chance of ≥ ${tgt}x on ${s}.`, spec, spec_hash: hash, tests: 1, p: tp.p, best: `${n1} matches, ${(k1 / n1 * 100).toFixed(1)}% vs ${(base * 100).toFixed(1)}%`, power: null, page: 'sequence', session: true });
      const q = bh(REG.map((e) => e.p));
      const myQ = q[q.length - 1];
      const last = ends.slice(-12).reverse();
      out.innerHTML = `
      <div class="grid g4">
        ${kpi('Matches', n0(n1), `last seen ${n0(x.length - ends[ends.length - 1])} rounds ago`)}
        ${kpi(`Next ≥ ${tgt}x after match`, pct(k1 / n1, 2), `95% CI [${pct(lo, 1)}, ${pct(hi, 1)}]`, 'acc')}
        ${kpi('Base rate / house', `${pct(base, 2)} / ${pct(houseP(tgt), 2)}`, `lift ${(k1 / n1 - base >= 0 ? '+' : '')}${((k1 / n1 - base) * 100).toFixed(2)} pp, z ${f(tp.z, 2)}`)}
        ${kpi('Session-corrected q', f(myQ, 3), `BH over ${REG.length} session search${REG.length > 1 ? 'es' : ''}; ${moreExtreme}/20 shuffles as extreme`, myQ < 0.05 ? 'bad' : 'ok')}
      </div>
      <div class="grid g2 mt">
        ${card('Observed vs shuffled tapes', 'Hit rate after the pattern on the real tape (orange) and on 20 shuffled copies (blue). If orange sits inside the blue cloud, the pattern carries no information.', '<div class="chart short"><canvas id="c-sq"></canvas></div>')}
        ${card('Most recent matches', 'The pattern, then the round that followed.', `<div class="tbl-wrap"><table><tbody>${last.map((i) => `<tr><td class="mono tiny faint">#${n0(i)}</td><td><div class="chips">${Array.from({ length: pat.length }, (_, j) => chip(x[i - pat.length + j])).join('')}<span class="faint">→</span>${chip(x[i])}</div></td></tr>`).join('')}</tbody></table></div>`)}
      </div>
      <div class="callout ${myQ < 0.05 ? 'bad' : 'ok'} mt"><b>${myQ < 0.05 ? 'Survives correction on this tape.' : 'No evidence the pattern matters.'}</b> ${myQ < 0.05 ? 'Before trusting it, rerun it on another stream, or register it and test it on data collected afterwards. One significant search out of many is expected by chance.' : `After the pattern the next round reached ${tgt}x ${pct(k1 / n1, 1)} of the time, against ${pct(base, 1)} overall. The interval covers the base rate${lo <= base && hi >= base ? '' : ' narrowly'} once the number of searches is taken into account.`}</div>`;
      ch = killChart(C, ch);
      ch = chart('c-sq', { type: 'scatter', data: { datasets: [
        { label: 'shuffled', data: shV.map((z, i) => ({ x: z * 100, y: 1 + ((i % 5) - 2) * 0.06 })), backgroundColor: '#8fb4ff99', pointRadius: 4 },
        { label: 'observed', data: [{ x: (k1 / n1) * 100, y: 1 }], backgroundColor: '#f0b34a', pointRadius: 7 },
        { label: 'base', data: [{ x: base * 100, y: 0.75 }, { x: base * 100, y: 1.25 }], type: 'line', borderColor: '#5d616b', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
      ] }, options: { scales: { y: { display: false, min: 0.7, max: 1.3 }, x: { title: { display: true, text: `% next round ≥ ${tgt}x` } } } } });
      document.getElementById('sq-msg').textContent = `logged as ${REG[REG.length - 1].id} · spec ${hash}`;
    };
    document.getElementById('sq-run').addEventListener('click', run);
    run();
  });
}

// ═══════════ F-22 / F-24 / F-21 / F-18 Predictor Arena ═══════════
export function pageArena(v, C) {
  withPL(v, C, async (seq) => {
    const P = await loadPredict();
    const { esc, n0, f, pct, kpi, card, chart, chip, sgn } = C;
    const s = C.state.source;
    const ser = await C.loadSeries(s);
    if (C.state.seq !== seq) return;
    const x = ser.x, t = ser.t;
    const W = P.streams[s].weights;
    const names = P.features;
    const lp = (w, vec) => { let z = w.b; for (let i = 0; i < vec.length; i++) z += w.coef[i] * ((vec[i] - w.mean[i]) / w.scale[i]); return 1 / (1 + Math.exp(-z)); };
    const r = rng(Date.now() % 1e9);
    let tgt = 2;
    let i = 0; // index of the round to predict
    const pickStart = () => { i = 400 + Math.floor(r() * (x.length - 1200)); };
    pickStart();
    const players = { you: [], model: [], house: [], gambler: [] };
    const sc = (p, y) => (y ? Math.log(Math.max(1e-6, p)) : Math.log(Math.max(1e-6, 1 - p)));
    v.innerHTML = `
    <p class="lede">${fid('F-22')} ${fid('F-24')} ${fid('F-21')} ${fid('F-18')} Replay the real tape from a random point you cannot see past. Commit a probability that the next round reaches the target, then the round is revealed. You are scored with the log score against the house curve, next to the logistic model, the house curve itself and a "gambler's fallacy" bot. Every commit is chained with SHA-256, so the track record cannot be edited after the fact.</p>
    <div class="grid g4" id="ar-kpi"></div>
    <div class="grid g2 mt">
      ${card('Replay', 'Last 30 rounds, then the forecast cone for the next 5 from the house curve (p10–p90 and p25–p75). The cone never depends on history: that is the point.', `
        <div class="chart"><canvas id="c-ar"></canvas></div>
        <div class="chips mt" id="ar-tail"></div>`)}
      ${card('Your call', 'Drag to your probability, then commit. The model and the bots commit at the same time.', `
        <div class="row">${seg('ar-t', [1.5, 2, 3, 10].map((m) => [m, `≥ ${m}x`]), 2)}</div>
        <div class="controls mt"><label class="small muted" for="ar-p">P(next round ≥ <span id="ar-tl">2</span>x) = <b class="acc mono" id="ar-pv">50%</b> <span class="faint">(house <span id="ar-hp"></span>)</span></label><input type="range" id="ar-p" min="1" max="99" value="49"></div>
        <div class="row mt"><button class="btn" id="ar-go">Commit and reveal</button><button class="btn ghost" id="ar-auto">Auto-play 50 at house odds</button><button class="btn ghost" id="ar-jump">Jump to a new point</button></div>
        <div class="mt small" id="ar-last"></div>`)}
    </div>
    <div class="grid g2 mt">
      ${card('Leaderboard', 'Mean log-score skill per round against the house curve, in milli-nats. Zero is the house. Positive means better than the house.', '<div class="tbl-wrap"><table id="ar-lb"></table></div><div class="chart short mt"><canvas id="c-arl"></canvas></div>')}
      ${card('Tamper-evident track record', 'Each row stores a SHA-256 hash of its content plus the previous hash. Verify recomputes the chain. Tamper edits one stored probability to show the check fail.', `<div class="row"><button class="btn ghost" id="ar-ver">Verify chain</button><button class="btn ghost" id="ar-tam">Tamper with a row</button><span class="small" id="ar-vmsg"></span></div><div class="tbl-wrap mt" style="max-height:300px;overflow:auto"><table id="ar-led"></table></div>`)}
    </div>`;
    let c1 = null, c2 = null;
    const slider = document.getElementById('ar-p');
    const syncSlider = () => { document.getElementById('ar-pv').textContent = `${slider.value}%`; document.getElementById('ar-tl').textContent = tgt; document.getElementById('ar-hp').textContent = pct(houseP(tgt), 1); };
    slider.addEventListener('input', syncSlider);
    bindSeg('ar-t', (k) => { tgt = +k; slider.value = Math.round(houseP(tgt) * 100); syncSlider(); });
    slider.value = Math.round(houseP(2) * 100);
    syncSlider();
    const TGI = { 1.5: '1.5', 2: '2', 3: '3', 10: '10' };
    const drawReplay = () => {
      c1 = killChart(C, c1);
      const lo = i - 30;
      const hist = x.slice(lo, i);
      const q = (p) => Math.max(1, RTP / (1 - p));
      const labels = [...hist.map((_, k) => `${k - 30}`), '+1', '+2', '+3', '+4', '+5'];
      const pad = Array(30).fill(null);
      const cone = (p) => [...pad.slice(0, 29), hist[29], q(p), q(p), q(p), q(p), q(p)];
      c1 = chart('c-ar', { type: 'bar', data: { labels, datasets: [
        { type: 'bar', label: 'round', data: [...hist, null, null, null, null, null], backgroundColor: hist.map((v2) => (v2 >= 10 ? '#c49bff' : v2 >= 2 ? '#62d2a2' : '#ff6f5e')).concat([]), barPercentage: 0.8 },
        { type: 'line', label: 'p90', data: cone(0.9), borderColor: 'transparent', backgroundColor: 'rgba(240,179,74,0.12)', fill: '+3', pointRadius: 0 },
        { type: 'line', label: 'p75', data: cone(0.75), borderColor: 'transparent', backgroundColor: 'rgba(240,179,74,0.22)', fill: '+1', pointRadius: 0 },
        { type: 'line', label: 'median', data: cone(0.5), borderColor: '#f0b34a', borderDash: [3, 3], pointRadius: 0, borderWidth: 1 },
        { type: 'line', label: 'p25', data: cone(0.25), borderColor: 'transparent', pointRadius: 0, fill: false },
        { type: 'line', label: 'p10', data: cone(0.1), borderColor: 'transparent', pointRadius: 0, fill: false },
      ] }, options: { plugins: { legend: { display: false } }, scales: { y: { type: 'logarithmic', min: 1, max: Math.max(20, ...hist) * 1.2, ticks: { callback: (vv) => ([1, 2, 5, 10, 20, 50, 100, 1000].includes(vv) ? vv + 'x' : '') } }, x: { ticks: { maxTicksLimit: 12 } } } } });
      document.getElementById('ar-tail').innerHTML = x.slice(i - 12, i).map(chip).join('') + '<span class="pill">next: ?</span>';
    };
    const commit = async (pYou, silent) => {
      const m = tgt;
      const F = nextFeatures(x.slice(0, i), t.slice(0, i));
      const pm = lp(W[TGI[m]], names.map((k) => F[k]));
      const ph = houseP(m);
      const run = F['run_below2'];
      const pg = Math.min(0.95, ph * (1 + 0.08 * run)); // "it's due" bot
      const y = x[i] >= m ? 1 : 0;
      const rec = { n: LEDGER.length + 1, stream: s, round: i, target: m, p: +pYou.toFixed(4), outcome: +x[i].toFixed(2), committed: new Date().toISOString() };
      const prev = LEDGER.length ? LEDGER[LEDGER.length - 1].hash : '0'.repeat(64);
      const body = JSON.stringify(rec);
      rec.hash = await sha256(prev + body);
      rec.body = body;
      LEDGER.push(rec);
      players.you.push(1000 * (sc(pYou, y) - sc(ph, y)));
      players.model.push(1000 * (sc(pm, y) - sc(ph, y)));
      players.house.push(0);
      players.gambler.push(1000 * (sc(pg, y) - sc(ph, y)));
      if (!silent) document.getElementById('ar-last').innerHTML = `Round #${n0(i)} settled at ${chip(x[i])}: ${y ? '<span class="ok">hit</span>' : '<span class="bad">miss</span>'} at ${m}x. You said ${pct(pYou, 0)}, the model ${pct(pm, 1)}, the house ${pct(ph, 1)}. Your score this round: <b class="${players.you.at(-1) >= 0 ? 'ok' : 'bad'}">${sgn(players.you.at(-1), 0)} mnat</b>.`;
      i++;
      if (i >= x.length - 1) pickStart();
    };
    const drawBoard = () => {
      const stat = (a) => { const n = a.length; if (!n) return { m: 0, se: 0, n }; const m = a.reduce((p, c) => p + c, 0) / n; const sd = Math.sqrt(a.reduce((p, c) => p + (c - m) ** 2, 0) / Math.max(1, n - 1)); return { m, se: sd / Math.sqrt(n), n }; };
      const rows = [['you', 'You'], ['model', 'Logistic model'], ['house', 'House curve (0.97/m)'], ['gambler', '"It\'s due" bot']].map(([k, lab]) => ({ k, lab, ...stat(players[k]) })).sort((a, b) => b.m - a.m);
      const n = players.you.length;
      document.getElementById('ar-lb').innerHTML = `<thead><tr><th>#</th><th>Player</th><th class="num">Rounds</th><th class="num">Skill (mnat/round)</th><th class="num">± 2 SE</th></tr></thead><tbody>${rows.map((rw, k) => `<tr class="${rw.k === 'you' ? 'sel' : ''}"><td class="mono">${k + 1}</td><td>${esc(rw.lab)}</td><td class="num">${rw.n}</td><td class="num ${rw.m > 0 ? 'acc' : ''}">${sgn(rw.m, 1)}</td><td class="num faint">${rw.k === 'house' ? '–' : f(2 * rw.se, 1)}</td></tr>`).join('')}</tbody>`;
      document.getElementById('ar-kpi').innerHTML = [
        kpi('Rounds played', n0(n), `${esc(C.label(s))}, from round #${n0(i - n)}`),
        kpi('Your skill', n ? `${sgn(stat(players.you).m, 1)}` : '–', 'mnat per round vs house', n && stat(players.you).m > 0 ? 'acc' : ''),
        kpi('Model skill', n ? sgn(stat(players.model).m, 1) : '–', 'same rounds, logistic weights'),
        kpi('Ledger', `${LEDGER.length} entries`, LEDGER.length ? `head ${LEDGER.at(-1).hash.slice(0, 10)}…` : 'nothing committed yet'),
      ].join('');
      c2 = killChart(C, c2);
      const cum = (a) => { let c = 0; return a.map((v2) => (c += v2)); };
      c2 = chart('c-arl', { type: 'line', data: { labels: players.you.map((_, k) => k + 1), datasets: [
        { label: 'You', data: cum(players.you), borderColor: '#f0b34a', pointRadius: 0, borderWidth: 1.6 },
        { label: 'Model', data: cum(players.model), borderColor: '#c49bff', pointRadius: 0, borderWidth: 1.2 },
        { label: '"It\'s due" bot', data: cum(players.gambler), borderColor: '#ff6f5e', pointRadius: 0, borderWidth: 1.2 },
        { label: 'House', data: players.house, borderColor: '#5d616b', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
      ] }, options: { interaction: { mode: 'index', intersect: false }, scales: { y: { title: { display: true, text: 'cumulative mnat vs house' } }, x: { ticks: { maxTicksLimit: 8 } } } } });
      document.getElementById('ar-led').innerHTML = `<thead><tr><th class="num">#</th><th>Stream</th><th class="num">Round</th><th class="num">Target</th><th class="num">P</th><th class="num">Result</th><th>Hash</th></tr></thead><tbody>${LEDGER.slice(-60).reverse().map((e) => { const b = JSON.parse(e.body); return `<tr><td class="num">${b.n}</td><td class="small">${esc(b.stream)}</td><td class="num">${n0(b.round)}</td><td class="num">${b.target}x</td><td class="num">${pct(b.p, 0)}</td><td class="num">${f(b.outcome)}x</td><td class="mono tiny">${e.hash.slice(0, 16)}…</td></tr>`; }).join('')}</tbody>`;
    };
    document.getElementById('ar-go').addEventListener('click', async () => { await commit(+slider.value / 100); drawReplay(); drawBoard(); });
    document.getElementById('ar-auto').addEventListener('click', async (e) => { e.target.disabled = true; for (let k = 0; k < 50; k++) await commit(houseP(tgt), true); document.getElementById('ar-last').innerHTML = `Auto-played 50 rounds at the house probability. Your line tracks the house exactly.`; drawReplay(); drawBoard(); e.target.disabled = false; });
    document.getElementById('ar-jump').addEventListener('click', () => { pickStart(); drawReplay(); drawBoard(); document.getElementById('ar-last').textContent = `Jumped to round #${n0(i)}.`; });
    document.getElementById('ar-ver').addEventListener('click', async () => {
      let prev = '0'.repeat(64), bad = null;
      for (let k = 0; k < LEDGER.length; k++) { const h = await sha256(prev + LEDGER[k].body); if (h !== LEDGER[k].hash) { bad = k + 1; break; } prev = h; }
      document.getElementById('ar-vmsg').innerHTML = !LEDGER.length ? '<span class="faint">Nothing to verify yet.</span>' : bad ? `<span class="bad">Chain broken at entry ${bad}: its content no longer matches its hash.</span>` : `<span class="ok">All ${LEDGER.length} entries verify. Head ${LEDGER.at(-1).hash.slice(0, 12)}…</span>`;
    });
    document.getElementById('ar-tam').addEventListener('click', () => {
      if (!LEDGER.length) return;
      const k = Math.floor(r() * LEDGER.length);
      const b = JSON.parse(LEDGER[k].body);
      b.p = b.outcome >= b.target ? 0.99 : 0.01; // rewrite history to look prescient
      LEDGER[k].body = JSON.stringify(b);
      document.getElementById('ar-vmsg').innerHTML = `<span class="acc">Entry ${k + 1} rewritten to look prescient. Press Verify.</span>`;
      drawBoard();
    });
    drawReplay();
    drawBoard();
  });
}

// ═══════════ F-32 Bankroll simulator + F-30 Decision ledger ═══════════
const STRATS = {
  flat2: { label: 'Flat 1 unit at 2x', stake: () => ({ m: 2, s: 1 }) },
  flat10: { label: 'Flat 1 unit at 10x', stake: () => ({ m: 10, s: 1 }) },
  flat15: { label: 'Flat 1 unit at 1.5x', stake: () => ({ m: 1.5, s: 1 }) },
  mart2: { label: 'Martingale at 2x (double after loss, cap 7)', stake: (st) => ({ m: 2, s: Math.pow(2, Math.min(7, st.losses)) }) },
  after3: { label: '2x only after three sub-2x rounds', stake: (st, hist) => (hist.length >= 3 && hist.slice(-3).every((v) => v < 2) ? { m: 2, s: 1 } : null) },
  drought10: { label: '10x only when 10x is "overdue" (gap ≥ 20)', stake: (st) => (st.since10 >= 20 ? { m: 10, s: 1 } : null) },
};
export function pageSimulator(v, C) {
  withPL(v, C, async (seq) => {
    const { esc, n0, f, pct, kpi, card, chart, sgn } = C;
    const s = C.state.source;
    const ser = await C.loadSeries(s);
    if (C.state.seq !== seq) return;
    const x = ser.x;
    // F-30 ledger over the full history
    const ledger = Object.entries(STRATS).map(([k, S]) => {
      const st = { losses: 0, since10: 0 };
      let bets = 0, hits = 0, pnl = 0, staked = 0, peak = 0, dd = 0, pBase = 0;
      const hist = [];
      const per = [];
      for (let j = 0; j < x.length; j++) {
        const b = S.stake(st, hist);
        if (b) { bets++; staked += b.s; pBase += houseP(b.m); const w = x[j] >= b.m; const g = w ? b.s * (b.m - 1) : -b.s; hits += w; pnl += g; per.push(g / b.s); peak = Math.max(peak, pnl); dd = Math.max(dd, peak - pnl); st.losses = w ? 0 : st.losses + 1; }
        st.since10 = x[j] >= 10 ? 0 : st.since10 + 1;
        hist.push(x[j]); if (hist.length > 20) hist.shift();
      }
      const m = per.reduce((a, c) => a + c, 0) / Math.max(1, per.length);
      const sd = Math.sqrt(per.reduce((a, c) => a + (c - m) ** 2, 0) / Math.max(1, per.length - 1));
      return { k, label: S.label, bets, hit: hits / Math.max(1, bets), base: pBase / Math.max(1, bets), roi: pnl / Math.max(1, staked), pnl, dd, per100: m * 100, lo: (m - 1.96 * sd / Math.sqrt(Math.max(1, per.length))) * 100, hi: (m + 1.96 * sd / Math.sqrt(Math.max(1, per.length))) * 100 };
    }).sort((a, b) => b.per100 - a.per100);
    v.innerHTML = `
    <p class="lede">${fid('F-30')} ${fid('F-32')} The decision ledger replays every strategy over the full clean ${esc(C.label(s))} stream and ranks them by profit per 100 units staked, with a 95% interval. The simulator then resamples real sessions in blocks to show the spread of outcomes for one bankroll: the equity fan, the chance of ruin and the typical drawdown. Spec: ${bookLink('11-decision-autopilot', null, 'chapter 11')}.</p>
    ${card('Decision ledger: producer leaderboard', 'Hit rate compared with what the house curve implies for the same bets. Profit per 100 staked is the realised return; the interval shows whether it could be luck.', `<div class="tbl-wrap"><table><thead><tr><th>#</th><th>Strategy</th><th class="num">Bets</th><th class="num">Hit rate</th><th class="num">House implies</th><th class="num">P&amp;L / 100 staked [95% CI]</th><th class="num">Max drawdown</th><th>Verdict</th></tr></thead><tbody>
      ${ledger.map((rw, k) => `<tr><td class="mono">${k + 1}</td><td>${esc(rw.label)}</td><td class="num">${n0(rw.bets)}</td><td class="num">${pct(rw.hit, 2)}</td><td class="num faint">${pct(rw.base, 2)}</td><td class="num ${rw.per100 >= 0 ? 'acc' : ''}">${sgn(rw.per100, 2)} <span class="faint">[${f(rw.lo, 1)}, ${f(rw.hi, 1)}]</span></td><td class="num">${n0(rw.dd)} u</td><td>${rw.lo > 0 ? '<span class="pill bad">profitable</span>' : rw.hi < 0 ? '<span class="pill">loses, significant</span>' : '<span class="pill acc">indistinguishable from −3%</span>'}</td></tr>`).join('')}</tbody></table></div>`)}
    ${card('Bankroll simulator', 'Block bootstrap: sessions are rebuilt from 250-round blocks of the real tape, so streaks and clustering are kept. Nothing is stored.', `
      <div class="grid g4">
        <label class="small muted">Strategy<select id="bs-s" style="width:100%">${Object.entries(STRATS).map(([k, S]) => `<option value="${k}">${esc(S.label)}</option>`).join('')}</select></label>
        <label class="small muted">Bankroll (units)<input id="bs-b" type="number" value="100" min="5" max="100000" style="width:100%"></label>
        <label class="small muted">Rounds per session<input id="bs-n" type="number" value="1000" min="50" max="20000" style="width:100%"></label>
        <label class="small muted">Simulated sessions<input id="bs-k" type="number" value="400" min="50" max="3000" style="width:100%"></label>
      </div>
      <div class="row mt"><button class="btn" id="bs-run">Simulate</button><span class="small muted" id="bs-msg"></span></div>
      <div class="grid g4 mt" id="bs-kpi"></div>
      <div class="chart tall mt"><canvas id="c-bs"></canvas></div>`, 'mt')}`;
    let ch = null;
    const run = () => {
      const sk = document.getElementById('bs-s').value, S = STRATS[sk];
      const B0 = Math.max(5, +document.getElementById('bs-b').value || 100);
      const N = Math.min(20000, Math.max(50, +document.getElementById('bs-n').value || 1000));
      const K = Math.min(3000, Math.max(50, +document.getElementById('bs-k').value || 400));
      const r = rng(1234 + N + K + B0);
      const BL = 250, nb = Math.floor(x.length / BL);
      const pts = 60, step = Math.max(1, Math.floor(N / pts));
      const paths = [];
      let ruin = 0, prof = 0;
      const finals = [], dds = [];
      for (let k = 0; k < K; k++) {
        let bank = B0, peak = B0, dd = 0, ruined = false;
        const st = { losses: 0, since10: 0 };
        const hist = [];
        const path = [];
        let j = 0, pos = Math.floor(r() * nb) * BL;
        for (let n = 0; n < N; n++) {
          if (j >= BL) { pos = Math.floor(r() * nb) * BL; j = 0; }
          const xv = x[pos + j++];
          if (!ruined) {
            const b = S.stake(st, hist);
            if (b) {
              const stake = Math.min(b.s, bank);
              const w = xv >= b.m;
              bank += w ? stake * (b.m - 1) : -stake;
              st.losses = w ? 0 : st.losses + 1;
              if (bank < 1) { ruined = true; bank = Math.max(0, bank); }
            }
            peak = Math.max(peak, bank); dd = Math.max(dd, (peak - bank) / peak);
          }
          st.since10 = xv >= 10 ? 0 : st.since10 + 1;
          hist.push(xv); if (hist.length > 20) hist.shift();
          if (n % step === 0) path.push(bank);
        }
        path.push(bank);
        paths.push(path); finals.push(bank); dds.push(dd);
        ruin += ruined; prof += bank > B0;
      }
      const qt = (a, q) => { const b = a.slice().sort((u, w) => u - w); return b[Math.min(b.length - 1, Math.floor(q * b.length))]; };
      const L = paths[0].length;
      const band = (q) => Array.from({ length: L }, (_, i2) => qt(paths.map((p) => p[i2]), q));
      const labs = Array.from({ length: L }, (_, i2) => Math.min(N, i2 * step));
      document.getElementById('bs-kpi').innerHTML = [
        kpi('P(ruin)', pct(ruin / K, 1), `bankroll under 1 unit within ${n0(N)} rounds`, ruin / K > 0.2 ? 'bad' : ''),
        kpi('P(ahead at end)', pct(prof / K, 1), `of ${n0(K)} sessions`),
        kpi('Median final', `${n0(qt(finals, 0.5))} u`, `from ${n0(B0)} · p5 ${n0(qt(finals, 0.05))} · p95 ${n0(qt(finals, 0.95))}`),
        kpi('Median max drawdown', pct(qt(dds, 0.5), 0), 'peak-to-trough, per session'),
      ].join('');
      ch = killChart(C, ch);
      ch = chart('c-bs', { type: 'line', data: { labels: labs, datasets: [
        { label: 'p95', data: band(0.95), borderColor: 'transparent', backgroundColor: 'rgba(240,179,74,0.10)', fill: '+4', pointRadius: 0 },
        { label: 'p75', data: band(0.75), borderColor: 'transparent', backgroundColor: 'rgba(240,179,74,0.22)', fill: '+2', pointRadius: 0 },
        { label: 'median', data: band(0.5), borderColor: '#f0b34a', pointRadius: 0, borderWidth: 1.8 },
        { label: 'p25', data: band(0.25), borderColor: 'transparent', pointRadius: 0, fill: false },
        { label: 'p5', data: band(0.05), borderColor: 'transparent', pointRadius: 0, fill: false },
        { label: 'start', data: labs.map(() => B0), borderColor: '#5d616b', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
        ...paths.slice(0, 6).map((p, k) => ({ label: `path ${k + 1}`, data: p, borderColor: 'rgba(143,180,255,0.35)', pointRadius: 0, borderWidth: 0.8 })),
      ] }, options: { plugins: { legend: { labels: { filter: (it) => !/^path|p25|p5$/.test(it.text) } } }, interaction: { mode: 'index', intersect: false }, scales: { x: { title: { display: true, text: 'round' }, ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: 'bankroll (units)' }, min: 0 } } } });
      document.getElementById('bs-msg').textContent = `${n0(K)} sessions × ${n0(N)} rounds from ${n0(nb)} real blocks.`;
    };
    document.getElementById('bs-run').addEventListener('click', run);
    run();
  });
}

// ═══════════ F-34 Experiment Registry ═══════════
export function pageExperiments(v, C) {
  withPL(v, C, async () => {
    const { esc, n0, f, pct, kpi, card } = C;
    const base = PL.registry;
    v.innerHTML = `
    <p class="lede">${fid('F-34')} Every hypothesis the suite has tested, with its specification, the hash of that specification plus the data, the power check, and the verdict after multiple-testing correction. New runs below are causal (they use only earlier rounds), come with a shuffle baseline and a planted-effect power check, and join a session-wide Benjamini–Hochberg correction together with your sequence searches. Spec: ${bookLink('12-research-suite', null, 'chapter 12')}.</p>
    <div class="grid g4" id="ex-kpi"></div>
    ${card('New experiment', 'Condition on the previous rounds, then test whether the next round hits the target more or less often than the rest of the time.', `
      <div class="grid g4">
        <label class="small muted">Stream<select id="ex-s" style="width:100%">${STREAMS.map((s) => `<option value="${s}" ${s === C.state.source ? 'selected' : ''}>${esc(C.label(s))}</option>`).join('')}</select></label>
        <label class="small muted">Condition<select id="ex-c" style="width:100%"><option value="runlow">last k rounds all below a</option><option value="runhigh">last k rounds all at or above a</option><option value="drought">no round ≥ a in the last k</option><option value="after">previous round ≥ a (k ignored)</option></select></label>
        <label class="small muted">k<input id="ex-k" type="number" value="4" min="1" max="200" style="width:100%"></label>
        <label class="small muted">a (multiplier)<input id="ex-a" type="number" value="1.5" step="0.1" min="1" max="1000" style="width:100%"></label>
        <label class="small muted">Target: next round ≥<input id="ex-m" type="number" value="2" step="0.5" min="1.01" max="1000" style="width:100%"></label>
        <label class="small muted">Planted effect for power check (pp)<input id="ex-pl" type="number" value="5" step="1" min="1" max="30" style="width:100%"></label>
      </div>
      <div class="row mt"><button class="btn" id="ex-run">Register and run</button><span class="small muted" id="ex-msg"></span></div>
      <div id="ex-out" class="mt"></div>`, 'mt')}
    ${card('Registry', 'Pipeline entries were produced by predict.py and platform_features.py. Session entries were run in this browser and are gone when you close the tab.', '<div class="tbl-wrap"><table id="ex-tbl"></table></div>', 'mt')}`;
    const draw = () => {
      const q = REG.length ? bh(REG.map((e) => e.p)) : [];
      const sess = REG.map((e, k) => ({ ...e, q: q[k], verdict: q[k] < 0.05 ? 'supported' : 'rejected' }));
      const all = [...base, ...sess];
      document.getElementById('ex-kpi').innerHTML = [
        kpi('Experiments', all.length, `${base.length} from the pipeline, ${sess.length} this session`),
        kpi('Supported', all.filter((e) => e.verdict === 'supported').length, 'after correction', ''),
        kpi('Controls that fired as designed', base.filter((e) => e.family === 'Control').length, 'shuffle, timestamp leak, burst leak', 'ok'),
        kpi('Data hash', `<span style="font-size:1rem">${esc(PL.data_hash)}</span>`, 'every spec hash includes it'),
      ].join('');
      const vp = (vd) => `<span class="pill ${vd === 'supported' ? 'bad' : vd === 'rejected' ? 'ok' : vd === 'fired' || vd === 'passed' ? 'info' : ''}">${esc(vd)}</span>`;
      document.getElementById('ex-tbl').innerHTML = `<thead><tr><th>ID</th><th>Family</th><th>Hypothesis</th><th class="num">Tests</th><th>Best result</th><th>Power check</th><th>Verdict</th><th>Spec</th></tr></thead><tbody>${all.map((e) => `<tr class="${e.session ? 'sel' : ''}"><td class="mono">${esc(e.id)}</td><td class="small">${esc(e.family)}</td><td class="wrap small">${esc(e.hypothesis)}</td><td class="num">${e.tests}</td><td class="small mono">${esc(e.best)}${e.q != null ? ` · q ${f(e.q, 3)}` : ''}</td><td class="small muted wrap" style="min-width:150px">${esc(e.power || '–')}</td><td>${vp(e.verdict)}</td><td><details><summary class="mono tiny acc">${esc(e.spec_hash)}</summary><pre class="code">${esc(e.spec)}</pre>${e.page ? `<a class="small acc" href="#/${e.page}">open page</a>` : ''}</details></td></tr>`).join('')}</tbody>`;
    };
    const cond = (x, i, c, k, a) => {
      if (c === 'after') return x[i - 1] >= a;
      if (i < k) return false;
      for (let j = i - k; j < i; j++) {
        if (c === 'runlow' && !(x[j] < a)) return false;
        if (c === 'runhigh' && !(x[j] >= a)) return false;
        if (c === 'drought' && x[j] >= a) return false;
      }
      return true;
    };
    const test = (x, c, k, a, m) => {
      let n1 = 0, k1 = 0, n0_ = 0, k0 = 0;
      for (let i = 200; i < x.length; i++) { const y = x[i] >= m; if (cond(x, i, c, k, a)) { n1++; k1 += y; } else { n0_++; k0 += y; } }
      if (n1 < 20 || n0_ < 20) return null;
      return { n1, k1, n0: n0_, k0, ...twoProp(k1, n1, k0, n0_) };
    };
    document.getElementById('ex-run').addEventListener('click', async (e) => {
      const s = document.getElementById('ex-s').value, c = document.getElementById('ex-c').value;
      const k = Math.max(1, Math.min(200, +document.getElementById('ex-k').value || 1));
      const a = Math.max(1, +document.getElementById('ex-a').value || 2);
      const m = Math.max(1.01, +document.getElementById('ex-m').value || 2);
      const eff = Math.max(1, Math.min(30, +document.getElementById('ex-pl').value || 5)) / 100;
      const out = document.getElementById('ex-out');
      e.target.disabled = true;
      document.getElementById('ex-msg').textContent = 'running…';
      await new Promise((res) => setTimeout(res, 20));
      const ser = await C.loadSeries(s);
      const x = ser.x;
      const res = test(x, c, k, a, m);
      if (!res) { out.innerHTML = '<div class="callout">The condition is too rare or too common on this stream (fewer than 20 rounds on one side). Loosen or tighten it.</div>'; e.target.disabled = false; document.getElementById('ex-msg').textContent = ''; return; }
      const r = rng(4242 + k * 7 + Math.round(a * 100) + Math.round(m * 10));
      const shz = [];
      for (let b = 0; b < 20; b++) { const t2 = test(shuffle(x, r), c, k, a, m); if (t2) shz.push(t2.z); }
      // planted power: fair tape of the same length; the condition raises P(next >= m) by eff
      let det = 0;
      const reps = 10;
      for (let b = 0; b < reps; b++) {
        const xf = new Array(x.length);
        for (let i = 0; i < x.length; i++) {
          if (i >= 1 && cond(xf, i, c, k, a) && r() < eff / (1 - houseP(m) + 1e-9)) xf[i] = Math.floor(m / (1 - r() * 0.5) * 100) / 100;
          else xf[i] = fairRound(r());
        }
        const t2 = test(xf, c, k, a, m);
        if (t2 && t2.p < 0.05 && t2.z > 0) det++;
      }
      const cl = { runlow: `last ${k} all < ${a}x`, runhigh: `last ${k} all ≥ ${a}x`, drought: `no ≥ ${a}x in last ${k}`, after: `previous ≥ ${a}x` }[c];
      const spec = `stream: ${s}\ncondition: ${cl}\noutcome: next round >= ${m}x\nrows: i >= 200 (warm-up), causal\ntest: two-proportion z, condition vs rest\nshuffle_baseline: 20 permutations\npower_check: planted +${(eff * 100).toFixed(0)} pp on a fair tape, ${reps} reps`;
      const hash = (await sha256(spec + PL.data_hash)).slice(0, 12);
      REG.push({ id: `S-${String(REG.length + 1).padStart(3, '0')}`, family: 'Custom (F-34)', hypothesis: `When ${cl}, the next round reaches ${m}x at a different rate on ${s}.`, spec, spec_hash: hash, tests: 1, p: res.p, best: `${n0(res.n1)} rounds, ${(res.p1 * 100).toFixed(2)}% vs ${(res.p0 * 100).toFixed(2)}%, z ${res.z.toFixed(2)}`, power: `+${(eff * 100).toFixed(0)} pp planted: detected ${det}/${reps}`, page: null, session: true });
      const q = bh(REG.map((e2) => e2.p)).at(-1);
      const ext = shz.filter((z) => Math.abs(z) >= Math.abs(res.z)).length;
      const [lo, hi] = wilson(res.k1, res.n1);
      out.innerHTML = `<div class="grid g4">
        ${kpi('When the condition holds', pct(res.p1, 2), `${n0(res.n1)} rounds · 95% CI [${pct(lo, 1)}, ${pct(hi, 1)}]`, 'acc')}
        ${kpi('Otherwise', pct(res.p0, 2), `house ${pct(houseP(m), 2)} · z ${f(res.z, 2)}`)}
        ${kpi('Shuffle baseline', `${ext}/${shz.length}`, 'shuffled tapes with |z| at least as large')}
        ${kpi('Power check', `${det}/${reps}`, `a real +${(eff * 100).toFixed(0)} pp effect would be found this often`, det >= 8 ? 'ok' : 'acc')}
      </div>
      <div class="callout ${q < 0.05 ? 'bad' : 'ok'} mt"><b>${q < 0.05 ? 'Supported on this tape after correction' : 'Rejected'}</b>, q = ${f(q, 3)} over ${REG.length} session experiment${REG.length > 1 ? 's' : ''}. ${det < 8 ? 'The power check is weak, so a small real effect could still be hiding. Say "not detected" rather than "absent".' : q < 0.05 ? 'Confirm it on data collected after this registration before acting on it.' : 'The test had the power to see an effect of that size and did not.'} Spec hash <span class="mono">${hash}</span>.</div>`;
      document.getElementById('ex-msg').textContent = `registered as ${REG.at(-1).id}`;
      e.target.disabled = false;
      draw();
    });
    draw();
  });
}

// ═══════════ F-35 Explain-this-forecast (used by Next Round) ═══════════
export function explainHTML(W, names, vec, m, C, top = 12) {
  const { esc, f, pct } = C;
  const contrib = names.map((k, i) => ({ k, v: vec[i], z: (vec[i] - W.mean[i]) / W.scale[i], c: W.coef[i] * ((vec[i] - W.mean[i]) / W.scale[i]) }));
  const tot = W.b + contrib.reduce((a, c) => a + c.c, 0);
  const p = 1 / (1 + Math.exp(-tot));
  const pb = 1 / (1 + Math.exp(-W.b));
  const srt = contrib.slice().sort((a, b) => Math.abs(b.c) - Math.abs(a.c));
  const shown = srt.slice(0, top);
  const rest = srt.slice(top).reduce((a, c) => a + c.c, 0);
  const mx = Math.max(...shown.map((c) => Math.abs(c.c)), Math.abs(rest), 1e-6);
  const house = Math.min(1, RTP / m);
  const bar = (c) => `<div class="wf"><i style="${c >= 0 ? 'left:50%' : `left:${50 - (Math.abs(c) / mx) * 50}%`};width:${(Math.abs(c) / mx) * 50}%;background:${c >= 0 ? 'var(--accent)' : 'var(--info)'}"></i></div>`;
  return `<p class="small muted">Starting from the model's intercept (${pct(pb, 2)}), each row adds its contribution in log-odds: weight × standardised value. The ${top} largest are shown and the rest are summed. Final: <b class="acc">${pct(p, 2)}</b> against the house ${pct(house, 2)}, a difference of ${((p - house) * 100).toFixed(2)} pp.</p>
    <div class="tbl-wrap"><table><thead><tr><th>Feature</th><th class="num">Value</th><th class="num">z</th><th class="num">Log-odds</th><th style="min-width:180px">Push</th></tr></thead><tbody>
    ${shown.map((c) => `<tr><td class="mono small">${esc(c.k)}</td><td class="num">${f(c.v, 3)}</td><td class="num faint">${f(c.z, 2)}</td><td class="num ${c.c >= 0 ? 'acc' : 'info'}">${c.c >= 0 ? '+' : ''}${c.c.toFixed(4)}</td><td>${bar(c.c)}</td></tr>`).join('')}
    <tr><td class="small muted">${names.length - top} other features</td><td></td><td></td><td class="num">${rest >= 0 ? '+' : ''}${rest.toFixed(4)}</td><td>${bar(rest)}</td></tr></tbody></table></div>
    <p class="small muted mt">What would change it: the biggest single push is ${esc(srt[0].k)} at ${srt[0].c >= 0 ? '+' : ''}${srt[0].c.toFixed(3)} log-odds, about ${(Math.abs(srt[0].c) * p * (1 - p) * 100).toFixed(2)} pp. Walk-forward testing found these weights do not beat the house curve out of sample, so this explains what the model believes, not what will happen.</p>`;
}
