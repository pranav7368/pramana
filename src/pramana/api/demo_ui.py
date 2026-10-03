"""Self-contained evidence desk: ask or audit, then trace every claim to its source."""

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="description" content="PRAMANA checks every claim in an AI answer against the source document, in English, Hindi and Tamil, and withholds what the source cannot support.">
<meta property="og:title" content="PRAMANA — Evidence desk">
<meta property="og:description" content="Answers you can check, line by line: multilingual hallucination detection and correction for RAG.">
<meta property="og:type" content="website">
<meta name="theme-color" content="#f5f2ea" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#141310" media="(prefers-color-scheme: dark)">
<title>PRAMANA — Evidence desk</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%231c1a15'/%3E%3Cpath d='M9 10h10M9 16h14M9 22h8' stroke='%23f4efe3' stroke-width='2.4' stroke-linecap='round'/%3E%3Cpath d='M24 8v16' stroke='%23d9622b' stroke-width='2.4' stroke-linecap='round'/%3E%3C/svg%3E">
<style>
:root {
  --bg:#f5f2ea; --rail:#efebe1; --surface:#fffdf8; --sunk:#f9f6ef;
  --ink:#1c1a15; --ink-2:#3d3a32; --muted:#77705f; --faint:#a39b88;
  --line:#e2dccd; --line-2:#d3cbb8;
  --accent:#c2501f; --accent-ink:#fff;
  --s:#2e6b3f; --s-bg:#e4efe1; --s-hl:#cfe4c9;
  --c:#b3261e; --c-bg:#f8e3df; --c-hl:#f2cbc4;
  --u:#9a6200; --u-bg:#f6ebd2; --u-hl:#efdcae;
  --mark:#fde9a6;
  --ui:"Segoe UI Variable Text","Segoe UI",-apple-system,BlinkMacSystemFont,"Helvetica Neue","Nirmala UI","Noto Sans Devanagari","Noto Sans Tamil",Arial,sans-serif;
  --serif:"Iowan Old Style","Charter","Bitstream Charter","Sitka Text",Cambria,Georgia,"Nirmala UI","Noto Serif Devanagari","Noto Serif Tamil",serif;
  --mono:ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,monospace;
  --shadow:0 1px 0 rgba(28,26,21,.04), 0 8px 24px -16px rgba(28,26,21,.18);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#141310; --rail:#181713; --surface:#1d1b17; --sunk:#171612;
    --ink:#eee8da; --ink-2:#cfc8b8; --muted:#9b937f; --faint:#6f6858;
    --line:#2c2a23; --line-2:#3a372e;
    --accent:#e2733f; --accent-ink:#141310;
    --s:#86c290; --s-bg:#1f2b20; --s-hl:#2b4430;
    --c:#f08a7f; --c-bg:#341d1a; --c-hl:#552a24;
    --u:#e3b25a; --u-bg:#30271a; --u-hl:#4f3d1c;
    --mark:#5b4a16;
    --shadow:0 1px 0 rgba(0,0,0,.3), 0 10px 28px -18px rgba(0,0,0,.7);
  }
}
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--bg); color:var(--ink); font:14.5px/1.5 var(--ui);
  -webkit-font-smoothing:antialiased; }
button, input, textarea { font:inherit; color:inherit; }
button { cursor:pointer; background:none; border:0; padding:0; }
button:disabled { cursor:progress; opacity:.55; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:4px; }
.sr { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }
.hidden { display:none !important; }
.mono { font-family:var(--mono); font-size:.86em; letter-spacing:-.01em; }

/* ── Bar ─────────────────────────────────────────────── */
.bar { position:sticky; top:0; z-index:5; height:56px; display:flex; align-items:center;
  gap:24px; padding:0 20px; background:var(--bg); border-bottom:1px solid var(--line); }
.brand { display:flex; align-items:center; gap:10px; min-width:0; }
.brand svg { flex:none; }
.word { font:600 15px/1 var(--serif); letter-spacing:.2em; }
.devan { color:var(--muted); font-size:13px; padding-left:10px; border-left:1px solid var(--line-2); }
.seg { display:inline-flex; border:1px solid var(--line-2); border-radius:7px; padding:2px; background:var(--surface); }
.seg button { padding:5px 12px; border-radius:5px; font-size:13px; font-weight:560; color:var(--muted); }
.seg button[aria-pressed="true"], .seg button[aria-selected="true"] { background:var(--ink); color:var(--bg); }
.seg button:not([aria-pressed="true"]):not([aria-selected="true"]):hover { color:var(--ink); }
.modes { margin-left:8px; }
.barright { margin-left:auto; display:flex; align-items:center; gap:14px; }
.conn { display:flex; align-items:center; gap:7px; font-size:12.5px; color:var(--muted); white-space:nowrap; }
.conn i { width:7px; height:7px; border-radius:50%; background:var(--faint); }
.conn.live i { background:var(--s); box-shadow:0 0 0 3px var(--s-bg); }
.conn.offline i { background:var(--u); box-shadow:0 0 0 3px var(--u-bg); }
.conn.down i { background:var(--c); }
.links { display:flex; gap:14px; font-size:12.5px; }
.links a { color:var(--muted); text-decoration:none; }
.links a:hover { color:var(--ink); text-decoration:underline; text-underline-offset:3px; }
.pill { font-size:10.5px; font-weight:650; letter-spacing:.08em; text-transform:uppercase; padding:3px 8px;
  border-radius:999px; border:1px solid var(--line-2); color:var(--ink-2); white-space:nowrap; }
.quota { margin-top:12px; font-size:12.5px; color:var(--muted); }
.quota.low { color:var(--c); }

/* ── Layout ──────────────────────────────────────────── */
.app { display:grid; grid-template-columns:292px minmax(0,1fr) minmax(330px,400px); min-height:calc(100vh - 56px); }
.rail, .ledger { position:sticky; top:56px; height:calc(100vh - 56px); overflow:auto; }
.rail { background:var(--rail); border-right:1px solid var(--line); padding:22px 20px 28px; }
.ledger { background:var(--sunk); border-left:1px solid var(--line); padding:22px 20px 28px; }
.desk { padding:30px clamp(20px,4vw,56px) 56px; min-width:0; }
.desk-inner { max-width:760px; margin:0 auto; }
.kicker { font-size:11px; font-weight:650; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
.panehead { display:flex; align-items:baseline; justify-content:space-between; gap:10px; margin-bottom:14px; }
.panehead h2 { margin:0; font-size:11px; font-weight:650; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }

/* ── Source rail ─────────────────────────────────────── */
.doc { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px; box-shadow:var(--shadow); }
.doctop { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }
.tag { font-size:10.5px; font-weight:650; letter-spacing:.08em; text-transform:uppercase; padding:2px 7px;
  border-radius:4px; background:var(--line); color:var(--ink-2); }
.tag.own { background:var(--ink); color:var(--bg); }
.docname { font:600 15.5px/1.3 var(--serif); overflow-wrap:anywhere; }
.docmeta { margin-top:6px; font-size:12px; color:var(--muted); }
.drop { display:block; margin-top:14px; border:1.5px dashed var(--line-2); border-radius:10px; padding:16px 14px;
  text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
.drop:hover, .drop.over, .drop:focus-within { border-color:var(--accent); background:var(--surface); }
.drop b { display:block; font-size:13px; font-weight:600; }
.drop span { display:block; font-size:12px; color:var(--muted); margin-top:3px; }
.pending { margin-top:10px; display:flex; align-items:center; gap:8px; background:var(--surface);
  border:1px solid var(--line); border-radius:8px; padding:8px 8px 8px 11px; }
.pending .name { flex:1; min-width:0; font-size:12.5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.btn { display:inline-flex; align-items:center; justify-content:center; gap:8px; border-radius:7px;
  font-weight:600; font-size:13.5px; padding:9px 15px; white-space:nowrap; }
.btn-ink { background:var(--ink); color:var(--bg); }
.btn-ink:hover:not(:disabled) { background:var(--accent); color:var(--accent-ink); }
.btn-line { border:1px solid var(--line-2); background:var(--surface); }
.btn-line:hover:not(:disabled) { border-color:var(--ink); }
.btn-sm { padding:6px 11px; font-size:12.5px; }
.iconbtn { width:28px; height:28px; border-radius:6px; color:var(--muted); display:grid; place-items:center; }
.iconbtn:hover { background:var(--line); color:var(--ink); }
.textlink { margin-top:12px; font-size:12.5px; color:var(--muted); text-decoration:underline;
  text-decoration-color:var(--line-2); text-underline-offset:3px; }
.textlink:hover:not(:disabled) { color:var(--ink); text-decoration-color:var(--ink); }
.status { min-height:18px; margin-top:10px; font-size:12.5px; color:var(--muted); }
.status.error { color:var(--c); }
.passages { margin-top:20px; border-top:1px solid var(--line); padding-top:14px; }
.passages summary { cursor:pointer; font-size:12.5px; font-weight:600; color:var(--ink-2); list-style:none;
  display:flex; justify-content:space-between; }
.passages summary::-webkit-details-marker { display:none; }
.passages summary::after { content:"+"; color:var(--muted); font-weight:400; }
.passages[open] summary::after { content:"–"; }
.passages ol { list-style:none; margin:12px 0 0; padding:0; }
.passages li { font:12.5px/1.55 var(--serif); color:var(--ink-2); padding:9px 0; border-top:1px dashed var(--line); }
.passages li .mono { color:var(--faint); display:block; margin-bottom:2px; font-size:10.5px; }
.fine { margin-top:20px; font-size:11.5px; line-height:1.55; color:var(--muted); }

/* ── Composer ────────────────────────────────────────── */
.composer { background:var(--surface); border:1px solid var(--line-2); border-radius:12px; box-shadow:var(--shadow);
  transition:border-color .15s; }
.composer:focus-within { border-color:var(--ink); }
.composer textarea, .composer input { width:100%; border:0; outline:none; background:transparent; resize:none; }
.composer .q { display:block; padding:16px 18px 6px; font:19px/1.45 var(--serif); min-height:64px; }
.composer .q::placeholder, .composer textarea::placeholder { color:var(--faint); }
.composer .field { padding:12px 18px 0; }
.composer .field + .field { border-top:1px dashed var(--line); margin-top:10px; }
.composer label.k { display:block; font-size:11px; font-weight:650; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); margin-bottom:4px; }
.composer .field input { font:16px/1.45 var(--serif); padding:2px 0 4px; }
.composer .field textarea { font:16px/1.55 var(--serif); min-height:84px; padding:2px 0 4px; }
.composer .foot { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:10px 12px 12px 18px; }
.hint { font-size:12px; color:var(--faint); }
kbd { font:11px var(--mono); border:1px solid var(--line-2); border-bottom-width:2px; border-radius:4px; padding:0 4px; color:var(--muted); }
.try { margin-top:14px; display:flex; flex-wrap:wrap; gap:6px 14px; align-items:baseline; }
.try .kicker { margin-right:2px; }
.try button { font:14px/1.4 var(--serif); color:var(--ink-2); text-align:left; text-decoration:underline;
  text-decoration-color:var(--line-2); text-underline-offset:4px; }
.try button:hover:not(:disabled) { color:var(--accent); text-decoration-color:var(--accent); }

/* ── Result ──────────────────────────────────────────── */
#output { margin-top:34px; }
.empty { border-top:1px solid var(--line); padding-top:26px; }
.empty h3 { font:400 26px/1.25 var(--serif); margin:0 0 10px; letter-spacing:-.01em; max-width:30ch; }
.empty p { color:var(--muted); margin:0 0 22px; max-width:58ch; }
.legend { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; background:var(--line);
  border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.legend div { background:var(--surface); padding:14px; }
.legend b { display:flex; align-items:center; gap:7px; font-size:13px; }
.legend span { display:block; margin-top:5px; font-size:12.5px; color:var(--muted); line-height:1.45; }
.dotv { width:8px; height:8px; border-radius:2px; flex:none; }
.dotv.s { background:var(--s); } .dotv.c { background:var(--c); } .dotv.u { background:var(--u); }

.running { border-top:1px solid var(--line); padding-top:22px; }
.progress { height:2px; background:var(--line); overflow:hidden; border-radius:2px; margin:10px 0 18px; }
.progress::after { content:""; display:block; height:100%; width:30%; background:var(--accent); animation:slide 1.1s ease-in-out infinite; }
@keyframes slide { from { transform:translateX(-100%); } to { transform:translateX(340%); } }
.stages { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13px; color:var(--muted); list-style:none; padding:0; margin:0; }
.stages li::before { content:"·"; margin-right:8px; color:var(--faint); }
.stages li:first-child::before { content:none; }

.verdict-line, .passage { scroll-margin-top:72px; }
.verdict-line { display:flex; align-items:center; gap:10px; flex-wrap:wrap; border-top:1px solid var(--line); padding-top:18px; }
.state { display:inline-flex; align-items:center; gap:7px; font-size:12px; font-weight:650; letter-spacing:.08em;
  text-transform:uppercase; }
.state::before { content:""; width:8px; height:8px; border-radius:50%; background:currentColor; }
.state.ok { color:var(--s); } .state.warn { color:var(--c); } .state.held { color:var(--u); } .state.neutral { color:var(--ink-2); }
.runmeta { margin-left:auto; display:flex; gap:12px; color:var(--faint); font-size:12px; align-items:center; }
.answer { margin:14px 0 0; font:21px/1.6 var(--serif); letter-spacing:-.003em; white-space:pre-wrap; overflow-wrap:anywhere; }
.answer.withheld { color:var(--ink-2); font-style:italic; }
.hl { border-radius:3px; padding:0 1px; cursor:pointer; box-decoration-break:clone; -webkit-box-decoration-break:clone;
  background:linear-gradient(transparent 62%, var(--hl) 62%); transition:background .12s; }
.hl.s { --hl:var(--s-hl); } .hl.c { --hl:var(--c-hl); } .hl.u { --hl:var(--u-hl); }
.hl:hover, .hl.on { background:var(--hl); }
.instr { margin-top:8px; font-size:12.5px; color:var(--muted); }

.gauges { display:grid; grid-template-columns:1fr 1fr; gap:28px; margin-top:28px; padding:16px 0;
  border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
.gauge .top { display:flex; align-items:baseline; justify-content:space-between; gap:8px; }
.gauge .big { font:500 24px/1 var(--serif); }
.gauge .sub { font-size:12px; color:var(--muted); }
.strip { display:flex; height:6px; border-radius:3px; overflow:hidden; background:var(--line); margin-top:12px; gap:2px; }
.strip i { display:block; height:100%; }
.strip .s { background:var(--s); } .strip .c { background:var(--c); } .strip .u { background:var(--u); }
.counts { display:flex; gap:14px; margin-top:8px; font-size:12px; color:var(--muted); }
.counts span { display:inline-flex; align-items:center; gap:5px; }
.ruler { position:relative; height:6px; margin-top:12px; border-radius:3px; display:flex; overflow:visible; gap:2px; }
.ruler i { height:100%; display:block; background:var(--line); }
.ruler i:first-child { border-radius:3px 0 0 3px; } .ruler i:last-child { border-radius:0 3px 3px 0; }
.ruler i.on { background:var(--line-2); }
.ruler b { position:absolute; top:-5px; width:2px; height:16px; background:var(--accent); border-radius:1px; transform:translateX(-1px); }
.ticks { display:flex; justify-content:space-between; margin-top:8px; font-size:11px; color:var(--faint); }

.section { margin-top:30px; }
.section h4 { margin:0 0 10px; font-size:11px; font-weight:650; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
.claims { list-style:none; margin:0; padding:0; counter-reset:claim; }
.claims li + li { border-top:1px solid var(--line); }
.claim { width:100%; display:grid; grid-template-columns:22px 1fr auto; gap:12px; align-items:start; text-align:left;
  padding:12px 10px; border-radius:8px; }
.claim:hover { background:var(--surface); }
.claim[aria-pressed="true"] { background:var(--surface); box-shadow:inset 3px 0 0 var(--vc); }
.claim .n { font:12px/1.9 var(--mono); color:var(--faint); }
.claim .t { font:15.5px/1.5 var(--serif); overflow-wrap:anywhere; }
.claim .cite { display:block; margin-top:4px; font-size:12px; color:var(--muted); }
.vtag { font-size:10.5px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; padding:3px 7px; border-radius:4px;
  color:var(--vc); background:var(--vbg); white-space:nowrap; }
.vs { --vc:var(--s); --vbg:var(--s-bg); } .vc { --vc:var(--c); --vbg:var(--c-bg); } .vu { --vc:var(--u); --vbg:var(--u-bg); }
.none { font-size:13px; color:var(--muted); padding:8px 0; }

.trail { list-style:none; margin:0; padding:0; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.trail li { font:12px var(--mono); padding:4px 8px; border:1px solid var(--line-2); border-radius:5px; background:var(--surface); color:var(--ink-2); }
.trail li.arrow { border:0; background:none; padding:0; color:var(--faint); }
.trail li.final-ok { border-color:var(--s); color:var(--s); }
.trail li.final-held { border-color:var(--u); color:var(--u); }
.reason { margin:10px 0 0; font-size:13px; color:var(--muted); }
.note { margin-top:12px; font-size:12.5px; padding:9px 12px; border-left:2px solid var(--u); background:var(--u-bg); border-radius:0 6px 6px 0; }
.error-box { border-top:1px solid var(--line); padding-top:22px; }
.error-box h3 { margin:0 0 6px; font:400 22px/1.3 var(--serif); }
.error-box p { margin:0; color:var(--muted); }

/* ── Evidence ledger ─────────────────────────────────── */
.ledger .lead { font-size:12.5px; color:var(--muted); margin:0 0 16px; }
.passage { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:13px 14px 14px;
  margin-bottom:10px; transition:opacity .15s, border-color .15s; }
.passage.cited { border-color:var(--line-2); }
.passage.focus { border-color:var(--vc, var(--ink)); box-shadow:0 0 0 1px var(--vc, var(--ink)); }
.passage.dim { opacity:.45; }
.phead { display:flex; align-items:baseline; gap:8px; }
.phead .no { font:600 12px var(--mono); color:var(--ink); }
.phead .src { flex:1; min-width:0; font-size:12px; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.phead .role { font-size:10.5px; font-weight:650; letter-spacing:.06em; text-transform:uppercase; color:var(--faint); white-space:nowrap; }
.passage.cited .role { color:var(--ink-2); }
.ptext { margin-top:8px; font:13.5px/1.62 var(--serif); color:var(--ink-2); white-space:pre-wrap; overflow-wrap:anywhere;
  max-height:240px; overflow:auto; }
.ptext mark { background:var(--mark); color:inherit; border-radius:2px; padding:0 1px; }
.ptext .best { color:var(--ink); }
.ptext mark.x { background:var(--c-hl); color:var(--c); font-weight:650; }
.key { display:inline-flex; align-items:center; gap:5px; margin-right:10px; }
.key i { width:10px; height:10px; border-radius:2px; background:var(--mark); }
.key i.x { background:var(--c-hl); }
.ledger .fine { margin-top:16px; }
.ghost { border:1px dashed var(--line-2); border-radius:10px; padding:18px; color:var(--muted); font-size:13px; }
.ghost .lines { margin-top:12px; display:grid; gap:7px; }
.ghost .lines i { display:block; height:7px; border-radius:4px; background:var(--line); }
.ghost .lines i:nth-child(2) { width:86%; } .ghost .lines i:nth-child(3) { width:64%; }

.foot-note { grid-column:1 / -1; }

/* ── Responsive ──────────────────────────────────────── */
@media (max-width:1180px) {
  .app { grid-template-columns:268px minmax(0,1fr); }
  .ledger { grid-column:2; position:static; height:auto; border-left:0; border-top:1px solid var(--line); }
}
@media (max-width:780px) {
  .bar { gap:12px; padding:0 16px; }
  .devan, .conn span { display:none; }
  .modes { margin-left:0; }
  .app { display:flex; flex-direction:column; }
  .desk { order:1; } .ledger { order:2; } .rail { order:3; }
  .rail, .ledger { position:static; height:auto; border:0; border-top:1px solid var(--line); padding:20px 16px; }
  .desk { padding:22px 16px 40px; }
  .legend, .gauges { grid-template-columns:1fr; }
  .gauges { gap:20px; }
  .answer { font-size:19px; }
  .runmeta { margin-left:0; width:100%; }
}
@media (max-width:960px) { .links { display:none; } }
@media (max-width:520px) { .word, .keys, .pill { display:none; } .seg button { padding:5px 9px; } }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
</style>
</head>
<body>
<header class="bar">
  <div class="brand">
    <svg width="24" height="24" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="currentColor"/><path d="M9 10h10M9 16h14M9 22h8" stroke="var(--bg)" stroke-width="2.4" stroke-linecap="round"/><path d="M24 8v16" stroke="var(--accent)" stroke-width="2.4" stroke-linecap="round"/></svg>
    <span class="word">PRAMANA</span><span class="devan" lang="hi">प्रमाण</span>
  </div>
  <div class="seg modes" role="tablist" aria-label="Mode">
    <button type="button" role="tab" id="tab-ask" aria-selected="true" aria-controls="askPane" data-mode="ask">Ask</button>
    <button type="button" role="tab" id="tab-audit" aria-selected="false" aria-controls="auditPane" data-mode="audit">Audit</button>
  </div>
  <div class="barright">
    <span class="pill hidden" id="publicPill" title="Shared public demonstration">Public demo</span>
    <nav class="links" aria-label="Project"><a href="/docs" target="_blank" rel="noopener">API</a><a href="https://github.com/pranav7368/pramana" target="_blank" rel="noopener">Source</a></nav>
    <div class="seg langs" role="group" aria-label="Language">
      <button type="button" data-lang="en" aria-pressed="true" title="English">EN</button>
      <button type="button" data-lang="hi" aria-pressed="false" title="Hindi" lang="hi">हि</button>
      <button type="button" data-lang="ta" aria-pressed="false" title="Tamil" lang="ta">த</button>
    </div>
    <div class="conn" id="conn" role="status"><i></i><span id="connText">Connecting</span></div>
  </div>
</header>

<div class="app">
  <aside class="rail" aria-labelledby="sourceTitle">
    <div class="panehead"><h2 id="sourceTitle">Source</h2><span class="mono" id="langName" style="color:var(--faint)">English</span></div>
    <div class="doc">
      <div class="doctop"><span class="tag" id="docTag">Sample</span></div>
      <div class="docname" id="docName">Loading…</div>
      <div class="docmeta" id="docMeta"></div>
    </div>
    <label class="drop" id="drop" for="file">
      <input class="sr" id="file" type="file" accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown">
      <b>Add a document</b><span>Text PDF, .txt or .md in English, हिन्दी or தமிழ் · 5 MB · 25 pages</span>
    </label>
    <div class="pending hidden" id="pending">
      <span class="name" id="pendingName"></span>
      <button class="btn btn-ink btn-sm" id="upload" type="button">Index</button>
      <button class="iconbtn" id="cancelFile" type="button" aria-label="Remove selected file">✕</button>
    </div>
    <button class="textlink" id="reset" type="button">Restore the sample document</button>
    <div class="status" id="uploadStatus" role="status" aria-live="polite"></div>
    <details class="passages"><summary><span>Indexed passages <span class="mono" id="passageCount" style="color:var(--faint);margin-left:4px"></span></span></summary><ol id="sourcePreview"></ol></details>
    <p class="fine" id="privacy"></p>
  </aside>

  <main class="desk"><div class="desk-inner">
    <section id="askPane" role="tabpanel" aria-labelledby="tab-ask">
      <form id="askForm" class="composer">
        <label class="sr" for="question">Question</label>
        <textarea class="q" id="question" rows="2" maxlength="2000" placeholder="Ask in English, हिन्दी or தமிழ் — the answer comes back in your language…" required></textarea>
        <div class="foot"><span class="hint keys"><kbd>Ctrl</kbd> <kbd>Enter</kbd> to check</span>
          <button class="btn btn-ink" id="askButton" type="submit">Get verified answer</button></div>
      </form>
      <div class="try" id="examples"></div>
    </section>
    <section id="auditPane" class="hidden" role="tabpanel" aria-labelledby="tab-audit">
      <form id="auditForm" class="composer">
        <div class="field"><label class="k" for="auditQuestion">Question</label>
          <input id="auditQuestion" type="text" maxlength="2000" value="How long can I appeal a rejected claim?" required></div>
        <div class="field"><label class="k" for="draft">Answer to audit</label>
          <textarea id="draft" maxlength="8000" rows="3" required>Rejected claims may be appealed within 90 days of the rejection notice.</textarea></div>
        <div class="foot"><span class="hint">Paste an answer from any assistant. It is checked, never edited.</span>
          <span style="display:flex;gap:8px"><button class="btn btn-line" id="askInstead" type="button">Answer it instead</button>
          <button class="btn btn-ink" id="auditButton" type="submit">Audit answer</button></span></div>
      </form>
    </section>
    <p class="quota hidden" id="quota" role="status"></p>
    <section id="output" aria-live="polite"></section>
  </div></main>

  <aside class="ledger" aria-labelledby="evidenceTitle">
    <div class="panehead"><h2 id="evidenceTitle">Evidence</h2><span class="mono" id="evidenceCount" style="color:var(--faint)"></span></div>
    <div id="proof"></div>
  </aside>
</div>

<script>
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const LANGS = {en:'English', hi:'Hindi · हिन्दी', ta:'Tamil · தமிழ்'};
const EXAMPLES = {
  en:['How long do I have to appeal a rejected claim?', 'Is maternity covered in the first year?', 'What is the WiFi password in the Chennai office?'],
  hi:['रिजेक्ट क्लेम की अपील कितने दिन में कर सकते हैं?', 'पहले साल में मातृत्व लाभ मिलता है क्या?'],
  ta:['மேல்முறையீடு எத்தனை நாட்களுக்குள் செய்யலாம்?', 'முதல் ஆண்டில் மகப்பேறு நலன்கள் கிடைக்குமா?']
};
const VERDICT = {
  SUPPORTED:{label:'Supported', k:'s'}, CONTRADICTED:{label:'Contradicted', k:'c'}, UNVERIFIABLE:{label:'Unverifiable', k:'u'}
};
const ACTION = {ACCEPT:'Accepted', REGENERATE:'Rewrote against evidence', RE_RETRIEVE:'Searched again', PRUNE:'Removed unsupported', ABSTAIN:'Withheld'};
const REASON = {
  accept:'Every claim held up against the source, so the answer was released.',
  abstain:'The source could not support an answer, so none was given.',
  audit_only:'Audit reports a verdict for each claim and never changes the submitted text.',
  generator_abstained:'No answer was found in the retrieved passages.',
  correction_limit:'Corrections ran out before every claim was supported, so the answer was withheld.',
  latency_budget:'The time budget ran out before the answer was fully supported, so it was withheld.',
  rolled_back:'A revision made the answer worse and was rolled back.',
  correction_failed:'A revision could not be produced, so the checked answer was kept.',
  correction_abstained:'Revision concluded the source cannot answer this.',
  unresolved_evidence:'Some claims stayed unsupported, so the answer was withheld.'
};
const STOP = new Set(['the','and','for','are','was','were','with','that','this','from','have','has','been','will','may','can','its','any','all','per','into','than','then','their','there','which','what','when','your','you','does','did','how','who']);
const WORD = /[\p{L}\p{M}\p{N}]+/gu;
const NEG = new Set(['not','no','never','cannot','without','नहीं','बिना','न','இல்லை','கிடைக்காது','முடியாது','வேண்டாம்']);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

const state = {lang:'en', mode:'ask', app:null, corpus:{}, busy:false, result:null, kind:null, sel:null, file:null};

// Uploads belong to this browser tab only; the server keys them by this id.
const SESSION = (() => {
  const make = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('');
  try {
    let id = sessionStorage.getItem('pramana-session');
    if (!/^[A-Za-z0-9_-]{16,64}$/.test(id || '')) { id = make(); sessionStorage.setItem('pramana-session', id); }
    return id;
  } catch (_) { return make(); }
})();

async function api(path, options={}) {
  const response = await fetch(path, {...options, headers:{...(options.headers || {}), 'X-Pramana-Session': SESSION}});
  if (!response.ok) {
    let detail = response.status + ' ' + response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    const error = new Error(detail); error.status = response.status; throw error;
  }
  return response.json();
}
function setStatus(message, error=false) {
  $('uploadStatus').textContent = message;
  $('uploadStatus').classList.toggle('error', error);
}
function setBusy(value) {
  state.busy = value;
  ['upload','reset','askButton','auditButton','askInstead','cancelFile'].forEach(id => $(id).disabled = value);
  document.querySelectorAll('#examples button, .langs button').forEach(b => b.disabled = value);
}

/* Mode and language */
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('.modes button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.mode === mode)));
  $('askPane').classList.toggle('hidden', mode !== 'ask');
  $('auditPane').classList.toggle('hidden', mode !== 'audit');
  (mode === 'ask' ? $('question') : $('draft')).focus();
}
document.querySelectorAll('.modes button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
function setLang(lang) {
  if (state.busy || lang === state.lang) return;
  state.lang = lang;
  document.querySelectorAll('.langs button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.lang === lang)));
  document.documentElement.lang = lang;
  renderDocument(); renderEmpty();
}
document.querySelectorAll('.langs button').forEach(b => b.addEventListener('click', () => setLang(b.dataset.lang)));

/* Source */
// An uploaded document answers questions in every language; otherwise the
// selected language's sample policy is the source.
const activeDoc = () => state.app?.uploaded || state.app?.documents?.[state.lang];
const docLang = () => state.app?.uploaded?.language || state.lang;
function renderExamples() {
  const box = $('examples'); box.innerHTML = '';
  const doc = activeDoc();
  if (!doc?.sample) return;
  const label = document.createElement('span'); label.className = 'kicker'; label.textContent = 'Try';
  box.appendChild(label);
  EXAMPLES[state.lang].forEach(q => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = q;
    b.addEventListener('click', () => { $('question').value = q; setMode('ask'); $('askForm').requestSubmit(); });
    box.appendChild(b);
  });
}
function renderDocument() {
  $('langName').textContent = LANGS[docLang()];
  const doc = activeDoc();
  if (!doc) return;
  $('docName').textContent = doc.sample ? 'Fictional sample policy' : doc.name;
  $('docTag').textContent = doc.sample ? 'Sample' : 'Your document';
  $('docTag').classList.toggle('own', !doc.sample);
  $('docMeta').textContent = doc.sample
    ? `${plural(doc.chunks, 'passage')} · safe to experiment with`
    : `${LANGS[doc.language] || doc.language} detected · ${plural(doc.pages, 'page')} · ${plural(doc.chunks, 'passage')} · held in memory · ask in any language`;
  const chunks = state.corpus[docLang()] || [];
  $('passageCount').textContent = chunks.length;
  $('sourcePreview').innerHTML = chunks.slice(0, 60).map((c, i) =>
    `<li><span class="mono">${esc(c.source || 'Sample policy')}${c.page ? ' · p.' + Number(c.page) : ''} · ${i + 1}</span>${esc(c.text.slice(0, 420))}${c.text.length > 420 ? '…' : ''}</li>`
  ).join('') || '<li>No indexed text.</li>';
  renderExamples();
}
async function refresh() {
  [state.app, state.corpus] = await Promise.all([api('/v1/demo/documents'), api('/v1/corpus')]);
  const conn = $('conn');
  conn.classList.remove('live', 'offline', 'down');
  conn.classList.add(state.app.offline ? 'offline' : 'live');
  $('connText').textContent = state.app.offline ? 'Offline · fixtures' : `Live · ${state.app.provider}`;
  const minutes = Math.round((state.app.session_ttl_s || 1800) / 60);
  $('publicPill').classList.toggle('hidden', !state.app.public);
  $('privacy').textContent = state.app.offline
    ? `Offline mode answers sample questions from fixtures. Uploaded documents can be audited; answering them needs a model key${state.app.public ? '' : ' in .env'}.`
    : `Questions and retrieved passages are sent to ${state.app.provider}.${state.app.api_embedding_model ? ' Indexed text is also sent to Google for embeddings.' : ''} ` +
      (state.app.public
        ? `This is a public demo: your uploads are visible only in this browser tab, held in memory, and dropped after ${minutes} idle minutes or a restart. Do not upload confidential or personal documents.`
        : "Documents stay in this server's memory until it restarts. Use approved material only.");
  renderDocument(); renderQuota();
}
function renderQuota() {
  const box = $('quota'), limits = state.app?.limits;
  box.classList.toggle('hidden', !limits);
  if (!limits) return;
  const parts = [];
  if (limits.visitor_limit) parts.push(`${limits.visitor_remaining} of ${limits.visitor_limit} checks left for you this ${limits.window_s === 3600 ? 'hour' : 'period'}`);
  if (limits.daily_limit) parts.push(`${limits.daily_remaining} left today for everyone`);
  box.textContent = parts.join(' · ');
  box.classList.toggle('low', limits.visitor_remaining === 0 || limits.daily_remaining === 0);
}
async function refreshLimits() {
  try { state.app = await api('/v1/demo/documents'); renderQuota(); } catch (_) {}
}

function chooseFile(file) {
  if (!file) return;
  const ok = /\.(pdf|txt|md)$/i.test(file.name);
  if (!ok) return setStatus('Use a text PDF, .txt or .md file.', true);
  if (file.size > 5000000) return setStatus('That file is larger than 5 MB.', true);
  state.file = file;
  $('pendingName').textContent = `${file.name} · ${Math.max(1, Math.round(file.size / 1024))} KB`;
  $('pending').classList.remove('hidden');
  setStatus('');
}
$('file').addEventListener('change', () => chooseFile($('file').files[0]));
['dragenter','dragover'].forEach(t => $('drop').addEventListener(t, e => { e.preventDefault(); $('drop').classList.add('over'); }));
['dragleave','drop'].forEach(t => $('drop').addEventListener(t, e => { e.preventDefault(); $('drop').classList.remove('over'); }));
$('drop').addEventListener('drop', e => chooseFile(e.dataTransfer.files[0]));
$('cancelFile').addEventListener('click', () => { state.file = null; $('file').value = ''; $('pending').classList.add('hidden'); });
$('upload').addEventListener('click', async () => {
  if (state.busy || !state.file) return;
  setBusy(true); setStatus('Reading the document and building its index…');
  try {
    const file = state.file;
    const doc = await api(`/v1/demo/documents?filename=${encodeURIComponent(file.name)}`,
      {method:'POST', headers:{'Content-Type': file.type || 'application/octet-stream'}, body:file});
    await refresh(); renderEmpty();
    state.file = null; $('file').value = ''; $('pending').classList.add('hidden');
    setStatus(`Indexed · ${LANGS[doc.language] || doc.language} detected. Ask in English, हिन्दी or தமிழ்.`);
    $('question').focus();
  } catch (error) { setStatus(friendly(error), true); refreshLimits(); }
  finally { setBusy(false); }
});
$('reset').addEventListener('click', async () => {
  if (state.busy) return;
  setBusy(true); setStatus('Restoring the sample…');
  try {
    await api('/v1/demo/documents', {method:'DELETE'});
    await refresh(); renderEmpty(); setStatus('Sample document restored.');
  } catch (error) { setStatus(error.message, true); }
  finally { setBusy(false); }
});

/* Running */
function renderEmpty() {
  state.result = null; state.sel = null;
  $('output').innerHTML = `<div class="empty">
    <h3>Answers you can check, line by line.</h3>
    <p>Each answer is split into claims and every claim is tested against the source before you see it. Select a claim to see the passage behind it.</p>
    <div class="legend">
      <div><b><i class="dotv s"></i>Supported</b><span>The source states it.</span></div>
      <div><b><i class="dotv c"></i>Contradicted</b><span>The source says otherwise.</span></div>
      <div><b><i class="dotv u"></i>Unverifiable</b><span>The source is silent.</span></div>
    </div></div>`;
  $('evidenceCount').textContent = '';
  $('proof').innerHTML = `<div class="ghost">Passages retrieved for your question appear here, with the words each claim relies on marked.
    <div class="lines"><i></i><i></i><i></i></div></div>`;
}
function renderRunning(audit) {
  const stages = audit ? ['Retrieving passages','Checking each claim','Reporting verdicts']
                       : ['Retrieving passages','Drafting an answer','Checking each claim','Applying policy'];
  $('output').innerHTML = `<div class="running"><span class="kicker">${audit ? 'Auditing' : 'Working'}</span>
    <div class="progress" role="progressbar" aria-label="Checking"></div>
    <ol class="stages">${stages.map(s => `<li>${s}</li>`).join('')}</ol></div>`;
  $('evidenceCount').textContent = '';
  $('proof').innerHTML = '<div class="ghost">Waiting for retrieved passages…<div class="lines"><i></i><i></i><i></i></div></div>';
}
function friendly(error) {
  if (error.status === 503 && /busy/i.test(error.message)) return 'Another check is still running. Try again in a moment.';
  if (error.status === 422 || error.status === 429) return error.message;
  if (error.status === 502 || error.status === 504 || error instanceof TypeError)
    return state.app?.public
      ? 'The service could not be reached. A free-tier server sleeps when idle and takes about a minute to wake; try again shortly.'
      : 'The local service could not be reached. Check that it is still running.';
  return error.message;
}
function renderError(error) {
  $('output').innerHTML = `<div class="error-box"><span class="state warn">Not completed</span>
    <h3>That check didn't finish.</h3><p>${esc(friendly(error))}</p></div>`;
  $('evidenceCount').textContent = '';
  $('proof').innerHTML = '<div class="ghost">No passages were returned.</div>';
}
async function run(kind, path, body) {
  if (state.busy) return;
  setBusy(true); renderRunning(kind === 'audit');
  try { renderResult(await api(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}), kind); }
  catch (error) { renderError(error); }
  finally { setBusy(false); if (state.app?.limits) refreshLimits(); }
}
$('askForm').addEventListener('submit', e => {
  e.preventDefault();
  const query = $('question').value.trim();
  if (query) run('answer', '/v1/ask', {query});
});
$('auditForm').addEventListener('submit', e => {
  e.preventDefault();
  const query = $('auditQuestion').value.trim(), answer = $('draft').value.trim();
  if (query && answer) run('audit', '/v1/verify', {query, answer});
});
$('askInstead').addEventListener('click', () => { $('question').value = $('auditQuestion').value; setMode('ask'); $('askForm').requestSubmit(); });
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && e.target.closest?.('form')) {
    e.preventDefault(); e.target.closest('form').requestSubmit();
  } else if (e.key === '/' && !e.target.closest?.('input, textarea')) {
    e.preventDefault(); setMode('ask');
  } else if (e.key === 'Escape' && state.sel !== null) { selectClaim(null); }
});

/* Result */
function normalisedClaims(result) {
  return (result.claims || []).map((c, i) => ({
    i, text:c.text, evidence:c.evidence || [],
    verdict: VERDICT[c.verdict] ? c.verdict : 'UNVERIFIABLE'
  }));
}
function markAnswer(text, claims) {
  const lower = text.toLowerCase();
  if (lower.length !== text.length) return esc(text);
  const ranges = [];
  claims.forEach(c => {
    const needle = c.text.trim().replace(/[.।!?]+$/u, '').toLowerCase();
    if (needle.length < 6) return;
    const start = lower.indexOf(needle);
    if (start < 0) return;
    const end = start + needle.length;
    if (ranges.some(r => start < r.end && end > r.start)) return;
    ranges.push({start, end, c});
  });
  ranges.sort((a, b) => a.start - b.start);
  let out = '', at = 0;
  for (const r of ranges) {
    out += esc(text.slice(at, r.start));
    out += `<span class="hl ${VERDICT[r.c.verdict].k}" data-claim="${r.c.i}" role="button" tabindex="0">${esc(text.slice(r.start, r.end))}</span>`;
    at = r.end;
  }
  return out + esc(text.slice(at));
}
function headline(result, claims, audit) {
  const flagged = claims.filter(c => c.verdict !== 'SUPPORTED').length;
  if (audit) return flagged ? ['warn', `Audit · ${flagged} of ${claims.length} claim${claims.length === 1 ? '' : 's'} flagged`]
                            : claims.length ? ['ok', 'Audit · every claim supported'] : ['neutral', 'Audit · no checkable claims'];
  if (result.abstained) return ['held', 'Answer withheld'];
  if (!claims.length) return ['neutral', 'Answer · no checkable claims'];
  return flagged ? ['warn', `Answer · ${flagged} claim${flagged === 1 ? '' : 's'} flagged`] : ['ok', 'Verified answer'];
}
function renderResult(result, kind) {
  const audit = kind === 'audit';
  state.result = result; state.kind = kind; state.sel = null;
  const claims = normalisedClaims(result);
  state.claims = claims;
  const ids = [...new Set([...(result.retrieved_chunk_ids || []), ...(result.evidence_chunk_ids || [])])];
  state.passageIds = ids;
  const no = id => ids.indexOf(id) + 1;
  const count = k => claims.filter(c => c.verdict === k).length;
  const [s, c, u] = [count('SUPPORTED'), count('CONTRADICTED'), count('UNVERIFIABLE')];
  const [tone, title] = headline(result, claims, audit);
  const seconds = Object.values(result.latency_ms || {}).reduce((a, b) => a + Number(b || 0), 0) / 1000;
  const score = Math.max(0, Math.min(1, Number(result.confidence?.score || 0)));
  const band = result.confidence?.band || 'LOW';
  const calibrated = result.confidence?.calibrator && result.confidence.calibrator !== 'none' && result.confidence.calibrator !== 'n/a';
  const withheld = !audit && result.abstained;

  const stripTotal = Math.max(1, claims.length);
  const strip = claims.length
    ? [['s', s], ['c', c], ['u', u]].filter(([, n]) => n).map(([k, n]) => `<i class="${k}" style="flex:${n / stripTotal}"></i>`).join('')
    : '';
  const zones = [['LOW', .45], ['MEDIUM', .30], ['HIGH', .25]].map(([z, w]) => `<i style="flex:${w}" class="${z === band ? 'on' : ''}"></i>`).join('');

  const actions = audit ? ['Retrieved','Checked','Reported']
    : ['Retrieved','Drafted','Checked', ...(result.action_history || []).map(a => ACTION[a] || a)];
  const last = actions.length - 1;
  const trail = actions.map((a, i) => {
    const cls = i === last && !audit ? (withheld ? 'final-held' : (result.action_history || []).includes('ACCEPT') ? 'final-ok' : '') : '';
    return (i ? '<li class="arrow" aria-hidden="true">→</li>' : '') + `<li class="${cls}">${esc(a)}</li>`;
  }).join('');

  const claimItems = claims.length ? claims.map(cl => {
    const v = VERDICT[cl.verdict];
    const cites = cl.evidence.map(no).filter(Boolean);
    const cite = cites.length ? `Source ${cites.map(n => '§' + n).join(', ')}` : 'No passage supports this';
    return `<li><button type="button" class="claim v${v.k}" data-claim="${cl.i}" aria-pressed="false">
      <span class="n">${String(cl.i + 1).padStart(2, '0')}</span>
      <span><span class="t">${esc(cl.text)}</span><span class="cite">${cite}</span></span>
      <span class="vtag">${v.label}</span></button></li>`;
  }).join('') : `<li class="none">${withheld ? 'Nothing was released, so there are no claims to show.' : 'No factual claims could be extracted from this text.'}</li>`;

  $('output').innerHTML = `
    <div class="verdict-line"><span class="state ${tone}">${esc(title)}</span>
      <span class="runmeta"><span title="Answer language">${esc(String(result.detected_language).toUpperCase())}${result.evidence_language && result.evidence_language !== result.detected_language ? ' · from ' + esc(String(result.evidence_language).toUpperCase()) + ' document' : ''}</span><span>${seconds.toFixed(1)} s</span>
      <span class="mono" title="Trace id">${esc(result.trace_id)}</span>
      <button class="btn btn-line btn-sm" id="copyAnswer" type="button">Copy</button></span></div>
    <p class="answer ${withheld ? 'withheld' : ''}">${withheld ? esc(result.answer) : markAnswer(result.answer, claims)}</p>
    ${claims.length && !withheld ? '<div class="instr">Select a highlighted phrase or a claim below to trace it to its source.</div>' : ''}
    ${withheld ? '' : `<div class="gauges">
      <div class="gauge"><div class="top"><span class="kicker">Claims</span><span class="sub">${claims.length} checked</span></div>
        <div class="big">${claims.length ? `${s}<span class="sub"> / ${claims.length} supported</span>` : '—'}</div>
        <div class="strip" aria-hidden="true">${strip}</div>
        <div class="counts"><span><i class="dotv s"></i>${s}</span><span><i class="dotv c"></i>${c}</span><span><i class="dotv u"></i>${u}</span></div></div>
      <div class="gauge"><div class="top"><span class="kicker">Confidence</span><span class="sub">${calibrated ? 'calibrated' : 'heuristic score'}</span></div>
        <div class="big">${score.toFixed(2)}<span class="sub"> ${esc(band.toLowerCase())}</span></div>
        <div class="ruler" aria-hidden="true">${zones}<b style="left:${(score * 100).toFixed(1)}%"></b></div>
        <div class="ticks"><span>0</span><span>.45</span><span>.75</span><span>1</span></div></div>
    </div>
    <div class="section"><h4>Claims</h4><ol class="claims">${claimItems}</ol></div>`}
    <div class="section"><h4>Decision</h4><ol class="trail">${trail}</ol>
      <p class="reason">${esc(REASON[result.stop_reason] || (audit ? REASON.audit_only : ''))}</p>
      ${result.rolled_back ? '<div class="note">A revision scored worse than the checked answer and was discarded.</div>' : ''}
      ${!calibrated && !withheld ? '<p class="reason">The confidence score is a transparent heuristic, not a calibrated probability.</p>' : ''}
    </div>`;
  $('copyAnswer').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(result.answer); $('copyAnswer').textContent = 'Copied'; }
    catch (_) { $('copyAnswer').textContent = 'Copy failed'; }
    setTimeout(() => { const b = $('copyAnswer'); if (b) b.textContent = 'Copy'; }, 1600);
  });
  renderEvidence();
}

/* Claim ↔ evidence tracing */
function tokens(text) {
  return new Set((text.toLowerCase().match(WORD) || []).filter(t => /\p{N}/u.test(t) || (t.length >= 3 && !STOP.has(t))));
}
// Marks the words a claim shares with a passage, emphasises the sentence it
// most likely rests on, and -- for a contradiction -- marks the numbers and
// negations in that sentence that the claim does not carry: where they differ.
function markPassage(text, claim) {
  if (!claim) return esc(text);
  const wanted = tokens(claim.text);
  const claimWords = new Set(claim.text.toLowerCase().match(WORD) || []);
  const parts = text.split(/(?<=[.!?।॥])(\s+)/u);
  let best = -1, bestScore = 0;
  parts.forEach((sentence, i) => {
    if (i % 2) return;
    const score = (sentence.toLowerCase().match(WORD) || []).filter(t => wanted.has(t)).length;
    if (score > bestScore) { bestScore = score; best = i; }
  });
  const conflict = claim.verdict === 'CONTRADICTED';
  return parts.map((sentence, i) => {
    if (i % 2) return esc(sentence);
    const html = sentence.split(/([\p{L}\p{M}\p{N}]+)/u).map((part, j) => {
      if (!(j % 2)) return esc(part);
      const low = part.toLowerCase();
      if (conflict && i === best && !claimWords.has(low) && (/^\p{N}+$/u.test(low) || NEG.has(low)))
        return `<mark class="x">${esc(part)}</mark>`;
      return wanted.has(low) ? `<mark>${esc(part)}</mark>` : esc(part);
    }).join('');
    return i === best ? `<span class="best">${html}</span>` : html;
  }).join('');
}
function selectClaim(index) {
  state.sel = state.sel === index ? null : index;
  document.querySelectorAll('.claim').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.claim) === state.sel)));
  document.querySelectorAll('.hl').forEach(h => h.classList.toggle('on', Number(h.dataset.claim) === state.sel));
  renderEvidence();
  if (state.sel !== null) document.querySelector('.passage.focus')?.scrollIntoView({block:'nearest', behavior:'smooth'});
}
$('output').addEventListener('click', e => {
  const target = e.target.closest('[data-claim]');
  if (target) selectClaim(Number(target.dataset.claim));
});
$('output').addEventListener('keydown', e => {
  const target = e.target.closest('.hl');
  if (target && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); selectClaim(Number(target.dataset.claim)); }
});
function renderEvidence() {
  const result = state.result;
  if (!result) return;
  // Evidence can be in a different language from the answer, so look in every index.
  const byId = Object.fromEntries(Object.values(state.corpus).flat().map(c => [c.chunk_id, c]));
  const used = new Set(result.evidence_chunk_ids || []);
  const claim = state.sel === null ? null : state.claims[state.sel];
  const vk = claim ? VERDICT[claim.verdict].k : null;
  const items = state.passageIds.map((id, idx) => {
    const chunk = byId[id];
    if (!chunk) return '';
    const citedBy = state.claims.filter(c => c.evidence.includes(id)).map(c => c.i + 1);
    const focus = claim && claim.evidence.includes(id);
    const cls = ['passage', used.has(id) ? 'cited' : '', focus ? 'focus' : '', claim && !focus ? 'dim' : ''].join(' ');
    const role = citedBy.length ? `Cited by ${citedBy.map(n => String(n).padStart(2, '0')).join(', ')}` : 'Retrieved';
    const style = focus ? ` style="--vc:var(--${vk})"` : '';
    return `<article class="${cls}"${style}><div class="phead"><span class="no">§${idx + 1}</span>
      <span class="src">${esc(chunk.source || 'Sample policy')}${chunk.page ? ' · p.' + Number(chunk.page) : ''}</span>
      <span class="role">${role}</span></div>
      <div class="ptext">${focus ? markPassage(chunk.text, claim) : esc(chunk.text)}</div></article>`;
  }).join('');
  $('evidenceCount').textContent = state.passageIds.length ? plural(state.passageIds.length, 'passage') : '';
  const keys = claim && claim.evidence.length
    ? `<span class="key"><i></i>shared with the claim</span>${claim.verdict === 'CONTRADICTED' ? '<span class="key"><i class="x"></i>differs from the claim</span>' : ''}`
    : 'No passage supports this claim; retrieved text is shown dimmed.';
  const lead = claim
    ? `<p class="lead">Claim ${String(claim.i + 1).padStart(2, '0')} · ${keys} <button class="textlink" type="button" id="clearSel" style="margin:0">Show all</button></p>`
    : `<p class="lead">${state.claims.length ? 'Select a claim to trace it here.' : 'Passages retrieved for this question.'}</p>`;
  $('proof').innerHTML = items ? lead + items + '<p class="fine">Citations point to indexed text. Read the passage to judge whether it really supports the claim.</p>'
                               : '<div class="ghost">No relevant passage was retrieved for this question.</div>';
  $('clearSel')?.addEventListener('click', () => selectClaim(state.sel));
}

renderEmpty();
refresh().catch(error => {
  $('conn').classList.add('down'); $('connText').textContent = 'Service unavailable';
  renderError(error);
});
</script>
</body></html>"""


def _script_hash(html: str) -> str:
    import base64
    import hashlib
    import re

    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    return " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode() + "'"
        for body in scripts
    )


# The single inline script is pinned by hash, so injected markup cannot run
# script even if an escaping bug slipped through. Inline style attributes are
# allowed; they cannot execute code.
DEMO_CSP = (
    "default-src 'none'; "
    f"script-src {_script_hash(DEMO_HTML)}; "
    "style-src 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
