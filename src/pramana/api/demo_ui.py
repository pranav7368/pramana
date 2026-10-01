"""Self-contained faculty demo: document, question, evidence, and audit."""

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PRAMANA · Evidence-first RAG demo</title>
<style>
  :root { --navy:#12284e; --ink:#18263e; --muted:#60718b; --line:#dbe3ef;
    --paper:#fff; --bg:#f4f7fc; --gold:#cf9b31; --blue:#2256a6;
    --green:#16805b; --red:#b73c42; --amber:#a76a0b; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.5 "Segoe UI", "Noto Sans Devanagari", "Noto Sans Tamil", Arial, sans-serif; }
  button, input, textarea, select { font:inherit; }
  button { cursor:pointer; }
  button:disabled { cursor:wait; opacity:.6; }
  .top { background:var(--navy); color:#fff; border-bottom:4px solid var(--gold); }
  .topinner { max-width:1500px; margin:auto; padding:18px 28px; display:flex;
    align-items:center; justify-content:space-between; gap:20px; }
  .brand { display:flex; gap:15px; align-items:center; }
  .mark { width:44px; height:44px; display:grid; place-items:center; border:2px solid var(--gold);
    border-radius:12px; font:700 22px Georgia,serif; color:#ffe2a1; }
  h1 { margin:0; font-size:22px; letter-spacing:.05em; }
  .subtitle { margin-top:2px; font-size:12px; color:#cbd8ed; }
  .topstatus { display:flex; align-items:center; gap:9px; font-size:12px; color:#d9e8fd; }
  .dot { width:8px; height:8px; border-radius:50%; background:#55cf9d; }
  .shell { max-width:1500px; margin:auto; padding:24px 28px 34px; }
  .intro { display:flex; align-items:flex-start; justify-content:space-between; gap:18px;
    margin-bottom:18px; }
  .intro h2 { margin:0 0 5px; font-size:24px; line-height:1.18; color:var(--navy); }
  .intro p { margin:0; max-width:750px; color:var(--muted); }
  .mode { border:1px solid #d6e0ef; background:#eaf1fb; border-radius:99px;
    padding:7px 12px; font-size:12px; color:var(--navy); white-space:nowrap; }
  .steps { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:20px; }
  .step { border:1px solid var(--line); background:#fff; border-radius:8px;
    padding:8px 11px; font-size:12px; color:var(--muted); }
  .step b { color:var(--blue); margin-right:5px; }
  .grid { display:grid; grid-template-columns:minmax(245px,.78fr) minmax(340px,1.18fr)
    minmax(290px,1fr); gap:16px; align-items:start; }
  .panel { background:var(--paper); border:1px solid var(--line); border-radius:14px;
    box-shadow:0 5px 22px rgba(18,40,78,.05); min-width:0; overflow:hidden; }
  .panelhead { padding:17px 18px 14px; border-bottom:1px solid var(--line); }
  .eyebrow { font-size:11px; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
    color:var(--blue); }
  .panelhead h3 { font-size:17px; line-height:1.2; margin:4px 0 0; color:var(--navy); }
  .body { padding:17px 18px 20px; }
  .small { color:var(--muted); font-size:12px; line-height:1.45; }
  label { display:block; color:var(--navy); font-weight:650; font-size:13px; margin:15px 0 6px; }
  input[type=text], textarea, select { width:100%; color:var(--ink); background:#fff;
    border:1px solid #cbd6e6; border-radius:8px; padding:10px 11px; outline:none; }
  input:focus, textarea:focus, select:focus { border-color:var(--blue);
    box-shadow:0 0 0 3px rgba(34,86,166,.12); }
  textarea { min-height:92px; resize:vertical; }
  input[type=file] { width:100%; font-size:12px; color:var(--muted); }
  .upload { border:1.5px dashed #aac0de; border-radius:10px; background:#f8fbff;
    padding:15px; margin-top:13px; }
  .upload strong { display:block; color:var(--navy); margin-bottom:5px; }
  .row { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin-top:12px; }
  .primary { background:var(--blue); border:1px solid var(--blue); color:#fff;
    border-radius:8px; font-weight:700; padding:10px 15px; }
  .secondary { background:#fff; border:1px solid #bfcde0; color:var(--navy);
    border-radius:8px; font-weight:650; padding:9px 13px; }
  .secondary:hover, .chip:hover { border-color:var(--blue); color:var(--blue); }
  .doccard { background:#f5f8fd; border:1px solid var(--line); border-radius:10px;
    margin-top:17px; padding:12px; }
  .docname { font-weight:750; overflow-wrap:anywhere; color:var(--navy); }
  .docmeta { font-size:12px; color:var(--muted); margin-top:4px; }
  .preview { margin-top:10px; border:1px solid var(--line); border-radius:9px; }
  .preview summary { color:var(--blue); font-size:12px; font-weight:750;
    padding:10px 11px; cursor:pointer; }
  .previewbody { border-top:1px solid var(--line); padding:10px 11px;
    max-height:280px; overflow:auto; }
  .previewitem { font-size:11px; color:var(--muted); margin-bottom:10px; }
  .previewitem:last-child { margin-bottom:0; }
  .previewitem strong { color:var(--navy); display:block; margin-bottom:3px; }
  .previewitem span { white-space:pre-wrap; overflow-wrap:anywhere; }
  .status { margin-top:12px; font-size:12px; min-height:20px; color:var(--muted); }
  .status.error { color:var(--red); }
  .notice { background:#fff7e7; border-left:3px solid var(--gold); color:#654b1a;
    padding:9px 11px; border-radius:5px; font-size:12px; margin-top:17px; }
  .tabs { display:flex; gap:5px; border-bottom:1px solid var(--line); padding:0 18px; }
  .tab { background:none; border:0; border-bottom:3px solid transparent;
    color:var(--muted); padding:14px 12px 10px; font-weight:700; }
  .tab.active { color:var(--blue); border-bottom-color:var(--gold); }
  .hidden { display:none !important; }
  .chips { display:flex; gap:7px; flex-wrap:wrap; margin:12px 0 0; }
  .chip { border:1px solid #cad7e9; border-radius:99px; padding:6px 10px;
    background:#f8fbff; color:#36547e; font-size:11.5px; text-align:left; }
  .output { margin-top:18px; border-top:1px solid var(--line); padding-top:17px; min-height:155px; }
  .placeholder { color:var(--muted); padding:22px 3px; font-size:13px; }
  .answer { color:var(--navy); font-size:17px; line-height:1.6; white-space:pre-wrap;
    overflow-wrap:anywhere; margin-top:8px; }
  .pill { display:inline-block; border-radius:99px; font-size:11px; font-weight:800;
    padding:4px 9px; background:#e7f0ff; color:var(--blue); }
  .pill.bad { background:#ffe9e8; color:var(--red); }
  .pill.ok { background:#e5f6ee; color:var(--green); }
  .resultmeta { display:flex; gap:7px; flex-wrap:wrap; margin-top:14px; }
  .resultmeta span { background:#f3f6fb; border:1px solid var(--line);
    border-radius:6px; padding:5px 8px; font-size:11px; color:#405778; }
  .sectiontitle { font-size:12px; text-transform:uppercase; letter-spacing:.08em;
    color:var(--muted); font-weight:800; margin:18px 0 8px; }
  .claim { border:1px solid var(--line); border-radius:8px; padding:10px 11px; margin:8px 0; }
  .verdict { display:inline-block; border-radius:5px; font-size:10px; font-weight:800;
    letter-spacing:.04em; padding:3px 6px; margin-bottom:5px; }
  .SUPPORTED { color:var(--green); background:#e5f6ee; }
  .CONTRADICTED { color:var(--red); background:#ffe9e8; }
  .UNVERIFIABLE { color:var(--amber); background:#fff3d9; }
  .claimtext { font-size:13px; color:var(--ink); overflow-wrap:anywhere; }
  .claimcite { font-size:11px; color:var(--muted); margin-top:5px; overflow-wrap:anywhere; }
  .evidence { border:1px solid var(--line); border-radius:9px; padding:11px;
    margin-bottom:9px; background:#fbfcff; }
  .evidence.used { border-left:3px solid var(--green); }
  .evidencehead { font-size:12px; font-weight:750; color:var(--navy); overflow-wrap:anywhere; }
  .evidencetext { margin-top:6px; font-size:12px; line-height:1.6; white-space:pre-wrap;
    color:#35465e; max-height:170px; overflow:auto; }
  .trail { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-top:9px; }
  .trail span { background:#eef3fc; color:var(--navy); border-radius:6px;
    padding:5px 8px; font-size:11px; font-weight:700; }
  .foot { color:var(--muted); font-size:11px; margin-top:20px; }
  @media (max-width:1100px) { .grid { grid-template-columns:1fr 1.15fr; }
    .grid>.panel:last-child { grid-column:1 / -1; } }
  @media (max-width:700px) { .topinner,.shell { padding-left:16px; padding-right:16px; }
    .grid { grid-template-columns:1fr; } .grid>.panel:last-child { grid-column:auto; }
    .intro { display:block; } .mode { display:inline-block; margin-top:10px; } }
</style>
</head>
<body>
<div class="top"><div class="topinner">
  <div class="brand"><div class="mark">P</div><div><h1>PRAMANA</h1>
    <div class="subtitle">Multilingual RAG assurance · faculty demonstration</div></div></div>
  <div class="topstatus"><span class="dot"></span><span id="server">Connecting…</span></div>
</div></div>
<main class="shell">
  <div class="intro"><div><h2>Ask a document. Inspect every claim.</h2>
    <p>Upload a policy, ask a question, and see the source passage behind the answer. Audit a deliberately wrong draft to show what PRAMANA detects.</p></div>
    <div class="mode" id="mode">Checking mode…</div></div>
  <div class="steps"><div class="step"><b>01</b> Document</div><div class="step"><b>02</b> Retrieve</div>
    <div class="step"><b>03</b> Generate</div><div class="step"><b>04</b> Verify claims</div>
    <div class="step"><b>05</b> Correct or abstain</div></div>
  <div class="grid">
    <section class="panel"><div class="panelhead"><div class="eyebrow">Step 01 · Source</div><h3>Document library</h3></div>
      <div class="body">
        <div class="small">Each language has its own searchable index. Upload replaces the selected language's sample policy for this running demo.</div>
        <label for="lang">Document / question language</label>
        <select id="lang"><option value="en">English</option><option value="hi">हिन्दी · Hindi</option>
          <option value="ta">தமிழ் · Tamil</option></select>
        <div class="upload"><strong>Add your document</strong>
          <div class="small">Text PDF, UTF-8 .txt or .md · up to 5 MB · PDF up to 25 pages</div>
          <div style="margin-top:10px"><input id="file" type="file" accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown"></div>
          <div class="row"><button class="primary" id="upload">Use this document</button>
            <button class="secondary" id="reset" type="button">Restore sample</button></div>
        </div>
        <div class="status" id="uploadStatus" role="status"></div>
        <div class="doccard"><div class="eyebrow">Active source</div><div class="docname" id="docName">Loading…</div>
          <div class="docmeta" id="docMeta"></div></div>
        <details class="preview"><summary>Preview indexed text</summary><div class="previewbody" id="sourcePreview"></div></details>
        <div class="notice" id="privacy">Live models receive questions and retrieved passages. If API embeddings are enabled, indexed document text also goes to Google. Use fictional or approved material.</div>
      </div></section>
    <section class="panel"><div class="panelhead"><div class="eyebrow">Steps 02–05 · Decision</div><h3>Question and answer</h3></div>
      <div class="tabs"><button class="tab active" data-tab="ask" type="button">Ask a question</button>
        <button class="tab" data-tab="audit" type="button">Audit a draft</button></div>
      <div class="body">
        <form id="askForm"><label for="question">Question</label>
          <input id="question" type="text" maxlength="2000" placeholder="What does this policy say about appeals?" required>
          <div class="chips" id="examples"></div>
          <div class="row"><button class="primary" id="askButton" type="submit">Get verified answer</button></div>
        </form>
        <form id="auditForm" class="hidden"><div class="small">Enter an answer from another chatbot or a deliberately wrong example. Audit checks it; it does not edit this text.</div>
          <label for="auditQuestion">Original question</label>
          <input id="auditQuestion" type="text" maxlength="2000" value="How long can I appeal a rejected claim?" required>
          <label for="draft">Draft answer to inspect</label>
          <textarea id="draft" maxlength="8000" required>Rejected claims may be appealed within 90 days of the rejection notice.</textarea>
          <div class="row"><button class="primary" id="auditButton" type="submit">Audit this draft</button>
            <button class="secondary" id="askInstead" type="button">Ask PRAMANA this question</button></div>
        </form>
        <div class="output" id="output"><div class="placeholder">Start with a sample question or upload a text PDF. The answer and verification trail will appear here.</div></div>
      </div></section>
    <section class="panel"><div class="panelhead"><div class="eyebrow">Proof · Inspectable</div><h3>Evidence and claim verdicts</h3></div>
      <div class="body" id="proof"><div class="placeholder">After a question, retrieved passages and the verdict for each answer claim appear here.</div></div></section>
  </div>
  <div class="foot">This is a fictional faculty demo, not a measured accuracy result. A heuristic confidence score is not a calibrated probability. Audit mode preserves submitted drafts; answer mode runs the correction policy.</div>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const examples = {
  en:['How long do I have to appeal a rejected claim?', 'Is maternity covered in the first year?', 'What is the WiFi password in the Chennai office?'],
  hi:['रिजेक्ट क्लेम की अपील कितने दिन में कर सकते हैं?', 'पहले साल में मातृत्व लाभ मिलता है क्या?'],
  ta:['மேல்முறையீடு எத்தனை நாட்களுக்குள் செய்யலாம்?', 'முதல் ஆண்டில் மகப்பேறு நலன்கள் கிடைக்குமா?']
};
let appState = null;
let corpus = {};
let busy = false;
let activeTab = 'ask';

async function api(path, options={}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.status + ' ' + response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}
function setStatus(message, error=false) {
  $('uploadStatus').textContent = message;
  $('uploadStatus').classList.toggle('error', error);
}
function setBusy(value) {
  busy = value;
  ['upload','reset','askButton','auditButton','askInstead'].forEach(id => $(id).disabled = value);
}
function showTab(name) {
  activeTab = name;
  document.querySelectorAll('.tab').forEach(button => button.classList.toggle('active', button.dataset.tab === name));
  $('askForm').classList.toggle('hidden', name !== 'ask');
  $('auditForm').classList.toggle('hidden', name !== 'audit');
}
document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => showTab(button.dataset.tab)));
function renderExamples() {
  const lang = $('lang').value;
  $('examples').innerHTML = '';
  if (appState?.documents?.[lang]?.sample === undefined) return;
  examples[lang].forEach(question => {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'chip'; button.textContent = question;
    button.onclick = () => { $('question').value = question; showTab('ask'); $('askForm').requestSubmit(); };
    $('examples').appendChild(button);
  });
}
function renderDocument() {
  const lang = $('lang').value;
  const doc = appState?.documents?.[lang];
  if (!doc) return;
  $('docName').textContent = doc.name;
  $('docMeta').textContent = doc.sample
    ? `${doc.chunks} passages · fictional sample · use sample questions`
    : `${doc.pages} page(s) · ${doc.chunks} passages · ${doc.characters.toLocaleString()} characters · in memory`;
  $('sourcePreview').innerHTML = (corpus[lang] || []).slice(0, 3).map(chunk =>
    `<div class="previewitem"><strong>${esc(chunk.source || 'Sample policy')}${chunk.page ? ' · page ' + Number(chunk.page) : ''}</strong>
    <span>${esc(chunk.text.slice(0, 650))}${chunk.text.length > 650 ? '…' : ''}</span></div>`
  ).join('') || '<div class="small">No indexed text available.</div>';
  renderExamples();
}
async function refresh() {
  [appState, corpus] = await Promise.all([api('/v1/demo/documents'), api('/v1/corpus')]);
  $('server').textContent = 'Local service ready';
  $('mode').textContent = appState.offline ? 'Offline fixture mode' : `Live model · ${appState.provider}`;
  $('privacy').textContent = appState.offline
    ? 'Offline mode uses fixtures for sample questions. Uploaded documents can be audited, but live question answering needs a model key in .env.'
    : `Questions and retrieved passages are sent to ${appState.provider}. ${appState.api_embedding_model ? 'Indexed document text is also sent to Google for API embeddings. ' : ''}The local index is held in server memory. Upload only fictional or approved material.`;
  renderDocument();
}
function clearResult() {
  $('output').innerHTML = '<div class="placeholder">Document changed. Ask a new question to inspect its evidence.</div>';
  $('proof').innerHTML = '<div class="placeholder">No answer has been checked against this document yet.</div>';
}
$('lang').onchange = () => { renderDocument(); clearResult(); };
$('upload').onclick = async () => {
  if (busy) return;
  const file = $('file').files[0];
  if (!file) return setStatus('Choose a PDF, .txt, or .md file first.', true);
  if (file.size > 5000000) return setStatus('The file is larger than 5 MB.', true);
  setBusy(true); setStatus('Reading text and building the evidence index…');
  try {
    const lang = $('lang').value;
    await api(`/v1/demo/documents?filename=${encodeURIComponent(file.name)}&language=${lang}`,
      {method:'POST', headers:{'Content-Type': file.type || 'application/octet-stream'}, body:file});
    await refresh(); clearResult();
    setStatus('Document ready. Ask a question about this file.');
    $('file').value = '';
  } catch (error) { setStatus(error.message, true); }
  finally { setBusy(false); }
};
$('reset').onclick = async () => {
  if (busy) return;
  setBusy(true); setStatus('Restoring sample policies…');
  try { await api(`/v1/demo/documents?language=${$('lang').value}`, {method:'DELETE'}); await refresh(); clearResult();
    setStatus('Fictional sample policies restored.'); }
  catch (error) { setStatus(error.message, true); }
  finally { setBusy(false); }
};
function working(label, audit=false) {
  $('output').innerHTML = `<div class="placeholder">${esc(label)}<br>Retrieving → generating → verifying claims → applying policy…</div>`;
  if (audit) $('output').innerHTML = `<div class="placeholder">${esc(label)}<br>Retrieving → checking submitted claims → reporting verdicts…</div>`;
  $('proof').innerHTML = '<div class="placeholder">Evidence and verdicts will appear when the run finishes.</div>';
}
$('askForm').onsubmit = async event => {
  event.preventDefault(); if (busy) return;
  const query = $('question').value.trim(); if (!query) return;
  setBusy(true); working('Checking the document for an answer');
  try { const result = await api('/v1/ask', {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query,language:$('lang').value})}); renderResult(result, 'answer'); }
  catch (error) { renderError(error); }
  finally { setBusy(false); }
};
$('auditForm').onsubmit = async event => {
  event.preventDefault(); if (busy) return;
  const query = $('auditQuestion').value.trim(), answer = $('draft').value.trim();
  if (!query || !answer) return;
  setBusy(true); working('Auditing the submitted draft', true);
  try { const result = await api('/v1/verify', {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query,answer,language:$('lang').value})}); renderResult(result, 'audit'); }
  catch (error) { renderError(error); }
  finally { setBusy(false); }
};
$('askInstead').onclick = () => { $('question').value = $('auditQuestion').value; showTab('ask'); $('askForm').requestSubmit(); };
function renderError(error) {
  $('output').innerHTML = `<div class="pill bad">Could not complete</div><div class="small" style="margin-top:10px">${esc(error.message)}</div>`;
  $('proof').innerHTML = '<div class="placeholder">No result was returned.</div>';
}
function evidenceTitle(chunk) {
  if (!chunk) return 'Source passage';
  return esc(chunk.source || 'Sample policy') + (chunk.page ? ` · page ${Number(chunk.page)}` : '');
}
function renderResult(result, kind) {
  const audit = kind === 'audit';
  const score = Number(result.confidence.score || 0);
  const seconds = Object.values(result.latency_ms || {}).reduce((a,b) => a + Number(b || 0), 0) / 1000;
  const label = audit ? 'Submitted draft · audit only' : result.abstained ? 'No supported answer' : 'Verified answer';
  const flag = result.abstained ? '<span class="pill bad">ABSTAINED</span>' :
    audit ? '<span class="pill">AUDIT ONLY</span>' : '<span class="pill ok">ANSWER</span>';
  $('output').innerHTML = `<div class="eyebrow">${esc(label)}</div><div class="answer">${esc(result.answer)}</div>
    <div style="margin-top:12px">${flag}</div>
    <div class="resultmeta"><span>Language: ${esc(result.detected_language)}</span>
      <span>Confidence: ${score.toFixed(2)} · ${esc(result.confidence.band)} · ${esc(result.confidence.calibrator === 'none' ? 'heuristic' : result.confidence.calibrator)}</span>
      <span>Time: ${seconds.toFixed(1)} s</span><span>Trace: ${esc(result.trace_id)}</span></div>
    <div class="sectiontitle">Decision trail</div>
    <div class="small">${audit ? 'Audit checks the submitted answer and leaves it unchanged. Use “Ask PRAMANA this question” for a generated answer.' :
      'Actions show what the correction policy actually did. A missing answer may be deliberately withheld.'}</div>
    <div class="trail">${(audit ? ['RETRIEVE','VERIFY','REPORT'] : ['RETRIEVE','GENERATE','VERIFY',...result.action_history]).map(s => `<span>${esc(s)}</span>`).join('')}</div>
    ${result.rolled_back ? '<div class="notice">A worse correction candidate was rolled back.</div>' : ''}`;
  const langCorpus = corpus[result.detected_language] || [];
  const byId = Object.fromEntries(langCorpus.map(chunk => [chunk.chunk_id, chunk]));
  const claims = result.claims.length ? result.claims.map(claim => {
    const verdict = ['SUPPORTED','CONTRADICTED','UNVERIFIABLE'].includes(claim.verdict) ? claim.verdict : 'UNVERIFIABLE';
    const cited = claim.evidence.length ? claim.evidence.map(id => evidenceTitle(byId[id])).join(', ') : 'No supporting passage';
    return `<div class="claim"><div class="verdict ${verdict}">${verdict}</div>
      <div class="claimtext">${esc(claim.text)}</div><div class="claimcite">${cited}</div></div>`;
  }).join('') : '<div class="small">No factual claims returned. The system withheld the answer or the draft had no extractable claims.</div>';
  const used = new Set(result.evidence_chunk_ids || []);
  const ids = [...new Set([...(result.retrieved_chunk_ids || []), ...(result.evidence_chunk_ids || [])])];
  const evidence = ids.length ? ids.map(id => {
    const chunk = byId[id];
    if (!chunk) return '';
    return `<div class="evidence ${used.has(id) ? 'used' : ''}"><div class="evidencehead">${evidenceTitle(chunk)}${used.has(id) ? ' · cited' : ' · retrieved'}</div>
      <div class="evidencetext">${esc(chunk.text)}</div></div>`;
  }).join('') : '<div class="small">No relevant passage was retrieved.</div>';
  $('proof').innerHTML = `<div class="sectiontitle" style="margin-top:0">Claim verification · ${result.claims.length}</div>${claims}
    <div class="sectiontitle">Retrieved source passages · ${ids.length}</div>${evidence}
    <div class="small">A citation points to indexed text; inspect the passage to judge whether it really supports the claim.</div>`;
}
refresh().catch(error => { $('server').textContent = 'Service unavailable'; renderError(error); });
</script>
</body></html>"""
