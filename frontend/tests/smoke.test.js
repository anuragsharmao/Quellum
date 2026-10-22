/**
 * Frontend wiring smoke test (no browser needed): runs the REAL script.js against a stub DOM and an in-memory fake of the Quellum API
 * contract, and checks the flows end to end: Home -> Workspace (text + PDF), chat persistence calls, History, restore, expiry, XSS escaping.
 * It proves the wiring and the absence of runtime errors; it does NOT replace a manual browser pass (layout, PDF.js rendering).
 *   node frontend/tests/smoke.test.js
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const html = [];                       // every innerHTML assignment, to check escaping
const els = {};
const alerts = [];
function makeEl(id) {
  const listeners = {};
  const el = {
    id, style: {}, dataset: {}, children: [], value: '', textContent: '', className: '', scrollTop: 0, scrollHeight: 0, files: [],
    classes: new Set(),
    classList: { add: (c) => el.classes.add(c), remove: (c) => el.classes.delete(c), contains: (c) => el.classes.has(c), toggle() {} },
    set innerHTML(v) { el._html = String(v); html.push(el._html); }, get innerHTML() { return el._html || ''; },
    addEventListener: (t, f) => { (listeners[t] = listeners[t] || []).push(f); },
    fire: async (t, ev) => { for (const f of listeners[t] || []) await f(Object.assign({ preventDefault() {}, stopPropagation() {}, target: el, key: '' }, ev || {})); await new Promise(r => setTimeout(r, 40)); },
    appendChild(c) { el.children.push(c); c.parent = el; return c; },
    remove() { el.removed = true; if (el.parent) { const i = el.parent.children.indexOf(el); if (i >= 0) el.parent.children.splice(i, 1); } },
    querySelector: () => makeEl('q'), querySelectorAll: () => [], setAttribute() {}, getAttribute: () => '1', closest: () => null,
    focus() {}, click() {}, contains: () => false, scrollIntoView() {}, getContext: () => ({})
  };
  return el;
}
const $ = (id) => (els[id] = els[id] || makeEl(id));

class ApiError extends Error { constructor(status, code, message) { super(message); this.name = 'ApiError'; this.status = status; this.code = code; } }
const calls = [];
const db = { docs: {}, convs: {}, n: 0, mode: 'ok' };
const now = () => new Date().toISOString();
const api = {
  ApiError,
  session: async () => ({ status: 'ok' }),
  uploadPdf: async (f) => { calls.push('uploadPdf'); const id = 'd' + (++db.n); db.docs[id] = { id, sourceType: 'pdf', name: f.name }; return db.docs[id]; },
  createTextDocument: async (t, n) => { calls.push('createTextDocument'); const id = 'd' + (++db.n); db.docs[id] = { id, sourceType: 'text', name: n, text: t }; return db.docs[id]; },
  getDocumentFile: async () => new ArrayBuffer(8),
  createConversation: async ({ documentId, title }) => {
    calls.push('createConversation'); const d = db.docs[documentId]; const id = 'c' + (++db.n);
    return (db.convs[id] = { id, title, sourceType: d.sourceType, documentId, documentName: d.name, textContent: d.text, createdAt: now(), updatedAt: now(), messages: [] });
  },
  sendMessage: async (cid, content) => {
    calls.push('sendMessage:' + cid);
    if (db.mode === 'noGen') throw new ApiError(503, 'generation_not_configured', 'Answer generation is not configured on this server.');
    if (db.mode === 'expired') throw new ApiError(410, 'expired', 'expired');
    const c = db.convs[cid];
    const u = { id: 'u' + (++db.n), role: 'user', content, createdAt: now(), sources: [], hallucinationResult: null };
    const a = { id: 'a' + (++db.n), role: 'assistant', content: 'Answer <img src=x onerror=alert(1)> to ' + content, createdAt: now(),
      sources: [{ page: null, section: 'Pasted text', excerpt: 'evidence <script>x</script>', score: 0.9 }],
      hallucinationResult: { status: 'ok', flagged: false, riskLabel: 'Lower hallucination risk', risk: 0.2, totalSentences: 1, flaggedSentences: null, sentences: [] } };
    c.messages.push(u, a); c.updatedAt = now();
    return { status: 'ok', messages: [u, a], conversation: { id: cid, updatedAt: c.updatedAt, expiresAt: now() } };
  },
  listConversations: async () => { calls.push('list'); return Object.values(db.convs).filter(c => c.messages.length).map(c => JSON.parse(JSON.stringify(c))); },
  getConversation: async (id) => {
    calls.push('get:' + id);
    if (db.mode === 'expired') throw new ApiError(410, 'expired', 'expired');
    if (!db.convs[id]) throw new ApiError(404, 'not_found', 'nf');
    return JSON.parse(JSON.stringify(db.convs[id]));
  },
  renameConversation: async (id, title) => { db.convs[id].title = title; return JSON.parse(JSON.stringify(db.convs[id])); },
  deleteConversation: async (id) => { calls.push('delete:' + id); delete db.convs[id]; return { ok: true }; }
};

let domReady;
const win = { location: { hash: '' }, addEventListener() {}, scrollTo() {}, QuellumAPI: api, pdfjsLib: undefined, indexedDB: undefined };
const ctx = vm.createContext({
  window: win, console, setTimeout, clearTimeout, alert: (m) => alerts.push(m), File: require('buffer').File, URL, Promise, Date, Math, JSON, ArrayBuffer,
  localStorage: { removeItem() {}, getItem: () => null, setItem() {} }, sessionStorage: { removeItem() {}, getItem: () => null, setItem() {} },
  IntersectionObserver: class { observe() {} disconnect() {} }, matchMedia: () => ({ matches: false, addEventListener() {} }),
  document: { getElementById: $, createElement: () => makeEl('new'), querySelector: () => makeEl('qs'), querySelectorAll: () => [],
    addEventListener: (t, f) => { if (t === 'DOMContentLoaded') domReady = f; }, body: { style: {}, classList: { add() {}, remove() {}, toggle() {} } },
    documentElement: { setAttribute() {}, classList: { add() {}, remove() {}, toggle() {} } } }
});
win.window = win;
vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'script.js'), 'utf8'), ctx, { filename: 'script.js' });

(async () => {
  domReady();
  await new Promise(r => setTimeout(r, 20));

  // 1. Paste text on Home -> Analyze -> Workspace shows the text, chat initialised, document registered once
  $('pasteInput').value = 'Paris is the capital of France. It is on the Seine.';
  await $('analyzeTextBtn').fire('click');
  assert.deepStrictEqual(calls.filter(c => c === 'createTextDocument').length, 1);
  assert.strictEqual(win.location.hash, '#workspace');
  assert.strictEqual($('wsPreviewHeader').style.display, 'flex');
  assert.strictEqual($('wsDocTitle').textContent, 'Text Source');
  assert.strictEqual($('wsInitialInputState').style.display, 'none');          // the user is not asked to paste again

  // 2. First question: conversation created ONCE, answer rendered from server data, HTML escaped
  $('wsChatTextarea').value = 'What is the capital of France?';
  await $('wsChatSendBtn').fire('click');
  assert.strictEqual(calls.filter(c => c === 'createConversation').length, 1);
  const thread = $('wsChatThreadList');
  assert.strictEqual(thread.children.length, 2);
  const bubble = thread.children[1].innerHTML;
  assert(bubble.includes('Lower hallucination risk') && !bubble.includes('<img') && !bubble.includes('<script'), bubble);
  assert(!bubble.includes('98%') && !bubble.includes('0 Unverified'), 'no fabricated grounding numbers');

  // 3. Follow-up: same conversation, no duplicate
  $('wsChatTextarea').value = 'And the river?';
  await $('wsChatSendBtn').fire('click');
  assert.strictEqual(calls.filter(c => c === 'createConversation').length, 1);
  assert.strictEqual(calls.filter(c => c.startsWith('sendMessage:')).length, 2);
  assert.strictEqual(new Set(calls.filter(c => c.startsWith('sendMessage:'))).size, 1);

  // 4. History: loads from the server, escapes hostile titles / names
  const cid = Object.keys(db.convs)[0];
  db.convs[cid].title = '<img src=x onerror=alert(1)>'; db.convs[cid].documentName = '<b>evil</b>.pdf';
  await $('navHistoryBtn').fire('click');
  await new Promise(r => setTimeout(r, 20));
  assert(calls.includes('list'));
  const itemHtml = html.filter(h => h.includes('history-item-title')).pop();
  assert(itemHtml && !itemHtml.includes('<img') && !itemHtml.includes('<b>evil'), itemHtml);

  // 5. Restore from History: loads by id, rebuilds the thread, keeps the same conversation id for follow-ups
  $('wsChatThreadList').children.length = 0;
  await $('historyOpenInWorkspaceBtn').fire('click');
  await new Promise(r => setTimeout(r, 20));
  assert(calls.includes('get:' + cid));
  assert.strictEqual($('wsChatThreadList').children.length, 4);
  $('wsChatTextarea').value = 'One more?';
  await $('wsChatSendBtn').fire('click');
  assert.strictEqual(calls.filter(c => c === 'createConversation').length, 1, 'restoring must not create a duplicate conversation');
  assert.strictEqual(db.convs[cid].messages.length, 6);

  // 6. Expired conversation: friendly message, nothing restored
  db.mode = 'expired';
  await $('historyOpenInWorkspaceBtn').fire('click');
  await new Promise(r => setTimeout(r, 20));
  assert.strictEqual($('expiredModalText').textContent, 'This conversation has expired after 30 days of inactivity.');
  assert($('expiredModal').classes.has('open'));
  db.mode = 'ok';

  // 7. Replacing the document starts a fresh conversation (no context mix-up)
  const before = calls.filter(c => c === 'createConversation').length;
  await $('wsUploadNewBtn').fire('click');
  $('wsChatTextarea').value = 'question with no document';
  await $('wsChatSendBtn').fire('click');
  assert(alerts.some(a => a.includes('Upload a PDF or paste some text first')));
  $('wsPasteTextarea').value = 'Completely different source text.';
  await $('wsUseTextBtn').fire('click');
  $('wsChatTextarea').value = 'About the new text?';
  await $('wsChatSendBtn').fire('click');
  assert.strictEqual(calls.filter(c => c === 'createConversation').length, before + 1);
  assert.notStrictEqual(Object.keys(db.convs).pop(), cid);

  // 8. Generation not configured: honest error, question not shown as saved, no fake answer
  db.mode = 'noGen';
  $('wsChatTextarea').value = 'will fail';
  const kids = $('wsChatThreadList').children.length;
  await $('wsChatSendBtn').fire('click');
  assert.strictEqual($('wsChatTextarea').value, 'will fail', 'question is kept for retry');
  const last = $('wsChatThreadList').children.slice(-1)[0];
  assert(/not configured/.test(last.textContent), last.textContent);
  db.mode = 'ok';

  // 9. Delete from History calls the server
  await $('navHistoryBtn').fire('click'); await new Promise(r => setTimeout(r, 20));
  console.log('frontend smoke test: ALL CHECKS PASSED');
})().catch((e) => { console.error('FAILED:', e && e.stack || e); process.exit(1); });
