/**
 * Quellum API client (vanilla JS). Talks to the Quellum backend with the anonymous-session cookie.
 * The cookie is HttpOnly: this file never sees or stores any credential, only the server does.
 *
 * API base: same origin by default (the backend serves this frontend). To host the frontend elsewhere set, BEFORE this script,
 *   <script>window.QUELLUM_API_BASE = 'https://api.example.com';</script>   (and see the cross-site cookie notes in the README)
 */
(function () {
  'use strict';
  const location = window.location;
  const localFrontend = location &&
    (location.hostname === 'localhost' || location.hostname === '127.0.0.1') &&
    (location.port === '3000' || location.port === '5173' || location.port === '5500');
  const defaultBase = localFrontend ? location.protocol + '//' + location.hostname + ':8000' : '';
  const BASE = String(window.QUELLUM_API_BASE || defaultBase).replace(/\/+$/, '');

  class ApiError extends Error {
    constructor(status, code, message) {
      super(message || 'Request failed');
      this.name = 'ApiError';
      this.status = status;
      this.code = code || 'error';
    }
  }

  async function request(method, path, opts) {
    opts = opts || {};
    const init = { method, credentials: 'include', headers: {} };
    if (opts.json !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.json);
    } else if (opts.form) {
      init.body = opts.form;                       // multipart: the browser sets the boundary
    }
    let res;
    try {
      res = await fetch(BASE + path, init);
    } catch (e) {
      throw new ApiError(0, 'network', 'Could not reach the Quellum server. Please check your connection and try again.');
    }
    if (!res.ok) {
      let body = null;
      try { body = await res.json(); } catch (e) { /* non-JSON error */ }
      const detail = body && (body.detail && typeof body.detail === 'string' ? body.detail : null);
      throw new ApiError(res.status, body && body.code, detail || ('Request failed (' + res.status + ').'));
    }
    return opts.binary ? res.arrayBuffer() : res.json();
  }

  const enc = encodeURIComponent;
  window.QuellumAPI = {
    ApiError,
    session: () => request('GET', '/api/session'),
    uploadPdf: async (file) => {
      const form = new FormData();
      form.append('file', file, file.name);
      return (await request('POST', '/api/documents', { form })).document;
    },
    createTextDocument: async (text, name) => (await request('POST', '/api/documents/text', { json: { text, name } })).document,
    getDocument: async (id) => (await request('GET', '/api/documents/' + enc(id))).document,
    getDocumentFile: (id) => request('GET', '/api/documents/' + enc(id) + '/file', { binary: true }),
    createConversation: async (payload) => (await request('POST', '/api/conversations', { json: payload })).conversation,
    listConversations: async () => (await request('GET', '/api/conversations?include=messages')).conversations,
    getConversation: async (id) => (await request('GET', '/api/conversations/' + enc(id))).conversation,
    renameConversation: async (id, title) => (await request('PATCH', '/api/conversations/' + enc(id), { json: { title } })).conversation,
    deleteConversation: (id) => request('DELETE', '/api/conversations/' + enc(id)),
    sendMessage: (id, content) => request('POST', '/api/conversations/' + enc(id) + '/messages', { json: { content } })
  };
})();
