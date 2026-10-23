const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const source = fs.readFileSync(path.join(__dirname, '..', 'api.js'), 'utf8');

function makeApi(location, apiBase) {
  const calls = [];
  const window = { location, QUELLUM_API_BASE: apiBase };
  const context = vm.createContext({
    window,
    fetch: async (url, options) => {
      calls.push({ url, options });
      return { ok: true, json: async () => ({ status: 'ok' }) };
    },
    FormData: class {}
  });
  vm.runInContext(source, context, { filename: 'api.js' });
  return { api: window.QuellumAPI, calls };
}

(async () => {
  for (const hostname of ['localhost', '127.0.0.1']) {
    for (const port of ['3000', '5173', '5500']) {
      const { api, calls } = makeApi({ protocol: 'http:', hostname, port });
      await api.session();
      assert.strictEqual(calls[0].url, `http://${hostname}:8000/api/session`);
      assert.strictEqual(calls[0].options.credentials, 'include');
    }
  }

  const sameOrigin = makeApi({ protocol: 'http:', hostname: 'localhost', port: '8000' });
  await sameOrigin.api.session();
  assert.strictEqual(sameOrigin.calls[0].url, '/api/session');

  const override = makeApi(
    { protocol: 'http:', hostname: 'localhost', port: '3000' },
    'https://api.example.test/'
  );
  await override.api.session();
  assert.strictEqual(override.calls[0].url, 'https://api.example.test/api/session');

  console.log('frontend API test: ALL CHECKS PASSED');
})().catch((error) => {
  console.error('FAILED:', error && error.stack || error);
  process.exit(1);
});
