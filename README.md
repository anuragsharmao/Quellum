# Quellum

Hallucination-aware RAG verification, login-free. Upload a PDF or paste text -> Analyze -> ask questions -> conversations are saved
(anonymously, per browser) -> History -> resume. Everything expires **30 days after the last real activity**.

```
frontend/   existing Quellum UI (vanilla HTML/CSS/JS, unchanged design) + api.js (client) + tests/smoke.test.js
backend/    FastAPI: anonymous sessions, SQLite storage, retrieval, answer generation hook, hallucination detector (8-seed ensemble), cleanup
```

## How it works
**Anonymous sessions.** First request -> the server creates a random 256-bit secret, stores only its SHA-256, and sends it in an
`HttpOnly; SameSite=Lax` (`Secure` in production) cookie. JavaScript never sees it. Every document/conversation query is scoped by the session
resolved from that cookie; ids in URLs are only names. Another session's id answers `404`, same as a missing id (no existence leak).
A guessed/forged cookie or an id used as a cookie resolves to nothing.

**Conversations & restore.** Analyze uploads the PDF (or text) once to a private store (not a public URL; served only through
`GET /api/documents/:id/file` after the ownership check). The first question creates a conversation bound to that document; each answer stores
the question, answer, the retrieved passages (page + excerpt) and the detector's result. History lists non-expired conversations of the session;
"Open in Workspace" fetches the conversation and the original PDF/text from the server and appends new questions to the same conversation.
Replacing the document clears the chat and starts a new conversation.

**Expiry.** `expires_at = last_activity + 30 days`, extended only by a stored question/answer. Opening History, refreshing or renaming does not extend it.
Every read filters `expires_at > now`, so expired data is unreachable immediately, even if cleanup has not run. An expired conversation returns `410`
and the UI shows "This conversation has expired after 30 days of inactivity." with a "Start a new conversation" button.

**Cleanup** (`app/services/cleanup.py`; idempotent, each step its own transaction, retries safe): expired conversations (+ messages/evidence, leaving a
content-free tombstone so the owner sees "expired") -> documents no conversation references (+ pasted text + embeddings) -> queued PDF unlinks (failures stay
queued and are retried) -> orphan files on disk -> orphan chunks, old tombstones, expired sessions that own nothing. A document shared by two
conversations survives until the last one is gone. Uploads never attached to a conversation are removed after 24 h.

## Run locally
```
cd backend
python -m venv .venv && .venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
copy ..\.env.example .env      # then export/set the variables (the app reads the process environment)
uvicorn app.main:app --port 8000
```
Open http://localhost:8000 (the backend serves `frontend/`, so everything is same-origin: no CORS, plain cookies).

* **Answers need provider credentials.** Groq is primary and Gemini is called once only if Groq fails. Set `GROQ_API_KEY` and, optionally, `GEMINI_API_KEY`; model ids and bounded prompt size are configurable in `.env`. OpenAI-compatible and Anthropic providers remain available through the legacy `QUELLUM_LLM_*` settings. Without a configured provider key, questions return 503 - nothing is faked.
* **Detector ensemble.** Models 0–7 are loaded by default (`QUELLUM_DETECTOR_SEEDS=0,1,2,3,4,5,6,7`); models 8–9 can be selected instead or in another configured ensemble. The published detector metrics and threshold in `artifacts/` still describe the original five-model validation run and are not recalibrated for a different seed selection.
* **Hallucination analysis** needs the detector set up as in `backend/README.md` (research files, `export_artifacts.py`, `verify_ensemble.py`, `parity_test.py`).
  Until then answers are saved with `hallucinationResult.status = "unavailable"` and the UI says so.
* **Retrieval embeddings**: `QUELLUM_EMBEDDER=sbert` (default, same MiniLM the detector uses) or `hashing` (no ML dependencies; lexical, for development).

## Tests
```
cd backend && python -m unittest tests.test_storage -v     # sessions, isolation, expiry (accelerated clock), cleanup, limits  - 25 tests
cd backend && pytest tests/test_api.py -v                   # HTTP layer; needs fastapi + httpx installed
node frontend/tests/smoke.test.js                           # real script.js against a stub DOM + fake API contract
```

## Deploy
Single instance with a **persistent disk** (SQLite + PDFs live in `QUELLUM_DATA_DIR`). `Dockerfile` provided (mount a volume at `/data`).
Production env: `QUELLUM_ENV=production` (Secure cookies), HTTPS in front (reverse proxy / platform), keep the default same-origin setup.
Split hosting (frontend and API on different sites): set `window.QUELLUM_API_BASE` in `index.html` before `api.js`, `QUELLUM_COOKIE_SAMESITE=none`,
`QUELLUM_COOKIE_SECURE=true`, `QUELLUM_CORS_ORIGINS=https://your-frontend` (exact origin, never `*`). Same-site subdomains can keep `lax`.
Set a body-size limit at the proxy (e.g. nginx `client_max_body_size 21m`): the app rejects oversized `Content-Length` but multipart is parsed before handlers run.
Rate limits are in-process (correct for one instance; use the platform/Redis limiter if you scale out - SQLite also means one writer instance).

**Scheduled cleanup.** The server runs it every `QUELLUM_CLEANUP_INTERVAL_MINUTES` (default 60) and once at startup. For a schedule independent of the web process run
`python -m app.cleanup_cli` (exit code 1 if a step failed) from: cron (`0 * * * *`), a systemd timer, Windows Task Scheduler, a platform cron job
(Render/Railway/Fly) sharing the same disk, or `docker exec <container> python -m app.cleanup_cli`. Overlapping runs are safe.

## Dashboard / credentials you must provide (nothing here is pre-configured)
LLM provider account + key + model id; a hosting platform with a persistent volume and HTTPS; (optional) a scheduler for `cleanup_cli`.
