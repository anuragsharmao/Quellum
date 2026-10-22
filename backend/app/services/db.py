"""SQLite persistence (stdlib only). One file DB in WAL mode, one connection per thread, explicit transactions.
All timestamps are integer UTC epoch seconds; the clock is injectable so tests can fast-forward instead of waiting 30 days."""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  token_hash TEXT NOT NULL UNIQUE,          -- sha256 of the cookie secret; the secret itself is never stored
  created_at INTEGER NOT NULL, last_seen INTEGER NOT NULL, expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  source_type TEXT NOT NULL CHECK (source_type IN ('pdf','text')),
  name TEXT NOT NULL, size_bytes INTEGER NOT NULL, page_count INTEGER NOT NULL,
  text_content TEXT,                        -- pasted text only
  file_name TEXT,                           -- PDF only: <uuid>.pdf inside the private files dir
  embedder TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  attached_at INTEGER,                      -- set when the first conversation is created on this document
  orphan_expires_at INTEGER NOT NULL        -- unattached documents are removed after this
);
CREATE INDEX IF NOT EXISTS ix_documents_session ON documents(session_id);
CREATE TABLE IF NOT EXISTS chunks (         -- the vector store: one embedding per passage
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL,
  dim INTEGER NOT NULL, embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(document_id);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  document_id TEXT NOT NULL REFERENCES documents(id),     -- RESTRICT: a referenced document can never be deleted by mistake
  title TEXT NOT NULL,
  created_at INTEGER NOT NULL, last_activity_at INTEGER NOT NULL, expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_conv_session ON conversations(session_id, expires_at);
CREATE INDEX IF NOT EXISTS ix_conv_doc ON conversations(document_id);
CREATE INDEX IF NOT EXISTS ix_conv_expiry ON conversations(expires_at);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  seq INTEGER NOT NULL, role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content TEXT NOT NULL, sources_json TEXT, analysis_json TEXT, created_at INTEGER NOT NULL,
  UNIQUE (conversation_id, seq)
);
CREATE TABLE IF NOT EXISTS tombstones (     -- lets the owner see "expired" instead of "not found"; holds no content
  conversation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, expired_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_file_deletes (   -- files whose rows are gone but whose unlink has not succeeded yet (retried by cleanup)
  file_name TEXT PRIMARY KEY, created_at INTEGER NOT NULL
);
"""


def iso(ts: int | float) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Database:
    def __init__(self, path, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self._local = threading.local()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def now(self) -> int:
        return int(self.clock())

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=30000")
            self._local.c = c
        return c

    @contextmanager
    def tx(self):
        c = self.conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def migrate(self):
        c = self.conn()
        c.executescript(SCHEMA)
        c.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)

    def ping(self) -> bool:
        try:
            self.conn().execute("SELECT 1").fetchone()
            return True
        except Exception:  # noqa: BLE001
            return False
