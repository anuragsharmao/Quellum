"""Anonymous sessions. The browser holds a random 256-bit secret in an HttpOnly cookie; the DB stores only its SHA-256.
Possessing a conversation/document id proves nothing: every query is scoped by the session id resolved from the cookie."""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass

from ..config import Settings
from .db import Database

REFRESH_AFTER_S = 24 * 3600     # sliding expiry is refreshed at most once a day (avoids a write per request)


@dataclass
class SessionCtx:
    id: str
    new_token: str | None = None      # set only when a cookie must be (re)issued


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf8")).hexdigest()    # input has 256 bits of entropy, so a fast hash is appropriate


def _ttl(settings: Settings) -> int:
    return int(settings.expiration_days * 86400)


def create_session(db: Database, settings: Settings) -> SessionCtx:
    token, sid, now = secrets.token_urlsafe(32), uuid.uuid4().hex, db.now()
    with db.tx() as c:
        c.execute("INSERT INTO sessions(id, token_hash, created_at, last_seen, expires_at) VALUES (?,?,?,?,?)",
                  (sid, hash_token(token), now, now, now + _ttl(settings)))
    return SessionCtx(sid, token)


def resolve_session(db: Database, settings: Settings, token: str | None) -> SessionCtx | None:
    if not token or len(token) > 200:
        return None
    now = db.now()
    row = db.conn().execute("SELECT id, last_seen FROM sessions WHERE token_hash=? AND expires_at>?", (hash_token(token), now)).fetchone()
    if row is None:
        return None
    if now - row["last_seen"] >= REFRESH_AFTER_S:
        with db.tx() as c:
            c.execute("UPDATE sessions SET last_seen=?, expires_at=? WHERE id=?", (now, now + _ttl(settings), row["id"]))
        return SessionCtx(row["id"], token)       # re-issue the same cookie so its Max-Age slides forward
    return SessionCtx(row["id"])


def extend_session(conn, settings: Settings, session_id: str, now: int):
    """Called inside the transaction of a meaningful action so a session never expires before its conversations."""
    conn.execute("UPDATE sessions SET last_seen=?, expires_at=MAX(expires_at, ?) WHERE id=?", (now, now + _ttl(settings), session_id))
