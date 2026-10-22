"""Conversations and messages. Every function takes the session id resolved from the cookie and filters on it, so an id alone grants nothing.
Expiry is enforced in SQL (expires_at > now) on every read, so data is inaccessible the instant it expires, before any cleanup run."""
from __future__ import annotations

import json
import uuid

from ..config import Settings
from . import documents as docs
from .db import Database, iso
from .errors import Expired, InvalidInput, LimitExceeded, NotFound
from .sessions import extend_session


def expiry_for(settings: Settings, now: int) -> int:
    return now + int(settings.expiration_days * 24 * 60 * 60)        # == Date.now() + EXPIRATION_DAYS*24*60*60*1000


def clean_title(title: str) -> str:
    t = " ".join((title or "").split())[:120]
    if not t:
        raise InvalidInput("Title must not be empty.")
    return t


def _msg(row) -> dict:
    return dict(id=row["id"], role=row["role"], content=row["content"], createdAt=iso(row["created_at"]),
                sources=json.loads(row["sources_json"]) if row["sources_json"] else [],
                hallucinationResult=json.loads(row["analysis_json"]) if row["analysis_json"] else None)


def _conv(row, messages=None, with_text=False) -> dict:
    d = dict(id=row["id"], title=row["title"], sourceType=row["source_type"], documentId=row["document_id"], documentName=row["doc_name"],
             createdAt=iso(row["created_at"]), updatedAt=iso(row["last_activity_at"]), expiresAt=iso(row["expires_at"]))
    if with_text and row["source_type"] == "text":
        d["textContent"] = row["text_content"]
    if messages is not None:
        d["messages"] = messages
        d["messageCount"] = len(messages)
    return d


_SEL = ("SELECT c.*, d.name AS doc_name, d.source_type AS source_type, d.text_content AS text_content "
        "FROM conversations c JOIN documents d ON d.id=c.document_id ")


def _load_messages(conn, conv_ids):
    out = {i: [] for i in conv_ids}
    if conv_ids:
        for r in conn.execute("SELECT * FROM messages WHERE conversation_id IN (%s) ORDER BY conversation_id, seq" % ",".join("?" * len(conv_ids)),
                              list(conv_ids)).fetchall():
            out[r["conversation_id"]].append(_msg(r))
    return out


def create(db: Database, settings: Settings, session_id: str, document_id: str, title: str) -> dict:
    title = clean_title(title) if title else "New Conversation"
    docs.get_document_meta(db, session_id, document_id)               # ownership + liveness (NotFound otherwise)
    now, cid = db.now(), uuid.uuid4().hex
    with db.tx() as c:
        n = c.execute("SELECT COUNT(*) FROM conversations WHERE session_id=? AND expires_at>?", (session_id, now)).fetchone()[0]
        if n >= settings.max_conversations_per_session:
            raise LimitExceeded("Conversation limit reached. Delete an old conversation first.")
        c.execute("INSERT INTO conversations(id, session_id, document_id, title, created_at, last_activity_at, expires_at) VALUES (?,?,?,?,?,?,?)",
                  (cid, session_id, document_id, title, now, now, expiry_for(settings, now)))
        c.execute("UPDATE documents SET attached_at=COALESCE(attached_at, ?) WHERE id=?", (now, document_id))
        extend_session(c, settings, session_id, now)
    return get(db, session_id, cid)


def _owned_live_row(db: Database, session_id: str, cid: str):
    now = db.now()
    row = db.conn().execute(_SEL + "WHERE c.id=? AND c.session_id=?", (cid, session_id)).fetchone()
    if row is not None:
        if row["expires_at"] <= now:
            raise Expired("This conversation has expired after 30 days of inactivity.")
        return row
    if db.conn().execute("SELECT 1 FROM tombstones WHERE conversation_id=? AND session_id=?", (cid, session_id)).fetchone():
        raise Expired("This conversation has expired after 30 days of inactivity.")
    raise NotFound("Conversation not found.")                         # also the answer for ids that belong to another session


def get(db: Database, session_id: str, cid: str) -> dict:
    row = _owned_live_row(db, session_id, cid)
    return _conv(row, _load_messages(db.conn(), [cid])[cid], with_text=True)


def list_conversations(db: Database, session_id: str, include_messages: bool = False) -> list:
    rows = db.conn().execute(_SEL + "WHERE c.session_id=? AND c.expires_at>? "
                             "AND EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id=c.id) "     # a conversation with no saved answer is not shown
                             "ORDER BY c.last_activity_at DESC, c.rowid DESC",
                             (session_id, db.now())).fetchall()
    msgs = _load_messages(db.conn(), [r["id"] for r in rows]) if include_messages else None
    out = []
    for r in rows:
        if msgs is not None:
            out.append(_conv(r, msgs[r["id"]]))
        else:
            cnt = db.conn().execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", (r["id"],)).fetchone()[0]
            d = _conv(r)
            d["messageCount"] = cnt
            out.append(d)
    return out


def rename(db: Database, settings: Settings, session_id: str, cid: str, title: str) -> dict:
    title = clean_title(title)
    _owned_live_row(db, session_id, cid)
    with db.tx() as c:                                                # renaming is NOT activity: expiry is left untouched
        c.execute("UPDATE conversations SET title=? WHERE id=? AND session_id=? AND expires_at>?", (title, cid, session_id, db.now()))
    return get(db, session_id, cid)


def delete(db: Database, settings: Settings, session_id: str, cid: str) -> None:
    row = db.conn().execute("SELECT document_id FROM conversations WHERE id=? AND session_id=?", (cid, session_id)).fetchone()
    if row is None:
        raise NotFound("Conversation not found.")
    with db.tx() as c:
        c.execute("DELETE FROM conversations WHERE id=? AND session_id=?", (cid, session_id))     # messages cascade
        docs.gc_documents(c, db.now(), [row["document_id"]])          # removes the PDF/text/embeddings only if no other conversation uses it
    docs.purge_pending_files(db, settings)


def append_exchange(db: Database, settings: Settings, session_id: str, cid: str, user_content: str, assistant: dict) -> list:
    """Atomically stores a question and its answer and slides the expiry. This is the only place (besides create) that counts as activity."""
    with db.tx() as c:
        now = db.now()
        row = c.execute("SELECT expires_at FROM conversations WHERE id=? AND session_id=?", (cid, session_id)).fetchone()
        if row is None:
            raise NotFound("Conversation not found.")
        if row["expires_at"] <= now:
            raise Expired("This conversation has expired after 30 days of inactivity.")
        seq = c.execute("SELECT COALESCE(MAX(seq),0) FROM messages WHERE conversation_id=?", (cid,)).fetchone()[0]
        if seq + 2 > settings.max_messages_per_conversation:
            raise LimitExceeded("This conversation reached its message limit. Start a new conversation.")
        ids = [uuid.uuid4().hex, uuid.uuid4().hex]
        c.execute("INSERT INTO messages(id, conversation_id, seq, role, content, sources_json, analysis_json, created_at) VALUES (?,?,?,?,?,NULL,NULL,?)",
                  (ids[0], cid, seq + 1, "user", user_content, now))
        c.execute("INSERT INTO messages(id, conversation_id, seq, role, content, sources_json, analysis_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
                  (ids[1], cid, seq + 2, "assistant", assistant["content"], json.dumps(assistant.get("sources") or []),
                   json.dumps(assistant.get("hallucinationResult")), now))
        c.execute("UPDATE conversations SET last_activity_at=?, expires_at=? WHERE id=?", (now, expiry_for(settings, now), cid))
        extend_session(c, settings, session_id, now)
        rows = c.execute("SELECT * FROM messages WHERE id IN (?,?) ORDER BY seq", ids).fetchall()
    return [_msg(r) for r in rows]


def require_live(db: Database, session_id: str, cid: str) -> dict:
    """Used by the chat flow before doing expensive work. Returns {documentId}."""
    row = _owned_live_row(db, session_id, cid)
    return dict(documentId=row["document_id"], sourceType=row["source_type"])
