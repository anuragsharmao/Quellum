"""Documents: validation, text extraction, chunking, embedding, private file storage, ownership/expiry-aware access, garbage collection."""
from __future__ import annotations

import io
import os
import re
import uuid
from pathlib import Path

import numpy as np

from ..config import Settings, files_dir
from .db import Database, iso
from .errors import InvalidInput, LimitExceeded, NotFound

_FILE_RE = re.compile(r"^[0-9a-f]{32}\.pdf$")
_CTRL = re.compile(r"[\x00-\x1f\x7f<>:\"|?*]")


def clean_name(name: str, default: str) -> str:
    n = os.path.basename((name or "").replace("\\", "/"))
    n = _CTRL.sub("", n).strip()[:150]
    return n or default


def chunk_pages(pages, words: int, overlap: int):
    out = []
    step = max(words - overlap, 1)
    for pno, text in enumerate(pages, 1):
        ws = text.split()
        i = 0
        while i < len(ws):
            out.append((pno, " ".join(ws[i:i + words])))
            if i + words >= len(ws):
                break
            i += step
    return out


def extract_pdf_pages(data: bytes, settings: Settings):
    if data[:1024].find(b"%PDF-") < 0:
        raise InvalidInput("This file is not a valid PDF.")
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise InvalidInput("Password-protected PDFs are not supported.")
        n = len(reader.pages)
        if n == 0:
            raise InvalidInput("This PDF has no pages.")
        if n > settings.max_pdf_pages:
            raise InvalidInput(f"This PDF has {n} pages; the limit is {settings.max_pdf_pages}.")
        pages = []
        for p in reader.pages:
            try:
                pages.append((p.extract_text() or "").strip())
            except Exception:  # noqa: BLE001 - one bad page must not sink the document
                pages.append("")
    except InvalidInput:
        raise
    except Exception:  # noqa: BLE001
        raise InvalidInput("This PDF could not be read. It may be corrupted.")
    if sum(len(p) for p in pages) < 20:
        raise InvalidInput("No selectable text was found in this PDF (scanned documents need OCR, which is not supported).")
    return pages


def _storage_used(conn, session_id: str, now: int) -> int:
    return conn.execute("SELECT COALESCE(SUM(size_bytes),0) FROM documents WHERE session_id=?", (session_id,)).fetchone()[0]


def _store(db: Database, settings: Settings, embedder, session_id, source_type, name, size, pages, text_content, file_bytes):
    chunks = chunk_pages(pages, settings.chunk_words, settings.chunk_overlap)
    if not chunks:
        raise InvalidInput("The document contains no usable text.")
    emb = embedder.embed([t for _, t in chunks]).astype(np.float32)          # slow part happens before any DB lock is taken
    doc_id, now = uuid.uuid4().hex, db.now()
    file_name = None
    fdir = files_dir(settings)
    if file_bytes is not None:
        fdir.mkdir(parents=True, exist_ok=True)
        file_name = f"{doc_id}.pdf"
        tmp = fdir / f".{file_name}.tmp"
        tmp.write_bytes(file_bytes)
        os.replace(tmp, fdir / file_name)
    try:
        with db.tx() as c:
            if _storage_used(c, session_id, now) + size > settings.max_storage_mb_per_session * 1024 * 1024:
                raise LimitExceeded("Storage limit for this session reached. Delete an old conversation and try again.")
            c.execute("INSERT INTO documents(id, session_id, source_type, name, size_bytes, page_count, text_content, file_name, embedder,"
                      " created_at, attached_at, orphan_expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,NULL,?)",
                      (doc_id, session_id, source_type, name, size, len(pages), text_content, file_name, embedder.name, now,
                       now + int(settings.orphan_doc_ttl_hours * 3600)))
            c.executemany("INSERT INTO chunks(document_id, idx, page, text, dim, embedding) VALUES (?,?,?,?,?,?)",
                          [(doc_id, i, pg, tx, emb.shape[1], emb[i].tobytes()) for i, (pg, tx) in enumerate(chunks)])
    except BaseException:
        if file_name:                                       # do not leave an orphan file behind a failed insert
            try:
                (fdir / file_name).unlink()
            except OSError:
                pass
        raise
    return get_document_meta(db, session_id, doc_id)


def ingest_pdf(db, settings, embedder, session_id: str, filename: str, data: bytes):
    if not data:
        raise InvalidInput("The uploaded file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise LimitExceeded(f"File exceeds the maximum size of {settings.max_upload_mb} MB.")
    pages = extract_pdf_pages(data, settings)
    return _store(db, settings, embedder, session_id, "pdf", clean_name(filename, "document.pdf"), len(data), pages, None, data)


def ingest_text(db, settings, embedder, session_id: str, text: str, name: str = "Text Source"):
    text = (text or "").strip()
    if len(text) < 1:
        raise InvalidInput("Please enter some text.")
    if len(text) > settings.max_text_chars:
        raise LimitExceeded(f"Text exceeds the maximum of {settings.max_text_chars} characters.")
    return _store(db, settings, embedder, session_id, "text", clean_name(name, "Text Source"), len(text.encode("utf8")), [text], text, None)


_LIVE = """(EXISTS (SELECT 1 FROM conversations c WHERE c.document_id=d.id AND c.expires_at>:now)
           OR (d.attached_at IS NULL AND d.orphan_expires_at>:now))"""


def _meta(row, with_text=False):
    d = dict(id=row["id"], sourceType=row["source_type"], name=row["name"], sizeBytes=row["size_bytes"], pages=row["page_count"],
             createdAt=iso(row["created_at"]))
    if with_text and row["source_type"] == "text":
        d["textContent"] = row["text_content"]
    return d


def _live_row(db: Database, session_id: str, doc_id: str):
    row = db.conn().execute(f"SELECT d.* FROM documents d WHERE d.id=:id AND d.session_id=:sid AND {_LIVE}",
                            dict(id=doc_id, sid=session_id, now=db.now())).fetchone()
    if row is None:
        raise NotFound("Document not found.")          # same answer for 'missing', 'someone else's' and 'expired'
    return row


def get_document_meta(db, session_id, doc_id, with_text=False):
    return _meta(_live_row(db, session_id, doc_id), with_text)


def read_pdf_bytes(db, settings, session_id, doc_id) -> tuple[bytes, str]:
    row = _live_row(db, session_id, doc_id)
    if row["source_type"] != "pdf" or not row["file_name"] or not _FILE_RE.match(row["file_name"]):
        raise NotFound("Document not found.")
    try:
        return (files_dir(settings) / row["file_name"]).read_bytes(), row["name"]
    except OSError:
        raise NotFound("Document not found.")


def retrieve(db: Database, embedder, doc_id: str, query: str, k: int):
    """Top-k passages of ONE document by cosine similarity. Returns [{page, text, score, idx}] best first."""
    rows = db.conn().execute("SELECT idx, page, text, dim, embedding FROM chunks WHERE document_id=? ORDER BY idx", (doc_id,)).fetchall()
    if not rows:
        return []
    doc = db.conn().execute("SELECT embedder FROM documents WHERE id=?", (doc_id,)).fetchone()
    if doc is not None and doc["embedder"] != embedder.name:      # configured embedder changed since upload: re-embed on the fly
        E = embedder.embed([r["text"] for r in rows]).astype(np.float32)
    else:
        E = np.stack([np.frombuffer(r["embedding"], np.float32, count=r["dim"]) for r in rows])
    q = embedder.embed([query]).astype(np.float32)[0]
    sims = E @ q
    order = np.argsort(-sims, kind="stable")[:k]
    return [dict(idx=rows[i]["idx"], page=rows[i]["page"], text=rows[i]["text"], score=float(sims[i])) for i in order]


def gc_documents(conn, now: int, doc_ids=None) -> int:
    """Deletes documents that no conversation references (and, if never attached, whose grace period is over).
    Files are queued in pending_file_deletes inside the same transaction; the unlink happens after commit and is retried by cleanup."""
    q = ("SELECT d.id, d.file_name FROM documents d WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.document_id=d.id) "
         "AND (d.attached_at IS NOT NULL OR d.orphan_expires_at<=?)")
    params = [now]
    if doc_ids is not None:
        doc_ids = list(doc_ids)
        if not doc_ids:
            return 0
        q += " AND d.id IN (%s)" % ",".join("?" * len(doc_ids))
        params += doc_ids
    rows = conn.execute(q, params).fetchall()
    for r in rows:
        if r["file_name"]:
            conn.execute("INSERT OR IGNORE INTO pending_file_deletes(file_name, created_at) VALUES (?,?)", (r["file_name"], now))
        conn.execute("DELETE FROM documents WHERE id=?", (r["id"],))        # chunks (embeddings) cascade
    return len(rows)


def purge_pending_files(db: Database, settings: Settings) -> int:
    fdir, done = files_dir(settings), 0
    for r in db.conn().execute("SELECT file_name FROM pending_file_deletes").fetchall():
        name = r["file_name"]
        try:
            if _FILE_RE.match(name):
                (fdir / name).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            continue                                       # keep the row; the next cleanup run retries
        with db.tx() as c:
            c.execute("DELETE FROM pending_file_deletes WHERE file_name=?", (name,))
        done += 1
    return done
