"""Scheduled cleanup. Idempotent, restart-safe, and independent of user traffic. Correctness never depends on it running:
expired rows are already invisible because every read filters on expires_at. Cleanup reclaims space and removes private data for good.

Order matters and each step is its own small transaction, so a crash leaves only orphans that the next run removes:
  1 expired conversations -> tombstone + delete (messages and evidence cascade)
  2 documents nothing references any more -> delete row (chunks/embeddings cascade) and queue the PDF for unlink
  3 unlink queued files (failures stay queued and are retried)
  4 sweep files on disk that no document row references (crash leftovers), with a grace period for in-flight uploads
  5 belt-and-braces: orphan chunks; old tombstones; expired sessions that own nothing"""
from __future__ import annotations

import logging
import time

from ..config import Settings, files_dir
from . import documents as docs
from .db import Database

log = logging.getLogger("quellum.cleanup")
SWEEP_GRACE_S = 3600


def run_cleanup(db: Database, settings: Settings, batch: int = 500, sweep_grace_s: int = SWEEP_GRACE_S) -> dict:
    stats = dict(conversations=0, documents=0, files=0, orphan_files=0, chunks=0, tombstones=0, sessions=0, errors=0)
    now = db.now()
    try:                                                                  # 1
        while True:
            with db.tx() as c:
                ids = [r["id"] for r in c.execute("SELECT id FROM conversations WHERE expires_at<=? LIMIT ?", (now, batch)).fetchall()]
                if not ids:
                    break
                q = ",".join("?" * len(ids))
                c.execute(f"INSERT OR IGNORE INTO tombstones(conversation_id, session_id, expired_at) "
                          f"SELECT id, session_id, ? FROM conversations WHERE id IN ({q})", [now] + ids)
                c.execute(f"DELETE FROM conversations WHERE id IN ({q})", ids)
                stats["conversations"] += len(ids)
    except Exception:  # noqa: BLE001
        log.exception("cleanup step 1 failed"); stats["errors"] += 1
    try:                                                                  # 2
        with db.tx() as c:
            stats["documents"] = docs.gc_documents(c, now)
    except Exception:  # noqa: BLE001
        log.exception("cleanup step 2 failed"); stats["errors"] += 1
    try:                                                                  # 3
        stats["files"] = docs.purge_pending_files(db, settings)
    except Exception:  # noqa: BLE001
        log.exception("cleanup step 3 failed"); stats["errors"] += 1
    try:                                                                  # 4
        fdir = files_dir(settings)
        if fdir.exists():
            known = {r[0] for r in db.conn().execute("SELECT file_name FROM documents WHERE file_name IS NOT NULL").fetchall()}
            wall = time.time()                       # file mtimes are real time, independent of the (testable) database clock
            for p in fdir.iterdir():
                if p.name in known or not p.is_file():
                    continue
                if wall - p.stat().st_mtime < sweep_grace_s:       # may be an upload whose DB insert is still in flight
                    continue
                try:
                    p.unlink(); stats["orphan_files"] += 1
                except OSError:
                    stats["errors"] += 1
    except Exception:  # noqa: BLE001
        log.exception("cleanup step 4 failed"); stats["errors"] += 1
    try:                                                                  # 5
        with db.tx() as c:
            stats["chunks"] = c.execute("DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)").rowcount
            stats["tombstones"] = c.execute("DELETE FROM tombstones WHERE expired_at<=?", (now - int(settings.tombstone_days * 86400),)).rowcount
            stats["sessions"] = c.execute(
                "DELETE FROM sessions WHERE expires_at<=? AND NOT EXISTS (SELECT 1 FROM conversations WHERE session_id=sessions.id) "
                "AND NOT EXISTS (SELECT 1 FROM documents WHERE session_id=sessions.id)", (now,)).rowcount
    except Exception:  # noqa: BLE001
        log.exception("cleanup step 5 failed"); stats["errors"] += 1
    log.info("cleanup: %s", stats)
    return stats
