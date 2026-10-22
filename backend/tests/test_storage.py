"""Storage / session / expiry / cleanup tests. Pure stdlib + numpy + pypdf + reportlab: no FastAPI, no model needed.
Time is accelerated with an injectable clock and a 60-second 'month'.   Run:  python -m unittest tests.test_storage -v"""
import io, shutil, sys, tempfile, threading, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, files_dir
from app.services import chat, cleanup, conversations as convs, documents as docs, sessions
from app.services.db import Database
from app.services.embedding import HashingEmbedder
from app.services.errors import Expired, InvalidInput, LimitExceeded, NotFound, GenerationNotConfigured, GenerationFailed
from app.services.ratelimit import RateLimiter

LIFE = 60            # seconds a conversation lives in these tests (stands in for 30 days)
TEXT_A = ("Paris is the capital of France. It sits on the Seine river and has about two million residents. " * 5 +
          "The Eiffel Tower was completed in 1889 for the World's Fair. ") * 3
TEXT_B = "Photosynthesis converts light energy into chemical energy stored in glucose. Chlorophyll absorbs red and blue light. " * 10


class Clock:
    def __init__(self): self.t = 1_800_000_000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def make_pdf(pages):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO(); c = canvas.Canvas(buf)
    for text in pages:
        y = 800
        for line in text:
            c.drawString(50, y, line); y -= 16
        c.showPage()
    c.save(); return buf.getvalue()


class FakeGen:
    def __init__(self): self.calls = []
    def generate(self, q, passages):
        self.calls.append((q, passages)); return "Paris is the capital of France. The tower dates from 1889."


def fake_analyzer(q, ctx, ans):
    return dict(summary=dict(overall_score=-1.2, overall_risk=0.2, decision_threshold=0.55, flagged=False, total_sentences=2,
                             potentially_hallucinated=None), sentences=[dict(text="Paris is the capital of France.", score=0.1, label=None),
                                                                        dict(text="The tower dates from 1889.", score=0.2, label=None)])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.clock = Clock()
        self.s = Settings(data_dir=self.tmp, expiration_days=LIFE / 86400, orphan_doc_ttl_hours=120 / 3600, tombstone_days=1000,
                          max_conversations_per_session=3, max_messages_per_conversation=6, max_storage_mb_per_session=1, max_upload_mb=1)
        self.db = Database(self.tmp / "q.sqlite3", clock=self.clock)
        self.emb = HashingEmbedder()
        self.a = sessions.create_session(self.db, self.s)
        self.b = sessions.create_session(self.db, self.s)

    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)

    def doc(self, sess=None, text=TEXT_A): return docs.ingest_text(self.db, self.s, self.emb, (sess or self.a).id, text)["id"]
    def conv(self, sess=None, doc=None, title="T"):
        sess = sess or self.a
        return convs.create(self.db, self.s, sess.id, doc or self.doc(sess), title)["id"]
    def ask(self, cid, q="What is the capital of France?", sess=None, gen=None):
        return chat.send_message(self.db, self.s, (sess or self.a).id, cid, q, embedder=self.emb, generator=gen or FakeGen(), analyzer=fake_analyzer)
    def count(self, table): return self.db.conn().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    def files(self): return sorted(p.name for p in files_dir(self.s).glob("*.pdf")) if files_dir(self.s).exists() else []


class TestSessions(Base):
    def test_secret_is_not_stored_and_resolves(self):
        row = self.db.conn().execute("SELECT * FROM sessions WHERE id=?", (self.a.id,)).fetchone()
        self.assertNotEqual(row["token_hash"], self.a.new_token)
        self.assertEqual(row["token_hash"], sessions.hash_token(self.a.new_token))
        self.assertGreaterEqual(len(self.a.new_token), 43)                       # 256 bits urlsafe
        self.assertEqual(sessions.resolve_session(self.db, self.s, self.a.new_token).id, self.a.id)
        self.assertNotEqual(self.a.new_token, self.b.new_token)

    def test_guessed_or_public_ids_do_not_authenticate(self):
        for bad in (None, "", "x" * 300, self.a.id, sessions.hash_token(self.a.new_token), "A" * 43):
            self.assertIsNone(sessions.resolve_session(self.db, self.s, bad))

    def test_session_expires_and_slides(self):
        self.clock.advance(LIFE - 5)
        self.assertIsNone(sessions.resolve_session(self.db, self.s, self.a.new_token).new_token)   # < refresh threshold: no re-issue
        self.clock.advance(sessions.REFRESH_AFTER_S + 1)
        self.assertIsNone(sessions.resolve_session(self.db, self.s, self.a.new_token))              # long expired by now


class TestIsolationAndExpiry(Base):
    def test_cross_session_access_is_denied_everywhere(self):
        d = self.doc(self.a); cid = self.conv(self.a, d); self.ask(cid)
        bs = self.b.id
        with self.assertRaises(NotFound): convs.get(self.db, bs, cid)
        with self.assertRaises(NotFound): convs.rename(self.db, self.s, bs, cid, "pwn")
        with self.assertRaises(NotFound): convs.delete(self.db, self.s, bs, cid)
        with self.assertRaises(NotFound): docs.get_document_meta(self.db, bs, d)
        with self.assertRaises(NotFound): docs.read_pdf_bytes(self.db, self.s, bs, d)
        with self.assertRaises(NotFound): convs.create(self.db, self.s, bs, d, "steal")           # cannot attach to A's document
        with self.assertRaises(NotFound): self.ask(cid, sess=self.b)
        self.assertEqual(convs.list_conversations(self.db, bs), [])
        self.assertEqual(len(convs.get(self.db, self.a.id, cid)["messages"]), 2)                  # A's data untouched

    def test_expired_is_invisible_immediately_without_cleanup(self):
        d = self.doc(); cid = self.conv(doc=d); self.ask(cid)
        self.clock.advance(LIFE + 1)
        self.assertEqual(convs.list_conversations(self.db, self.a.id, True), [])
        with self.assertRaises(Expired): convs.get(self.db, self.a.id, cid)
        with self.assertRaises(NotFound): docs.get_document_meta(self.db, self.a.id, d)
        with self.assertRaises(Expired): self.ask(cid)
        with self.assertRaises(Expired): convs.rename(self.db, self.s, self.a.id, cid, "x")
        self.assertEqual(self.count("messages"), 2)                         # still physically there: invisible, not yet cleaned

    def test_activity_slides_expiry_but_reading_and_renaming_do_not(self):
        cid = self.conv(); first = convs.get(self.db, self.a.id, cid)["expiresAt"]
        self.clock.advance(LIFE // 2)
        convs.list_conversations(self.db, self.a.id, True); convs.get(self.db, self.a.id, cid)
        convs.rename(self.db, self.s, self.a.id, cid, "renamed")
        self.assertEqual(convs.get(self.db, self.a.id, cid)["expiresAt"], first)           # opening/refreshing/renaming: no extension
        self.ask(cid)
        self.assertGreater(convs.get(self.db, self.a.id, cid)["expiresAt"], first)         # a real question: extended
        self.clock.advance(LIFE - 5)
        self.assertEqual(len(convs.get(self.db, self.a.id, cid)["messages"]), 2)            # alive thanks to the new activity
        self.clock.advance(10)
        with self.assertRaises(Expired): convs.get(self.db, self.a.id, cid)

    def test_expiry_is_exactly_N_days_of_inactivity(self):
        s30 = Settings(data_dir=self.tmp)                                                   # production default
        self.assertEqual(s30.expiration_days, 30)
        self.assertEqual(convs.expiry_for(s30, 1000), 1000 + 30 * 24 * 60 * 60)

    def test_unattached_document_is_private_and_time_limited(self):
        d = self.doc()
        docs.get_document_meta(self.db, self.a.id, d)
        self.clock.advance(120 + 1)
        with self.assertRaises(NotFound): docs.get_document_meta(self.db, self.a.id, d)


class TestPersistenceAndChat(Base):
    def test_exchange_is_stored_with_sources_and_analysis_and_order(self):
        cid = self.conv(); out = self.ask(cid); self.ask(cid, "When was the Eiffel Tower completed?")
        full = convs.get(self.db, self.a.id, cid)
        self.assertEqual([m["role"] for m in full["messages"]], ["user", "assistant", "user", "assistant"])
        asst = full["messages"][1]
        self.assertTrue(asst["sources"] and asst["sources"][0]["page"] is None and asst["sources"][0]["section"] == "Pasted text")
        self.assertEqual(asst["hallucinationResult"]["status"], "ok")
        self.assertEqual(asst["hallucinationResult"]["riskLabel"], "Lower hallucination risk")
        self.assertEqual(full["textContent"], TEXT_A.strip())
        self.assertEqual(full["id"], cid); self.assertEqual(len(convs.list_conversations(self.db, self.a.id)), 1)   # same conversation, no duplicates

    def test_retrieval_uses_only_this_documents_chunks(self):
        da, db_ = self.doc(self.a, TEXT_A), self.doc(self.a, TEXT_B)
        cid_a, cid_b = self.conv(doc=da), self.conv(doc=db_)
        gen = FakeGen(); self.ask(cid_a, "capital of France Eiffel Tower", gen=gen)
        self.assertTrue(all("Photosynthesis" not in p for p in gen.calls[0][1]))
        gen2 = FakeGen(); self.ask(cid_b, "How does chlorophyll absorb light?", gen=gen2)
        self.assertTrue(all("Paris" not in p for p in gen2.calls[0][1]) and "chlorophyll" in " ".join(gen2.calls[0][1]).lower())

    def test_no_generator_means_503_and_nothing_stored(self):
        cid = self.conv(); before = convs.get(self.db, self.a.id, cid)["expiresAt"]
        self.clock.advance(5)
        with self.assertRaises(GenerationNotConfigured):
            chat.send_message(self.db, self.s, self.a.id, cid, "q?", embedder=self.emb, generator=None, analyzer=None)
        self.assertEqual(self.count("messages"), 0)
        self.assertEqual(convs.get(self.db, self.a.id, cid)["expiresAt"], before)

    def test_generator_failure_stores_nothing(self):
        class Boom:
            def generate(self, q, p): raise GenerationFailed("down")
        cid = self.conv()
        with self.assertRaises(GenerationFailed): self.ask(cid, gen=Boom())
        self.assertEqual(self.count("messages"), 0)

    def test_analyzer_states_are_honest(self):
        cid = self.conv()                                   # one conversation, three exchanges (message cap is 6)
        for analyzer, status in ((None, "unavailable"), (lambda *a: (_ for _ in ()).throw(ValueError("too long")), "skipped"),
                                 (lambda *a: (_ for _ in ()).throw(RuntimeError("x")), "error")):
            out = chat.send_message(self.db, self.s, self.a.id, cid, "capital?", embedder=self.emb, generator=FakeGen(), analyzer=analyzer)
            self.assertEqual(out[1]["hallucinationResult"]["status"], status)
            self.assertIn("reason", out[1]["hallucinationResult"])

    def test_empty_conversations_are_hidden_from_history_but_resumable(self):
        cid = self.conv()
        self.assertEqual(convs.list_conversations(self.db, self.a.id), [])                  # created, nothing answered yet
        self.ask(cid)
        self.assertEqual([c["id"] for c in convs.list_conversations(self.db, self.a.id)], [cid])

    def test_limits(self):
        ids = [self.conv() for _ in range(3)]
        with self.assertRaises(LimitExceeded): self.conv()                                  # max 3 conversations
        cid = ids[0]
        for _ in range(3): self.ask(cid)
        with self.assertRaises(LimitExceeded): self.ask(cid)                                # max 6 messages
        with self.assertRaises(InvalidInput): self.ask(cid, "   ")
        with self.assertRaises(InvalidInput): self.ask(cid, "x" * 5000)

    def test_concurrent_messages_keep_unique_sequence(self):
        cid = self.conv(); errs = []
        def go():
            try: self.ask(cid)
            except Exception as e: errs.append(e)
        ts = [threading.Thread(target=go) for _ in range(3)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(errs, [])
        seqs = [r[0] for r in self.db.conn().execute("SELECT seq FROM messages ORDER BY seq")]
        self.assertEqual(seqs, [1, 2, 3, 4, 5, 6])

    def test_rate_limiter(self):
        rl = RateLimiter(clock=self.clock)
        for _ in range(3): rl.check("k", 3, 10)
        from app.services.errors import RateLimited
        with self.assertRaises(RateLimited): rl.check("k", 3, 10)
        self.clock.advance(11); rl.check("k", 3, 10)


class TestPdf(Base):
    def test_pdf_roundtrip_pages_private_file_and_rejections(self):
        pdf = make_pdf([["Machine learning is a subset of artificial intelligence that learns from data."] * 1,
                        ["Healthcare and finance use machine learning for prediction and fraud detection."]])
        m = docs.ingest_pdf(self.db, self.s, self.emb, self.a.id, "../../evil<name>.pdf", pdf)
        self.assertEqual((m["pages"], m["sourceType"]), (2, "pdf")); self.assertNotIn("/", m["name"]); self.assertNotIn("<", m["name"])
        self.assertEqual(len(self.files()), 1)
        data, name = docs.read_pdf_bytes(self.db, self.s, self.a.id, m["id"]); self.assertEqual(data, pdf)
        with self.assertRaises(NotFound): docs.read_pdf_bytes(self.db, self.s, self.b.id, m["id"])
        hits = docs.retrieve(self.db, self.emb, m["id"], "fraud detection in finance", 1)
        self.assertEqual(hits[0]["page"], 2)                                                # real page numbers for citations
        with self.assertRaises(InvalidInput): docs.ingest_pdf(self.db, self.s, self.emb, self.a.id, "a.pdf", b"not a pdf at all")
        with self.assertRaises(InvalidInput): docs.ingest_pdf(self.db, self.s, self.emb, self.a.id, "a.pdf", b"")
        with self.assertRaises(LimitExceeded): docs.ingest_pdf(self.db, self.s, self.emb, self.a.id, "a.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024))
        with self.assertRaises(InvalidInput): docs.ingest_pdf(self.db, self.s, self.emb, self.a.id, "blank.pdf", make_pdf([[]]))   # no text layer
        self.assertEqual(len(self.files()), 1)                                              # rejected uploads leave no files

    def test_storage_quota_and_no_orphan_file_on_failure(self):
        pdf = make_pdf([["Some sample sentence about quotas and limits in storage."]])
        s = Settings(data_dir=self.tmp, max_storage_mb_per_session=0, max_upload_mb=1)
        with self.assertRaises(LimitExceeded): docs.ingest_pdf(self.db, s, self.emb, self.a.id, "q.pdf", pdf)
        self.assertEqual(self.files(), []); self.assertEqual(self.count("documents"), 0)


class TestCleanup(Base):
    def run_cleanup(self): return cleanup.run_cleanup(self.db, self.s, sweep_grace_s=0)

    def test_cleanup_removes_everything_for_expired_and_is_idempotent(self):
        pdf = make_pdf([["Machine learning learns patterns from data and makes predictions."]])
        d = docs.ingest_pdf(self.db, self.s, self.emb, self.a.id, "ml.pdf", pdf)["id"]
        cid = self.conv(doc=d); self.ask(cid, "machine learning data")
        keep = self.conv(self.b)                                         # other session, will stay alive
        self.clock.advance(LIFE // 2); self.ask(keep, sess=self.b)         # keep B alive; A is left idle
        self.clock.advance(LIFE // 2 + 5)
        st = self.run_cleanup()
        self.assertEqual((st["conversations"], st["documents"], st["files"], st["errors"]), (1, 1, 1, 0))
        self.assertEqual(self.files(), [])
        self.assertEqual(self.db.conn().execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", (cid,)).fetchone()[0], 0)
        self.assertEqual(self.db.conn().execute("SELECT COUNT(*) FROM chunks WHERE document_id=?", (d,)).fetchone()[0], 0)    # embeddings gone
        self.assertEqual(self.db.conn().execute("SELECT COUNT(*) FROM documents WHERE id=?", (d,)).fetchone()[0], 0)
        self.assertEqual(len(convs.get(self.db, self.b.id, keep)["messages"]), 2)                                        # B untouched
        with self.assertRaises(Expired): convs.get(self.db, self.a.id, cid)                                              # tombstone: friendly message
        with self.assertRaises(NotFound): convs.get(self.db, self.b.id, cid)                                             # tombstone is session-scoped
        st2 = self.run_cleanup()
        self.assertEqual((st2["conversations"], st2["documents"], st2["files"], st2["orphan_files"], st2["errors"]), (0, 0, 0, 0, 0))

    def test_shared_document_survives_until_last_conversation_goes(self):
        pdf = make_pdf([["Shared document text about neural networks and training."]])
        d = docs.ingest_pdf(self.db, self.s, self.emb, self.a.id, "s.pdf", pdf)["id"]
        c1, c2 = self.conv(doc=d), self.conv(doc=d)
        convs.delete(self.db, self.s, self.a.id, c1)
        self.assertEqual(self.files(), [d + ".pdf"]); docs.get_document_meta(self.db, self.a.id, d)                       # still needed by c2
        self.assertTrue(docs.retrieve(self.db, self.emb, d, "neural networks", 1))
        convs.delete(self.db, self.s, self.a.id, c2)
        self.assertEqual(self.files(), []); self.assertEqual(self.count("chunks"), 0)

    def test_document_used_by_active_conversation_is_not_deleted_when_other_expired(self):
        d = self.doc(); old = self.conv(doc=d)
        self.clock.advance(LIFE - 10); fresh = self.conv(doc=d); self.ask(fresh)       # same document, newer conversation
        self.clock.advance(15)                                                          # `old` expired, `fresh` alive
        st = self.run_cleanup()
        self.assertEqual((st["conversations"], st["documents"]), (1, 0))
        self.assertEqual(len(convs.get(self.db, self.a.id, fresh)["messages"]), 2)
        self.assertTrue(docs.retrieve(self.db, self.emb, d, "capital", 1))

    def test_unattached_documents_removed_only_after_grace_period(self):
        d = self.doc()
        self.assertEqual(self.run_cleanup()["documents"], 0)
        self.clock.advance(120 + 1)
        self.assertEqual(self.run_cleanup()["documents"], 1); self.assertEqual(self.count("chunks"), 0)

    def test_partial_failure_is_retried_and_orphan_files_are_swept(self):
        pdf = make_pdf([["Retry semantics matter when file deletion fails halfway."]])
        d = docs.ingest_pdf(self.db, self.s, self.emb, self.a.id, "r.pdf", pdf)["id"]; cid = self.conv(doc=d)
        self.clock.advance(LIFE + 1)
        with mock.patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            st = self.run_cleanup()
        self.assertEqual(st["documents"], 1); self.assertEqual(st["files"], 0)                       # rows gone, file still queued
        self.assertEqual(self.count("pending_file_deletes"), 1); self.assertEqual(len(self.files()), 1)
        with self.assertRaises(NotFound): docs.read_pdf_bytes(self.db, self.s, self.a.id, d)          # inaccessible although the file lingers
        st = self.run_cleanup()
        self.assertEqual(st["files"], 1); self.assertEqual(self.files(), []); self.assertEqual(self.count("pending_file_deletes"), 0)
        (files_dir(self.s) / ("f" * 32 + ".pdf")).write_bytes(b"%PDF- stray")                         # crash leftover with no row
        self.assertEqual(self.run_cleanup()["orphan_files"], 1)

    def test_expired_sessions_without_data_are_removed_but_not_with_data(self):
        cid = self.conv(self.b)
        self.clock.advance(LIFE - 5); self.ask(cid, sess=self.b)                                      # B is active, A is idle
        self.clock.advance(10)
        st = self.run_cleanup()
        self.assertEqual(st["sessions"], 1)                                                           # only A (idle, owns nothing)
        self.assertIsNone(self.db.conn().execute("SELECT id FROM sessions WHERE id=?", (self.a.id,)).fetchone())
        self.assertIsNotNone(self.db.conn().execute("SELECT id FROM sessions WHERE id=?", (self.b.id,)).fetchone())
        self.assertEqual(len(convs.get(self.db, self.b.id, cid)["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
