"""One question -> retrieve -> generate -> hallucination analysis -> persist. Pure functions with injected collaborators (embedder,
generator, analyzer) so the whole flow is testable without a model. Nothing is persisted unless an answer was really produced."""
from __future__ import annotations

import logging
import time

from ..config import MAX_QUESTION_CHARS, Settings
from . import conversations as convs
from . import documents as docs
from .errors import GenerationNotConfigured, InvalidInput

log = logging.getLogger("quellum.chat")
EXCERPT_CHARS = 700


def to_hallucination_result(res: dict) -> dict:
    s = res["summary"]
    return dict(status="ok", flagged=s["flagged"], riskLabel="Elevated hallucination risk" if s["flagged"] else "Lower hallucination risk",
                risk=s["overall_risk"], score=s["overall_score"], threshold=s["decision_threshold"],
                totalSentences=s["total_sentences"], flaggedSentences=s.get("potentially_hallucinated"),
                sentences=[dict(text=x["text"], score=x["score"], label=x.get("label"), truncated=x.get("truncated", False)) for x in res["sentences"]],
                note="Model scores from the Quellum detector. They are not calibrated probabilities and do not prove a statement true or false.")


def analyze_answer(analyzer, question: str, context: str, answer: str) -> dict:
    if analyzer is None:
        return dict(status="unavailable", reason="The hallucination detector is not loaded on this server.")
    try:
        return to_hallucination_result(analyzer(question, context, answer))
    except ValueError as e:                                   # preprocessing.InputError (answer/context cannot be analysed)
        return dict(status="skipped", reason=str(e))
    except Exception:  # noqa: BLE001
        log.exception("hallucination analysis failed")
        return dict(status="error", reason="The hallucination analysis failed for this answer.")


def send_message(db, settings: Settings, session_id: str, cid: str, content: str, *, embedder, generator, analyzer) -> list:
    content = (content or "").strip()
    if not content:
        raise InvalidInput("Please enter a question.")
    if len(content) > MAX_QUESTION_CHARS:
        raise InvalidInput(f"Questions are limited to {MAX_QUESTION_CHARS} characters.")
    info = convs.require_live(db, session_id, cid)                        # ownership + expiry, before any expensive work
    if generator is None:
        raise GenerationNotConfigured("Answer generation is not configured on this server.")
    passages = docs.retrieve(db, embedder, info["documentId"], content, settings.top_k)
    if not passages:
        raise InvalidInput("This document has no searchable text.")
    t0 = time.perf_counter()
    answer = generator.generate(content, [p["text"] for p in passages])
    t1 = time.perf_counter()
    context = "\n\n".join(p["text"] for p in sorted(passages, key=lambda p: p["idx"]))
    analysis = analyze_answer(analyzer, content, context, answer)
    log.info("chat timing: generation %.1fs, hallucination analysis %.1fs (%s)", t1 - t0, time.perf_counter() - t1, analysis.get("status"))
    sources = [dict(page=p["page"] if info["sourceType"] == "pdf" else None,
                    section=f"Page {p['page']}" if info["sourceType"] == "pdf" else "Pasted text",
                    excerpt=p["text"][:EXCERPT_CHARS], score=round(p["score"], 4)) for p in passages]
    return convs.append_exchange(db, settings, session_id, cid, content,
                                 dict(content=answer, sources=sources, hallucinationResult=analysis))
