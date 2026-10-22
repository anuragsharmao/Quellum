from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas.requests import DetectRequest, DetectResponse
from ..services import preprocessing as P
from ..services.sessions import SessionCtx
from .session import require_session

log = logging.getLogger("quellum.api")
router = APIRouter(prefix="/api/v1")


def _detector(request: Request):
    det = getattr(request.app.state, "detector", None)
    if det is None or not det.ready:
        raise HTTPException(status_code=503, detail="The detector is not loaded. See /health.")
    return det


def _short_info(det) -> dict:
    return dict(name="Quellum BiGRU-attn ensemble", ensemble_size=len(det.nets), aggregation="mean sentence logit -> noisy-OR",
                model_ids=list(det.seed_ids), nli_features=True)


def _run(request: Request, body: DetectRequest, detail: bool, ctx: SessionCtx) -> dict:
    det = _detector(request)
    st = request.app.state
    st.limiter.check(f"msg:{ctx.id}", st.settings.messages_per_10min, 600)       # same budget as chat: inference is the expensive part
    try:
        res = det.analyze(body.question, body.context, body.answer, detail=detail)
    except P.InputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:  # noqa: BLE001 - never leak internals
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail="Inference failed. Details were written to the server log.")
    flagged = res["summary"]["flagged"]
    res["summary"]["risk_label"] = "Elevated hallucination risk" if flagged else "Lower hallucination risk"
    res["status"] = "success"
    res["model_info"] = _short_info(det)
    return res


@router.get("/model-info")
def model_info(request: Request):
    det = _detector(request)
    c = det.cfg
    return dict(
        name="Quellum BiGRU-attn ensemble (RAGTruth)", ensemble_size=len(det.nets), model_ids=list(det.seed_ids),
        architecture="BiGRU answer/evidence encoders + cross-attention + sentence-context BiGRU (model6.Net), frozen-NLI and grounding features",
        aggregation=c["ensemble"]["agg"], decision_threshold=c["ensemble"]["thr"],
        nli_model=c["nli"]["model"], sbert_model=c["sbert"]["model"], feature_groups=det.a["feat_groups"],
        research_evaluation=dict(c["ensemble"]["reported"], dataset="RAGTruth official test split, response level",
                                 note="original five-model research run; not re-evaluated for the configured model ids"),
        capabilities=dict(sentence_scores=True, response_level_decision=True, sentence_labels=det.sentence_thr is not None,
                          token_level_spans=False, calibrated_probabilities=False, retrieval=False, answer_generation=False),
        limitations=[
            "Trained on RAGTruth (QA, Data2txt, Summary). Behaviour on other domains has not been measured.",
            "Scores are model scores, not calibrated probabilities, and do not prove a statement true or false.",
            "Only the first 64 tokens of each sentence and the 3 best-matching 40-token evidence chunks are seen by the model.",
            "The stored threshold and reported benchmark results were validated for the original five-model run, not this model-id selection.",
        ])


@router.post("/detect", response_model=DetectResponse, response_model_exclude_none=True)
def detect(body: DetectRequest, request: Request, ctx: SessionCtx = Depends(require_session)):
    return _run(request, body, False, ctx)


@router.post("/analyze", response_model=DetectResponse, response_model_exclude_none=True)
def analyze(body: DetectRequest, request: Request, ctx: SessionCtx = Depends(require_session)):
    return _run(request, body, True, ctx)
