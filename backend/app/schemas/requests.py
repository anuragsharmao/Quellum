from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from ..config import MAX_ANSWER_CHARS, MAX_CONTEXT_CHARS, MAX_QUESTION_CHARS


class DetectRequest(BaseModel):
    question: str = Field("", max_length=MAX_QUESTION_CHARS, description="Optional. Placed in the source like RAGTruth QA sources.")
    context: str = Field(..., max_length=MAX_CONTEXT_CHARS, description="Evidence the answer is checked against.")
    answer: str = Field(..., max_length=MAX_ANSWER_CHARS, description="The answer to verify.")

    @field_validator("context", "answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class SentenceResult(BaseModel):
    index: int
    text: str
    start: int
    end: int
    truncated: bool = Field(..., description="Only the first 64 tokens of a sentence are seen by the model (training setting).")
    logit: float
    score: float = Field(..., description="sigmoid(mean ensemble logit). A model score, NOT a calibrated probability.")
    label: Optional[str] = Field(None, description="Present only when a validation-derived sentence threshold is installed.")
    seed_logits: Optional[List[float]] = None
    features: Optional[dict] = None


class Summary(BaseModel):
    overall_score: float = Field(..., description="noisy-OR response logit (the quantity the saved threshold is applied to)")
    overall_risk: float = Field(..., description="same quantity on a 0..1 scale: 1 - prod(1 - sigmoid(logit_i)). Uncalibrated.")
    decision_threshold: float
    decision_threshold_risk: float
    flagged: bool
    risk_label: str
    total_sentences: int
    potentially_hallucinated: Optional[int] = None
    supported: Optional[int] = None


class DetectResponse(BaseModel):
    status: str = "success"
    analysis_id: str
    summary: Summary
    sentences: List[SentenceResult]
    model_info: dict
    timing_ms: dict
    per_seed_response_scores: Optional[List[float]] = None
    seed_ids: Optional[List[int]] = None
