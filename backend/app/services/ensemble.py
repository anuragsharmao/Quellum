"""Five-seed ensemble aggregation, exactly as in model6.py's evaluation:
  1. ensemble sentence score = MEAN of the seeds' sentence-head logits         (np.mean([r["tlog"] for r in runs], 0))
  2. response score = noisy-OR of the sentence probabilities, as a logit        (agg_scores(..., "noisyor"))
         t = sum_i softplus(l_i);  score = t + log(1 - exp(-t)) = logit(1 - prod_i (1 - sigmoid(l_i)))
  3. the response is flagged when score >= thr; thr was chosen on VALIDATION only (results.json, ensemble row).
No probability averaging, no stacking, no per-task thresholds (those variants scored lower and are not used)."""
from __future__ import annotations

import math

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def mean_logits(per_model) -> np.ndarray:
    """per_model: [n_models, n_sentences] -> [n_sentences]."""
    a = np.asarray(per_model, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] == 0:
        raise ValueError("per_model must be [n_models, n_sentences] with at least one sentence")
    return a.mean(0)


def _t(logits) -> float:
    return max(float(np.logaddexp(0.0, np.asarray(logits, dtype=np.float64)).sum()), 1e-6)


def noisy_or_logit(logits) -> float:
    """Identical to model6.agg_scores(..., 'noisyor') for ONE response."""
    t = _t(logits)
    return t + math.log(-math.expm1(-t))


def noisy_or_risk(logits) -> float:
    """1 - prod(1 - sigmoid(l_i)): the same quantity as noisy_or_logit on the 0..1 scale. Uncalibrated."""
    return -math.expm1(-_t(logits))
