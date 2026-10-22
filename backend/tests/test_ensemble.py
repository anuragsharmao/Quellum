"""Pure-numpy tests of app.services.ensemble. Run:  python tests/test_ensemble.py   (or pytest)"""
import math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import ensemble as E


def test_noisy_or_matches_definition():
    rng = np.random.default_rng(0)
    for _ in range(200):
        l = rng.normal(0, 3, rng.integers(1, 12))
        p = 1 / (1 + np.exp(-l))
        want = 1 - np.prod(1 - p)
        assert abs(E.noisy_or_risk(l) - want) < 1e-9
        assert abs(1 / (1 + math.exp(-E.noisy_or_logit(l))) - want) < 1e-9


def test_matches_model6_agg_scores_formula():
    # model6.agg_scores(..., "noisyor"): t = sum softplus(l); t + log(-expm1(-t))
    l = np.array([-3.0, 0.2, -1.1])
    t = np.logaddexp(0.0, l).sum()
    assert abs(E.noisy_or_logit(l) - (t + np.log(-np.expm1(-t)))) < 1e-12


def test_mean_of_logits_not_probabilities():
    per = np.array([[4.0, -2.0], [-4.0, -2.0]])
    assert np.allclose(E.mean_logits(per), [0.0, -2.0])


def test_more_flagged_sentences_never_lower_risk():
    base = E.noisy_or_logit([-2.0])
    assert E.noisy_or_logit([-2.0, -2.0]) > base


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
