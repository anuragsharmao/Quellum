#!/usr/bin/env python3
"""Checks that app.services.ensemble reproduces the research ensemble row (F1 0.741 / AUROC 0.873 / AUPRC 0.805, thr 0.5507) from the saved
predictions_seed*.npz, and (optionally) derives a validation-only SENTENCE threshold into artifacts/sentence_threshold.json.
Test data is used only to REPORT, never to choose a threshold.

  python scripts/verify_ensemble.py --predictions_dir results_v6_no_d2t_no_synth_nli --results results.json [--write_sentence_threshold]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resp_scores(logits, rids, n):
    from app.services import ensemble as E
    out = np.zeros(n)
    for i in range(n):
        m = rids == i
        out[i] = E.noisy_or_logit(logits[m]) if m.any() else logits.min() - 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--research_dir", default=str(ROOT / "research"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--write_sentence_threshold", action="store_true")
    a = ap.parse_args()
    from app import research
    M, _, _ = research.load(a.research_dir)
    R = json.load(open(a.results, encoding="utf8")); seeds = R["args"]["seeds"]
    P = [np.load(Path(a.predictions_dir) / f"predictions_seed{R['args']['seed'] + k}.npz") for k in range(seeds)]
    vl, tl = np.mean([p["val_sent_logit"] for p in P], 0), np.mean([p["test_sent_logit"] for p in P], 0)
    yv, yt = P[0]["val_y"], P[0]["test_y"]
    sv, st = resp_scores(vl, P[0]["val_sent_rid"], len(yv)), resp_scores(tl, P[0]["test_sent_rid"], len(yt))
    thr = M.best_thr(yv, sv)
    m = M.metrics(yt, st, thr)
    ref = [v for k, v in R["results"].items() if k.endswith(f"ENSEMBLE x{seeds}")][0]["overall"]
    print(f"threshold  recomputed {thr:.4f}   results.json {ref['thr']:.4f}")
    for k in ("f1", "auroc", "auprc"):
        print(f"{k:6s} recomputed {m[k]:.4f}   results.json {ref[k]:.4f}   diff {abs(m[k] - ref[k]):.2e}")
    ok = abs(thr - ref["thr"]) < 1e-6 and all(abs(m[k] - ref[k]) < 1e-6 for k in ("f1", "auroc", "auprc"))
    print("ENSEMBLE REPRODUCED" if ok else "MISMATCH: do not trust the backend ensemble until this is explained")

    # sentence level (NOT part of results.json): threshold on VALIDATION sentences only
    ys_v, ys_t = P[0]["val_sent_y"], P[0]["test_sent_y"]
    sthr = M.best_thr(ys_v, vl)
    mv, mt = M.metrics(ys_v, vl, sthr), M.metrics(ys_t, tl, sthr)
    print(f"sentence-level: val-selected logit threshold {sthr:.4f}; val F1 {mv['f1']:.3f}; test F1 {mt['f1']:.3f} "
          f"(P {mt['precision']:.3f} R {mt['recall']:.3f}, AUROC {mt['auroc']:.3f}, AUPRC {mt['auprc']:.3f})")
    if a.write_sentence_threshold:
        json.dump(dict(thr_logit=sthr, val_f1=mv["f1"], test_f1=mt["f1"], test_precision=mt["precision"], test_recall=mt["recall"],
                       note="sentence-level threshold chosen on validation sentences only; not part of the 0.741 response-level result"),
                  open(Path(a.artifacts) / "sentence_threshold.json", "w"), indent=1)
        print("wrote sentence_threshold.json")


if __name__ == "__main__":
    main()
