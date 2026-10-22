#!/usr/bin/env python3
"""Does the backend pipeline reproduce the research pipeline on RAGTruth TEST responses?  Three independent checks, in order:

 1. features : backend preprocessing vs the cached prep_v7_*.pkl (tokens, 14 token features, evidence, per-sentence vector)  -> exact
 2. nli      : live NLI features vs nli_cache/nli_feats.npz (confirms the cross-encoder name + retrieval logic)               -> ~1e-2 (float16 cache)
 3. logits   : the five backend models vs predictions_seed*.npz sentence logits, plus response decisions of the ensemble
               --nli cache  feeds the cached NLI rows (isolates models/features from the live NLI model);  --nli live uses the live model.

  python scripts/parity_test.py --data_dir data/ragtruth --cache_dir cache_v2 --predictions_dir results_v6_... --nli_feats nli_cache/nli_feats.npz --n 40
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--predictions_dir", required=True); ap.add_argument("--nli_feats", required=True)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--nli", choices=["cache", "live"], default="cache")
    ap.add_argument("--skip_nli_check", action="store_true")
    a = ap.parse_args()

    from app.config import get_settings
    from app.services import ensemble as E, preprocessing as P
    from app.services.detector import Detector
    det = Detector(get_settings()); det.load()
    M, XF = det.M, det.XF
    R = json.load(open(ROOT / "artifacts" / "config.json")); A = R["model_args"]

    import argparse as ap_
    full = json.load(open(next(Path(a.predictions_dir).glob("results.json"))))["args"]
    ns = ap_.Namespace(**full); ns.data_dir, ns.cache_dir = a.data_dir, a.cache_dir
    data = M.prepare(ns)
    te_sents, te_R = data["test"]
    by_key = {s["key"]: s for s in te_sents}

    src = {}
    for l in open(os.path.join(a.data_dir, "source_info.jsonl"), encoding="utf8"):
        d = json.loads(l); src[d["source_id"]] = d
    resp = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.data_dir, "response.jsonl"), encoding="utf8"))}
    rng = np.random.default_rng(0)
    ids = [r["id"] for r in te_R]
    pick = [ids[i] for i in sorted(rng.choice(len(ids), min(a.n, len(ids)), replace=False))]

    z = np.load(a.nli_feats); nli_idx = {k: i for i, k in enumerate(z["keys"].tolist())}; nli_X = z["X"]
    preds = [np.load(Path(a.predictions_dir) / f"predictions_seed{full['seed'] + k}.npz") for k in range(A["seeds"])]
    pid = preds[0]["test_resp_id"].tolist(); prid = preds[0]["test_sent_rid"]

    f_bad = 0; f_tot = 0; nli_diff = []; lg_diff = []; agree = 0; agree_tot = 0
    thr = det.cfg["ensemble"]["thr"]
    for rid in pick:
        r = resp[rid]; si = src[r["source_id"]]["source_info"]
        S = P.Source(si, M, XF)
        sents = P.featurize_answer(S, r["response"], det.idf, M, XF, A["topk"], A["max_ans"], A["max_ev"])
        # ---- 1. features
        for s in sents:
            ref = by_key[f"{rid}|{s['idx']}"]; f_tot += 1
            ok = (s["low"] == ref["low"] and s["ev"] == ref["ev"] and np.allclose(s["F"], ref["F"], atol=1e-6)
                  and np.allclose(s["evF"], ref["evF"]) and np.allclose(s["xf"][:9], ref["xf"][:9], atol=1e-5)
                  and np.allclose(s["xf"][11:], ref["xf"][11:], atol=1e-5))
            f_bad += (not ok)
        sims = det.sbert.sims(S.segments, [s["text"] for s in sents])
        cached = np.stack([nli_X[nli_idx[f"{rid}|{s['idx']}"]] for s in sents])
        # ---- 2. NLI
        if not a.skip_nli_check:
            live = det.nli.features(si, [s["text"] for s in sents])
            nli_diff.append(np.abs(live - cached).max())
        nli_raw = cached if a.nli == "cache" else det.nli.features(si, [s["text"] for s in sents])
        P.attach_vectors(sents, sims, nli_raw, det.norm, det.keep)
        # ---- 3. logits
        per = np.stack([np.asarray(M.predict(net, sents, det.w2i, det.mode, bs=256, sent_score=det.sent_score)[1]) for net in det.nets])
        ref_rows = np.where(prid == pid.index(rid))[0]
        ref_per = np.stack([p["test_sent_logit"][ref_rows] for p in preds])
        assert ref_per.shape == per.shape, f"{rid}: sentence count differs {per.shape} vs {ref_per.shape}"
        lg_diff.append(np.abs(per - ref_per).max())
        a1 = E.noisy_or_logit(E.mean_logits(per)) >= thr
        a2 = E.noisy_or_logit(E.mean_logits(ref_per)) >= thr
        agree += int(a1 == a2); agree_tot += 1

    print(f"responses checked: {len(pick)}   sentences: {f_tot}")
    print(f"[1 features] sentences with ANY mismatch vs prep cache: {f_bad}/{f_tot}")
    if nli_diff:
        print(f"[2 nli     ] max |live - cached| NLI feature, per response: median {np.median(nli_diff):.4f}  worst {max(nli_diff):.4f}")
    print(f"[3 logits  ] ({a.nli} NLI) max |logit diff| per response: median {np.median(lg_diff):.5f}  worst {max(lg_diff):.5f}")
    print(f"[3 decision] ensemble flag agrees with the research run on {agree}/{agree_tot} responses")


if __name__ == "__main__":
    main()
