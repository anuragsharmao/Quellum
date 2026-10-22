#!/usr/bin/env python3
"""Run ONCE on the machine that has the research data. Extracts everything the trained models need at inference time but that is not in the
.h5 files: vocabulary, TRAIN-fitted IDF tables, TRAIN standardisation statistics, the validation-selected ensemble threshold.
It uses the research code itself (model6.prepare / split_val / attach_sbert / finalize), so the numbers are the ones training used.

  python scripts/export_artifacts.py --research_dir research --results path/to/results.json --vocab path/to/vocab.json \
      --data_dir data/ragtruth --cache_dir cache_v2 --nli_feats nli_cache/nli_feats.npz
"""
import argparse, argparse as ap_, collections, json, math, os, shutil, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEYS = ["emb", "hidden", "dropout", "emb_drop", "ablate", "ctx", "kd_weight", "nli_dim", "topk", "max_ans", "max_ev", "no_features",
        "sent_score", "resp_agg", "feat_groups", "sbert_model", "seeds", "xf_names"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research_dir", default=str(ROOT / "research"))
    ap.add_argument("--results", required=True)
    ap.add_argument("--vocab", required=True, help="vocab.json written by the training run (out_dir)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True, help="the cache_v2 folder (prep_v7_*.pkl, sbert_cache.pkl)")
    ap.add_argument("--nli_feats", required=True)
    ap.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-xsmall", help="MUST be the model that produced nli_feats.npz")
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    from app import research
    from app.services import preprocessing as P
    M, XF, NF = research.load(args.research_dir)
    R = json.load(open(args.results, encoding="utf8"))
    A = R["args"]
    ens = [k for k in R["results"] if k.endswith(f"ENSEMBLE x{A['seeds']}")]
    assert len(ens) == 1, f"expected exactly one plain ensemble row, found {ens}"
    ov = R["results"][ens[0]]["overall"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    vocab = json.load(open(args.vocab, encoding="utf8"))
    shutil.copyfile(args.vocab, out / "vocab.json")

    ns = ap_.Namespace(**A); ns.data_dir, ns.cache_dir = args.data_dir, args.cache_dir
    ns.sbert_cache = os.path.join(args.cache_dir, "sbert_cache.pkl")

    # ---- raw data
    src = {}
    for l in open(os.path.join(args.data_dir, "source_info.jsonl"), encoding="utf8"):
        d = json.loads(l); src[d["source_id"]] = d
    resp = [json.loads(l) for l in open(os.path.join(args.data_dir, "response.jsonl"), encoding="utf8")]
    train = [r for r in resp if r["split"] == "train"]

    # ---- chunk-level IDF (model6.prepare): TRAIN sources minus the grouped validation split
    val_sids = M.split_val(train, src, A["split"])
    fit = set(r["source_id"] for r in train if r["source_id"] not in val_sids)
    df, N = collections.Counter(), 0
    for sid in fit:
        ch = P.Source(src[sid]["source_info"], M, XF).chunks
        N += len(ch)
        for c in ch:
            df.update(set(c))
    idf = {w: math.log((N + 1) / (c + 1)) for w, c in df.items()}
    json.dump(dict(idf=idf, n_chunks=N, n_sources=len(fit)), open(out / "idf.json", "w", encoding="utf8"))
    print(f"chunk IDF: {len(idf)} words from {len(fit)} sources / {N} chunks")

    # ---- passage-level IDF for the NLI retrieval (nli_features.py): ALL official-train sources
    tsids = set(r["source_id"] for r in train)
    ndf, nN = collections.Counter(), 0
    for sid in tsids:
        for p in NF.make_passages(src[sid]["source_info"], 90):
            nN += 1; ndf.update(set(w.lower() for w in NF.WORD.findall(p)))
    json.dump(dict(idf={w: math.log((nN + 1) / (c + 1)) for w, c in ndf.items()}, unk=math.log(nN + 1)), open(out / "nli_idf.json", "w", encoding="utf8"))
    print(f"NLI passage IDF: {len(ndf)} words, {nN} passages")

    # ---- standardisation statistics (attach_nli / merge_into_nli): TRAIN sentences, after SBERT + finalize
    data = M.prepare(ns)
    groups = XF.parse_groups(A["feat_groups"])
    segs = XF.load_source_segments(args.data_dir, M.flatten, M.split_sents)
    assert XF.attach_sbert(data, segs, A["sbert_model"], ns.sbert_cache), "sentence-transformers is required"
    XF.finalize(data, groups)
    z = np.load(args.nli_feats)
    idx = {k: i for i, k in enumerate(z["keys"].tolist())}; X = z["X"].astype(np.float32)
    tr = np.stack([X[idx[s["key"]]] if s["key"] in idx else np.zeros(X.shape[1], np.float32) for s in data["train"][0]])
    xf = np.stack([s["xf"] for s in data["train"][0]])
    np.savez(out / "norm.npz", nli_mu=tr.mean(0), nli_sd=tr.std(0) + 1e-6, xf_mu=xf.mean(0), xf_sd=xf.std(0) + 1e-6)
    print(f"normalisation from {len(tr)} train sentences; NLI dim {tr.shape[1]}, sentence-feature dim {xf.shape[1]}")

    cfg = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), source_results=os.path.basename(args.results), n_vocab=len(vocab),
        model_args={k: A[k] for k in KEYS},
        ensemble=dict(name=ens[0], size=A["seeds"], agg=A["resp_agg"], thr=ov["thr"],
                      reported={k: ov[k] for k in ("f1", "precision", "recall", "auroc", "auprc")}),
        nli=dict(model=args.nli_model, topk=3, max_words=90, max_len=256, bs=32),
        sbert=dict(model=A["sbert_model"]), xf_names=A["xf_names"])
    json.dump(cfg, open(out / "config.json", "w", encoding="utf8"), indent=1)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
