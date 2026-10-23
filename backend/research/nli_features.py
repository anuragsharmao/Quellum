#!/usr/bin/env python3
"""
Frozen-NLI features for RAGTruth answer sentences (CPU, cached, resumable). Runs ONCE; the detector then reads the .npz.

For every response sentence (same split_sents segmentation + index as rnn_ragtruth_v2.prepare, key = "<response id>|<sentence idx>"):
  1. the source is cut into ~90-word passages of consecutive sentences (1-sentence overlap),
  2. the top-K passages are retrieved by IDF overlap (IDF fit on TRAIN sources only, no labels used),
  3. a frozen NLI cross-encoder scores (premise = passage, hypothesis = answer sentence),
  4. 8 features per sentence are saved:
     ent_max, ent_mean, contra_max, contra_mean, neutral_min, ent_top1, contra_top1, idf_coverage_top1

Setup:  pip install torch transformers sentencepiece
Smoke (measures speed):  python nli_features.py --limit 200
Full:                    python nli_features.py
Resumes automatically if interrupted (pair cache next to the output file).
"""
import argparse, hashlib, json, math, os, pickle, random, re, time
import numpy as np

TOK = re.compile(r"[\w\u0900-\u097F]+|[^\w\s]")
WORD = re.compile(r"[\w\u0900-\u097F]+")
FEATS = ["ent_max", "ent_mean", "contra_max", "contra_mean", "neutral_min", "ent_top1", "contra_top1", "idf_cov_top1"]


# ---- identical to rnn_ragtruth_v2.py (kept here so this script does not import TensorFlow next to torch)
def split_sents(text):
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        seg = text[start:m.start()]
        if seg.strip():
            out.append((start, m.start(), seg))
        start = m.end()
    seg = text[start:]
    if seg.strip():
        out.append((start, len(text), seg))
    return out


def flatten(o, prefix=""):
    if isinstance(o, dict):
        lines = []
        for k, v in o.items():
            lines += flatten(v, k if not prefix else f"{prefix} {k}")
        return lines
    if isinstance(o, list):
        return [x for v in o for x in flatten(v, prefix)]
    return [f"{prefix}: {o}" if prefix else str(o)]
# ----


def make_passages(source_info, max_words=90):
    sents = []
    for line in flatten(source_info):
        sents += [seg.strip() for _, _, seg in split_sents(line)]
    out, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur = cur[-1:] if len(cur[-1].split()) < max_words // 2 else []
            n = sum(len(c.split()) for c in cur)
        cur.append(s); n += w
    if cur:
        out.append(" ".join(cur))
    return out or [""]


def pair_key(prem, hyp):
    return hashlib.md5((prem + "\x00" + hyp).encode("utf8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/ragtruth")
    ap.add_argument("--out", default="nli_cache/nli_feats.npz")
    ap.add_argument("--model", default="cross-encoder/nli-deberta-v3-xsmall",
                    help="any HF NLI sequence-classification model; labels are read from its config. "
                         "Stronger/slower option: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max_words", type=int, default=90, help="passage size in words")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only this many random responses")
    args = ap.parse_args()

    src = {}
    for l in open(os.path.join(args.data_dir, "source_info.jsonl"), encoding="utf8"):
        d = json.loads(l); src[d["source_id"]] = d
    resp = [json.loads(l) for l in open(os.path.join(args.data_dir, "response.jsonl"), encoding="utf8")]
    if args.limit:
        resp = random.Random(0).sample(resp, min(args.limit, len(resp)))

    # passages for every needed source; IDF from TRAIN sources only
    need = sorted(set(r["source_id"] for r in resp))
    P = {sid: make_passages(src[sid]["source_info"], args.max_words) for sid in need}
    Pw = {sid: [set(w.lower() for w in WORD.findall(p)) for p in P[sid]] for sid in need}
    train_sids = set(r["source_id"] for r in resp if r["split"] == "train")
    df, N = {}, 0
    for sid in train_sids:
        for pw in Pw[sid]:
            N += 1
            for w in pw:
                df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N + 1) / (c + 1)) for w, c in df.items()}
    unk = math.log(N + 1) if N else 1.0
    print(f"{len(resp)} responses, {len(need)} sources, {sum(len(v) for v in P.values())} passages, IDF from {len(train_sids)} train sources")

    # jobs: per sentence the retrieved passages
    jobs, pairs = [], {}
    for r in resp:
        sid = r["source_id"]
        for si, (_, _, seg) in enumerate(split_sents(r["response"])):
            if not TOK.findall(seg):
                continue
            q = set(w.lower() for w in WORD.findall(seg))
            qmass = sum(idf.get(w, unk) for w in q) + 1e-6
            sc = [sum(idf.get(w, unk) for w in q & pw) / math.sqrt(len(pw) + 1) for pw in Pw[sid]]
            order = np.argsort(sc)[::-1][:args.topk]
            hyp = " ".join(seg.split()[:100])
            keys, cov = [], 0.0
            for j, oi in enumerate(order):
                k = pair_key(P[sid][oi], hyp)
                keys.append(k); pairs[k] = (P[sid][oi], hyp)
                if j == 0:
                    cov = sum(idf.get(w, unk) for w in q & Pw[sid][oi]) / qmass
            jobs.append((f"{r['id']}|{si}", keys, cov))
    print(f"{len(jobs)} sentences, {len(pairs)} unique (passage, sentence) pairs")

    # NLI (resumable)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cache_path = args.out + ".pairs.pkl"
    done = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
    todo = [k for k in pairs if k not in done]
    print(f"{len(done)} pairs cached, {len(todo)} to score with {args.model}")
    if todo:
        tokz = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
        names = {int(i): str(l).lower() for i, l in model.config.id2label.items()}
        ci = [i for i, n in names.items() if n.startswith("contra")][0]
        ei = [i for i, n in names.items() if n.startswith("entail")][0]
        ni = [i for i, n in names.items() if n.startswith("neutral")][0]
        print(f"label map: {names}")
        todo.sort(key=lambda k: len(pairs[k][0]) + len(pairs[k][1]))
        t0, last_save = time.time(), 0
        with torch.inference_mode():
            for b in range(0, len(todo), args.bs):
                ks = todo[b:b + args.bs]
                enc = tokz([pairs[k][0] for k in ks], [pairs[k][1] for k in ks], truncation="only_first",
                           max_length=args.max_len, padding=True, return_tensors="pt")
                pr = torch.softmax(model(**enc).logits, -1).numpy()
                for k, p in zip(ks, pr):
                    done[k] = np.array([p[ci], p[ei], p[ni]], np.float16)
                nd = b + len(ks)
                if nd - last_save >= 5000 or nd == len(todo):
                    pickle.dump(done, open(cache_path, "wb")); last_save = nd
                    rate = nd / (time.time() - t0)
                    print(f"  {nd}/{len(todo)} pairs  {rate:.1f} pairs/s  ETA {(len(todo) - nd) / rate / 60:.1f} min", flush=True)

    # features
    X = np.zeros((len(jobs), len(FEATS)), np.float32)
    for i, (_, keys, cov) in enumerate(jobs):
        p = np.stack([done[k] for k in keys]).astype(np.float32)          # [k, (contra, ent, neutral)]
        X[i] = [p[:, 1].max(), p[:, 1].mean(), p[:, 0].max(), p[:, 0].mean(), p[:, 2].min(), p[0, 1], p[0, 0], cov]
    np.savez_compressed(args.out, keys=np.array([j[0] for j in jobs], dtype="U"), X=X, feature_names=np.array(FEATS))
    print(f"saved {args.out}: {X.shape}; feature means {dict(zip(FEATS, np.round(X.mean(0), 3).tolist()))}")


if __name__ == "__main__":
    main()