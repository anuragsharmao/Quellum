#!/usr/bin/env python3
"""
[v6] = v5 (EMA, AdamW, label smoothing, span loss, seed ensemble) + the changes below. Every NEW structural option is its own flag and is OFF
     unless you pass it (or pass --v6 = all of them) so each one can be ablated:
     --resp_loss W        response-level loss on a pooled (noisy-OR / log-sum-exp) response logit; batches are built from WHOLE responses
     --ctx                sentence-sequence BiGRU: every sentence sees the other sentences of its response
     --synth_ratio R      synthetic hard negatives (number / entity / negation corruption of faithful TRAIN responses, see synth.py)
     --topk_d2t/--max_ev_d2t  larger evidence budget + line-packed chunks for Data2txt (structured JSON) sources
     --nli_feats f.npz    per-sentence verifier features from verifier_features.py (a verifier NOT trained on RAGTruth labels)
     --kd_feats/--kd_col/--kd_weight   distil the verifier into the GRU (verifier is NOT an input -> not needed at deployment)
     second-stage rows (on by default; --no_stack / --no_task_thr): "+stack" = logistic regression over sentence-score statistics fitted on
     validation only (OOF threshold), "+task-thr" = per-task thresholds tuned on validation.  Test data is never used for fitting/selection.
     --resp_agg auto      response score from sentence logits: max (default) | noisyor | lse   (auto = pooling used by --resp_loss)
[v4] = v3 + extra grounding features from extra_feats.py (stemmed/normalised-number/date/n-gram/entity matches +
     pretrained sentence-embedding similarity). New flags: --feat_groups (default all; "none" reproduces v3), --sbert_model, --sbert_cache.

RAGTruth hallucination detector v3 = v2 + TOKEN-LEVEL SPAN SUPERVISION (BiGRU + cross-attention, CPU friendly).  Keep rnn_ragtruth.py / _v2.py as they are.

What changed vs the original (every model-changing factor is its OWN flag, default = original behaviour):
  MEASUREMENT ONLY (do not change the weight trajectory)
    --split grouped      content-hash grouped, task-stratified train/val split (default). --split original reproduces the old split.
    --eval_every N       validate every N optimizer steps (default 200; 0 = once per epoch). Patience counts evaluations.
    --select             checkpoint criterion: auprc (default) | f1 | loss. Best checkpoint per criterion is tracked simultaneously.
    predictions_seed*.npz  val/test sentence + response scores, labels, task, source ids, baselines, pos_weight  (input of analyze_predictions.py)
    history_seed*.json   full step-level training history;   vocab.json
    --cache_dir          caches retrieval/preprocessing (keyed by split/topk/max_ev/max_ans/flags)
  MODEL-CHANGING (change one at a time)
    --lr, --dropout, --emb, --hidden (as before)
    --ablate no_evidence     retrieved evidence removed (answer + whole-source exact-match features only; evidence-flag zeroed)
    --ablate no_cross_attn   evidence encoder + cross-attention removed (answer BiGRU only, evidence enters only via the in-evidence flag)
    --no_features            zero exact-match features (as before; combine with --ablate no_evidence for an ANSWER-ONLY model)
    --min_count K            vocabulary min frequency (default 2)
    --word_dropout P         replace answer/evidence tokens by <unk> with prob P during training only (default 0)
    --token_loss W           NEW: auxiliary per-token loss on the labelled hallucination SPANS (weight W; 0 = off = v2 behaviour).
                             A small Dense head on the post-comparison BiGRU outputs predicts, per answer token, 'inside a hallucinated span'.
                             Token labels come from the RAGTruth char spans (train labels only are used for training).
    --tok_pos_weight_cap C   cap on the token-level positive weight (default 10)
    --sent_score head|token_max   which logit is THE sentence score for selection/threshold/response aggregation:
                             the original sentence head (default) or the max over the token logits. Both are saved in predictions_seed*.npz.
Removed as decided: GloVe, L2, --compare, --use_task.   Removed plots 02-06 (regenerate from the saved predictions).

Run:   python rnn_ragtruth_v2.py --data_dir data/ragtruth --out_dir results_v2
Smoke: python rnn_ragtruth_v2.py --max_train 300 --max_test 120 --epochs 1 --eval_every 20
"""
import argparse, collections, hashlib, json, math, os, pickle, random, re, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import HistGradientBoostingClassifier
import extra_feats as XF
import synth as SY
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

TOK = re.compile(r"[\w\u0900-\u097F]+|[^\w\s]")
NUM = re.compile(r"^\d+$")
TASKS = ["QA", "Data2txt", "Summary"]
TYPES = ["Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info"]
REPORTED = {"GPT-4 prompt (reported)": 63.4, "Luna (reported)": 65.4,
            "LettuceDetect-base (reported)": 76.8, "LettuceDetect-large (reported)": 79.2}
MAXIDF, CHUNK = 12.0, 40
NF = 7 + XF.NF_TOK
# token-feature columns: 0 in-context, 1 in-evidence, 2 is-number, 3 number-not-in-context, 4 capitalized-not-in-context, 5 idf, 6 punct
EM_COLS = [0, 1, 3, 4] + XF.EM_EXTRA


def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)


# ----------------------------------------------------------------------------- data
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


def tok_feats(raw, ctxset, evset, idf):
    low = [t.lower() for t in raw]
    rows = []
    for i, (r, l) in enumerate(zip(raw, low)):
        punct = not (r[0].isalnum() or r[0] == "_" or "\u0900" <= r[0] <= "\u097f")
        inc = punct or l in ctxset
        isn = bool(NUM.match(r))
        isc = r[:1].isupper() and i > 0
        rows.append([inc, l in evset, isn, isn and not inc, isc and not inc,
                     min(idf.get(l, MAXIDF), MAXIDF) / MAXIDF, punct])
    return low, np.array(rows, dtype=np.float32)


def hand_feats(low, Fm, top_score, idf):
    content = Fm[:, 6] == 0
    n = max(int(content.sum()), 1)
    w = np.array([min(idf.get(t, MAXIDF), MAXIDF) for t in low]) * content
    inc = Fm[:, 0]
    return np.array([(inc * content).sum() / n, (w * inc).sum() / max(w.sum(), 1e-6),
                     Fm[:, 3].sum(), Fm[:, 4].sum(), len(low),
                     top_score, (Fm[:, 1] * content).sum() / n], dtype=np.float32)


def content_hash(src, sid):
    return hashlib.md5(json.dumps(src[sid]["source_info"], sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest()


def split_val(train, src, mode, frac=0.1):
    """Validation source ids carved out of the official train split.
    original: random sources (identical to rnn_ragtruth.py).  grouped: whole content-duplicate groups, stratified by task."""
    sids = sorted(set(r["source_id"] for r in train))
    if mode == "original":
        random.Random(0).shuffle(sids)
        return set(sids[:max(1, len(sids) // 10)])
    groups = collections.defaultdict(list)
    for s in sids:
        groups[content_hash(src, s)].append(s)
    by_task = collections.defaultdict(list)
    for g in groups.values():
        by_task[src[g[0]]["task_type"]].append(g)
    rng, val = random.Random(0), set()
    for t in sorted(by_task):
        gs = by_task[t]; rng.shuffle(gs)
        need, got = max(1, round(frac * sum(map(len, gs)))), 0
        for g in gs:
            if got >= need:
                break
            val.update(g); got += len(g)
    return val


def check_leakage(splits, src, strict_dups=False, grouped=False):
    """source_ids must be disjoint (assert). Identical source content under different ids across splits: warn, or assert (--strict_dups).
    With the grouped split, train/val content duplicates are asserted impossible; train/test duplicates cannot be fixed (official test)."""
    sids = {k: set(r["source_id"] for r in rs) for k, rs in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = sids[a] & sids[b]
        assert not inter, f"source_id leakage between {a} and {b}: {len(inter)} shared"
    where = collections.defaultdict(lambda: collections.defaultdict(set))
    for k, ss in sids.items():
        for sid in ss:
            where[content_hash(src, sid)][k].add(sid)
    cross = {h: v for h, v in where.items() if len(v) > 1}
    tv = [v for v in cross.values() if "val" in v and "train" in v]
    tt = [v for v in cross.values() if "test" in v]
    if grouped:
        assert not tv, "grouped split failed: train/val content duplicates remain"
    info = dict(n_sources={k: len(v) for k, v in sids.items()}, cross_split_duplicate_contents=len(cross),
                train_val_duplicate_groups=len(tv), test_duplicate_groups=len(tt),
                cross_groups=[{k: sorted(s) for k, s in v.items()} for v in cross.values()][:20])
    if cross:
        msg = f"{len(cross)} cross-split duplicate-content group(s): train/val={len(tv)}, involving test={len(tt)}"
        assert not strict_dups, "content leakage: " + msg
        print("[WARN] " + msg)
    else:
        print("leakage checks passed: source_ids disjoint, no cross-split duplicate content")
    return info


def prepare(args):
    key = hashlib.md5(json.dumps({k: getattr(args, k) for k in ["split", "topk", "max_ans", "max_ev", "drop_implicit",
                       "drop_bad_quality", "max_train", "max_test", "strict_dups", "synth_ratio", "synth_seed", "topk_d2t", "max_ev_d2t"]}, sort_keys=True).encode()).hexdigest()[:12]
    cpath = os.path.join(args.cache_dir, f"prep_v7_{key}.pkl") if args.cache_dir else None
    if cpath and os.path.exists(cpath):
        print(f"loading cached preprocessing {cpath}")
        return pickle.load(open(cpath, "rb"))
    if args.max_ev < args.topk * CHUNK:
        print(f"[WARN] max_ev={args.max_ev} < topk*{CHUNK}={args.topk * CHUNK}: evidence is cut after ranking; raise --max_ev together with --topk")
    src = {}
    for l in open(os.path.join(args.data_dir, "source_info.jsonl"), encoding="utf8"):
        d = json.loads(l); src[d["source_id"]] = d
    chunks, ctxsets, csets, xctx = {}, {}, {}, {}
    for sid, d in src.items():
        cs, segraw = [], []
        pack = d["task_type"] == "Data2txt" and args.topk_d2t > 0     # structured source: pack consecutive "key: value" lines into one chunk
        cur = []
        for line in flatten(d["source_info"]):
            for _, _, seg in split_sents(line):
                t = TOK.findall(seg.lower()); segraw.append(seg)
                if pack and len(t) <= CHUNK:
                    if cur and len(cur) + len(t) > CHUNK:
                        cs.append(cur); cur = []
                    cur = cur + t
                else:
                    if cur:
                        cs.append(cur); cur = []
                    cs += [t[i:i + CHUNK] for i in range(0, len(t), CHUNK) if t[i:i + CHUNK]]
        if cur:
            cs.append(cur)
        chunks[sid] = cs
        xctx[sid] = XF.build_ctx(segraw, TOK)
        csets[sid] = [set(c) for c in cs]
        ctxsets[sid] = set(t for c in cs for t in c)
    resp = [json.loads(l) for l in open(os.path.join(args.data_dir, "response.jsonl"), encoding="utf8")]
    if args.drop_bad_quality:
        resp = [r for r in resp if r.get("quality", "good") == "good"]   # NOTE: also filters the official test set
    for r in resp:
        if args.drop_implicit:
            r["labels"] = [l for l in r["labels"] if not l.get("implicit_true")]  # NOTE: also changes test labels
    train = [r for r in resp if r["split"] == "train"]
    test = [r for r in resp if r["split"] == "test"]
    if args.max_train:
        train = random.Random(1).sample(train, min(args.max_train, len(train)))
    if args.max_test:
        test = random.Random(2).sample(test, min(args.max_test, len(test)))
    val_sids = split_val(train, src, args.split)
    splits = {"train": [r for r in train if r["source_id"] not in val_sids],
              "val": [r for r in train if r["source_id"] in val_sids], "test": test}
    leak = check_leakage(splits, src, args.strict_dups, args.split == "grouped")
    if args.synth_ratio > 0:        # synthetic hard negatives: TRAIN sources only, never validation/test
        stxt = {}

        def src_text_of(sid):
            if sid not in stxt:
                stxt[sid] = " ".join(flatten(src[sid]["source_info"]))
            return stxt[sid]
        syn = SY.make_synthetic(splits["train"], src_text_of, args.synth_ratio, args.synth_seed)
        splits["train"] = splits["train"] + syn
        print(f"synthetic hard negatives added to TRAIN: {len(syn)} responses "
              f"({collections.Counter(x['labels'][0]['kind'] for x in syn)})")
    fit_sids = set(r["source_id"] for r in splits["train"])           # IDF fit on TRAIN sources only
    df = collections.Counter()
    for sid in fit_sids:
        for c in chunks[sid]:
            df.update(set(c))
    N = sum(len(chunks[s]) for s in fit_sids) or 1
    idf = {w: math.log((N + 1) / (c + 1)) for w, c in df.items()}
    data = {}
    for name, rs in splits.items():
        sents, R = [], []
        for ri, r in enumerate(rs):
            sid = r["source_id"]; task = src[sid]["task_type"]
            R.append(dict(id=r["id"], sid=sid, task=task, y=int(bool(r["labels"])), model=r["model"],
                          types=sorted(set(l["label_type"] for l in r["labels"]))))
            for si, (s0, s1, seg) in enumerate(split_sents(r["response"])):
                ms = list(TOK.finditer(seg))[:args.max_ans]
                raw = [m.group() for m in ms]
                if not raw:
                    continue
                tl = np.array([float(any(l["start"] < s0 + m.end() and l["end"] > s0 + m.start() for l in r["labels"])) for m in ms], dtype=np.float32)
                qs = set(t.lower() for t in raw)
                sc = [(sum(idf.get(w, MAXIDF) for w in qs & cs) / math.sqrt(len(cs) + 1), j) for j, cs in enumerate(csets[sid])]
                d2t = task == "Data2txt" and args.topk_d2t > 0
                top = sorted(sc, reverse=True)[:(args.topk_d2t if d2t else args.topk)]
                ev = [t for _, j in top for t in chunks[sid][j]][:(args.max_ev_d2t if d2t else args.max_ev)] or ["<unk>"]
                low, Fm = tok_feats(raw, ctxsets[sid], set(ev), idf)
                Xc, xf = XF.sent_feats(seg, ms, raw, low, xctx[sid], ev, idf, MAXIDF)
                Fm = np.hstack([Fm, Xc]).astype(np.float32)
                lab = int(any(l["start"] < s1 and l["end"] > s0 for l in r["labels"]))
                sents.append(dict(rid=ri, ry=R[ri]["y"], low=low, F=Fm, ev=ev, evF=np.array([[t in qs] for t in ev], dtype=np.float32),
                                  y=lab, task=TASKS.index(task), hand=hand_feats(low, Fm, top[0][0] if top else 0, idf), tl=tl, xf=xf, text=seg.strip(), key=f"{r['id']}|{si}"))
        data[name] = (sents, R)
        yp = [x for x in sents if x["y"]]
        print(f"  token labels: positive-token rate {np.mean(np.concatenate([x['tl'] for x in sents])):.4f}; "
              f"positive sentences with >=1 positive token {np.mean([x['tl'].max() > 0 for x in yp]) if yp else float('nan'):.3f}")
        print(f"{name}: {len(R)} responses, {len(sents)} sentences, {len(set(x['sid'] for x in R))} sources, "
              f"{np.mean([x['y'] for x in R]):.3f} hallucinated")
    data["leakage"] = leak
    if cpath:
        os.makedirs(args.cache_dir, exist_ok=True)
        pickle.dump(data, open(cpath, "wb"))
    return data


# ----------------------------------------------------------------------------- model
class Net(keras.Model):
    """BiGRU(answer) [+ BiGRU(evidence) + cross-attention] -> comparison -> BiGRU -> max+mean pool -> MLP.
    [v6] --ctx: a BiGRU over the sentence vectors of one response, so each sentence sees its neighbours.
    [v6] --kd_weight: auxiliary head regressing a verifier's score (distillation; the verifier is not an input)."""

    def __init__(self, V, E, H, drop=0.3, use_attn=True, nli_dim=0, emb_drop=0.2, use_ctx=False, use_kd=False):
        super().__init__()
        self.use_attn = use_attn
        self.nli_dim = nli_dim
        self.use_ctx, self.use_kd = use_ctx, use_kd
        self.nli_drop = layers.Dropout(0.2)
        self.nli_lin = layers.Dense(1)   # linear skip: NLI features -> sentence logit
        self.emb = layers.Embedding(V, E)
        self.drop = layers.SpatialDropout1D(emb_drop)   # drops whole embedding dims: stops memorising word identities
        self.ln = layers.LayerNormalization()
        self.ea = layers.Bidirectional(layers.GRU(H, return_sequences=True))
        self.ee = layers.Bidirectional(layers.GRU(H, return_sequences=True))
        self.cmp = layers.Dense(H, activation="relu")
        self.cmp_drop = layers.Dropout(drop)
        self.post = layers.Bidirectional(layers.GRU(H, return_sequences=True))
        self.hid = layers.Dense(H, activation="relu")
        self.hid_drop = layers.Dropout(drop)
        self.sgru = layers.Bidirectional(layers.GRU(max(H // 2, 8), return_sequences=True))
        self.ctx_drop = layers.Dropout(drop)
        self.out = layers.Dense(1)
        self.kd_out = layers.Dense(1)
        self.tok_drop = layers.Dropout(drop)
        self.tok_out = layers.Dense(1)       # per-token 'inside a hallucinated span' head

    def call(self, inputs, training=False):
        a, af, al, e, ef, el, nli = inputs[:7]
        am = tf.sequence_mask(al, tf.shape(a)[1])
        A = self.ea(tf.concat([self.drop(self.emb(a), training=training), af], -1), mask=am, training=training)
        if self.use_attn:
            em = tf.sequence_mask(el, tf.shape(e)[1])
            E = self.ee(tf.concat([self.drop(self.emb(e), training=training), ef], -1), mask=em, training=training)
            scores = tf.matmul(A, E, transpose_b=True)
            scores += (1.0 - tf.cast(em, tf.float32))[:, None, :] * -1e9
            C = tf.matmul(tf.nn.softmax(scores, -1), E)
        else:
            C = tf.zeros_like(A)
        X = self.cmp_drop(self.ln(self.cmp(tf.concat([A, C, A * C, A - C], -1))), training=training)
        X = self.post(X, mask=am, training=training)
        amf = tf.cast(am, tf.float32)[..., None]
        mx = tf.reduce_max(X + (1.0 - amf) * -1e9, axis=1)
        mean = tf.reduce_sum(X * amf, axis=1) / tf.cast(al, tf.float32)[:, None]
        pooled = [mx, mean] + ([self.nli_drop(nli, training=training)] if self.nli_dim else [])
        z = self.hid_drop(self.hid(tf.concat(pooled, -1)), training=training)
        zc = z
        if self.use_ctx:
            seg, pos, nr, smax = inputs[7:11]
            idx = tf.stack([seg, pos], 1)
            padded = tf.scatter_nd(idx, z, tf.stack([nr, smax, tf.shape(z)[1]]))          # [responses, sentences, H]
            lens = tf.math.unsorted_segment_sum(tf.ones_like(seg), seg, nr)
            seq = self.sgru(padded, mask=tf.sequence_mask(lens, smax), training=training)
            zc = tf.concat([z, self.ctx_drop(tf.gather_nd(seq, idx), training=training)], -1)
        sent = tf.squeeze(self.out(zc), -1)
        if self.nli_dim:
            sent = sent + tf.squeeze(self.nli_lin(nli), -1)
        tok = tf.squeeze(self.tok_out(self.tok_drop(X, training=training)), -1)    # [B, La]
        kd = tf.squeeze(self.kd_out(z), -1) if self.use_kd else tf.zeros_like(sent)
        return sent, tok, kd


def collate(batch, w2i, nofeat=False, noev=False):
    """Dynamic padding. nofeat zeroes exact-match features; noev removes the retrieved evidence entirely.
    Sentences of one response must be consecutive (they are, in prepare() and in the response-grouped training batches).
    Returns the 7 model inputs, y, tl, then seg/pos/nr/smax (sentence -> response bookkeeping), yr, kd, kdm."""
    def pad(seqs, dtype):
        L = max(len(s) for s in seqs)
        out = np.zeros((len(seqs), L) + tuple(np.shape(seqs[0])[1:]), dtype=dtype)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = np.asarray(s, dtype=dtype)
        return out
    n = len(batch)
    a = pad([[w2i.get(t, 1) for t in b["low"]] for b in batch], np.int32)
    af = pad([b["F"] for b in batch], np.float32)
    al = np.array([len(b["low"]) for b in batch], np.int32)
    if noev:
        e, ef, el = np.ones((n, 1), np.int32), np.zeros((n, 1, 1), np.float32), np.ones(n, np.int32)
        af[..., XF.EV_COLS] = 0.0
    else:
        e = pad([[w2i.get(t, 1) for t in b["ev"]] for b in batch], np.int32)
        ef = pad([b["evF"] for b in batch], np.float32)
        el = np.array([len(b["ev"]) for b in batch], np.int32)
    if nofeat:
        af[..., EM_COLS] = 0.0
        ef = np.zeros_like(ef)
    nli = np.stack([b["nli"] for b in batch]).astype(np.float32)
    seg, pos, last, k, p = [], [], None, -1, 0
    for b in batch:
        if b["rid"] != last:
            k += 1; p = 0; last = b["rid"]
        seg.append(k); pos.append(p); p += 1
    seg, pos = np.array(seg, np.int32), np.array(pos, np.int32)
    yr = np.zeros(k + 1, np.float32)
    for i, b in enumerate(batch):
        yr[seg[i]] = b.get("ry", b["y"])
    kd = np.array([b.get("kd", 0.0) for b in batch], np.float32)
    kdm = np.array([1.0 if "kd" in b else 0.0 for b in batch], np.float32)
    return (a, af, al, e, ef, el, nli, np.array([b["y"] for b in batch], np.float32), pad([b["tl"] for b in batch], np.float32),
            seg, pos, np.asarray(k + 1, np.int32), np.asarray(int(pos.max()) + 1, np.int32), yr, kd, kdm)


def model_inputs(x):
    return tuple(x[:7]) + tuple(x[9:13])


def resp_chunks(sents, bs):
    """[i, j) ranges of consecutive sentences (about bs each) that never split a response."""
    out, i, n = [], 0, len(sents)
    while i < n:
        j = min(i + bs, n)
        while j < n and sents[j]["rid"] == sents[j - 1]["rid"]:
            j += 1
        out.append((i, j)); i = j
    return out


def predict(model, sents, w2i, mode, bs=256, sent_score="head"):
    """Returns (prob, logit) of the chosen sentence score and a dict with both sentence logits and the flat per-token probabilities."""
    hl, tm, tp, ts = [], [], [], []
    for i, j in resp_chunks(sents, bs):
        chunk = sents[i:j]
        sl, tk, _ = model(model_inputs(collate(chunk, w2i, **mode)), training=False)
        sl, tk = sl.numpy(), tk.numpy()
        hl.append(sl)
        tm.append(np.array([tk[k, :len(b["low"])].max() for k, b in enumerate(chunk)]))
        for k, b in enumerate(chunk):
            n = len(b["low"]); tp.append(1 / (1 + np.exp(-tk[k, :n]))); ts.append(np.full(n, i + k))
    cat = lambda x, d=0: np.concatenate(x) if x else np.zeros(d)
    ex = dict(head=cat(hl), tokmax=cat(tm), tok_prob=cat(tp), tok_sent=cat(ts).astype(int))
    logit = ex["head"] if sent_score == "head" else ex["tokmax"]
    return 1 / (1 + np.exp(-logit)), logit, ex


def weighted_bce(logits, y, pos_weight):
    x, z = np.asarray(logits, np.float64), np.asarray(y, np.float64)
    return float(np.mean((1 - z) * x + (1 + (pos_weight - 1) * z) * np.logaddexp(0.0, -x))) if len(x) else float("nan")


def pool_resp(logit, seg, nr, kind):
    """Sentence logits -> response logit. noisyor: P = 1 - prod(1 - p_i) (the response is hallucinated if ANY sentence is); lse: smooth max."""
    if kind == "lse":
        mx = tf.math.unsorted_segment_max(logit, seg, nr)
        return mx + tf.math.log(tf.math.unsorted_segment_sum(tf.exp(logit - tf.gather(mx, seg)), seg, nr))
    s_ = tf.maximum(tf.math.unsorted_segment_sum(tf.math.softplus(logit), seg, nr), 1e-6)
    return s_ + tf.math.log(-tf.math.expm1(-s_))          # logit(1 - exp(-s)), numerically stable


def make_train_step(model, opt, pos_weight, tok_pos_weight, token_w, smooth=0.0, resp_w=0.0, resp_pw=1.0, resp_pool="noisyor", kd_w=0.0):
    sig = [tf.TensorSpec([None, None], tf.int32), tf.TensorSpec([None, None, NF], tf.float32), tf.TensorSpec([None], tf.int32),
           tf.TensorSpec([None, None], tf.int32), tf.TensorSpec([None, None, 1], tf.float32), tf.TensorSpec([None], tf.int32),
           tf.TensorSpec([None, None], tf.float32), tf.TensorSpec([None], tf.float32), tf.TensorSpec([None, None], tf.float32),
           tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32), tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32),
           tf.TensorSpec([None], tf.float32), tf.TensorSpec([None], tf.float32), tf.TensorSpec([None], tf.float32)]

    @tf.function(input_signature=sig)
    def step(a, af, al, e, ef, el, nli, y, tl, seg, pos, nr, smax, yr, kd, kdm):
        with tf.GradientTape() as tape:
            logit, tlog, klog = model((a, af, al, e, ef, el, nli, seg, pos, nr, smax), training=True)
            loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y * (1.0 - smooth) + 0.5 * smooth, logits=logit, pos_weight=pos_weight))
            m = tf.cast(tf.sequence_mask(al, tf.shape(a)[1]), tf.float32)
            tloss = tf.reduce_sum(m * tf.nn.weighted_cross_entropy_with_logits(labels=tl, logits=tlog, pos_weight=tok_pos_weight)) / tf.reduce_sum(m)
            total = loss
            if token_w > 0:
                total += token_w * tloss
            if resp_w > 0:
                rlog = pool_resp(logit, seg, nr, resp_pool)
                rloss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=yr * (1.0 - smooth) + 0.5 * smooth, logits=rlog, pos_weight=resp_pw))
                total += resp_w * rloss
            if kd_w > 0:
                kl = tf.nn.sigmoid_cross_entropy_with_logits(labels=kd, logits=klog)
                total += kd_w * tf.reduce_sum(kl * kdm) / tf.maximum(tf.reduce_sum(kdm), 1.0)
        tv = model.trainable_variables
        pairs = [(g, v) for g, v in zip(tape.gradient(total, tv), tv) if g is not None]   # unused layers/heads get no gradient
        gs, _ = tf.clip_by_global_norm([g for g, _ in pairs], 1.0)
        opt.apply_gradients(zip(gs, [v for _, v in pairs]))
        return loss, tloss
    return step


def resp_scores(sent_scores, sents, n):
    s = np.zeros(n)
    for sc, st in zip(sent_scores, sents):
        s[st["rid"]] = max(s[st["rid"]], sc)
    return s


def agg_scores(logits, sents, n, agg="max"):
    """Response score from SENTENCE LOGITS. max | lse (smooth max) | noisyor (logit of 1 - prod(1 - p_i)). Responses without sentences get the lowest score."""
    l = np.asarray(logits, np.float64)
    rid = np.array([x["rid"] for x in sents], dtype=int)
    floor = (l.min() - 1.0) if len(l) else 0.0
    if agg == "noisyor":
        t = np.zeros(n); np.add.at(t, rid, np.logaddexp(0.0, l)); t = np.maximum(t, 1e-6)
        return t + np.log(-np.expm1(-t))
    mx = np.full(n, -np.inf); np.maximum.at(mx, rid, l)
    if agg == "lse":
        ex = np.zeros(n); ok = np.isfinite(mx[rid]); np.add.at(ex, rid[ok], np.exp(l[ok] - mx[rid[ok]]))
        out = np.where(np.isfinite(mx), mx + np.log(np.maximum(ex, 1e-300)), floor)
        return out
    return np.where(np.isfinite(mx), mx, floor)


def stack_feats(logits, sents, R):
    """Per-response statistics of the sentence logits (+ length + task one-hot). Inputs available at inference time only."""
    g, ntok = collections.defaultdict(list), collections.defaultdict(int)
    l = np.asarray(logits, np.float64)
    for v, x in zip(l, sents):
        g[x["rid"]].append(v); ntok[x["rid"]] += len(x["low"])
    floor = float(l.min()) - 1.0 if len(l) else -10.0
    X = np.zeros((len(R), 10 + len(TASKS)))
    for i, r in enumerate(R):
        v = np.sort(np.array(g.get(i, [floor])))[::-1]
        X[i, :10] = [v[0], v[1] if len(v) > 1 else v[0], v.mean(), v.std(), v[0] + np.log(np.exp(v - v[0]).sum()),
                     np.logaddexp(0.0, v).sum(), len(v), np.log1p(ntok.get(i, 0)), (v > 0).mean(), (v > 0).sum()]
        X[i, 10 + TASKS.index(r["task"])] = 1.0
    return X


def stack_scores(vlog, tlog, va, te, Rva, Rte, yv, folds=5):
    """Second stage: logistic regression on stack_feats, fitted on VALIDATION responses only. The validation score returned is
    out-of-fold (grouped by source), the threshold is chosen on it, the test score comes from the model fitted on all of validation."""
    Xv, Xt = stack_feats(vlog, va, Rva), stack_feats(tlog, te, Rte)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    Zv, Zt = (Xv - mu) / sd, (Xt - mu) / sd
    groups = np.array([r["sid"] for r in Rva])
    oof = np.zeros(len(yv))
    for trn, tst in GroupKFold(n_splits=min(folds, len(set(groups)))).split(Zv, yv, groups):
        oof[tst] = LogisticRegression(C=0.5, max_iter=2000).fit(Zv[trn], yv[trn]).decision_function(Zv[tst])
    clf = LogisticRegression(C=0.5, max_iter=2000).fit(Zv, yv)
    return oof, clf.decision_function(Zt), best_thr(yv, oof)


def task_thr_scores(vs, ts, yv, tv, tt, min_n=40):
    """Per-task decision thresholds tuned on validation. Returned scores are shifted so that the decision threshold is 0 for every task.
    NOTE: AUROC/AUPRC of these shifted scores describe the final decision rule, not the base model."""
    glob = best_thr(yv, vs)
    ov, ot = np.full(len(vs), glob), np.full(len(ts), glob)
    for t in TASKS:
        mv = tv == t
        if mv.sum() >= min_n and len(set(yv[mv])) > 1:
            th = best_thr(yv[mv], vs[mv]); ov[mv] = th; ot[tt == t] = th
    return vs - ov, ts - ot


# ----------------------------------------------------------------------------- metrics
def best_thr(y, s):
    cands = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    f = [precision_recall_fscore_support(y, s >= t, average="binary", zero_division=0)[2] for t in cands]
    return float(cands[int(np.argmax(f))])


def metrics(y, s, thr):
    p = (s >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    two = len(set(y)) > 1
    return dict(precision=pr, recall=rc, f1=f1, fnr=1 - rc, acc=accuracy_score(y, p),
                macro_f1=precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2],
                auroc=roc_auc_score(y, s) if two else float("nan"),
                auprc=average_precision_score(y, s) if two else float("nan"), thr=thr)


def boot_f1(y, s, thr, groups=None, n=1000, seed=0):
    rng, p, out = np.random.default_rng(seed), (s >= thr).astype(int), []
    if groups is not None:
        by = collections.defaultdict(list)
        for k, g in enumerate(groups):
            by[g].append(k)
        blocks = [np.array(v) for v in by.values()]
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)) if groups is None else \
            np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        tp, fp, fn = ((p[i] == 1) & (y[i] == 1)).sum(), ((p[i] == 1) & (y[i] == 0)).sum(), ((p[i] == 0) & (y[i] == 1)).sum()
        out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ----------------------------------------------------------------------------- training
def make_batches(n_sent, resp_groups, bs, by_resp):
    """Lists of sentence indices. by_resp: shuffle whole responses and fill each batch to ~bs sentences (needed for --resp_loss / --ctx)."""
    if not by_resp:
        order = list(range(n_sent)); random.shuffle(order)
        return [order[i:i + bs] for i in range(0, n_sent, bs)]
    order = list(range(len(resp_groups))); random.shuffle(order)
    out, cur = [], []
    for g in order:
        cur += resp_groups[g]
        if len(cur) >= bs:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def train_run(args, seed, tr, va, yv, w2i, V, pw, mode, tpw):
    """One training run. Validates every --eval_every steps, tracks the best checkpoint for each criterion (auprc/f1/loss),
    early-stops on --select, restores that checkpoint. Reads train + validation only."""
    set_seeds(seed)
    model = Net(V, args.emb, args.hidden, args.dropout, use_attn=args.ablate != "no_cross_attn", nli_dim=args.nli_dim, emb_drop=args.emb_drop,
                use_ctx=args.ctx, use_kd=args.kd_weight > 0)
    try:
        opt = keras.optimizers.AdamW(args.lr, weight_decay=args.weight_decay)
    except AttributeError:
        opt = keras.optimizers.Adam(args.lr)
    cur_lr = args.lr
    model(model_inputs(collate(tr[:2], w2i, **mode)), training=False)
    opt.build(model.trainable_variables)
    step = make_train_step(model, opt, pw, tpw, args.token_loss, args.label_smooth, args.resp_loss, args.resp_pw, args.resp_pool, args.kd_weight)
    groups_ = collections.OrderedDict()
    for j_, x_ in enumerate(tr):
        groups_.setdefault(x_["rid"], []).append(j_)
    resp_groups = list(groups_.values())
    by_resp = args.resp_loss > 0 or args.ctx          # these need whole responses in every batch
    ema = [tf.Variable(v, trainable=False) for v in model.weights]      # exponential moving average of the weights

    @tf.function
    def ema_update(d):
        for e_, v_ in zip(ema, model.weights):
            e_.assign(d * e_ + (1.0 - d) * v_)
    wrng = np.random.default_rng(seed)                       # separate RNG: word dropout never perturbs the shuffle stream
    yv_sent = [x["y"] for x in va]
    ytok_va = np.concatenate([x["tl"] for x in va])
    best = {c: dict(score=-1e18) for c in ["auprc", "f1", "loss"]}
    st = dict(step=0, bad=0, run_loss=0.0, run_tok=0.0, run_n=0)
    hist, t_start = [], time.time()

    def evaluate(epoch):
        backup = model.get_weights()
        if args.ema > 0:
            model.set_weights([e_.numpy() for e_ in ema])             # evaluate (and keep) the averaged weights
        vp, vlog, vx = predict(model, va, w2i, mode, sent_score=args.sent_score)
        vloss = weighted_bce(vlog, yv_sent, pw)
        vs = agg_scores(vlog, va, len(yv), args.resp_agg)
        thr = best_thr(yv, vs); vm = metrics(yv, vs, thr)
        rec = dict(step=st["step"], epoch=round(epoch, 3), loss=st["run_loss"] / max(st["run_n"], 1), val_loss=vloss,
                   f1=vm["f1"], precision=vm["precision"], recall=vm["recall"], auroc=vm["auroc"], auprc=vm["auprc"], thr=thr,
                   tok_loss=st["run_tok"] / max(st["run_n"], 1),
                   tok_auprc=float(average_precision_score(ytok_va, vx["tok_prob"])) if ytok_va.max() > 0 else float("nan"))
        st["run_loss"], st["run_tok"], st["run_n"] = 0.0, 0.0, 0
        vals = dict(auprc=vm["auprc"], f1=vm["f1"], loss=-vloss)
        improved = vals[args.select] > best[args.select]["score"]
        for c in best:
            if vals[c] > best[c]["score"]:
                best[c] = dict(score=vals[c], weights=model.get_weights(), rec=rec)
        model.set_weights(backup)                                        # continue training from the raw weights
        hist.append(rec)
        st["bad"] = 0 if improved else st["bad"] + 1
        nonlocal cur_lr
        if st["bad"] and st["bad"] % args.lr_patience == 0 and cur_lr > 1e-5:
            cur_lr *= 0.5; opt.learning_rate.assign(cur_lr); print(f"  lr -> {cur_lr:.2e}")
        print(f"seed {seed} step {st['step']} ep {epoch:.2f}: loss {rec['loss']:.4f} | val loss {vloss:.4f} AUPRC {vm['auprc']:.4f} "
              f"AUROC {vm['auroc']:.4f} tokAUPRC {rec['tok_auprc']:.4f} F1 {vm['f1']:.4f} P {vm['precision']:.3f} R {vm['recall']:.3f} thr {thr:.3f}{' *best' if improved else ''}")
        if args.patience > 0 and st["bad"] >= args.patience:
            print(f"seed {seed}: early stop (patience {args.patience} evals on {args.select}); best step {best[args.select]['rec']['step']}")
            return True
        return False

    stop = False
    for ep in range(args.epochs):
        batches = make_batches(len(tr), resp_groups, args.bs, by_resp)
        nb = len(batches)
        for bi, idxs in enumerate(batches):
            x = list(collate([tr[j] for j in idxs], w2i, **mode))
            if args.word_dropout > 0:
                for k in (0, 3):
                    m = (wrng.random(x[k].shape) < args.word_dropout) & (x[k] > 1)
                    x[k] = np.where(m, 1, x[k]).astype(np.int32)
            l_, lt_ = step(*x); n_ = len(x[7]); st["run_loss"] += float(l_) * n_; st["run_tok"] += float(lt_) * n_; st["run_n"] += n_; st["step"] += 1
            if args.ema > 0:
                ema_update(tf.constant(min(args.ema, (1.0 + st["step"]) / (10.0 + st["step"])), tf.float32))
            if args.eval_every and st["step"] % args.eval_every == 0:
                if evaluate(ep + (bi + 1) / nb):
                    stop = True; break
        if stop:
            break
        if not args.eval_every or st["step"] % args.eval_every != 0:
            if evaluate(ep + 1):
                break
    train_sec = time.time() - t_start
    sel = best[args.select]
    model.set_weights(sel["weights"])
    vp, vlog, vx = predict(model, va, w2i, mode, sent_score=args.sent_score)
    vs = agg_scores(vlog, va, len(yv), args.resp_agg)
    chk = metrics(yv, vs, sel["rec"]["thr"])["f1"]
    if abs(chk - sel["rec"]["f1"]) > 1e-4:
        print(f"[WARN] restored checkpoint val F1 {chk:.4f} != recorded {sel['rec']['f1']:.4f}")
    r = sel["rec"]
    alt = {c: dict(step=b["rec"]["step"], val_f1=b["rec"]["f1"], val_auprc=b["rec"]["auprc"], val_auroc=b["rec"]["auroc"],
                   val_loss=b["rec"]["val_loss"]) for c, b in best.items()}
    stem = os.path.join(args.out_dir, f"best_seed{seed}")
    try:
        model.save_weights(stem + ".weights.h5")
    except Exception as e:
        print(f"[warn] could not write checkpoint ({e})")
    info = dict(seed=seed, select=args.select, step=r["step"], epoch=r["epoch"], evals_run=len(hist), thr=r["thr"],
                val_f1=r["f1"], val_auprc=r["auprc"], val_auroc=r["auroc"], val_loss=r["val_loss"],
                val_precision=r["precision"], val_recall=r["recall"], lr=args.lr, dropout=args.dropout,
                train_seconds=train_sec, alt_criteria=alt, val_tok_auprc=r["tok_auprc"], token_loss=args.token_loss, sent_score=args.sent_score)
    json.dump(dict(info=info, history=hist), open(os.path.join(args.out_dir, f"history_seed{seed}.json"), "w"), indent=1, default=float)
    return dict(model=model, info=info, hist=hist, vs=vs, vp=vp, vlog=vlog, vx=vx)


def save_predictions(path, runout, Rva, Rte, va, te, base_scores, thr_of, pw):
    out = {}
    for sp, R, sents, probs, logits in [("val", Rva, va, runout["vp"], runout["vlog"]), ("test", Rte, te, runout["tp"], runout["tlog"])]:
        out[f"{sp}_resp_id"] = np.array([r["id"] for r in R], dtype="U"); out[f"{sp}_sid"] = np.array([r["sid"] for r in R], dtype="U")
        out[f"{sp}_task"] = np.array([r["task"] for r in R], dtype="U"); out[f"{sp}_model"] = np.array([r["model"] for r in R], dtype="U")
        out[f"{sp}_types"] = np.array(["|".join(r["types"]) for r in R], dtype="U"); out[f"{sp}_y"] = np.array([r["y"] for r in R])
        out[f"{sp}_sent_rid"] = np.array([s["rid"] for s in sents]); out[f"{sp}_sent_y"] = np.array([s["y"] for s in sents])
        out[f"{sp}_sent_prob"] = probs; out[f"{sp}_sent_logit"] = logits
        ex = runout["vx"] if sp == "val" else runout["tx"]
        out[f"{sp}_sent_logit_head"], out[f"{sp}_sent_logit_tokmax"] = ex["head"], ex["tokmax"]
        out[f"{sp}_tok_prob"], out[f"{sp}_tok_sent"] = ex["tok_prob"], ex["tok_sent"]
        out[f"{sp}_tok_y"] = np.concatenate([x["tl"] for x in sents])
    for n, (vs, ts) in base_scores.items():
        k = re.sub(r"\W+", "_", n).strip("_")
        out[f"val_score__{k}"], out[f"test_score__{k}"], out[f"thr__{k}"] = vs, ts, thr_of[n]
    out["pos_weight"] = np.array(pw)
    np.savez_compressed(path, **out)


# ----------------------------------------------------------------------------- main
def attach_nli(path, data, names=("train", "val", "test")):
    """Adds s["nli"] to every sentence. path=None -> a 1-dim zero placeholder (feature path disabled). Returns the feature dim (0 = off)."""
    if not path:
        for nm in names:
            for s in data[nm][0]:
                s["nli"] = np.zeros(1, np.float32)
        return 0
    z = np.load(path)
    idx = {k: i for i, k in enumerate(z["keys"].tolist())}
    X = z["X"].astype(np.float32); d = X.shape[1]
    miss = tot = 0
    for nm in names:
        for s in data[nm][0]:
            tot += 1
            i = idx.get(s["key"])
            if i is None:
                miss += 1; s["nli"] = np.zeros(d, np.float32)
            else:
                s["nli"] = X[i]
    print(f"NLI features {list(z['feature_names'])}: {miss}/{tot} sentences missing")
    assert miss <= 0.01 * tot, "more than 1% of sentences have no NLI features: rerun nli_features.py on the same data_dir (no --limit)"
    tr = np.stack([s["nli"] for s in data["train"][0]])
    mu, sd = tr.mean(0), tr.std(0) + 1e-6                      # standardise with TRAIN statistics only
    for nm in names:
        for s in data[nm][0]:
            s["nli"] = ((s["nli"] - mu) / sd).astype(np.float32)
    return d


def attach_kd(path, col, data, names=("train", "val", "test")):
    """s["kd"] = P(unsupported) = 1 - verifier support probability (column `col` of a verifier_features.py npz). Used ONLY as a training target."""
    z = np.load(path)
    fn = [str(x) for x in z["feature_names"]]
    assert col in fn, f"--kd_col {col!r} not in {fn}"
    j = fn.index(col); idx = {k: i for i, k in enumerate(z["keys"].tolist())}; X = z["X"].astype(np.float32)
    hit = tot = 0
    for nm in names:
        for s_ in data[nm][0]:
            tot += 1; i = idx.get(s_["key"])
            if i is not None:
                s_["kd"] = float(np.clip(1.0 - X[i, j], 0.0, 1.0)); hit += 1
    print(f"distillation targets: {hit}/{tot} sentences have a verifier score (column {col})")
    assert hit >= 0.9 * tot, "fewer than 90% of sentences have verifier scores: rerun verifier_features.py with the same data_dir / --synth_* / --drop_implicit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/ragtruth"); ap.add_argument("--out_dir", default="results_v2")
    ap.add_argument("--cache_dir", default="cache_v2")
    ap.add_argument("--split", choices=["grouped", "original"], default="grouped")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval_every", type=int, default=200, help="validate every N steps (0 = per epoch)")
    ap.add_argument("--patience", type=int, default=8, help="stop after this many EVALUATIONS without improvement on --select (<=0 off)")
    ap.add_argument("--select", choices=["auprc", "f1", "loss"], default="auprc")
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb", type=int, default=64); ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--topk", type=int, default=3); ap.add_argument("--max_ans", type=int, default=64)
    ap.add_argument("--max_ev", type=int, default=100); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--min_count", type=int, default=3); ap.add_argument("--word_dropout", type=float, default=0.1)
    ap.add_argument("--token_loss", type=float, default=0.5, help="weight of the auxiliary span (token-level) loss; 0 = off")
    ap.add_argument("--emb_drop", type=float, default=0.2, help="spatial dropout on embeddings")
    ap.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW decoupled weight decay")
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999, help="weight EMA decay (0 = off)")
    ap.add_argument("--lr_patience", type=int, default=2, help="halve LR after this many evals without improvement")
    ap.add_argument("--v6", action="store_true", help="enable resp_loss 0.5 + ctx + synth_ratio 0.3 + Data2txt evidence 6x40 (each can still be overridden)")
    ap.add_argument("--resp_loss", type=float, default=0.0, help="weight of the response-level pooled loss (0 = off)")
    ap.add_argument("--resp_pool", choices=["noisyor", "lse"], default="noisyor")
    ap.add_argument("--resp_agg", choices=["auto", "max", "noisyor", "lse"], default="auto", help="response score from sentence logits (auto = --resp_pool if --resp_loss > 0 else max)")
    ap.add_argument("--ctx", action="store_true", help="sentence-sequence BiGRU (each sentence sees the rest of its response)")
    ap.add_argument("--synth_ratio", type=float, default=0.0, help="fraction of faithful TRAIN responses that get a synthetic corrupted copy (0 = off)")
    ap.add_argument("--synth_seed", type=int, default=0)
    ap.add_argument("--topk_d2t", type=int, default=0, help="evidence chunks for Data2txt sources (0 = same as --topk); also packs JSON lines into 40-token chunks")
    ap.add_argument("--max_ev_d2t", type=int, default=0, help="evidence token budget for Data2txt sources")
    ap.add_argument("--kd_feats", default="", help="verifier npz used ONLY as a distillation target")
    ap.add_argument("--kd_col", default="", help="column of --kd_feats holding P(supported) in [0,1]")
    ap.add_argument("--kd_weight", type=float, default=0.0)
    ap.add_argument("--no_stack", action="store_true"); ap.add_argument("--no_task_thr", action="store_true")
    ap.add_argument("--tok_pos_weight_cap", type=float, default=10.0)
    ap.add_argument("--sent_score", choices=["head", "token_max"], default="head")
    ap.add_argument("--ablate", choices=["none", "no_evidence", "no_cross_attn"], default="none")
    ap.add_argument("--no_features", action="store_true")
    ap.add_argument("--nli_feats", default="", help="npz from nli_features.py; adds frozen-NLI features to the model and the GBM")
    ap.add_argument("--feat_groups", default="all", help="extra grounding features: comma list of stem,num,date,ngram,entity,sbert,align,neg,claim,detail | all (= first six) | new (= align,neg,claim,detail) | none (= v3). Example: all,new")
    ap.add_argument("--sbert_model", default="all-MiniLM-L6-v2"); ap.add_argument("--sbert_cache", default="cache_v2/sbert_cache.pkl")
    ap.add_argument("--drop_implicit", action="store_true", help="WARNING: also changes official test labels")
    ap.add_argument("--drop_bad_quality", action="store_true", help="WARNING: also filters the official test set")
    ap.add_argument("--strict_dups", action="store_true"); ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max_train", type=int, default=0); ap.add_argument("--max_test", type=int, default=0)
    pre, _ = ap.parse_known_args()
    if pre.v6:
        ap.set_defaults(resp_loss=0.5, ctx=True, synth_ratio=0.3, topk_d2t=6, max_ev_d2t=240)
    args = ap.parse_args()
    if args.resp_agg == "auto":
        args.resp_agg = args.resp_pool if args.resp_loss > 0 else "max"
    if args.topk_d2t > 0 and args.max_ev_d2t <= 0:
        args.max_ev_d2t = args.topk_d2t * CHUNK
    assert not (args.kd_weight > 0 and not (args.kd_feats and args.kd_col)), "--kd_weight needs --kd_feats and --kd_col"
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    os.makedirs(args.out_dir, exist_ok=True)
    data = prepare(args)
    groups = XF.parse_groups(args.feat_groups)
    if "sbert" in groups:
        segs = XF.load_source_segments(args.data_dir, flatten, split_sents)
        if not XF.attach_sbert(data, segs, args.sbert_model, args.sbert_cache):
            groups.remove("sbert")
    args.xf_names = XF.finalize(data, groups)
    args.nli_dim = attach_nli(args.nli_feats, data)
    if args.kd_weight > 0:
        attach_kd(args.kd_feats, args.kd_col, data)
    args.nli_dim = XF.merge_into_nli(data, args.nli_dim)
    print(f"extra feature groups: {groups}; sentence features: {args.xf_names}; sentence-vector dim {args.nli_dim}")
    sfx = " +extra" if args.xf_names else ""
    tr, va, te = data["train"][0], data["val"][0], data["test"][0]
    Rva, Rte = data["val"][1], data["test"][1]
    yv, yt = np.array([r["y"] for r in Rva]), np.array([r["y"] for r in Rte])
    gt, tt, tv = np.array([r["sid"] for r in Rte]), np.array([r["task"] for r in Rte]), np.array([r["task"] for r in Rva])

    # ---- baselines (fit on train only, thresholds on val only)
    scores = {"Majority (all hallucinated)": (np.ones(len(yv)), np.ones(len(yt))),
              "Task-type heuristic": ((tv == "Data2txt").astype(float), (tt == "Data2txt").astype(float))}

    def lex(sents, n):
        s = np.zeros(n)
        for st_ in sents:
            s[st_["rid"]] = max(s[st_["rid"]], 1 - st_["hand"][1])
        return s
    scores["Lexical overlap"] = (lex(va, len(yv)), lex(te, len(yt)))
    gbm_cfg = [("GBM hand features" + sfx, False, False), ("GBM hand features + task" + sfx, True, False)]
    if args.nli_feats:
        gbm_cfg.append(("GBM hand + NLI + task" + sfx, True, True))
    for nm, add_task, add_nli in gbm_cfg:
        mk = lambda ss: np.array([np.r_[x["hand"], np.eye(3)[x["task"]] if add_task else [], x["nli0"] if add_nli else []] for x in ss])
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
        gb.fit(mk(tr), [x["y"] for x in tr])
        scores[nm] = (resp_scores(gb.predict_proba(mk(va))[:, 1], va, len(yv)), resp_scores(gb.predict_proba(mk(te))[:, 1], te, len(yt)))
    base_thr = {n: 0.5 if n.startswith(("Majority", "Task-type")) else best_thr(yv, v[0]) for n, v in scores.items()}

    # ---- vocabulary / loss weight (train only)
    cnt = collections.Counter(t for x in tr for t in x["low"] + x["ev"])
    vocab = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common(40000) if c >= args.min_count]
    w2i = {w: i for i, w in enumerate(vocab)}
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    rp = float(np.mean([r["y"] for r in data["train"][1]]))
    args.resp_pw = float(min((1 - rp) / max(rp, 1e-6), 3.0))
    print(f"response positive rate {rp:.3f}, response pos_weight {args.resp_pw:.2f}, resp_loss {args.resp_loss} ({args.resp_pool}), ctx {args.ctx}, "
          f"response aggregation {args.resp_agg}, kd_weight {args.kd_weight}")
    pos = np.mean([x["y"] for x in tr])
    pw = float(min((1 - pos) / max(pos, 1e-6), 3.0))
    print(f"vocab {len(vocab)}, sentence pos rate {pos:.4f}, pos_weight {pw:.2f}")
    tok_pos = float(np.mean(np.concatenate([x["tl"] for x in tr])))
    tpw = float(min((1 - tok_pos) / max(tok_pos, 1e-6), args.tok_pos_weight_cap))
    print(f"token positive rate {tok_pos:.4f}, token pos_weight {tpw:.2f}, token_loss weight {args.token_loss}, sentence score = {args.sent_score}")
    mode = dict(nofeat=args.no_features, noev=args.ablate == "no_evidence")
    name = "BiGRU-attn" + (f" [{args.ablate}]" if args.ablate != "none" else "") + (" no-EM-features" if args.no_features else "") + XF.name_suffix(groups) \
        + (f" +resp-loss({args.resp_pool})" if args.resp_loss > 0 else "") + (" +ctx" if args.ctx else "") \
        + (f" +synth{args.synth_ratio:g}" if args.synth_ratio > 0 else "") + (" +d2t-ev" if args.topk_d2t > 0 else "") \
        + (" +kd" if args.kd_weight > 0 else "") + (" +verifier" if args.nli_feats else "")

    # ---- train; ALL selection (checkpoint, threshold) is finished before the test set is scored
    runs = [train_run(args, args.seed + k, tr, va, yv, w2i, len(vocab), pw, mode, tpw) for k in range(args.seeds)]
    seed_f1, seed_info = [], []
    for k, r in enumerate(runs):
        t0 = time.time()
        r["tp"], r["tlog"], r["tx"] = predict(r["model"], te, w2i, mode, sent_score=args.sent_score)
        r["info"]["infer_ms_per_response"] = 1000 * (time.time() - t0) / max(len(Rte), 1)
        r["ts"] = agg_scores(r["tlog"], te, len(yt), args.resp_agg)
        seed_f1.append(metrics(yt, r["ts"], r["info"]["thr"])["f1"]); seed_info.append(r["info"])
        sc = dict(scores); sc[name] = (r["vs"], r["ts"])
        save_predictions(os.path.join(args.out_dir, f"predictions_seed{r['info']['seed']}.npz"), r, Rva, Rte, va, te, sc,
                         {**base_thr, name: r["info"]["thr"]}, pw if args.sent_score == "head" else tpw)
    first = runs[0]
    scores[name] = (first["vs"], first["ts"]); thr_of = {**base_thr, name: first["info"]["thr"]}
    if len(runs) > 1:      # seed ensemble: mean sentence logit over seeds, threshold chosen on val only
        ev_ = agg_scores(np.mean([r["vlog"] for r in runs], 0), va, len(yv), args.resp_agg); et_ = agg_scores(np.mean([r["tlog"] for r in runs], 0), te, len(yt), args.resp_agg)
        ename = name + f" ENSEMBLE x{len(runs)}"; scores[ename] = (ev_, et_); thr_of[ename] = best_thr(yv, ev_)

    def second_stage(label, vlog_, tlog_):
        vs_, ts_ = scores[label]
        if not args.no_stack:
            try:
                ov, ot, th = stack_scores(vlog_, tlog_, va, te, Rva, Rte, yv)
                scores[label + " +stack"] = (ov, ot); thr_of[label + " +stack"] = th
            except Exception as ex_:
                print(f"[warn] stacking failed for {label}: {ex_}")
        if not args.no_task_thr:
            a_, b_ = task_thr_scores(vs_, ts_, yv, tv, tt)
            scores[label + " +task-thr"] = (a_, b_); thr_of[label + " +task-thr"] = 0.0
    second_stage(name, first["vlog"], first["tlog"])
    if len(runs) > 1:
        second_stage(ename, np.mean([r["vlog"] for r in runs], 0), np.mean([r["tlog"] for r in runs], 0))

    results = {}
    for n, (vs, ts) in scores.items():
        thr = thr_of[n]
        res = {"overall": metrics(yt, ts, thr), "ci95_f1": boot_f1(yt, ts, thr, gt), "per_task": {}}
        for t in TASKS:
            m = tt == t
            if m.sum() == 0 or len(set(yt[m])) < 2:
                res["per_task"][t] = dict(f1=float("nan"), auroc=float("nan"), auprc=float("nan")); continue
            res["per_task"][t] = metrics(yt[m], ts[m], thr)
            res["per_task"][t]["ci95_f1"] = boot_f1(yt[m], ts[m], thr, gt[m])
        res["recall_by_type"] = {}
        for ty in TYPES:
            idx = [i for i, r in enumerate(Rte) if ty in r["types"]]
            res["recall_by_type"][ty] = (float(np.mean(ts[idx] >= thr)) if idx else None, len(idx))
        results[n] = res
    results[name]["seed_f1"], results[name]["seed_selection"] = seed_f1, seed_info
    tx, tok_y = first["tx"], np.concatenate([x["tl"] for x in te])
    tsk = np.array([te[i]["task"] for i in tx["tok_sent"]])
    tl_res = {}
    for lab, m in [("all", np.ones(len(tok_y), bool))] + [(t, tsk == TASKS.index(t)) for t in TASKS]:
        tl_res[lab] = dict(auprc=float(average_precision_score(tok_y[m], tx["tok_prob"][m])) if tok_y[m].max() > 0 else float("nan"),
                           auroc=float(roc_auc_score(tok_y[m], tx["tok_prob"][m])) if len(set(tok_y[m])) > 1 else float("nan"), pos_rate=float(tok_y[m].mean()))
    results[name]["token_level"] = tl_res
    env = dict(python=sys.version.split()[0], numpy=np.__version__, tensorflow=tf.__version__)
    json.dump(dict(args=vars(args), env=env, leakage_checks=data["leakage"], val_prevalence=float(yv.mean()), test_prevalence=float(yt.mean()),
                   results=results, reported_literature_f1=REPORTED), open(os.path.join(args.out_dir, "results.json"), "w"), indent=1, default=float)

    # ---- training curves (step-level) of the first seed
    h, sel = first["hist"], first["info"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4)); xs = [x["step"] for x in h]
    ax[0].plot(xs, [x["loss"] for x in h], label="train (running)"); ax[0].plot(xs, [x["val_loss"] for x in h], label="val (sentence)")
    ax[1].plot(xs, [x["auprc"] for x in h], label="val AUPRC"); ax[1].plot(xs, [x["auroc"] for x in h], label="val AUROC")
    ax[2].plot(xs, [x["f1"] for x in h], c="C2")
    for a_, t_ in zip(ax, ["Loss", "Validation AUPRC / AUROC (response)", "Validation F1 (response)"]):
        a_.axvline(sel["step"], ls=":", c="k"); a_.set(title=t_, xlabel="step")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "01_training_curves.png"), dpi=140); plt.close(fig)

    # ---- markdown summary
    L = ["# Results summary (v2)\n", f"Args: `{vars(args)}`\n", f"Env: `{env}`\n",
         f"Prevalence: val {yv.mean():.3f}, test {yt.mean():.3f}.  Leakage: `{ {k: v for k, v in data['leakage'].items() if k != 'cross_groups'} }`\n",
         "## Overall (hallucinated = positive)\n", "| Model | F1 | 95% CI | P | R | Macro-F1 | AUROC | AUPRC | FNR |", "|---|---|---|---|---|---|---|---|---|"]
    for n in scores:
        o, c = results[n]["overall"], results[n]["ci95_f1"]
        L.append(f"| {n} | {o['f1']:.3f} | {c[0]:.3f}-{c[1]:.3f} | {o['precision']:.3f} | {o['recall']:.3f} | {o['macro_f1']:.3f} | {o['auroc']:.3f} | {o['auprc']:.3f} | {o['fnr']:.3f} |")
    L += ["", "## Per task: F1 / AUROC / AUPRC (compare tasks by AUROC/AUPRC: prevalence differs)\n", "| Model | " + " | ".join(TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for n in scores:
        L.append(f"| {n} | " + " | ".join(f"{results[n]['per_task'][t]['f1']:.3f} / {results[n]['per_task'][t].get('auroc', float('nan')):.3f} / "
                                          f"{results[n]['per_task'][t].get('auprc', float('nan')):.3f}" for t in TASKS) + " |")
    L += ["", f"{name}: seed F1 {np.round(seed_f1, 4).tolist()} (mean {np.mean(seed_f1):.4f}, std {np.std(seed_f1):.4f})",
          f"Selected (seed {sel['seed']}, by {sel['select']}): step {sel['step']} (epoch {sel['epoch']}), {sel['evals_run']} evals, thr {sel['thr']:.4f}, "
          f"val F1 {sel['val_f1']:.4f} AUPRC {sel['val_auprc']:.4f} AUROC {sel['val_auroc']:.4f} loss {sel['val_loss']:.4f}; "
          f"train {sel['train_seconds']:.0f}s, inference {sel['infer_ms_per_response']:.1f} ms/response",
          "Token-level test AUPRC all/QA/Data2txt/Summary: " + " / ".join(f"{results[name]['token_level'][k]['auprc']:.3f}" for k in ["all"] + TASKS)
          + f" (token pos rate {results[name]['token_level']['all']['pos_rate']:.4f}; token_loss={args.token_loss}, sent_score={args.sent_score})",
          f"Best checkpoint by other criteria (val only): `{sel['alt_criteria']}`", "",
          "## Reported literature F1 (not re-run)\n"] + [f"- {k}: {v}" for k, v in REPORTED.items()]
    open(os.path.join(args.out_dir, "results.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()