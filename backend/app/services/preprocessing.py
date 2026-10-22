"""Inference-time preprocessing for ONE (source, answer) pair. It re-runs the per-sentence logic of model6.prepare() using the research
functions themselves (split_sents, flatten, tok_feats, XF.build_ctx, XF.sent_feats), so only the loop around them is new.
scripts/parity_test.py compares its output with your cached prep_v7_*.pkl to prove the two agree.

Settings reproduced from the best run: topk=3 evidence chunks of <=40 tokens, max_ev=100, max_ans=64, no D2T packing (topk_d2t=0),
IDF fitted on TRAIN sources only (artifacts/idf.json), feature groups = the six original ones (feat_groups="all")."""
from __future__ import annotations

import math

import numpy as np


class InputError(ValueError):
    """The request is well-formed but cannot be analysed (nothing to score, too many sentences, ...)."""


def build_source_info(question: str, context: str):
    """RAGTruth QA sources are {"question":..., "passages":...}; Summary sources are a plain string. model6.flatten() turns either into lines.
    The detector never sees a task label, so this is only a formatting choice (see README: domain-shift caveat)."""
    q = (question or "").strip()
    return {"question": q, "passages": context} if q else context


class Source:
    """Per-source objects that prepare() builds once per source id."""

    def __init__(self, source_info, M, XF):
        segraw, chunks = [], []
        for line in M.flatten(source_info):
            for _, _, seg in M.split_sents(line):
                t = M.TOK.findall(seg.lower())
                segraw.append(seg)
                chunks += [t[i:i + M.CHUNK] for i in range(0, len(t), M.CHUNK) if t[i:i + M.CHUNK]]   # prepare(), non-packed branch
        self.chunks = chunks
        self.csets = [set(c) for c in chunks]
        self.ctxset = set(t for c in chunks for t in c)
        self.xctx = XF.build_ctx(segraw, M.TOK)
        self.segments = [s.strip() for s in segraw if s.strip()]      # == XF.load_source_segments (used for SBERT)


def featurize_answer(src: Source, answer: str, idf: dict, M, XF, topk: int, max_ans: int, max_ev: int):
    """Sentence records with the SAME fields model6.prepare() creates, plus idx/offsets/truncated for the UI. NLI vector is attached later."""
    out = []
    for si, (s0, s1, seg) in enumerate(M.split_sents(answer)):
        allm = list(M.TOK.finditer(seg))
        ms = allm[:max_ans]
        raw = [m.group() for m in ms]
        if not raw:
            continue
        qs = set(t.lower() for t in raw)
        sc = [(sum(idf.get(w, M.MAXIDF) for w in qs & cs) / math.sqrt(len(cs) + 1), j) for j, cs in enumerate(src.csets)]
        top = sorted(sc, reverse=True)[:topk]
        ev = [t for _, j in top for t in src.chunks[j]][:max_ev] or ["<unk>"]
        low, Fm = M.tok_feats(raw, src.ctxset, set(ev), idf)
        Xc, xf = XF.sent_feats(seg, ms, raw, low, src.xctx, ev, idf, M.MAXIDF)
        Fm = np.hstack([Fm, Xc]).astype(np.float32)
        out.append(dict(rid=0, ry=0, y=0, idx=si, start=s0, end=s1, text=seg.strip(), low=low, F=Fm, ev=ev,
                        evF=np.array([[t in qs] for t in ev], dtype=np.float32), tl=np.zeros(len(low), np.float32),
                        xf=xf, truncated=len(allm) > max_ans))
    return out


def feature_keep(XF, groups) -> list:
    """Which of the 21 sentence features XF.finalize() keeps (sorted union over the enabled groups)."""
    return sorted(set(j for g in groups for j in XF.XF_GROUP.get(g, [])))


def attach_vectors(sents, sbert, nli_raw, norm, keep):
    """Builds s['nli'] = [standardised NLI features (8) | standardised, clipped sentence features (11)], i.e. what attach_nli +
    XF.finalize + XF.merge_into_nli produce for the training data. sbert: [n,2] (max cosine, mean of top-3) written into xf[9], xf[10]."""
    nli_raw = np.asarray(nli_raw, np.float32)
    for i, s in enumerate(sents):
        xf = s["xf"].copy()
        xf[9], xf[10] = sbert[i, 0], sbert[i, 1]
        xf = xf[keep].astype(np.float32)
        z = np.clip((xf - norm["xf_mu"]) / norm["xf_sd"], -5, 5).astype(np.float32)
        n0 = ((nli_raw[i] - norm["nli_mu"]) / norm["nli_sd"]).astype(np.float32)
        s["nli"] = np.r_[n0, z].astype(np.float32)
    return sents
