"""Frozen-NLI features, reproducing nli_features.py for one response: source cut into ~90-word passages, top-k passages per answer sentence
by IDF overlap (passage-level IDF from official-TRAIN sources: artifacts/nli_idf.json), a frozen NLI cross-encoder scores
(premise = passage, hypothesis = sentence), 8 features per sentence in the order of nli_features.FEATS.
The cross-encoder MUST be the one that produced nli_feats.npz (artifacts/config.json -> nli.model); parity_test.py checks this."""
from __future__ import annotations

import math

import numpy as np


class NliScorer:
    def __init__(self, NF, cfg: dict, idf: dict, unk: float, threads: int = 0):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.NF, self.torch = NF, torch
        self.cfg, self.idf, self.unk = cfg, idf, unk
        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(cfg["model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg["model"]).eval()
        names = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        self.ci = [i for i, n in names.items() if n.startswith("contra")][0]
        self.ei = [i for i, n in names.items() if n.startswith("entail")][0]
        self.ni = [i for i, n in names.items() if n.startswith("neutral")][0]

    def features(self, source_info, texts) -> np.ndarray:
        NF, idf, unk, c = self.NF, self.idf, self.unk, self.cfg
        passages = NF.make_passages(source_info, c["max_words"])
        Pw = [set(w.lower() for w in NF.WORD.findall(p)) for p in passages]
        jobs, pairs = [], {}
        for seg in texts:
            q = set(w.lower() for w in NF.WORD.findall(seg))
            qmass = sum(idf.get(w, unk) for w in q) + 1e-6
            sc = [sum(idf.get(w, unk) for w in q & pw) / math.sqrt(len(pw) + 1) for pw in Pw]
            order = np.argsort(sc)[::-1][:c["topk"]]
            hyp = " ".join(seg.split()[:100])
            keys, cov = [], 0.0
            for j, oi in enumerate(order):
                k = NF.pair_key(passages[oi], hyp)
                keys.append(k)
                pairs[k] = (passages[oi], hyp)
                if j == 0:
                    cov = sum(idf.get(w, unk) for w in q & Pw[oi]) / qmass
            jobs.append((keys, cov))
        probs = self._score(pairs)
        X = np.zeros((len(jobs), len(NF.FEATS)), np.float32)
        for i, (keys, cov) in enumerate(jobs):
            p = np.stack([probs[k] for k in keys]).astype(np.float32)         # [k, (contra, ent, neutral)]
            X[i] = [p[:, 1].max(), p[:, 1].mean(), p[:, 0].max(), p[:, 0].mean(), p[:, 2].min(), p[0, 1], p[0, 0], cov]
        return X

    def _score(self, pairs):
        torch, c = self.torch, self.cfg
        todo = sorted(pairs, key=lambda k: len(pairs[k][0]) + len(pairs[k][1]))
        done = {}
        with torch.inference_mode():
            for b in range(0, len(todo), c["bs"]):
                ks = todo[b:b + c["bs"]]
                enc = self.tok([pairs[k][0] for k in ks], [pairs[k][1] for k in ks], truncation="only_first",
                               max_length=c["max_len"], padding=True, return_tensors="pt")
                pr = torch.softmax(self.model(**enc).logits, -1).numpy()
                for k, p in zip(ks, pr):
                    done[k] = np.array([p[self.ci], p[self.ei], p[self.ni]], np.float16)    # stored as float16 in the research cache too
        return done
