"""SBERT similarity features xf[9] (max cosine to any source sentence) and xf[10] (mean of the top-3), as XF.attach_sbert computes them:
all-MiniLM-L6-v2 on CPU, max_seq_length 128, texts cut to 600 chars, normalised embeddings stored as float16."""
from __future__ import annotations

import numpy as np


class SbertScorer:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = 128

    def _encode(self, texts):
        return self.model.encode([t[:600] for t in texts], batch_size=128, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float16)

    def sims(self, source_segments, sentence_texts) -> np.ndarray:
        """-> [n_sentences, 2]. A source with no segments leaves both at 0.0 (same as attach_sbert)."""
        out = np.zeros((len(sentence_texts), 2), np.float32)
        if not source_segments:
            return out
        uniq = sorted(set(source_segments) | set(sentence_texts), key=len)
        E = self._encode(uniq)
        index = {t: i for i, t in enumerate(uniq)}
        S = E[[index[t] for t in source_segments]].astype(np.float32)
        for i, t in enumerate(sentence_texts):
            sims = S @ E[index[t]].astype(np.float32)
            out[i] = (sims.max(), np.sort(sims)[-3:].mean())
        return out
