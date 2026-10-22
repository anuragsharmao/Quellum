"""Passage embeddings for retrieval. `sbert` uses the same all-MiniLM-L6-v2 the detector already needs (no extra model).
`hashing` is a dependency-free signed hashing bag-of-words embedder for development/tests: deterministic but lexical, weaker than SBERT."""
from __future__ import annotations

import re
import threading
import zlib

import numpy as np

_WORD = re.compile(r"[\w]+", re.UNICODE)


class HashingEmbedder:
    name = "hashing-512"
    dim = 512

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in _WORD.findall(t.lower()):
                h = zlib.crc32(w.encode("utf8"))
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            out[i] = np.sign(out[i]) * np.sqrt(np.abs(out[i]))
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class SbertEmbedder:
    name = "sbert-all-MiniLM-L6-v2"
    dim = 384

    def __init__(self, model=None, model_name="all-MiniLM-L6-v2"):
        self._model, self._model_name, self._lock = model, model_name, threading.Lock()

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device="cpu")
            self._model.max_seq_length = 256
        return self._model

    def embed(self, texts):
        with self._lock:
            e = self._get().encode([t[:2000] for t in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(e, np.float32)


def build_embedder(kind: str, shared_sbert_model=None):
    return HashingEmbedder() if kind == "hashing" else SbertEmbedder(model=shared_sbert_model)
