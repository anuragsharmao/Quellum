"""Loads the configured trained models ONCE and runs the full inference pipeline:
   question/context/answer -> sentence features -> SBERT + NLI features -> Net ensemble -> mean logits -> noisy-OR -> threshold.
The architecture is built from the research run's own arguments (artifacts/config.json <- results.json["args"])."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .. import research
from ..config import Settings
from . import ensemble
from . import preprocessing as P

log = logging.getLogger("quellum.detector")

REQUIRED_ARTIFACTS = ["config.json", "vocab.json", "idf.json", "nli_idf.json", "norm.npz"]
SENTENCE_LABELS = ("Supported", "Potentially Hallucinated")


class ModelLoadError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


class Detector:
    def __init__(self, settings: Settings):
        self.s = settings
        self.ready = False
        self._lock = threading.Lock()
        self.nli_provider = None          # overridable (parity_test uses cached NLI rows)

    # ------------------------------------------------------------------ loading
    def load(self):
        s = self.s
        self.seed_ids = s.detector_seeds
        if not self.seed_ids or len(set(self.seed_ids)) != len(self.seed_ids) or any(k < 0 or k > 9 for k in self.seed_ids):
            raise ModelLoadError("invalid_model_seeds", "QUELLUM_DETECTOR_SEEDS must contain unique ids from 0 through 9")
        missing = [n for n in REQUIRED_ARTIFACTS if not (s.artifacts_dir / n).exists()]
        if missing:
            raise ModelLoadError("artifacts_missing", ", ".join(missing))
        art = s.artifacts_dir
        self.cfg = json.load(open(art / "config.json", encoding="utf8"))
        a = self.cfg["model_args"]
        self.a = a
        self.vocab = json.load(open(art / "vocab.json", encoding="utf8"))
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.idf = json.load(open(art / "idf.json", encoding="utf8"))["idf"]
        nj = json.load(open(art / "nli_idf.json", encoding="utf8"))
        z = np.load(art / "norm.npz")
        self.norm = {k: z[k].astype(np.float32) for k in ("nli_mu", "nli_sd", "xf_mu", "xf_sd")}
        st = art / "sentence_threshold.json"
        self.sentence_thr = json.load(open(st, encoding="utf8")) if st.exists() else None

        try:
            import torch  # noqa: F401  (imported BEFORE tensorflow: avoids DLL/OpenMP clashes seen on Windows)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError("torch_unavailable", repr(e))
        try:
            self.M, self.XF, self.NF = research.load(s.research_dir)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError("research_import_failed", repr(e))
        M, XF = self.M, self.XF
        self.groups = XF.parse_groups(a["feat_groups"])
        self.keep = P.feature_keep(XF, self.groups)
        if len(self.keep) + 8 != a["nli_dim"]:
            raise ModelLoadError("feature_dim_mismatch", f"{len(self.keep)}+8 != {a['nli_dim']}")
        self.mode = dict(nofeat=a["no_features"], noev=a["ablate"] == "no_evidence")
        self.sent_score = a["sent_score"]

        if s.tf_threads:
            import tensorflow as tf
            try:
                tf.config.threading.set_intra_op_parallelism_threads(s.tf_threads)
                tf.config.threading.set_inter_op_parallelism_threads(s.tf_threads)
            except RuntimeError:
                pass
        self.nets = [self._build_net(k) for k in self.seed_ids]

        from .nli import NliScorer
        from .sbert import SbertScorer
        self.sbert = SbertScorer(self.cfg["sbert"]["model"])
        self.nli = NliScorer(self.NF, self.cfg["nli"], nj["idf"], nj["unk"], s.torch_threads)
        self.nli_provider = lambda source_info, sents: self.nli.features(source_info, [x["text"] for x in sents])
        self.ready = True

    def _weights_path(self, k: int) -> Path:
        for name in (f"seed_{k}.weights.h5", f"seed{k}.weights.h5", f"seed_{k}.h5", f"seed{k}.h5",
                     f"best_seed{k}.weights.h5", f"best_seed{k}_weights.h5"):
            p = self.s.models_dir / name
            if p.exists():
                if p.name.endswith(".weights.h5"):         # Keras 3 insists on this suffix
                    return p
                if not hasattr(self, "_tmp"):
                    self._tmp = tempfile.mkdtemp(prefix="quellum_w_")
                q = Path(self._tmp) / f"seed_{k}.weights.h5"
                shutil.copyfile(p, q)
                return q
        raise ModelLoadError("weights_missing", f"seed {k}")

    def _build_net(self, k: int):
        M, a = self.M, self.a
        net = M.Net(len(self.vocab), a["emb"], a["hidden"], a["dropout"], use_attn=a["ablate"] != "no_cross_attn",
                    nli_dim=a["nli_dim"], emb_drop=a["emb_drop"], use_ctx=a["ctx"], use_kd=a["kd_weight"] > 0)
        dummy = dict(rid=0, ry=0, y=0, low=["a"], F=np.zeros((1, M.NF), np.float32), ev=["a"], evF=np.zeros((1, 1), np.float32),
                     tl=np.zeros(1, np.float32), nli=np.zeros(a["nli_dim"], np.float32))
        net(M.model_inputs(M.collate([dummy], self.w2i, **self.mode)), training=False)      # create the variables
        try:
            net.load_weights(str(self._weights_path(k)))
        except ModelLoadError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError("weights_incompatible", f"seed {k}: {e!r}")
        return net

    # ------------------------------------------------------------------ inference
    def analyze(self, question: str, context: str, answer: str, detail: bool = False) -> dict:
        return self.analyze_source_info(P.build_source_info(question, context), answer, detail)

    def analyze_source_info(self, source_info, answer: str, detail: bool = False) -> dict:
        if not self.ready:
            raise ModelLoadError("not_ready")
        M, a, T = self.M, self.a, {}
        t0 = time.perf_counter()
        src = P.Source(source_info, M, self.XF)
        if not src.chunks:
            raise P.InputError("the context contains no analysable text")
        sents = P.featurize_answer(src, answer, self.idf, M, self.XF, a["topk"], a["max_ans"], a["max_ev"])
        if not sents:
            raise P.InputError("the answer contains no analysable text")
        from ..config import MAX_SENTENCES
        if len(sents) > MAX_SENTENCES:
            raise P.InputError(f"the answer has {len(sents)} sentences; the limit is {MAX_SENTENCES}")
        t1 = time.perf_counter(); T["preprocess_ms"] = 1000 * (t1 - t0)
        sims = self.sbert.sims(src.segments, [x["text"] for x in sents])
        t2 = time.perf_counter(); T["sbert_ms"] = 1000 * (t2 - t1)
        nli_raw = self.nli_provider(source_info, sents)
        t3 = time.perf_counter(); T["nli_ms"] = 1000 * (t3 - t2)
        P.attach_vectors(sents, sims, nli_raw, self.norm, self.keep)
        with self._lock:                                   # Keras models are not guaranteed thread-safe
            per = np.stack([np.asarray(M.predict(net, sents, self.w2i, self.mode, bs=256, sent_score=self.sent_score)[1])
                            for net in self.nets])
        t4 = time.perf_counter(); T["models_ms"] = 1000 * (t4 - t3)
        T["total_ms"] = 1000 * (t4 - t0)

        mean = ensemble.mean_logits(per)
        thr = float(self.cfg["ensemble"]["thr"])
        resp_logit = ensemble.noisy_or_logit(mean)
        sthr = self.sentence_thr["thr_logit"] if self.sentence_thr else None
        out_s = []
        for i, x in enumerate(sents):
            row = dict(index=x["idx"], text=x["text"], start=x["start"], end=x["end"], truncated=x["truncated"],
                       logit=float(mean[i]), score=float(ensemble.sigmoid(mean[i])),
                       label=None if sthr is None else SENTENCE_LABELS[int(mean[i] >= sthr)])
            if detail:
                row["seed_logits"] = [float(v) for v in per[:, i]]
                row["features"] = dict(zip(self.cfg["xf_names"], [float(v) for v in x["xf"][self.keep]]))
            out_s.append(row)
        n_flag = None if sthr is None else sum(1 for r in out_s if r["label"] == SENTENCE_LABELS[1])
        res = dict(
            analysis_id=uuid.uuid4().hex,
            summary=dict(overall_score=resp_logit, overall_risk=ensemble.noisy_or_risk(mean), decision_threshold=thr,
                         decision_threshold_risk=float(ensemble.sigmoid(thr)), flagged=bool(resp_logit >= thr),
                         total_sentences=len(out_s), potentially_hallucinated=n_flag,
                         supported=None if n_flag is None else len(out_s) - n_flag),
            sentences=out_s, timing_ms={k: round(v, 1) for k, v in T.items()})
        if detail:
            res["per_seed_response_scores"] = [ensemble.noisy_or_logit(per[m]) for m in range(per.shape[0])]
            res["seed_ids"] = list(self.seed_ids)
        return res
