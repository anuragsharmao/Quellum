# Quellum backend

FastAPI service around the configured BiGRU ensemble (models 0–7 by default; seeds 8–9 are available as alternatives). The evaluated eight-model ensemble reports response-level F1 0.745 / AUROC 0.873 / AUPRC 0.804 on RAGTruth. No LLM judge, no retraining, no substituted classifier.

## What happens on a request
`question + context + answer` -> source = `{"question","passages"}` (or plain context if no question) -> answer split into sentences
-> per sentence: top-3 evidence chunks, 14 token features, 11 grounding features (+ SBERT all-MiniLM-L6-v2 similarity),
8 frozen-NLI features (cross-encoder, top-3 passages) -> configured `model6.Net` ensemble (8 models by default) -> **mean sentence logit** -> **noisy-OR** response score
-> flagged if the score is at least the threshold for that model set. The results report gives the 0–7 ensemble a validation-selected
threshold of 0.2439; do not use the old five-model threshold of 0.5507 for this eight-model ensemble.

## Setup (Windows, CPU)
The backend runtime is pinned to Python 3.12.11 (also used by the Docker image) because the pinned TensorFlow 2.21.0 does not provide a stable build for Python 3.14. Keep the Render service on Python 3.12.11 as well.

1. `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
2. Copy `extra_feats.py`, `nli_features.py`, and `synth.py` into `research/` (see [research/README.md](research/README.md)).
3. Weights for model ids 0–4 are in `models/seed_{0..4}.weights.h5`; ids 5–9 are in `models/seed{k}.weights.h5` (Keras 3 requires the suffix). The server defaults to ids 0–7; set `QUELLUM_DETECTOR_SEEDS` to a comma-separated unique list from 0–9 to choose an alternative combination.
4. Export training-time inference artifacts using the **run-level** results file and vocabulary from the relevant training output
   (plus `data/ragtruth`, `cache_v2/`, and `nli_cache/nli_feats.npz`):
   `python scripts/export_artifacts.py --results <run dir>/results.json --vocab <run dir>/vocab.json --data_dir <...>/data/ragtruth --cache_dir <...>/cache_v2 --nli_feats <...>/nli_cache/nli_feats.npz --nli_model cross-encoder/nli-deberta-v3-xsmall`
   (`--nli_model` must identify the model that generated the NLI cache.) The exporter writes `config.json`, `vocab.json`, `idf.json`,
   `nli_idf.json`, and `norm.npz` into `artifacts/`. The aggregate `artifacts/results.json` is an evaluation summary, not a run-level
   exporter input: it lacks training arguments and vocabulary references.
5. Ensure the generated runtime config and threshold match the configured seed ids. For models 0–7, the aggregate research report gives
   threshold 0.2439. The run-level five-model export selects a different threshold and is not calibrated for the eight-model ensemble.
6. Run `python scripts/verify_ensemble.py --predictions_dir <run dir> --results <run dir>/results.json --write_sentence_threshold`
   and require `ENSEMBLE REPRODUCED` for the corresponding five-seed run. Then run
   `python scripts/parity_test.py --data_dir ... --cache_dir ... --predictions_dir <run dir> --nli_feats ... --n 40`.
   Check feature mismatches, NLI/logit differences, and decision agreement before deployment.
7. `python -m uvicorn app.main:app --reload --port 8000` -> open `http://localhost:8000` for the integrated frontend/API and `/docs` for API documentation.

For a separately served local frontend, use the frontend instructions in [README.frontend.md](../frontend/README.frontend.md). The client
automatically targets the backend at port 8000 when served from localhost/127.0.0.1 on port 3000 or 5173.

### Interpreting startup logs

Uvicorn can start while optional subsystems are unavailable; a successful startup line does not mean the detector, retrieval, or answer
generation is ready.

- `artifacts_missing: config.json, vocab.json, idf.json, nli_idf.json, norm.npz`: export the runtime artifacts as above. Detector routes
  remain unavailable until these files and the required research modules and model weights are present.
- `embedder 'sbert' is unavailable`: install the backend requirements and ensure the configured SBERT model can be loaded. For local
  development only, `QUELLUM_EMBEDDER=hashing` avoids the SBERT dependency but provides weaker retrieval.
- `answer generation is NOT configured`: provide `GROQ_API_KEY` and optionally `GEMINI_API_KEY` in the process environment before
  starting Uvicorn. Copying `.env.example` to `.env` alone does not load variables; the application reads the process environment.
- The Pydantic `model_info` protected-namespace message is a warning; it is separate from the detector-artifact, embedder, and provider
  availability errors.

## API
`GET /`, `GET /health` (503 until the configured model set is loaded), `GET /api/v1/model-info`, `POST /api/v1/detect`, `POST /api/v1/analyze`
(adds per-seed logits, per-seed response scores, grounding features). Body: `{"question": "", "context": "...", "answer": "..."}`.
Scores are model scores, not calibrated probabilities. `label` on sentences appears only if `sentence_threshold.json` exists.

## Answer generation
- Groq is the primary answer provider (`GROQ_API_KEY`); Gemini (`GEMINI_API_KEY`) is an optional fallback and is called only when the Groq request fails. The default models can be overridden with `QUELLUM_LLM_MODEL` and `QUELLUM_LLM_FALLBACK_MODEL`.
- Prompts are capped by `QUELLUM_LLM_MAX_PROMPT_CHARS` (default 12,000 characters), with the question retained and retrieved passage text trimmed as needed. One call is made to the primary and at most one to the fallback; there are no automatic retries that spend extra quota.
- Existing OpenAI-compatible and Anthropic configurations remain supported with `QUELLUM_LLM_PROVIDER`, `QUELLUM_LLM_MODEL`, and `QUELLUM_LLM_API_KEY`.

## Research results

The checked-in aggregate report evaluates 2,700 held-out RAGTruth responses (900 each for QA, Data2txt, and Summary). Results below use
the validation-selected threshold for each prefix ensemble:

| Ensemble | Seeds | F1 | AUROC | AUPRC | Threshold |
| -------- | ----- | -: | ----: | ----: | --------: |
| Primary | 0–7 | 0.7451 | 0.8732 | 0.8043 | 0.2439 |
| All evaluated models | 0–9 | 0.7394 | 0.8725 | 0.8038 | 0.4741 |

The complete analysis, per-task breakdown, baselines, and reproduction checks are in [`research/README.md`](research/README.md) and
[`artifacts/results.json`](artifacts/results.json).

## Honest limits
- Trained on RAGTruth only; other domains unmeasured. Wrapping the context as `{"question","passages"}` is a formatting choice, untested off-distribution.
- Metrics and thresholds are tied to the evaluated seed set; use validation-derived calibration for any other selection.
- Model sees the first 64 tokens of each sentence (`truncated: true` flags this) and 3 evidence chunks of <=40 tokens.
- Sentence labels come from a new validation-selected threshold; the published 0.741 is response-level only.
- Latency / memory are NOT measured yet: run `parity_test.py`, then read `timing_ms` in responses on your CPU.
- Torch is imported before TensorFlow on purpose (Windows DLL conflicts). If both still clash, the NLI scorer can be moved to a subprocess.
