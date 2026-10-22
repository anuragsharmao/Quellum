# Quellum backend

FastAPI service around the configured BiGRU ensemble (models 0–7 by default; seeds 8–9 are available as alternatives). The published five-seed research run reported F1 0.741 / AUROC 0.873 / AUPRC 0.805 at response level (best configuration: NLI on, no stacking). No LLM judge, no retraining, no substituted classifier.

## What happens on a request
`question + context + answer` -> source = `{"question","passages"}` (or plain context if no question) -> answer split into sentences
-> per sentence: top-3 evidence chunks, 14 token features, 11 grounding features (+ SBERT all-MiniLM-L6-v2 similarity),
8 frozen-NLI features (cross-encoder, top-3 passages) -> configured `model6.Net` ensemble (8 models by default) -> **mean sentence logit** -> **noisy-OR** response score
-> flagged if `score >= 0.5507` (validation-selected, from results.json).

## Setup (Windows, CPU)
1. `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
2. Copy `extra_feats.py`, `nli_features.py`, `synth.py` into `research/` (see research/README.txt).
3. Weights for model ids 0–4 are in `models/seed_{0..4}.weights.h5`; ids 5–9 are in `models/seed{k}.weights.h5` (Keras 3 requires the suffix). The server defaults to ids 0–7; set `QUELLUM_DETECTOR_SEEDS` to a comma-separated unique list from 0–9 to choose an alternative combination.
4. Export the training-time artifacts once (needs `data/ragtruth`, `cache_v2/`, `nli_cache/nli_feats.npz`):
   `python scripts/export_artifacts.py --results artifacts/results.json --vocab <results dir>/vocab.json --data_dir <...>/data/ragtruth --cache_dir <...>/cache_v2 --nli_feats <...>/nli_cache/nli_feats.npz --nli_model cross-encoder/nli-deberta-v3-xsmall`
   (`--nli_model` must be what produced nli_feats.npz; step 6 verifies it.)
5. `python scripts/verify_ensemble.py --predictions_dir <results dir> --results artifacts/results.json --write_sentence_threshold`
   must print `ENSEMBLE REPRODUCED`. It also writes the validation-derived sentence threshold.
6. `python scripts/parity_test.py --data_dir ... --cache_dir ... --predictions_dir <results dir> --nli_feats ... --n 40`
   Expect: features 0 mismatches; NLI diff ~1e-2; logit diffs small; decisions agree. Do not ship if these fail.
7. `uvicorn app.main:app --port 8000` -> `/docs`.

## API
`GET /`, `GET /health` (503 until the configured model set is loaded), `GET /api/v1/model-info`, `POST /api/v1/detect`, `POST /api/v1/analyze`
(adds per-seed logits, per-seed response scores, grounding features). Body: `{"question": "", "context": "...", "answer": "..."}`.
Scores are model scores, not calibrated probabilities. `label` on sentences appears only if `sentence_threshold.json` exists.

## Answer generation
- Groq is the primary answer provider (`GROQ_API_KEY`); Gemini (`GEMINI_API_KEY`) is an optional fallback and is called only when the Groq request fails. The default models can be overridden with `QUELLUM_LLM_MODEL` and `QUELLUM_LLM_FALLBACK_MODEL`.
- Prompts are capped by `QUELLUM_LLM_MAX_PROMPT_CHARS` (default 12,000 characters), with the question retained and retrieved passage text trimmed as needed. One call is made to the primary and at most one to the fallback; there are no automatic retries that spend extra quota.
- Existing OpenAI-compatible and Anthropic configurations remain supported with `QUELLUM_LLM_PROVIDER`, `QUELLUM_LLM_MODEL`, and `QUELLUM_LLM_API_KEY`.

## Honest limits
- Trained on RAGTruth only; other domains unmeasured. Wrapping the context as `{"question","passages"}` is a formatting choice, untested off-distribution.
- The saved threshold and reported benchmark metrics came from the original five-seed validation/test run. They remain in the artifacts for provenance but have not been recalibrated or remeasured for the default eight-model ensemble or other seed selections.
- Model sees the first 64 tokens of each sentence (`truncated: true` flags this) and 3 evidence chunks of <=40 tokens.
- Sentence labels come from a new validation-selected threshold; the published 0.741 is response-level only.
- Latency / memory are NOT measured yet: run `parity_test.py`, then read `timing_ms` in responses on your CPU.
- Torch is imported before TensorFlow on purpose (Windows DLL conflicts). If both still clash, the NLI scorer can be moved to a subprocess.
