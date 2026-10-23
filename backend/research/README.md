# Quellum hallucination-detection research

This folder contains the research implementation and manuscript for Quellum's RAG hallucination detector. The study evaluates whether a
compact evidence-grounding model can distinguish unsupported from supported generated responses without using an LLM as a judge.

## Research question and approach

The detector is evaluated on RAGTruth response annotations across its QA, Data2txt, and Summary tasks. Each response is represented at the
sentence level and scored against its source evidence. The model combines:

- A BiGRU answer/evidence architecture with cross-attention and response-level sentence context (`model6.Net`).
- Lexical and grounding features, including stemmed unigram, bigram and trigram coverage; numeric, date and entity checks; novel-token runs;
  and all-MiniLM-L6-v2 sentence-embedding similarity.
- Eight frozen-NLI features derived from a cross-encoder and retrieved passages.
- Auxiliary token and response-level training losses. At inference, sentence logits are averaged across models and pooled into a response
  score with noisy-OR; the response decision threshold is selected on validation data.

The reported primary configuration uses NLI features, response-level noisy-OR loss, sentence-context modelling, and no synthetic training
examples. Ten independently trained seeds (0–9) are evaluated. Seeds 0–4 and 5–9 were trained in separate runs using matching model
configurations; the aggregate report records no configuration differences.

## Test-set results

The aggregate report evaluates 2,700 held-out responses: 900 each for QA, Data2txt, and Summary. The requested eight-model primary
ensemble corresponds to the first eight seeds (0–7). Results are reported at the response level; thresholds shown were selected using
validation data for the corresponding ensemble size.

| Ensemble | Seeds | Test F1 | Test AUROC | Test AUPRC | Validation-selected threshold |
| -------- | ----- | ------: | ---------: | ---------: | ----------------------------: |
| Primary | 0–7 | 0.7451 | 0.8732 | 0.8043 | 0.2439 |
| All evaluated models | 0–9 | 0.7394 | 0.8725 | 0.8038 | 0.4741 |
| Original five-model run | 0–4 | 0.7407 | 0.8729 | 0.8052 | 0.5507 |

For the ten-model result, the report gives 95% source-bootstrap intervals of 0.7146–0.7636 for F1, 0.8578–0.8872 for AUROC, and
0.7749–0.8335 for AUPRC. The first-eight result has the highest F1 among the reported first-k ensembles; the report also evaluates
random seed subsets and does not claim that adding models always improves performance.

On the ten-model evaluation, per-task F1 is 0.6188 (QA), 0.8515 (Data2txt), and 0.4682 (Summary). Performance therefore varies by task;
the aggregate score should not obscure the weaker Summary result. The ten-model F1 is 0.0647 above the strongest listed hand-feature
GBM baseline (0.6747).

The saved `reproduction_check` reports that the backend aggregation reproduces the recorded F1 and AUROC for both five-seed runs. See
[`../artifacts/results.json`](../artifacts/results.json) for the full metrics, per-task and per-generator breakdowns, seed comparisons,
baselines, ensemble-size analysis, and figure captions. The manuscript preview is
[`paper_final_revised_PREVIEW.pdf`](paper_final_revised_PREVIEW.pdf).

## Research files and runtime dependencies

| File | Role |
| ---- | ---- |
| `model6.py` | Research architecture, preprocessing/training utilities, and prediction functions used by the backend. |
| `extra_feats.py` | Grounding-feature definitions imported by `model6.py`. |
| `nli_features.py` | Frozen-NLI feature extraction code imported by the backend. |
| `synth.py` | Module imported by `model6.py`, even when the synthetic-data ratio is zero. |
| `paper_final_revised_PREVIEW.pdf` | Manuscript preview. |

Keep the original research modules together and compatible; the backend imports them directly rather than maintaining a second architecture.
The `nli_feats.npz` used in artifact export is a generated feature cache and is not a substitute for the `nli_features.py` module.

## Reproducing the experiment and exporting inference artifacts

Training and evaluation require the RAGTruth data, the original run directories (including their vocabularies and per-seed predictions),
the preprocessing/SBERT caches, and the NLI feature cache. The aggregate `artifacts/results.json` is an evaluation summary; it does not
contain the run arguments and vocabulary expected by `scripts/export_artifacts.py`. Use the run-level `results.json` and `vocab.json`
from the corresponding training output when exporting.

From `backend/`, the workflow is:

1. Install `requirements.txt` and place the required research modules listed above in this directory.
2. Run `scripts/export_artifacts.py` with the run-level results, vocabulary, RAGTruth data, matching cache, NLI feature cache, and the NLI
   model that generated that cache. The exporter writes `config.json`, `vocab.json`, `idf.json`, `nli_idf.json`, and `norm.npz` into
   `artifacts/`.
3. Run `scripts/verify_ensemble.py` with the matching run's predictions and results. Require `ENSEMBLE REPRODUCED` before trusting
   parity with that five-seed research run.
4. Run `scripts/parity_test.py` on held-out examples. Check feature mismatches, live-versus-cached NLI differences, sentence-logit
   differences, and response decisions before deployment.

The checked-in artifact summary evaluates the 0–7 prefix with a validation-selected response threshold of 0.2439. Ensure any runtime
configuration for the eight-model set uses its matching threshold and feature/model configuration; the original five-seed export uses a
different threshold and must not be reused as if it calibrated eight models.

## Scope and limitations

- These are benchmark results on RAGTruth, not a guarantee of generalization to other domains or production RAG systems.
- The threshold is specific to the evaluated model set and validation split. Scores are not calibrated probabilities.
- Sentence and token metrics are additional evaluation views; the production API returns sentence scores and a response-level decision, not
  token-level hallucination spans.
- The aggregate results are retrospective benchmark evidence. They do not establish latency, memory use, or performance under live traffic.
