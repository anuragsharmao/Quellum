# Quellum — Hallucination Detection Research

This repository contains the research implementation of a lightweight hallucination detector for Retrieval-Augmented Generation (RAG). The research investigates how grounding-related features, neural sequence modelling, and ensemble scoring can help identify unsupported content in generated responses.

The objective is to develop a computationally practical approach to hallucination detection without relying on an LLM-based judge.

## Research Objective

Retrieval-Augmented Generation provides language models with external evidence, but generated answers may still contain unsupported claims or contradict the retrieved context.

This research explores a feature-based detection approach that evaluates the relationship between generated responses and their supporting evidence.

The main areas of investigation are:

* Evidence-grounding feature extraction.
* BiGRU-based sequence modelling.
* NLI-derived features for evidence comparison.
* Ensemble prediction across independently trained models.
* Experimental evaluation of detection performance.

## Methodology

### Grounding Features

The detector uses features that capture different aspects of the relationship between a response and its supporting evidence. These include lexical coverage, semantic similarity, entity mismatches, numerical inconsistencies, and other grounding-related signals.

### BiGRU-Based Detection

A bidirectional gated recurrent unit (BiGRU) processes response-level features to learn patterns associated with supported and unsupported content.

### NLI-Derived Features

Additional features derived from natural language inference provide signals about relationships between generated content and retrieved evidence, including entailment and contradiction.

### Ensemble Scoring

Predictions from independently trained models can be combined to form an ensemble. The selected models, score aggregation procedure, and decision threshold affect the resulting performance and should be documented for each reported configuration.

## Dataset and Evaluation

Experiments use **RAGTruth**, a benchmark dataset for hallucination detection in retrieval-augmented generation.

Source-disjoint data splits are used to reduce the risk of leakage between training and evaluation examples.

Performance is evaluated using metrics including:

* **F1-score:** Measures the balance between precision and recall at a selected decision threshold.
* **AUROC:** Measures discrimination between hallucinated and faithful examples across thresholds.
* **AUPRC:** Summarizes the precision-recall trade-off across thresholds and is particularly informative when class distributions are imbalanced.

### Reported Results

| Metric   | Score |
| -------- | ----: |
| F1-score |  73.9 |
| AUROC    | 0.873 |

These results must be interpreted in the context of the specific evaluation configuration from which they were obtained.

The metrics and threshold currently stored in `artifacts/` correspond to the original five-model validation run. They should not be presented as results for the eight-model ensemble unless that configuration has been evaluated separately.

## Implementation

The research implementation consists of the following modules:

| File              | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `model6.py`       | Model architecture, training logic, and inference utilities |
| `extra_feats.py`  | Additional feature extraction                               |
| `nli_features.py` | NLI-derived feature processing                              |
| `synth.py`        | Synthetic training-data utilities                           |

These files should remain compatible with one another. In particular, `model6.py` imports `synth.py` at module load time, even when synthetic augmentation is disabled.

### Inference and Artifact Utilities

The implementation also includes utilities for preparing and validating the detector artifacts:

* `export_artifacts.py`
* `verify_ensemble.py`
* `parity_test.py`

These support artifact export, ensemble verification, and consistency testing of the inference implementation.

## Reproducibility

Reproducing the reported results requires matching the original experimental configuration, including:

* Dataset preprocessing and source-disjoint splits.
* Feature groups and feature-generation settings.
* NLI feature configuration.
* Training hyperparameters and random seeds.
* Decision-threshold selection.
* Ensemble membership and score aggregation.
* Evaluation metrics on the held-out test set.

Cached features must correspond to the preprocessing and feature configuration used in the relevant experiment. Synthetic augmentation, when enabled, must be restricted to training data.

For reliable comparisons, ablation experiments should change one principal component at a time while keeping the remaining evaluation conditions consistent.

## Limitations

* Performance on RAGTruth does not guarantee equivalent performance on other datasets or domains.
* Feature extraction and NLI processing introduce additional computational costs.
* The detector may produce false positives and false negatives.
* Ensemble performance depends on model selection, score aggregation, and threshold calibration.
* Further evaluation on independent datasets is needed to assess generalization.

## Research Scope

This work investigates a lightweight, feature-based approach to hallucination detection in RAG systems. It focuses on evidence-grounding signals, recurrent sequence modelling, and ensemble-based evaluation.

The aim is to assess whether these components can provide useful hallucination detection performance while avoiding dependence on an LLM-based judge.

