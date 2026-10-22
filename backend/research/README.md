# Lightweight Hallucination Detection for Retrieval-Augmented Generation

This repository contains the research implementation of Quellum's hallucination detection component. The project investigates whether a lightweight, feature-based neural model can identify unsupported content in Retrieval-Augmented Generation (RAG) responses by examining the relationship between generated answers and retrieved evidence.

## Research Objective

Retrieval-Augmented Generation improves access to external knowledge but does not guarantee that every generated claim is supported by the retrieved context. This research explores a computationally efficient approach to hallucination detection using grounding-related features, sequence modelling, and ensemble prediction.

The objective is to develop and evaluate a detector that provides useful hallucination signals without relying on an LLM-based judge.

## Methodology

The detection pipeline combines three main components:

* **Grounding features:** Capture lexical overlap, semantic similarity, entity mismatches, numerical inconsistencies, and other signals related to evidence support.
* **BiGRU-based modelling:** Processes response-level features to learn patterns associated with supported and unsupported content.
* **Ensemble prediction:** Combines predictions from independently trained models to improve detection performance and reduce dependence on a single training run.

Additional experiments investigate NLI-derived features and alternative feature configurations to measure their contribution to detection performance.

## Dataset

Experiments use **RAGTruth**, a benchmark dataset containing responses generated in retrieval-augmented generation settings, with annotations identifying hallucinated content.

The evaluation uses source-disjoint data splits to reduce the risk of leakage between training and test examples.

## Experimental Results

The evaluated detector configuration achieved the following results on RAGTruth:

| Metric   | Result |
| -------- | -----: |
| F1-score |   73.9 |
| AUROC    |  0.873 |

The F1-score measures the balance between precision and recall at the selected decision threshold. AUROC measures how well the model distinguishes hallucinated responses from faithful ones across different thresholds.

These results apply to the evaluated configuration and dataset; they should not be interpreted as guaranteed performance on other datasets.

## Implementation

The research implementation is organized into the following modules:

| File              | Purpose                                                            |
| ----------------- | ------------------------------------------------------------------ |
| `model6.py`       | Model architecture, training logic, and inference utilities        |
| `extra_feats.py`  | Additional grounding-related feature extraction                    |
| `nli_features.py` | NLI-derived feature processing                                     |
| `synth.py`        | Synthetic training-data utilities used by the model implementation |

The backend imports the research implementation directly rather than duplicating the model architecture.

**Important:** Keep these modules compatible with one another. In particular, `model6.py` imports `synth.py` at module load time, even when synthetic augmentation is disabled.

## Reproducibility

To reproduce the reported results, use the corresponding model configuration, dataset preprocessing pipeline, grouped data splits, feature caches, random seeds, and ensemble aggregation procedure.

The evaluation configuration should specify:

* Dataset split strategy and preprocessing.
* Feature groups and NLI feature configuration.
* Training hyperparameters and random seeds.
* Decision-threshold selection procedure.
* Ensemble composition and score aggregation.
* Evaluation metrics and held-out test results.

Cached features must correspond to the same preprocessing and feature-generation configuration used in the experiment. Training-only synthetic augmentation must not introduce information from validation or test data.

## Limitations

* Performance on RAGTruth does not establish generalization to every RAG application or domain.
* Feature extraction and NLI processing introduce computational costs beyond the core neural model.
* Detection errors remain possible, including unsupported claims classified as faithful and supported claims classified as hallucinations.
* Further evaluation on independent datasets is needed to assess broader generalization.

## Integration with Quellum

This research implementation serves as the hallucination detection component of Quellum. The application uses the detector to assess generated responses against retrieved evidence, complementing the generation and retrieval stages of the RAG pipeline.

The research model and the application’s LLM generation providers serve different purposes: the detector evaluates evidence support, while the generation providers produce responses.

