# PIP-06: metrics and dummy baselines

Generated on **2026-11-09** with `python -m src.baselines` (Python 3.14.8,
scikit-learn 1.9.1), run `baselines-7636d7f13ae2`, split `split-8b25e438ab683c89`,
seed **42**. Both dummies were fitted on **training labels only** (8,632 rows) and
scored on the **validation partition only** (1,849 rows, 286 purchases). The test
partition was not scored.

## What the baselines are

| Model | sklearn `DummyClassifier` strategy | Purchase score for every session |
| --- | --- | --- |
| `majority_class` | `most_frequent` | 0 (training majority is non-purchase) |
| `class_prior` | `prior` | 0.154773 = 1,336 / 8,632 training purchase rate |

AP, ROC-AUC and Brier come from these **scores**, never from hard classes. Hard
classes use the protocol rule "purchase when score ≥ threshold", under two rules:
the fixed **0.5 reference** and the protocol's **validation max-F1** threshold
(distinct scores plus 0/0.5/1, ties to the largest threshold).

## Validation results

| Model | Threshold rule | Threshold | AP | ROC-AUC | Brier | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority_class | reference 0.5 | 0.5 | 0.1547 | 0.5000 | 0.1547 | 0.8453 | 0.0000 | 0.0000 | 0.0000 | 1,563 | 0 | 286 | 0 |
| majority_class | validation max-F1 | 0 | 0.1547 | 0.5000 | 0.1547 | 0.1547 | 0.1547 | 1.0000 | 0.2679 | 0 | 1,563 | 0 | 286 |
| class_prior | reference 0.5 | 0.5 | 0.1547 | 0.5000 | 0.1308 | 0.8453 | 0.0000 | 0.0000 | 0.0000 | 1,563 | 0 | 286 | 0 |
| class_prior | validation max-F1 | 0.1548 | 0.1547 | 0.5000 | 0.1308 | 0.1547 | 0.1547 | 1.0000 | 0.2679 | 0 | 1,563 | 0 | 286 |

Confusion matrices: rows actual, columns predicted, ordered non-purchase then
purchase (`[[TN, FP], [FN, TP]]`).

## Why accuracy is misleading here

On the **full CSV**, always predicting no purchase is correct for 10,422 of 12,330
sessions: **84.53% accuracy with zero purchase recall**. That figure is label-count
arithmetic stated for context; it includes test labels and is not an evaluation
result. The **actual validation value** is 1,563 / 1,849 = **84.53% accuracy, 0
recall** (table above). A model with ~85% accuracy may therefore find no purchases
at all, which is why AP is the primary metric.

## Constant scores: AP equals prevalence

A constant score ties every session, so the precision-recall curve has one point
(recall 1, precision = purchase rate). **AP therefore equals the evaluation-set
prevalence** (286 / 1,849 = 0.154678), not the training prior used as the score,
and ROC-AUC is exactly 0.5. Both dummies match this to within 1e-12; the run fails
if they do not. The prior dummy is better only on Brier (0.1308 vs 0.1547),
because its constant is close to the true rate.

**Floors for later models on validation:** AP 0.1547, ROC-AUC 0.5, Brier 0.1308,
and purchase F1 0.2679, the F1 of predicting every session as a purchase, which the
max-F1 threshold rule selects for any constant score.

## Zero-division handling

When no session is predicted as a purchase, precision is undefined (0/0), and so
is F1. As the protocol states, **undefined precision, recall or F1 is reported as
0** (`zero_division=0`). AP and ROC-AUC are undefined with only one class present;
evaluation then fails instead of reporting a value.

## Outputs and reproduction

```bash
python -m src.baselines
python -m unittest tests.test_baselines
```

Files in `results/baselines/` (byte-identical on rerun; no timestamps):

| File | Contents |
| --- | --- |
| `metrics.csv` | Standard result table: one row per model × threshold rule |
| `predictions.csv` | Per validation session: score, threshold and predicted class |
| `confusion_matrices.csv` | Long format: actual, predicted, count |
| `results.json` | Run configuration, conventions, metrics, checks and notes |

Every CSV row starts with `run_id`, `split_id` and `seed`. The split ID hashes the
frozen metadata and manifests; the run ID hashes the run configuration, including
split ID, source and code checksums and library versions. Later models should
reuse `src.evaluation` (`evaluate_scores`, `select_f1_threshold`, `write_outputs`)
so their outputs share these columns and conventions.
