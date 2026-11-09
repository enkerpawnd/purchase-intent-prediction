# PIP-07: logistic regression

Run on **2026-11-09** with `python -m src.train_logistic` (Python 3.14.8,
scikit-learn 1.9.1), run `logreg-b34da387f891`, split `split-8b25e438ab683c89`,
seed **42**. The model was fitted on the **8,632 training rows only** and scored
on the **1,849 validation rows only**. The test partition was not scored.

## What was trained

| Setting | Value |
| --- | --- |
| Inputs | `all_17` through the PIP-05 pipeline: validation, one-hot categories, standardized numerics (fitted on training rows) |
| Estimator | `sklearn.linear_model.LogisticRegression` |
| Objective | Log-loss (binary cross-entropy) plus an L2 penalty (`l1_ratio=0`); smaller `C` = stronger regularization |
| Solver | `lbfgs`, `tol=1e-4`, `max_iter=2000` |
| Seed | `random_state=42` (lbfgs is deterministic; recorded for consistency) |
| Selected | **`C=0.1`, `class_weight="balanced"`** |
| Convergence | All 40 CV fits and the final fit converged; final fit took 44 iterations, CV fits at most 126 |
| Runtime | 5.8 s for the CV comparison, 0.15 s for the final fit, 8.0 s in total (Apple M1 Pro; varies by machine, see `run_log.json`) |

## Bounded training-fold comparison

Eight predeclared settings were compared: four L2 strengths × unweighted vs
class-weighted (`balanced` weights purchases by about 5.5× in the loss). Each was
scored on the frozen five training CV folds. The whole pipeline was refitted
within each fold, so validation rows were never used. Selection rule: the highest
mean CV AP.

| Setting | Mean CV AP | SD | Mean ROC-AUC | Mean log-loss | Mean Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| C=0.01, unweighted | 0.6428 | 0.0295 | 0.8979 | 0.2959 | 0.0871 |
| C=0.1, unweighted | 0.6418 | 0.0285 | 0.8953 | **0.2920** | **0.0851** |
| C=1, unweighted | 0.6370 | 0.0283 | 0.8917 | 0.2933 | 0.0853 |
| C=10, unweighted | 0.6343 | 0.0278 | 0.8891 | 0.2946 | 0.0856 |
| C=0.01, balanced | 0.6430 | 0.0293 | 0.9012 | 0.4363 | 0.1264 |
| **C=0.1, balanced (selected)** | **0.6469** | 0.0279 | **0.9029** | 0.4366 | 0.1212 |
| C=1, balanced | 0.6447 | 0.0280 | 0.9010 | 0.4397 | 0.1214 |
| C=10, balanced | 0.6430 | 0.0283 | 0.8988 | 0.4412 | 0.1218 |

**Interpretation:** all eight settings fall within 0.013 AP of each other, well
inside one fold-to-fold standard deviation (~0.028). The selection follows the
predeclared rule, but the data do **not** show that class weighting or this `C`
ranks purchases meaningfully better. Weighting clearly **worsens probability
quality**: log-loss rises from ~0.29 to ~0.44 and Brier from ~0.085 to ~0.12.

## Validation results against the dummies

| Model | Threshold rule | Threshold | AP | ROC-AUC | Brier | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | reference 0.5 | 0.5 | **0.6644** | **0.9008** | 0.1197 | 0.8551 | 0.5216 | 0.7587 | 0.6182 | 1,364 | 199 | 69 | 217 |
| logistic_regression | validation max-F1 | 0.6057 | 0.6644 | 0.9008 | 0.1197 | 0.8864 | 0.6242 | 0.6678 | **0.6453** | 1,448 | 115 | 95 | 191 |
| majority_class | reference 0.5 | 0.5 | 0.1547 | 0.5000 | 0.1547 | 0.8453 | 0.0000 | 0.0000 | 0.0000 | 1,563 | 0 | 286 | 0 |
| class_prior | reference 0.5 | 0.5 | 0.1547 | 0.5000 | 0.1308 | 0.8453 | 0.0000 | 0.0000 | 0.0000 | 1,563 | 0 | 286 | 0 |
| either dummy | validation max-F1 | ≤0.155 | 0.1547 | 0.5000 | — | 0.1547 | 0.1547 | 1.0000 | 0.2679 | 0 | 1,563 | 0 | 286 |

- **Ranking:** AP is 4.3× the dummy floor (0.664 vs 0.155) and ROC-AUC is 0.90 vs 0.5.
  Validation AP is close to the CV estimate (0.647), with no sign of overfitting.
- **Decisions:** at 0.5 the model finds 76% of purchases (217 / 286), and 52% of its
  purchase predictions are correct. The validation max-F1 threshold (0.606) trades
  recall for precision (67% / 62%, F1 0.645 vs the 0.268 floor). That threshold was
  chosen on these same validation rows, so its F1 is optimistic.
- **Accuracy** is only 0.855 at 0.5, barely above the 0.845 of always predicting
  no purchase, even though the model is far better. This is why AP is primary.

## Probability quality (preliminary)

Brier is **0.1197**, only slightly better than the class-prior dummy (0.1308).
The reliability bins show **systematic overprediction**, the expected effect of
`class_weight="balanced"`:

| Predicted | 0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sessions | 348 | 437 | 301 | 184 | 163 | 108 | 62 | 39 | 51 | 156 |
| Mean predicted | 0.05 | 0.15 | 0.24 | 0.35 | 0.45 | 0.54 | 0.65 | 0.76 | 0.86 | 0.98 |
| Observed rate | 0.01 | 0.01 | 0.06 | 0.10 | 0.17 | 0.24 | 0.37 | 0.56 | 0.71 | 0.71 |

These probabilities rank sessions well but should **not** be read as purchase
chances. For the app or any probability claim, use the unweighted setting
(CV Brier ~0.085) or calibrate on out-of-fold training predictions as the protocol
requires. This is a weak result for calibration, recorded rather than hidden.

## Save, reload and reproduce

The complete fitted pipeline (raw-input validation, encoding, scaling, model) is
saved to `models/logistic_regression/all_17_logistic.joblib` (8 KB, ignored by
git like other model artifacts; rerunning recreates it byte for byte). The run
reloads it and checks that all 1,849 validation probabilities are **exactly
identical**. A test also reloads it in a fresh Python process and reproduces a
raw-session prediction (source row 18: probability 0.0532, no purchase at 0.5).
Only load trusted joblib files.

## Outputs and reproduction

```bash
python -m src.baselines          # dummy results the comparison reads
python -m src.train_logistic
python -m unittest tests.test_train_logistic
```

Files in `results/logistic_regression/` start with `run_id`, `split_id` and `seed`:

| File | Contents |
| --- | --- |
| `comparison.csv` | Validation result table: logistic regression and both dummies |
| `metrics.csv`, `predictions.csv`, `confusion_matrices.csv`, `results.json` | Standard PIP-06 outputs (`src.evaluation.write_outputs`) |
| `cv_summary.csv`, `cv_folds.csv` | Training-fold comparison per setting and per fold, including iterations and convergence |
| `pr_curve.csv`, `pr_curve.png` | Validation precision-recall curve with the dummy AP line |
| `reliability.csv`, `reliability.png` | Ten equal-width bins with counts, mean predicted and observed rate |
| `confusion_matrix.png` | Confusion matrices at both thresholds |
| `run_log.json` | Runtimes; the only file that changes between reruns |

All other files, including the PNGs and the saved pipeline, are byte-identical on rerun.
