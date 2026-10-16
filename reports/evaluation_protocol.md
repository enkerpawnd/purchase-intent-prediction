# PIP-03: split and evaluation protocol

Frozen and verified on **2026-11-08**. Split checks are complete; model training
and test performance evaluation have not been performed. See the
[README](../README.md#frozen-split-and-evaluation-protocol-pip-03) for usage.

## Why this split?

We reserve approximately **70% for training, 15% for validation and 15% for
final testing**. Most sessions support learning; separate held-out data supports
selection and final evaluation. These proportions are a practical choice,
not a demonstrated optimum.

Only **15.47% of sessions end in a purchase**. Stratification preserves that
minority rate across partitions, reducing differences in class composition
that could obscure model comparisons.

We retain all sessions but keep identical predictor rows together. The 201
records in repeated groups are not proven duplicate users, so deletion is
unjustified. Grouping prevents identical predictors appearing on both sides
of evaluation and potentially inflating performance. Groups use all 17
predictors and remain fixed for feature comparisons.

A fixed seed of **42** makes experiments comparable. Twenty grouped, stratified
parts are combined as 14/3/3; grouping can alter exact sizes. No seed search
was used to obtain favorable results.

| Partition | Sessions | Non-purchases | Purchases | Purchase rate |
| --- | ---: | ---: | ---: | ---: |
| Training | 8,632 | 7,296 | 1,336 | 15.4773% |
| Validation | 1,849 | 1,563 | 286 | 15.4678% |
| Test | 1,849 | 1,563 | 286 | 15.4678% |

## Why five-fold cross-validation?

A single internal holdout can favor a hyperparameter setting by chance.
**Five-fold cross-validation within training** compares settings across five
held-out subsets, revealing variation across them. Each session supports
fitting in four folds and evaluation in one, efficiently reusing the training
data and its 1,336 purchases.

Five folds balance data reuse, purchases per holdout and computation.
Cross-validation does not remove dataset bias. Folds preserve purchase
representation and predictor groups; validation/test never enter CV.
Every model and feature comparison reuses these folds.

| Fold | Fit sessions | Fit purchases | Holdout sessions | Holdout purchases |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6,905 | 1,068 | 1,727 | 268 |
| 1 | 6,905 | 1,069 | 1,727 | 267 |
| 2 | 6,906 | 1,069 | 1,726 | 267 |
| 3 | 6,906 | 1,069 | 1,726 | 267 |
| 4 | 6,906 | 1,069 | 1,726 | 267 |

## Why these metrics and threshold?

**Purchase average precision (AP) is primary** to assess finding minority
purchases while limiting false alarms. Accuracy can look high when always
predicting no purchase. Calculate non-interpolated AP from purchase scores.

| Secondary measure | What it adds |
| --- | --- |
| ROC-AUC | Ranking performance across decision thresholds |
| Purchase precision, recall and F1 | False-alarm and missed-purchase tradeoffs at the chosen threshold |
| Confusion matrix | Counts of correct/incorrect decisions; rows actual, columns predicted, ordered non-purchase then purchase |
| Brier score and reliability plot | Probability quality and agreement between predicted probabilities and observed purchase rates |

Reliability plots use ten equal-width bins with counts; Brier alone does not
prove better calibration. Undefined precision, recall or F1 is reported as zero.

After model/feature/calibration selection, **maximize purchase F1 on
validation** for a precision/recall compromise without invented business costs.
Consider distinct validation probabilities plus 0, 0.5 and 1; predict purchase
when probability is at least the threshold. Break equal-F1 ties with the
**largest threshold**, favoring fewer purchase predictions. Report **0.5 as a
reference**. Changes need a documented rationale before final testing.

## How evaluation stays separate from development

Development EDA uses **training data only**. Fit preprocessing and tune within
training CV folds to prevent evaluation rows influencing learned transforms.
Fit calibration on out-of-fold training predictions or a grouped training
holdout. When assessing calibration within CV, fit it inside that fold's
training side.

Training CV and validation guide model, feature and calibration choices;
validation sets the threshold. Their scores are development estimates.
Before **one final test evaluation**, freeze each comparator's fitted pipeline
and threshold, code version and split. Predeclare the selected model, dummy
baselines and required feature-study counterparts. Do not reselect after test
results or refit on validation/test. Disclose evaluation bugs and corrections.

## Evidence and limits

Versioned manifests in `data/processed/splits/` record partition, predictor group
and training CV fold. One source checksum links saved row positions to their
CSV. One-time verification passed **14 checks**, covering row coverage, group
separation, training-only CV, reproducibility and source matching. The split
command's built-in checks remain available for later runs. No rows were removed
or source data changed.

Random session splits do **not** establish future-period or cross-site
performance. The source lacks precise timestamps and user/site identifiers;
predictor grouping cannot establish user-level independence. Cross-validation
cannot correct those limits or establish feature availability at prediction time.
