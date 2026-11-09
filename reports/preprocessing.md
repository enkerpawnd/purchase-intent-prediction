# Reusable preprocessing and raw-session inference

Implemented on **2026-11-09** in [src/preprocessing.py](../src/preprocessing.py).
The local issue plan calls these acceptance criteria **PIP-05**; PIP-04 covers
the training exploration already in this repository. This implementation follows
the requested preprocessing criteria.

## Input and target contract

`split_features_target(frame, feature_set)` separates `Revenue` and returns
`(X, y)`, with purchase encoded as 1 and non-purchase as 0. It preserves row
order/index and does not mutate the source frame. It validates types but learns
no statistics. A prediction pipeline rejects `Revenue` in its inputs to catch
accidental label inclusion. Raw inference uses a pandas DataFrame; a single
session is `pd.DataFrame([session_dict])`.

All raw column lists are defined in [src/feature_sets.py](../src/feature_sets.py):

| Variant | Required raw inputs | Numeric inputs |
| --- | ---: | ---: |
| `all_17` | 17 | 10 |
| `without_pagevalues_16` | 16 (exclude PageValues) | 9 |
| `without_analytics_14` | 14 (also exclude BounceRates, ExitRates) | 7 |

All variants retain six nominal fields and one Boolean. Missing required,
duplicate or unexpected column names raise `ValueError` naming the problem.
Columns may arrive in any order. A smaller variant also accepts a full raw
predictor frame: known excluded fields are discarded before validation and
never influence the result. Unknown extra fields and `Revenue` are rejected.
Omitting a feature requires choosing and fitting the corresponding variant;
omitting required columns does not select a smaller model automatically.

| Field group | Accepted values and behavior |
| --- | --- |
| Administrative, Informational, ProductRelated | Finite nonnegative whole numeric counts. |
| Their durations and PageValues | Finite nonnegative real numbers. Preserve zero and large valid values. |
| BounceRates, ExitRates, SpecialDay | Finite numeric fractions in [0, 1]. A rate of 20 is rejected, not interpreted as 20%. |
| OperatingSystems, Browser, Region, TrafficType | Positive whole numeric category codes. Integral floats such as 2.0 normalize identically to 2. Numeric text such as `"2"` is rejected. No ordinal meaning is assigned. |
| Month, VisitorType | Nonblank strings. Exact spelling/case is preserved; no aliasing or substitution of `Other`. |
| Weekend and Revenue | Actual Python/NumPy Booleans, numeric 0/1 (including 0.0/1.0), or case-insensitive, whitespace-trimmed `"true"`/`"false"`. Normalize to integer 0/1. `"0"`, `"1"`, `"yes"` and other ambiguous text are rejected. |

Numeric measurements accept Python/NumPy real numbers, excluding Booleans,
numeric strings and complex numbers. Missing cells (`None`, NaN, pandas NA),
blank strings and infinities in required fields raise `ValueError`. There is
**no imputation**, clipping, replacement by zero, or missing-Boolean conversion
to false. The audited source has no missing values and needs no imputation.

Validity limits are independent of observed training ranges. For example, a
rate of 0.9 is allowed even though source rates top out near 0.2. Counts and
durations have no learned upper bound. The future app should show training-range
and unsupported-category warnings as described in
[feature availability](feature_availability.md); this pipeline implements
validation, not an app warning interface or live collection rules.

## Encoded inputs and model choices

`make_preprocessor(feature_set, scale_numeric=...)` returns an **unfitted**
sklearn Pipeline: `schema` (validation/Boolean normalization/column selection),
then `columns` (a ColumnTransformer). Output is a dense floating-point array in
this order: numeric inputs in source order, one-hot category blocks in the
explicit nominal-field order, and Weekend as one unscaled 0/1 indicator.
`get_feature_names_out()` exposes the fitted encoded names.

The six nominal fields are Month, OperatingSystems, Browser, Region,
TrafficType and VisitorType. OneHotEncoder learns their vocabulary from fit
rows only, with `drop=None` and `handle_unknown="ignore"`. An unseen level
activates **no indicator in that field's block**; it does not become `Other`, a
missing value or a known code. Other feature blocks remain unchanged. January,
April and new positive browser/region codes use this behavior. It prevents an
encoding failure but supplies no learned effect for the unfamiliar level.
Encoded width can differ across CV folds; each fold has its own fitted model.

Use `scale_numeric=True` (the default) for logistic regression. StandardScaler
learns means and variances from fit rows only. Use `scale_numeric=False` for
random forest or XGBoost: numeric values pass through unchanged and categories
remain one-hot, so neither model treats integer category codes as ordered
measurements. Weekend and category indicators are never standardized.

`make_model_pipeline(estimator, feature_set, scale_numeric=...)` clones the
estimator and places it after the preprocessor under step name `model`.
It works with sklearn-compatible estimators without importing XGBoost or
requiring its OpenMP runtime just to use preprocessing. Random forest and
logistic regression are runtime-verified here. XGBoost model fitting remains
pending its environment prerequisite and PIP-13; no XGBoost runtime success is
claimed.

## Fitting inside training folds

Factories do not load or split data. Callers must use the frozen assignments.
Pass the **whole pipeline and raw X** to CV, never a matrix transformed before
CV. This ensures both scaler and vocabulary are fitted separately on each
fold's fit rows. For example, from the repository root:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.preprocessing import make_model_pipeline, split_features_target
from src.split_data import load_frozen_splits, iter_training_cv

data, assignments, _ = load_frozen_splits()
rows = assignments.loc[
    assignments["partition"].eq("train"), "row_index"
].sort_values()
train = data.iloc[rows.to_numpy()]
X_train, y_train = split_features_target(train, "without_pagevalues_16")
pipeline = make_model_pipeline(
    LogisticRegression(max_iter=2000, random_state=42),
    "without_pagevalues_16",
    scale_numeric=True,
)
search = GridSearchCV(
    pipeline,
    {"model__C": [0.1, 1.0]},
    cv=list(iter_training_cv(assignments)),
    scoring="average_precision",
    refit=True,
)
# Optional model-development example; not run as part of preprocessing checks.
search.fit(X_train, y_train)
fitted = search.best_estimator_  # refitted on training only
```

Use `scale_numeric=False` when constructing the same pipeline with
`RandomForestClassifier(...)` or `XGBClassifier(...)`. Choose model settings in
their respective modeling tasks. Transform/predict validation or future raw
rows with the fitted object; never call fit/fit_transform on those rows.
Test evaluation remains reserved for the frozen final comparison.

Save the complete fitted pipeline and reload it with the project source
available on the Python import path (run from the repository root):

```python
from pathlib import Path
import joblib

Path("models").mkdir(exist_ok=True)
joblib.dump(fitted, "models/development_pipeline.joblib")
restored = joblib.load("models/development_pipeline.joblib")
raw_session = train.drop(columns="Revenue").iloc[:1]
prediction = restored.predict(raw_session)
probability = restored.predict_proba(raw_session)[:, 1]
```

Use the same dependency versions and keep `src.preprocessing` importable.
Only load trusted joblib artifacts, since loading can execute Python code.
Saved pipelines contain input validation as well as learned preprocessing
and the classifier; no manual inference encoding is needed.

## Reproduction and verification

After activating `.venv`, run:

```bash
python -m unittest discover -s tests
python -m src.check_preprocessing
```

The second command verifies the source-bound frozen split, selects **only the
8,632 training rows**, then fits logistic regression and a small random forest
for all three feature variants. It saves six smoke artifacts under
`models/preprocessing_smoke/` (ignored), reloads them, and checks exact equality
of probabilities for a raw training session and a copy with unseen Month/Browser
levels. Its JSON output records the source checksum, row count, feature widths
and inference examples. Change the artifact location with `--output-dir PATH`.
These artifacts are development checks, not selected final models or measured
validation performance. Re-running replaces only the named smoke artifacts.

The focused checks in [tests/test_preprocessing.py](../tests/test_preprocessing.py)
cover target separation, all feature variants, schema/value failures,
consistent Booleans, numeric-only scaling, unseen-category blocks, fresh
unfitted estimator clones, and saved/reloaded predictions in a **fresh Python
process**. The frozen five-fold check asserts each fitted scaler's mean,
variance and row count and each encoder's exact vocabulary against only that
fold's fit rows. A category injected exclusively into fold zero's holdout is
absent from its encoder. Predicting extreme/unseen future inputs leaves the
learned state unchanged.

Both commands passed in the project environment on 2026-11-09: **28 tests**
(including 10 new preprocessing checks) and **six saved/reloaded pipelines**.
The training-fitted encoded widths were 74, 73 and 71 for the 17-, 16- and
14-input variants respectively. No validation or test performance was
evaluated, and neither source data nor split manifests were changed. The
existing EDA, source attribution and feature-availability limitations remain
applicable.
