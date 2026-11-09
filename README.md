# Purchase Intent Prediction

Predict whether an online shopping session ends in a purchase (`Revenue`) using
the UCI Online Shoppers Purchasing Intention Dataset. The source CSV has 12,330
sessions, 17 predictors and one Boolean target; this is classification, not a
prediction of the amount spent.

## Project layout

```text
data/
  raw/          Original, attributed source CSV; keep unchanged
  processed/    Versioned frozen split manifests; other generated data ignored
models/         Generated fitted pipelines and model artifacts (ignored)
notebooks/      Exploratory notebooks; put reusable logic in src/
reports/        Written reports and manual reproduction evidence
results/        Generated metrics, tables and figures (ignored)
src/            Reusable source code and runnable entry points
```

Run the commands below from the repository root unless stated otherwise. Use
repository-relative paths in code, notebooks and configuration. The frozen split
manifests are versioned; other generated artifacts are ignored. Intentionally
select any small results to publish later.

## Environment and installation

The selected baseline is **Homebrew CPython 3.14.8**, recorded in
`.python-version`. The project `.venv` was created from that interpreter and its
Python dependencies were installed successfully on macOS arm64 using pip
26.2.1. Homebrew manages the base Python installation, so project dependencies
are installed in `.venv`.

The full model/application environment has **not been runtime-tested**. The
dataset audit, frozen split, training exploration and preprocessing checks ran
successfully. Preprocessing checks also fit/save/reload small logistic regression
and random forest pipelines on training rows; they do not verify XGBoost or the
application libraries.
Installation of XGBoost's
separate OpenMP runtime (`libomp`) was
deferred. Runtime checks and second-teammate reproduction remain manual. See
[reports/reproduction.md](reports/reproduction.md) for the installation record
and pending checks.

Pinned project libraries in `requirements.txt`:

| Library      | Version |
| ------------ | ------- |
| pandas       | 3.1.1   |
| scikit-learn | 1.9.1   |
| XGBoost      | 3.4.1   |
| matplotlib   | 3.11.2  |
| Streamlit    | 1.67.0  |

`requirements.txt` also pins NumPy, SciPy and the resolved transitive Python
dependencies installed on this Mac. It is an environment snapshot, without
package hashes or a cross-platform resolver lock. Installation on Windows and
Linux has not been checked; platform-specific dependencies may differ.

### macOS with Homebrew

From the repository root, use Homebrew's versioned Python formula. If it is not
installed, run `brew install python@3.14`. Homebrew may supply a newer 3.14 patch
in future; the interpreter used for the recorded installation reports
`Python 3.14.8`. The `.python-version` file records this baseline but does not
make Homebrew install or select an exact patch.

```bash
"$(brew --prefix python@3.14)/bin/python3.14" --version
"$(brew --prefix python@3.14)/bin/python3.14" -m venv .venv
source .venv/bin/activate
python -m pip install pip==26.2.1
python -m pip install -r requirements.txt
```

For the environment already created in this checkout, just run
`source .venv/bin/activate`. In Zed, open a Python file, run
**toolchain: select** from the command palette and select `.venv/bin/python`.

### Other platforms

Use a separately installed CPython 3.14.8. For Linux, create the environment
with `python3.14 -m venv .venv`, then follow the activation and pip commands
above. On Windows (PowerShell):

```powershell
py -3.14 --version
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install pip==26.2.1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of
`python` in the remaining commands. On macOS, XGBoost requires the OpenMP
runtime. Before using XGBoost, install it with `brew install libomp`; installing
the Python wheel alone does not supply this runtime. This remains outstanding
on the setup machine. See the
[XGBoost installation documentation](https://xgboost.readthedocs.io/en/stable/install.html).

## Dataset location and attribution

The unchanged team-supplied CSV is included at
`data/raw/online_shoppers_intention.csv`, so no separate download or path outside
the checkout is needed. See [data/raw/README.md](data/raw/README.md) for the
source, license, citation, expected checksum and replacement instructions.

The [PIP-02 source audit](reports/source_dataset_audit.md) records verified
counts, the feature dictionary, category coverage, range flags and the policy
to retain all sessions while grouping identical predictors for future splits.
The [executed audit notebook](notebooks/02_source_dataset_audit.ipynb) provides
rerunnable verification; the report includes a terminal command that uses the
existing environment without Jupyter.

Source: Sakar, C. & Kastro, Y. (2018). _Online Shoppers Purchasing Intention
Dataset_ [Dataset]. UCI Machine Learning Repository.
[doi:10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).
The dataset is distributed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
as stated on the [UCI dataset page](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
This attribution applies to the dataset; it does not assign a license to the
project's code.

## Load the CSV

After activating the environment, manually run:

```bash
python src/load_data.py
```

Expected: `Loaded 12330 rows and 18 columns.` followed by the first five rows.
The loader finds the bundled CSV relative to its own source file, so its default
data location does not depend on the working directory. It only reads the CSV;
it does not clean, split or train on the data. To use a different file:

```bash
python src/load_data.py --data-path data/raw/online_shoppers_intention.csv
```

A supplied relative path is resolved from the current working directory.

## Training, evaluation and application entry points

### Frozen split and evaluation protocol (PIP-03)

The frozen split manifests are included in the repository. After activating
the environment, verify them:

```bash
python src/split_data.py --check
```

The reusable implementation is in `src/split_data.py`. The fixed-seed,
purchase-stratified manifests are at
`data/processed/splits/{train,validation,test}.csv`, with `metadata.json` recording
configuration, class counts, checks and one source checksum linking saved row
positions to their CSV. Identical predictor groups stay together. These four
files are versioned so every teammate uses the same assignments.

To reproduce the manifests if absent, run `python src/split_data.py`.
Existing manifests are checked without being overwritten; `--check` only
validates an existing bundle. Other generated processed data remains ignored.

The frozen sizes are **8,632 train / 1,849 validation / 1,849 test**. Five
training-only CV folds are saved in `train.csv`; validation/test never enter CV.
Use `load_frozen_splits` and `iter_training_cv` from `src.split_data` instead of
making new splits. The [evaluation protocol](reports/evaluation_protocol.md)
explains the split and CV choices, records actual class/fold counts, and defines
the metrics, validation F1 threshold rule and one-time final-test boundary.
Development EDA uses training rows only. No validation or test performance has
been evaluated.

### Training exploration and feature availability (PIP-04)

Read [what the training sessions tell us](reports/training_data_exploration.md)
for five figures and explanations of purchase rates, long-tailed activity,
recorded zeros, rare categories and relationships between measurements. All
summaries use the 8,632 frozen training sessions. To reproduce them:

```bash
python -m src.explore_training_data
python -m unittest discover -s tests
```

Reusable analysis lives in `src/explore_training_data.py`; the selected figures,
small summary tables and provenance are retained in `results/eda/`. No notebook
or additional dependency is needed. The script checks the frozen split before
selecting training rows, preserves the source CSV and fits no preprocessing or
models.

The [feature availability notes](reports/feature_availability.md) define raw
types, likely collection needs and missing/unseen-input handling for the future
app. [src/feature_sets.py](src/feature_sets.py) provides the required 17-input
and 16-input (without PageValues) lists, plus the optional 14-input list without
PageValues, BounceRates and ExitRates. PageValues is strongly associated with
purchase in training, but its computation timing is unresolved; this exploration
does not establish label leakage or live-site readiness.

### Reusable preprocessing

The preprocessing request corresponds to **PIP-05** in the local issue plan;
PIP-04 above is the completed training exploration. Implementation is in
[src/preprocessing.py](src/preprocessing.py), with the input contract, usage and
verification evidence in [reports/preprocessing.md](reports/preprocessing.md).

Use `split_features_target` to separate Revenue, then `make_model_pipeline` to
combine an estimator with raw-input validation, one-hot category encoding,
consistent Boolean handling and numeric scaling. All three feature variants
are supported. Use `scale_numeric=True` for logistic regression and `False` for
tree models. Pass the whole unfitted pipeline to training CV so learned
statistics and vocabularies stay within each fit fold. Missing/invalid required
values raise an explicit error; unseen categories produce an all-zero block
for that field.

To verify raw-session prediction after saving/reloading:

```bash
python -m unittest discover -s tests
python -m src.check_preprocessing
```

The check command fits six small training-only pipelines (logistic regression
and random forest, each feature variant), saves them to the ignored
`models/preprocessing_smoke/` directory, and verifies normal and unseen-category
inference after reloading. These are smoke artifacts; model selection and
validation/test performance evaluation belong to later modeling tasks.

### Model and application status

| Task                  | Current status                                                                |
| --------------------- | ----------------------------------------------------------------------------- |
| Load source CSV       | Available: `python src/load_data.py`                                          |
| Freeze/check split    | Available: `python src/split_data.py` / `python src/split_data.py --check`    |
| Explore training data | Available: `python -m src.explore_training_data`                            |
| Preprocess/save/reload | Available: `python -m src.check_preprocessing`; reusable factories in `src/preprocessing.py` |
| Train models          | Pending PIP-07, PIP-12 and PIP-13; no training entry point yet                |
| Evaluate models       | Protocol frozen in PIP-03; metric execution pending PIP-06 and models         |
| Launch prediction app | Pending PIP-18; no Streamlit app yet                                          |

When the app exists, launch its documented source file with
`python -m streamlit run <app-entry-point.py>` from the repository root; the
placeholder is not a runnable command. Replace this guidance with the actual
file and model requirements in PIP-18. Add training/evaluation commands, their
inputs and output paths when those entry points are implemented.

## Manual reproduction handoff

Python dependency installation completed. The PIP-01 installation record did
not include automated tests, library-import checks or project executions;
PIP-02 subsequently ran the pandas-only dataset audit linked above. A second teammate should use a
fresh checkout and environment, follow the installation and CSV instructions
(including the OpenMP prerequisite when applicable), then manually run:

```bash
python --version
python -m pip check
python -c "import pandas, sklearn, xgboost, matplotlib, streamlit; print('All five project libraries imported.')"
python -m pip freeze > reports/environment-freeze.txt
```

Record the teammate, date, operating system, Python version, installed versions,
CSV shape, checksum and actual outcome in
[reports/reproduction.md](reports/reproduction.md). Keep any failure and fix in
the record. The tested-environment and successful peer-reproduction acceptance
criteria for PIP-01 remain pending until those manual results are recorded.
