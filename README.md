# Purchase Intent Prediction

Predict whether an online shopping session ends in a purchase (`Revenue`) using
the UCI Online Shoppers Purchasing Intention Dataset. The source CSV has 12,330
sessions, 17 predictors and one Boolean target; this is classification, not a
prediction of the amount spent.

## Project layout

```text
data/
  raw/          Original, attributed source CSV; keep unchanged
  processed/    Generated data and split manifests (ignored)
models/         Generated fitted pipelines and model artifacts (ignored)
notebooks/      Exploratory notebooks; put reusable logic in src/
reports/        Written reports and manual reproduction evidence
results/        Generated metrics, tables and figures (ignored)
src/            Reusable source code and runnable entry points
```

Run the commands below from the repository root unless stated otherwise. Use
repository-relative paths in code, notebooks and configuration. Generated
artifacts are ignored; intentionally select any small results to publish later.

## Environment and installation

The selected baseline is **Homebrew CPython 3.14.8**, recorded in
`.python-version`. The project `.venv` was created from that interpreter and its
Python dependencies were installed successfully on macOS arm64 using pip
26.2.1. Homebrew manages the base Python installation, so project dependencies
are installed in `.venv`.

The environment has **not been runtime-tested**. On this Mac, XGBoost's
separate OpenMP runtime (`libomp`) is still missing; its installation was
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

| Task                  | Current status                                                                |
| --------------------- | ----------------------------------------------------------------------------- |
| Load source CSV       | Available: `python src/load_data.py`                                          |
| Train models          | Pending PIP-07, PIP-12 and PIP-13; no training entry point yet                |
| Evaluate models       | Pending PIP-03/PIP-06 and model implementation; no evaluation entry point yet |
| Launch prediction app | Pending PIP-18; no Streamlit app yet                                          |

When the app exists, launch its documented source file with
`python -m streamlit run <app-entry-point.py>` from the repository root; the
placeholder is not a runnable command. Replace this guidance with the actual
file and model requirements in PIP-18. Add training/evaluation commands, their
inputs and output paths when those entry points are implemented.

## Manual reproduction handoff

Python dependency installation completed; no automated tests, library-import
checks or project executions were performed. A second teammate should use a
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
