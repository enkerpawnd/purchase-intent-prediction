# PIP-02: source dataset audit

Verified on **2026-10-13** using Python 3.14.8 and pandas 3.1.1 against
`data/raw/online_shoppers_intention.csv`. All six PIP-02 acceptance criteria are
covered below. The original file is unchanged; all 12,330 records are retained.

## Placement and reproduction

This report includes the data dictionary and audit decisions. The supporting
[executed notebook](../notebooks/02_source_dataset_audit.ipynb) belongs in
`notebooks/`. Original data and attribution remain in `data/raw/`; this audit
creates no processed data, split manifests or model artifacts.

Activate the project environment and run the notebook's cells in order, or run
this command from the repository root. The terminal route requires only the
existing environment, without installing Jupyter; it executes the same code
cells and assertions but does not refresh the notebook's saved outputs.

```bash
python - <<'PY'
import json
from pathlib import Path

notebook = json.loads(Path('notebooks/02_source_dataset_audit.ipynb').read_text())
namespace = {'__name__': '__main__'}
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), f'audit-cell-{index}', 'exec'), namespace)
PY
```

Success ends with `Audit passed; original CSV bytes unchanged. All 12,330 records
retained.` Assertion failures require investigation rather than updating
expected values silently. Source checksum, schema, label counts, missingness,
duplicate counts, label conflicts and hash equivalence are checked. Saved
outputs were obtained by executing every code cell in order in one namespace.

## Provenance and integrity

Sakar, C. & Kastro, Y. (2018). _Online Shoppers Purchasing Intention Dataset_
[Dataset]. UCI Machine Learning Repository.
[DOI: 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).
The [UCI dataset page](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
provides definitions and lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Retain attribution when redistributing. See the existing
[raw-data record](../data/raw/README.md) for restoration instructions.

The bundled team-supplied file's observed SHA-256 matches the pre-existing
expected checksum. This verifies the local source bytes; this audit did not
download and compare a fresh UCI archive.

```text
b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9
```

## Acceptance evidence

| Check | Observed result |
| --- | --- |
| Source preservation | SHA-256 above matches; byte-for-byte comparison also passes after the audit |
| Shape | 12,330 rows × 18 columns; 17 predictors plus `Revenue` |
| Target | `True`: 1,908 (15.474453%); `False`: 10,422 |
| Missingness | 0 pandas-missing cells; 0 blank/whitespace cells in raw CSV records |
| Full-row repeats beyond first occurrences | 125 |
| All rows in repeated full-row groups | 201, across 76 groups |
| Predictor-only repeats | 125 beyond first; 201 records in repeated groups |
| Identical predictors with conflicting `Revenue` | 0 groups |
| Retention and grouping | 12,330 records retained; 12,205 predictor-hash groups |

## Data dictionary

Meanings below paraphrase UCI's variable information; types reflect observed
CSV values and semantic roles, rather than blindly copying its table types.
Every row except `Revenue` is a predictor. Nominal integer codes have no
documented name mapping here and must not imply numeric order or distance.

| Column | Meaning | Units | Role |
| --- | --- | --- | --- |
| `Administrative` | Administrative pages visited | Pages | Numeric count |
| `Administrative_Duration` | Total administrative-page time | Time; source unit unspecified | Numeric continuous |
| `Informational` | Informational pages visited | Pages | Numeric count |
| `Informational_Duration` | Total informational-page time | Time; source unit unspecified | Numeric continuous |
| `ProductRelated` | Product-related pages visited | Pages | Numeric count |
| `ProductRelated_Duration` | Total product-related-page time | Time; source unit unspecified | Numeric continuous |
| `BounceRates` | Page-level bounce metric | Fraction, dimensionless | Numeric continuous |
| `ExitRates` | Page-level exit metric | Fraction, dimensionless | Numeric continuous |
| `PageValues` | Average pre-transaction page value | Monetary value; currency unspecified | Numeric continuous |
| `SpecialDay` | Proximity to a special day | Dimensionless score [0, 1] | Numeric score |
| `Month` | Session month | Month label | Categorical nominal |
| `OperatingSystems` | Operating system | Category code | Categorical nominal |
| `Browser` | Browser | Category code | Categorical nominal |
| `Region` | Region | Category code | Categorical nominal |
| `TrafficType` | Traffic source/type | Category code | Categorical nominal |
| `VisitorType` | Visitor status | Category label | Categorical nominal |
| `Weekend` | Weekend visit | True/False | Boolean predictor |
| `Revenue` | Purchase occurred | True/False | Boolean target |

**`Revenue` is a purchase label, not an amount spent.** The three duration
columns, `PageValues` and `SpecialDay` contain fractional values. Their observed
values contradict treating them all as integers. The source page does not
establish seconds or a currency, so those units remain explicitly unknown.
The CSV cannot establish when each input becomes available for prediction;
`PageValues` availability and the matched feature study remain later work.

## Duplicate policy and split-group contract

**Keep every source record by default.** Removing 125 repeats would reduce the
dataset to 12,205 rows and alter its empirical distribution without an identity
or ingestion-error justification. The largest repeated group has 14 records.
Group-size frequencies are: 12,129 singleton groups; 54 groups of 2; 14 of 3;
3 of 4; 1 of 5; 1 of 6; 2 of 7; and 1 of 14.

UCI describes sessions from different users over a year. Independently, this CSV
has no user ID or event log, which means different people can produce identical aggregate
counts, codes and zero-duration summaries. Equality therefore cannot establish
duplicate users; the absence of label conflicts does not establish identity.

For PIP-03, use the notebook's **`predictor-json-sha256-v1`** contract:

1. Exclude `Revenue` and preserve the 17 predictor names in source header order.
2. For each pandas-parsed row, serialize an object with `columns` then `values`,
   using Python JSON, `ensure_ascii=False`, `allow_nan=False` and compact
   separators `(',', ':')`; encode as UTF-8 and take SHA-256.
3. Retain the full 64-character hash as the group ID. The notebook checks hash
   equality against exact predictor-tuple equality, including collision checks.
   Row order and target labels do not enter the hash. Keep this schema and parser
   environment fixed when reproducing groups.
4. Assign each whole group to one train/validation/test partition and to one
   validation fold within training; verify disjoint group sets at each boundary.
   PIP-03 must record the actual assignment, seed and group contract in its
   manifest. No split is chosen or generated in this audit.

These are groups of identical summaries, not user groups. They prevent exact
predictor overlap across partitions; they cannot establish user-level separation.

## Observed categories and calendar coverage

| Field | Observed value: count |
| --- | --- |
| Month | `Aug`: 433; `Dec`: 1,727; `Feb`: 184; `Jul`: 432; `June`: 288; `Mar`: 1,907; `May`: 3,364; `Nov`: 2,998; `Oct`: 549; `Sep`: 448 |
| OperatingSystems | `1`: 2,585; `2`: 6,601; `3`: 2,555; `4`: 478; `5`: 6; `6`: 19; `7`: 7; `8`: 79 |
| Browser | `1`: 2,462; `2`: 7,961; `3`: 105; `4`: 736; `5`: 467; `6`: 174; `7`: 49; `8`: 135; `9`: 1; `10`: 163; `11`: 6; `12`: 10; `13`: 61 |
| Region | `1`: 4,780; `2`: 1,136; `3`: 2,403; `4`: 1,182; `5`: 318; `6`: 805; `7`: 761; `8`: 434; `9`: 511 |
| TrafficType | `1`: 2,451; `2`: 3,913; `3`: 2,052; `4`: 1,069; `5`: 260; `6`: 444; `7`: 40; `8`: 343; `9`: 42; `10`: 450; `11`: 247; `12`: 1; `13`: 738; `14`: 13; `15`: 38; `16`: 3; `17`: 1; `18`: 10; `19`: 17; `20`: 198 |
| VisitorType | `New_Visitor`: 1,694; `Other`: 85; `Returning_Visitor`: 10,551 |
| Weekend | `False`: 9,462; `True`: 2,868 |
| SpecialDay | `0.0`: 11,079; `0.2`: 178; `0.4`: 243; `0.6`: 351; `0.8`: 325; `1.0`: 154 |

January and April are absent. Preserve the actual `June` spelling. There is no
year, timestamp or user identifier from which to reconstruct a chronological
split or establish why months are absent; do not invent missing sessions.

With a descriptive flag of **fewer than 20 rows**, rare nominal levels are:
`OperatingSystems` 5 (6), 6 (19), 7 (7); `Browser` 9 (1), 11 (6), 12 (10);
`TrafficType` 12 (1), 14 (13), 16 (3), 17 (1), 18 (10), 19 (17).
`VisitorType=Other` has 85 rows and remains a distinct minority category.
These flags do not authorize merging categories. Any future mapping or
preprocessing must be fitted within training folds and handle unseen levels.

## Numeric ranges, skew and investigation flags

Values below are computed from the CSV, rounded to three decimal places.
Skew is pandas' sample skewness; the notebook also prints the 99th percentile.

| Field | Min | Median | 95th percentile | Max | Zeros | Skew |
| --- | --- | --- | --- | --- | --- | --- |
| `Administrative` | 0.000 | 1.000 | 9.000 | 27.000 | 5,768 | 1.960 |
| `Administrative_Duration` | 0.000 | 7.500 | 348.266 | 3,398.750 | 5,903 | 5.616 |
| `Informational` | 0.000 | 0.000 | 3.000 | 24.000 | 9,699 | 4.036 |
| `Informational_Duration` | 0.000 | 0.000 | 195.000 | 2,549.375 | 9,925 | 7.579 |
| `ProductRelated` | 0.000 | 18.000 | 109.000 | 705.000 | 38 | 4.342 |
| `ProductRelated_Duration` | 0.000 | 598.937 | 4,300.289 | 63,973.522 | 755 | 7.263 |
| `BounceRates` | 0.000 | 0.003 | 0.200 | 0.200 | 5,518 | 2.948 |
| `ExitRates` | 0.000 | 0.025 | 0.200 | 0.200 | 76 | 2.149 |
| `PageValues` | 0.000 | 0.000 | 38.161 | 361.764 | 9,600 | 6.383 |
| `SpecialDay` | 0.000 | 0.000 | 0.600 | 1.000 | 11,079 | 3.303 |

All ten numerical fields are nonnegative and positively skewed. The large
duration/count maxima and `PageValues` upper tail merit investigation; the
audit does not establish that they are errors. `PageValues` is zero for 9,600
records, so zero must not automatically be treated as missing.

`BounceRates` and `ExitRates` both stop at 0.2, with respectively 700 and 710
records exactly there. This suggests a bounded or processed source metric but
does not prove clipping; do not use 0.2 as a general validity ceiling.
`SpecialDay` has six observed levels from 0 to 1 in steps of 0.2.

There are 135 administrative, 226 informational and 717 product-related
records with positive page counts but zero corresponding duration. None has
a zero count with positive corresponding duration. These combinations are
investigation flags, not proof of corruption. No outliers are removed, values
capped, rows deduplicated, or labels changed.

## Handoff

PIP-02's dataset evidence is complete. PIP-01's separate runtime and peer
reproduction requirements are not established by this pandas-only audit.
PIP-03 owns the frozen split; PIP-04 owns training-only exploration. Full-source
descriptive statistics above must not be used to fit preprocessing, tune models
or choose thresholds against held-out data.
