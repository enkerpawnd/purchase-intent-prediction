# From a dataset row to a practical prediction

The dataset gives us completed session summaries. A working application needs
to know what can be measured **at the moment a prediction is requested**. A
session with 20 product pages at its end may have only two at an earlier
decision point. Even familiar inputs can change meaning when collected earlier.

For PIP-04, keep a clear distinction between a demonstration using a saved or
manually entered session and a live website integration. The former can show
how a model behaves on dataset-compatible inputs. The latter needs a defined
decision point, compatible measurements and data captured before that point.
The recommendations below were the PIP-04 design contract. The reusable
[preprocessing implementation](preprocessing.md) now enforces required inputs,
type/range validation and unseen-category handling. An input form and its
training-support warnings remain future work.

## The three feature lists

These are **raw column names in source order**, before encoding or scaling.
`Revenue` is the label and is never an input. The same lists are importable from
[src/feature_sets.py](../src/feature_sets.py) and exported to
[feature_sets.json](../results/eda/feature_sets.json).

**Required: all 17 inputs (`all_17`).** This is the full-dataset comparator.

```text
Administrative, Administrative_Duration, Informational, Informational_Duration,
ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues,
SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType,
Weekend
```

**Required: 16 inputs without PageValues (`without_pagevalues_16`).** This tests
how much performance depends on the input with the clearest reproducibility
concern.

```text
Administrative, Administrative_Duration, Informational, Informational_Duration,
ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, SpecialDay,
Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend
```

**Optional: 14 inputs without PageValues, BounceRates or ExitRates
(`without_analytics_14`).** This also removes two inputs that depend on page
analytics definitions and history. It is a more practical candidate to
investigate, not a certified live-deployment feature set.

```text
Administrative, Administrative_Duration, Informational, Informational_Duration,
ProductRelated, ProductRelated_Duration, SpecialDay, Month, OperatingSystems,
Browser, Region, TrafficType, VisitorType, Weekend
```

Use the same PIP-03 assignments and training CV folds for all versions. Do not
regenerate groups after dropping columns: the frozen groups use all 17 inputs.
PIP-09/PIP-15 own the model comparisons; no performance difference is measured
here. A model trained with 17 inputs cannot become a 16-input model by filling
PageValues with zero at prediction time. Each version needs its own fitted
pipeline and a matching input form.

## What each raw input would require

“Could collect” below is a proposed source for a new site, not evidence that
the original dataset used the same implementation. Use the
[source audit](source_dataset_audit.md#data-dictionary) for the complete dataset
dictionary. In particular, duration units, currency and category-code mappings
are not established by this CSV.

| Raw input | Expected raw type | What a practical app or website would need |
| --- | --- | --- |
| `Administrative` | Nonnegative integer page count | Classify URLs as administrative and count visits observed so far; match the source counting convention before transferring a model. |
| `Administrative_Duration` | Finite nonnegative float | Accumulate administrative-page time up to the decision point. Require a documented unit and a rule for active/idle time. |
| `Informational` | Nonnegative integer page count | Equivalent visit tracking for informational URLs. |
| `Informational_Duration` | Finite nonnegative float | Equivalent timing for informational pages; final totals cannot be substituted for earlier totals. |
| `ProductRelated` | Nonnegative integer page count | Equivalent visit tracking for product-related URLs. |
| `ProductRelated_Duration` | Finite nonnegative float | Equivalent timing for product-related pages. |
| `BounceRates` | Finite float fraction in [0, 1] | Compatible page analytics, a historical window ending before prediction and a documented rule for combining page metrics into a session input. It is not simply “did this current session bounce?” |
| `ExitRates` | Finite float fraction in [0, 1] | Compatible page analytics with a documented aggregation and update window. A current session's eventual exit is not known mid-session. |
| `PageValues` | Finite nonnegative float; currency unknown | Comparable historical page valuation, transaction attribution, aggregation and refresh rules. Verify which transaction outcomes contribute before claiming pre-purchase availability. |
| `SpecialDay` | Finite float score in [0, 1] | A documented calendar and closeness rule matching training semantics. Date alone does not establish the source-specific score. |
| `Month` | Category string | Derive from the session date in a chosen timezone; preserve labels, including `June`. Training covers Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec. |
| `OperatingSystems` | Integer category code, treated as nominal | Browser/device information could identify an OS, but translating it into source codes requires a codebook that is not supplied. |
| `Browser` | Integer category code, treated as nominal | Browser information plus a compatible mapping to source codes; do not guess which number names a browser. |
| `Region` | Integer category code, treated as nominal | Region information and a compatible codebook; neither numeric order nor geographic distance follows from these codes. |
| `TrafficType` | Integer category code, treated as nominal | Referral/campaign information and a matching source taxonomy; another site's traffic labels need an explicit mapping. |
| `VisitorType` | Category string | Prior-visit state under a documented identity/cookie rule. Training values: `New_Visitor`, `Returning_Visitor`, `Other`. Tracking limitations may leave status unknown. |
| `Weekend` | Boolean | Derive from the same session date and timezone used for Month; accept an actual true/false choice. |

Page analytics definitions are paraphrased from
[UCI's additional variable information](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset),
consulted on 2026-11-09. UCI describes count/time updates during navigation,
but the released aggregate rows do not reveal their snapshot time. Consequently
the six count/time fields also need timing verification for an early-session
prediction. The PageValues concern is explained in the
[exploration report](training_data_exploration.md#pagevalues-is-useful-evidence-with-an-availability-question).
Availability remains uncertain; leakage has not been proven for any input.

## Missing, unfamiliar and out-of-range inputs

The training data has no missing cells, so it supplies no evidence for how a
model performs when a user leaves a required measurement blank. For the first
demonstration, require every input in the chosen feature set. Show a useful
validation message for a blank, nonfinite numeric value or missing Boolean;
do not silently turn it into zero or false. Omitted fields belonging to a
smaller feature set need no replacement value. If partial inputs are supported
later, fit and evaluate the imputation rule within training CV.

Preserve actual zeros. Require whole counts, nonnegative durations/PageValues,
and fractions rather than percentage numbers for the two rates. The observed
0.2 rate maximum is a training-range warning, not a validity ceiling; a valid
fraction above it is outside observed support and should trigger an explanatory
warning. Large valid counts/durations deserve the same warning, not silent
clipping. SpecialDay has six observed levels; an intermediate score in [0, 1]
would also be unfamiliar and needs a compatible collection rule.

For a saved-row/manual demonstration, offer the training-observed categories
and explain that the integer codes are dataset codes. Keep unfamiliar values
distinct from missing values. In particular, `VisitorType=Other` is an observed
source category, not an automatic replacement for every unknown visitor.

For the future pipeline, fit categorical vocabularies only on each training
fold and use an encoder that tolerates unseen levels, such as one-hot encoding
with `handle_unknown="ignore"`. An unseen level then contributes no active
indicator for that field; this prevents an error but does not teach the model
what that category means. The app should identify the unsupported category and
make the limitation visible. January and April need this path too. Never map
an unfamiliar browser or region to a convenient known code, and never infer
ordinal meaning from numeric codes.

Calendar fields and page tracking are plausible to obtain with site
instrumentation, but codebooks, visitor identity, special-day scoring and
analytics history still require work. A new site's live performance would need
its own evaluation, even if it can supply all 14 inputs. Until those definitions
are verified, present the app as a dataset-based demonstration.
