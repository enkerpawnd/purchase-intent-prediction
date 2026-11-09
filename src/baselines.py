"""Fit dummy baselines on training labels and evaluate them on validation.

Run from the repository root:  python -m src.baselines
The test partition is never scored here.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/baselines.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier

from src.evaluation import (
    REFERENCE_THRESHOLD,
    config_id,
    evaluate_scores,
    file_sha256,
    predict_from_scores,
    select_f1_threshold,
    split_id,
    write_outputs,
)
from src.split_data import (
    DEFAULT_DATA_PATH,
    DEFAULT_SPLIT_DIR,
    ROOT,
    SEED,
    load_frozen_splits,
)

DEFAULT_OUTPUT_DIR = ROOT / "results/baselines"
FIT_PARTITION = "train"
EVAL_PARTITION = "validation"
MODELS = {
    # Always predicts the training majority class; purchase score is 0.
    "majority_class": "most_frequent",
    # Purchase score is the training purchase prior for every session.
    "class_prior": "prior",
}
CHECK_TOLERANCE = 1e-12


def _partition(data: pd.DataFrame, assignments: pd.DataFrame, name: str):
    rows = assignments.loc[assignments["partition"].eq(name), "row_index"].to_numpy()
    frame = data.iloc[rows]
    return rows, frame.drop(columns="Revenue"), frame["Revenue"].to_numpy(dtype=bool)


def full_csv_always_no_purchase(data: pd.DataFrame) -> dict:
    """Label-count arithmetic for the proposal's 84.5% figure (no model scoring)."""
    revenue = data["Revenue"].to_numpy(dtype=bool)
    return {
        "description": "Always predicting no purchase on all 12,330 CSV labels. "
        "Computed from label counts only; this is not a validation or test result.",
        "sessions": int(len(revenue)),
        "non_purchases": int((~revenue).sum()),
        "purchases": int(revenue.sum()),
        "accuracy": float((~revenue).mean()),
        "purchase_recall": 0.0,
    }


def run_baselines(
    data_path: str | Path = DEFAULT_DATA_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    data, assignments, metadata = load_frozen_splits(data_path, split_dir)
    _, x_fit, y_fit = _partition(data, assignments, FIT_PARTITION)
    eval_rows, x_eval, y_eval = _partition(data, assignments, EVAL_PARTITION)

    config = {
        "task": "dummy baselines",
        "fit_partition": FIT_PARTITION,
        "evaluation_partition": EVAL_PARTITION,
        "models": {
            name: {"estimator": "sklearn.dummy.DummyClassifier", "strategy": strategy}
            for name, strategy in MODELS.items()
        },
        "threshold_rules": {
            "reference_0.5": "fixed reference threshold 0.5",
            "validation_max_f1": "protocol rule: maximize validation purchase F1 over "
            "distinct scores plus 0/0.5/1; ties to the largest threshold",
        },
        "seed": SEED,
        "split_id": split_id(split_dir),
        "source_sha256": metadata["source"]["sha256"],
        "code_sha256": {
            name: file_sha256(ROOT / "src" / name)
            for name in ("baselines.py", "evaluation.py", "split_data.py")
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    run = {"run_id": config_id("baselines", config), **config}

    metrics_rows, prediction_frames, checks = [], [], {}
    train_prior = float(y_fit.mean())
    for name, strategy in MODELS.items():
        model = DummyClassifier(strategy=strategy, random_state=SEED)
        model.fit(x_fit, y_fit)  # training labels only
        purchase_column = list(model.classes_).index(True)
        scores = model.predict_proba(x_eval)[:, purchase_column]
        thresholds = {
            "reference_0.5": REFERENCE_THRESHOLD,
            "validation_max_f1": select_f1_threshold(y_eval, scores),
        }
        for rule, threshold in thresholds.items():
            metrics = evaluate_scores(y_eval, scores, threshold)
            metrics_rows.append(
                {
                    "model": name,
                    "partition": EVAL_PARTITION,
                    "threshold_rule": rule,
                    "threshold": threshold,
                    **metrics,
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_index": eval_rows,
                        "partition": EVAL_PARTITION,
                        "model": name,
                        "threshold_rule": rule,
                        "threshold": threshold,
                        "y_true": y_eval,
                        "purchase_score": scores,
                        "predicted_purchase": predict_from_scores(scores, threshold),
                    }
                )
            )

        reference = evaluate_scores(y_eval, scores, REFERENCE_THRESHOLD)
        expected_score = 0.0 if strategy == "most_frequent" else train_prior
        checks[name] = {
            "fit_rows_are_training_only": int(len(y_fit))
            == metadata["evidence"]["partitions"][FIT_PARTITION]["rows"],
            "constant_score": bool(np.ptp(scores) == 0),
            "score_equals_training_derived_value": bool(
                abs(scores[0] - expected_score) <= CHECK_TOLERANCE
            ),
            "ap_equals_validation_prevalence": bool(
                abs(reference["average_precision"] - reference["prevalence"])
                <= CHECK_TOLERANCE
            ),
            "roc_auc_is_0.5": reference["roc_auc"] == 0.5,
            "reference_predictions_match_dummy_predict": bool(
                np.array_equal(
                    predict_from_scores(scores, REFERENCE_THRESHOLD),
                    model.predict(x_eval).astype(bool),
                )
            ),
        }
    test_rows = set(assignments.loc[assignments["partition"].eq("test"), "row_index"])
    checks["test_partition_not_scored"] = not (set(eval_rows) & test_rows)
    checks["training_purchase_prior"] = train_prior
    checks["validation_prevalence"] = float(y_eval.mean())

    failed = [
        f"{model}.{check}"
        for model, result in checks.items()
        if isinstance(result, dict)
        for check, passed in result.items()
        if not passed
    ]
    if not checks["test_partition_not_scored"]:
        failed.append("test_partition_not_scored")
    if failed:
        raise ValueError(f"Baseline checks failed: {', '.join(failed)}")

    notes = {
        "full_csv_always_no_purchase": full_csv_always_no_purchase(data),
        "constant_score_ap": "A constant score ranks every session equally, so AP "
        "equals the evaluation-set purchase prevalence and ROC-AUC is 0.5, "
        "regardless of the constant value. Any useful model must exceed both.",
    }
    paths = write_outputs(
        output_dir,
        run,
        metrics_rows,
        pd.concat(prediction_frames, ignore_index=True),
        checks,
        notes,
    )
    return {"run": run, "metrics": metrics_rows, "checks": checks, "paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = run_baselines(args.data_path, args.split_dir, args.output_dir)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Baseline run failed: {exc}\n")
    table = pd.DataFrame(result["metrics"])[
        [
            "model",
            "threshold_rule",
            "threshold",
            "average_precision",
            "roc_auc",
            "brier",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ]
    print(f"Run {result['run']['run_id']} on {result['run']['split_id']}, seed {SEED}")
    print("Validation results (test partition not scored):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(json.dumps({k: str(v) for k, v in result["paths"].items()}, indent=2))


if __name__ == "__main__":
    main()
