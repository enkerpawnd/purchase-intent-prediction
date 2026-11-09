"""PIP-07: train and evaluate regularized logistic regression (all 17 inputs).

Run from the repository root:  python -m src.train_logistic

1. Compare a small, fixed grid of L2 strengths, unweighted vs class-weighted,
   on the frozen five training CV folds (training rows only, primary metric AP).
2. Refit the best setting on all training rows and save the whole pipeline.
3. Evaluate on the validation partition against both dummy baselines and export
   metrics, predictions, confusion matrix, PR curve and reliability bins.
4. Reload the saved pipeline and reproduce its predictions exactly.

The test partition is never scored.
"""

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/train_logistic.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

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
from src.preprocessing import make_model_pipeline, split_features_target
from src.split_data import (
    DEFAULT_DATA_PATH,
    DEFAULT_SPLIT_DIR,
    ROOT,
    SEED,
    iter_training_cv,
    load_frozen_splits,
)

MODEL_NAME = "logistic_regression"
FEATURE_SET = "all_17"
DEFAULT_OUTPUT_DIR = ROOT / "results/logistic_regression"
DEFAULT_MODEL_PATH = ROOT / "models/logistic_regression/all_17_logistic.joblib"
DEFAULT_BASELINE_METRICS = ROOT / "results/baselines/metrics.csv"
# Bounded, predeclared grid: 4 regularization strengths x 2 weightings = 8 settings.
C_VALUES = (0.01, 0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")
FIXED_PARAMS = {
    "l1_ratio": 0.0,  # pure L2 penalty
    "solver": "lbfgs",
    "max_iter": 2000,
    "tol": 1e-4,
    "fit_intercept": True,
    "random_state": SEED,
}
RELIABILITY_BINS = 10


def _settings() -> list[dict]:
    return [
        {"C": c, "class_weight": weight}
        for weight in CLASS_WEIGHTS
        for c in C_VALUES
    ]


def _label(setting: dict) -> str:
    weight = setting["class_weight"] or "unweighted"
    return f"C={setting['C']:g}, {weight}"


def _pipeline(setting: dict):
    estimator = LogisticRegression(**FIXED_PARAMS, **setting)
    return make_model_pipeline(estimator, FEATURE_SET, scale_numeric=True)


def _fit(pipeline, X, y) -> dict:
    """Fit and record convergence (iterations and any ConvergenceWarning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        pipeline.fit(X, y)
        seconds = time.perf_counter() - start
    n_iter = int(pipeline.named_steps["model"].n_iter_.max())
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return {
        "n_iter": n_iter,
        "converged": bool(not warned and n_iter < FIXED_PARAMS["max_iter"]),
        "seconds": seconds,
    }


def _purchase_scores(pipeline, X) -> np.ndarray:
    classes = list(pipeline.named_steps["model"].classes_)
    return pipeline.predict_proba(X)[:, classes.index(1)]


def compare_settings(X: pd.DataFrame, y: pd.Series, assignments) -> tuple:
    """Score every setting on the frozen training CV folds (no validation rows)."""
    folds = list(iter_training_cv(assignments))
    fold_rows, timings = [], []
    for setting in _settings():
        for fold, (fit, holdout) in enumerate(folds):
            pipeline = _pipeline(setting)
            fit_info = _fit(pipeline, X.iloc[fit], y.iloc[fit])
            y_hold = y.iloc[holdout].to_numpy().astype(bool)
            scores = _purchase_scores(pipeline, X.iloc[holdout])
            fold_rows.append(
                {
                    "setting": _label(setting),
                    "C": setting["C"],
                    "class_weight": setting["class_weight"] or "none",
                    "fold": fold,
                    "average_precision": average_precision_score(y_hold, scores),
                    "roc_auc": roc_auc_score(y_hold, scores),
                    "log_loss": log_loss(y_hold, scores, labels=[False, True]),
                    "brier": brier_score_loss(y_hold, scores, pos_label=True),
                    "n_iter": fit_info["n_iter"],
                    "converged": fit_info["converged"],
                }
            )
            timings.append({"setting": _label(setting), "fold": fold,
                            "fit_seconds": fit_info["seconds"]})
    folds_frame = pd.DataFrame(fold_rows)
    summary = (
        folds_frame.groupby(["setting", "C", "class_weight"], sort=False)
        .agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
            max_n_iter=("n_iter", "max"),
            all_converged=("converged", "all"),
        )
        .reset_index()
    )
    # Highest mean CV AP; exact ties keep grid order (stronger L2, unweighted first).
    best = int(summary["mean_average_precision"].to_numpy().argmax())
    summary["selected"] = summary.index == best
    return summary, folds_frame, _settings()[best], timings


def _baseline_rows(path: Path, expected_split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python -m src.baselines` first."
        )
    rows = pd.read_csv(path)
    if set(rows["split_id"]) != {expected_split}:
        raise ValueError("Baseline results were produced on a different split.")
    return rows


def reliability_table(y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Ten equal-width probability bins with counts (protocol)."""
    edges = np.arange(RELIABILITY_BINS + 1) / RELIABILITY_BINS
    # Right-closed final bin so a probability of exactly 1 is counted.
    index = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, RELIABILITY_BINS - 1)
    rows = []
    for b in range(RELIABILITY_BINS):
        mask = index == b
        count = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "lower": edges[b],
                "upper": edges[b + 1],
                "count": count,
                "purchases": int(y[mask].sum()),
                "mean_predicted": float(scores[mask].mean()) if count else np.nan,
                "observed_rate": float(y[mask].mean()) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _figures(directory: Path, y, scores, curve, reliability, metrics, prevalence):
    style = {"figure.facecolor": "white", "savefig.facecolor": "white"}
    meta = {"Software": None}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(curve["recall"], curve["precision"], drawstyle="steps-post",
                label=f"Logistic regression (AP {metrics['average_precision']:.3f})")
        ax.axhline(prevalence, color="grey", linestyle="--",
                   label=f"Dummy baselines (AP {prevalence:.3f})")
        ax.set(xlabel="Purchase recall", ylabel="Purchase precision",
               xlim=(0, 1), ylim=(0, 1.02),
               title="Validation precision-recall curve")
        ax.legend(loc="upper right")
        fig.savefig(directory / "pr_curve.png", dpi=150, bbox_inches="tight",
                    metadata=meta)
        plt.close(fig)

        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(5.5, 6), sharex=True, height_ratios=[3, 1]
        )
        filled = reliability.dropna(subset=["mean_predicted"])
        top.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect calibration")
        top.plot(filled["mean_predicted"], filled["observed_rate"], marker="o",
                 label=f"Logistic regression (Brier {metrics['brier']:.3f})")
        top.set(ylabel="Observed purchase rate", ylim=(0, 1),
                title="Validation reliability (10 equal-width bins)")
        top.legend(loc="upper left")
        centers = (reliability["lower"] + reliability["upper"]) / 2
        bottom.bar(centers, reliability["count"], width=0.09, color="tab:blue")
        bottom.set(xlabel="Predicted purchase probability", ylabel="Sessions",
                   xlim=(0, 1))
        fig.savefig(directory / "reliability.png", dpi=150, bbox_inches="tight",
                    metadata=meta)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, (rule, row) in zip(axes, metrics["by_rule"].items()):
            matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
            ax.imshow(matrix, cmap="Blues")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:,}", ha="center", va="center",
                        color="white" if value > matrix.max() / 2 else "black")
            ax.set(xticks=[0, 1], yticks=[0, 1],
                   xticklabels=["No purchase", "Purchase"],
                   yticklabels=["No purchase", "Purchase"],
                   xlabel="Predicted", ylabel="Actual",
                   title=f"{rule} (threshold {row['threshold']:.3f})")
        fig.tight_layout()
        fig.savefig(directory / "confusion_matrix.png", dpi=150, bbox_inches="tight",
                    metadata=meta)
        plt.close(fig)


def train_and_evaluate(
    data_path: str | Path = DEFAULT_DATA_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    baseline_metrics: str | Path = DEFAULT_BASELINE_METRICS,
) -> dict:
    total_start = time.perf_counter()
    data, assignments, metadata = load_frozen_splits(data_path, split_dir)
    the_split = split_id(split_dir)
    baselines = _baseline_rows(Path(baseline_metrics), the_split)

    def rows(name):
        idx = assignments.loc[assignments["partition"].eq(name), "row_index"]
        return np.sort(idx.to_numpy())

    train_rows, val_rows = rows("train"), rows("validation")
    X_train, y_train = split_features_target(data.iloc[train_rows], FEATURE_SET)
    X_val, y_val_int = split_features_target(data.iloc[val_rows], FEATURE_SET)
    y_val = y_val_int.to_numpy().astype(bool)

    # 1. Bounded comparison on training CV folds only.
    cv_start = time.perf_counter()
    cv_summary, cv_folds, selected, cv_timings = compare_settings(
        X_train, y_train, assignments
    )
    cv_seconds = time.perf_counter() - cv_start

    # 2. Refit the selected setting on all training rows; save the whole pipeline.
    pipeline = _pipeline(selected)
    final_fit = _fit(pipeline, X_train, y_train)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    # 3. Validation scores from the fitted pipeline (validation never fitted on).
    scores = _purchase_scores(pipeline, X_val)
    thresholds = {
        "reference_0.5": REFERENCE_THRESHOLD,
        "validation_max_f1": select_f1_threshold(y_val, scores),
    }
    metrics_rows, prediction_frames, by_rule = [], [], {}
    for rule, threshold in thresholds.items():
        metrics = evaluate_scores(y_val, scores, threshold)
        row = {"model": MODEL_NAME, "partition": "validation",
               "threshold_rule": rule, "threshold": threshold, **metrics}
        metrics_rows.append(row)
        by_rule[rule] = row
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": val_rows,
                    "partition": "validation",
                    "model": MODEL_NAME,
                    "threshold_rule": rule,
                    "threshold": threshold,
                    "y_true": y_val,
                    "purchase_score": scores,
                    "predicted_purchase": predict_from_scores(scores, threshold),
                }
            )
        )

    # 4. Reload and reproduce predictions exactly.
    restored = joblib.load(model_path)
    restored_scores = _purchase_scores(restored, X_val)
    raw_session = data.iloc[val_rows[:1]].drop(columns="Revenue")
    example = {
        "source_row_index": int(val_rows[0]),
        "purchase_probability": float(_purchase_scores(restored, raw_session)[0]),
        "predicted_purchase_at_0.5": bool(restored.predict(raw_session)[0] == 1),
    }

    config = {
        "task": "PIP-07 logistic regression",
        "feature_set": FEATURE_SET,
        "fit_partition": "train",
        "evaluation_partition": "validation",
        "estimator": "sklearn.linear_model.LogisticRegression",
        "objective": "L2-regularized log-loss (binary cross-entropy); "
        "class_weight reweights each session's loss term",
        "fixed_params": FIXED_PARAMS,
        "grid": {"C": list(C_VALUES),
                 "class_weight": [w or "none" for w in CLASS_WEIGHTS]},
        "selection_rule": "highest mean average precision over the five frozen "
        "training CV folds; exact ties keep grid order",
        "selected": {"C": selected["C"],
                     "class_weight": selected["class_weight"] or "none"},
        "preprocessing": "src.preprocessing.make_model_pipeline(scale_numeric=True)",
        "seed": SEED,
        "split_id": the_split,
        "source_sha256": metadata["source"]["sha256"],
        "code_sha256": {
            name: file_sha256(ROOT / "src" / name)
            for name in ("train_logistic.py", "evaluation.py", "preprocessing.py",
                         "feature_sets.py", "split_data.py")
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    run = {"run_id": config_id("logreg", config), **config}

    test_rows = set(rows("test"))
    checks = {
        "fit_rows_are_training_only": len(X_train)
        == metadata["evidence"]["partitions"]["train"]["rows"],
        # Every CV holdout position indexes X_train, and X_train holds exactly
        # the training source rows, so no validation/test row enters CV.
        "cv_uses_training_rows_only": bool(
            np.array_equal(
                np.sort(np.concatenate([h for _, h in iter_training_cv(assignments)])),
                np.arange(len(X_train)),
            )
            and np.array_equal(X_train.index.to_numpy(), train_rows)
        ),
        "test_partition_not_scored": not (set(val_rows) & test_rows),
        "all_cv_fits_converged": bool(cv_folds["converged"].all()),
        "final_fit_converged": final_fit["converged"],
        "reloaded_predictions_identical": bool(np.array_equal(restored_scores, scores)),
        "beats_dummy_ap": by_rule["reference_0.5"]["average_precision"]
        > float(baselines["average_precision"].max()),
    }
    required = ("fit_rows_are_training_only", "cv_uses_training_rows_only",
                "test_partition_not_scored", "reloaded_predictions_identical")
    failed = [name for name in required if not checks[name]]
    if failed:
        raise ValueError(f"PIP-07 checks failed: {', '.join(failed)}")

    directory = Path(output_dir)
    curve_precision, curve_recall, curve_thresholds = precision_recall_curve(
        y_val, scores, pos_label=True
    )
    curve = pd.DataFrame(
        {
            "threshold": np.append(curve_thresholds, np.nan),
            "precision": curve_precision,
            "recall": curve_recall,
        }
    )
    reliability = reliability_table(y_val, scores)
    notes = {
        "final_fit": {"n_iter": final_fit["n_iter"], "converged": final_fit["converged"]},
        "saved_pipeline": {
            "path": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT)
            else str(model_path),
            "reload_example": example,
        },
        "dummy_baseline_run_id": sorted(set(baselines["run_id"]))[0],
    }
    paths = write_outputs(
        directory, run, metrics_rows,
        pd.concat(prediction_frames, ignore_index=True), checks, notes,
    )
    keys = {"run_id": run["run_id"], "split_id": the_split, "seed": SEED}
    for name, frame in {
        "cv_summary": cv_summary,
        "cv_folds": cv_folds,
        "pr_curve": curve,
        "reliability": reliability,
    }.items():
        frame = frame.copy()
        for offset, (column, value) in enumerate(keys.items()):
            frame.insert(offset, column, value)
        paths[name] = directory / f"{name}.csv"
        frame.to_csv(paths[name], index=False, lineterminator="\n")

    comparison_columns = ["model", "threshold_rule", "threshold", "average_precision",
                          "roc_auc", "brier", "accuracy", "precision", "recall", "f1",
                          "tn", "fp", "fn", "tp"]
    comparison = pd.concat(
        [pd.DataFrame(metrics_rows)[comparison_columns], baselines[comparison_columns]],
        ignore_index=True,
    )
    for offset, (column, value) in enumerate(keys.items()):
        comparison.insert(offset, column, value)
    paths["comparison"] = directory / "comparison.csv"
    comparison.to_csv(paths["comparison"], index=False, lineterminator="\n")

    _figures(directory, y_val, scores, curve, reliability,
             {**by_rule["reference_0.5"], "by_rule": by_rule}, float(y_val.mean()))
    for name in ("pr_curve", "reliability", "confusion_matrix"):
        paths[f"{name}_png"] = directory / f"{name}.png"

    # Runtimes vary between runs, so they live in a separate log; everything
    # else above is deterministic.
    run_log = {
        "run_id": run["run_id"],
        "cv_seconds": round(cv_seconds, 3),
        "final_fit_seconds": round(final_fit["seconds"], 3),
        "total_seconds": round(time.perf_counter() - total_start, 3),
        "cv_fit_seconds": [{**t, "fit_seconds": round(t["fit_seconds"], 4)}
                           for t in cv_timings],
        "machine": {"platform": platform.platform(), "processor": platform.machine()},
    }
    paths["run_log"] = directory / "run_log.json"
    paths["run_log"].write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
    return {"run": run, "metrics": metrics_rows, "cv_summary": cv_summary,
            "comparison": comparison, "checks": checks, "run_log": run_log,
            "paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    args = parser.parse_args()
    try:
        result = train_and_evaluate(args.data_path, args.split_dir, args.output_dir,
                                    args.model_path, args.baseline_metrics)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Logistic regression run failed: {exc}\n")
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print(f"Run {result['run']['run_id']} on {result['run']['split_id']}, seed {SEED}")
    print("Training CV comparison (5 frozen folds, training rows only):")
    print(result["cv_summary"][["setting", "mean_average_precision",
                                "std_average_precision", "mean_roc_auc",
                                "mean_log_loss", "mean_brier", "max_n_iter",
                                "all_converged", "selected"]]
          .to_string(index=False, float_format=fmt))
    print("\nValidation results (test partition not scored):")
    print(result["comparison"].drop(columns=["run_id", "split_id", "seed"])
          .to_string(index=False, float_format=fmt))
    print(f"\nChecks: {json.dumps(result['checks'])}")
    print(f"Runtime: {result['run_log']['total_seconds']} s")


if __name__ == "__main__":
    main()
