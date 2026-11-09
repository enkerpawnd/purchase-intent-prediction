"""Common evaluation output for purchase scores (frozen PIP-03 protocol).

Every model, including dummy baselines, is evaluated through these functions so
metrics, threshold handling, confusion-matrix layout and exported columns stay
identical across experiments.

Conventions
-----------
* Positive class: ``Revenue=True`` (purchase). Scores are purchase
  probabilities/scores; AP, ROC-AUC and Brier are computed from scores, never
  from hard classes.
* Hard class: purchase when ``score >= threshold``.
* Undefined precision, recall or F1 (no predicted or no actual purchases) is
  reported as 0 (``zero_division=0``). AP and ROC-AUC require both classes in
  the evaluation set; otherwise evaluation fails instead of reporting a value.
* Confusion matrix: rows actual, columns predicted, ordered non-purchase then
  purchase, i.e. ``[[tn, fp], [fn, tp]]``.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ZERO_DIVISION = 0
REFERENCE_THRESHOLD = 0.5
CLASS_LABELS = (False, True)
CLASS_NAMES = ("non_purchase", "purchase")
TIE_TOLERANCE = 1e-12
METRIC_COLUMNS = [
    "n",
    "n_purchases",
    "prevalence",
    "average_precision",
    "roc_auc",
    "brier",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "predicted_purchases",
    "tn",
    "fp",
    "fn",
    "tp",
]


def _as_arrays(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    if y.dtype != bool:
        raise ValueError("y_true must be Boolean Revenue labels.")
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional and equal length.")
    if not np.isfinite(s).all() or s.min() < 0 or s.max() > 1:
        raise ValueError("Scores must be finite purchase probabilities in [0, 1].")
    if y.all() or not y.any():
        raise ValueError("Evaluation set must contain both classes for AP/ROC-AUC.")
    return y, s


def predict_from_scores(scores, threshold: float) -> np.ndarray:
    """Purchase when the score is at least the threshold."""
    return np.asarray(scores, dtype=float) >= threshold


def evaluate_scores(y_true, scores, threshold: float = REFERENCE_THRESHOLD) -> dict:
    """Return the protocol's metrics for purchase scores at one threshold."""
    y, s = _as_arrays(y_true, scores)
    predicted = predict_from_scores(s, threshold)
    (tn, fp), (fn, tp) = confusion_matrix(y, predicted, labels=list(CLASS_LABELS))
    pos = dict(pos_label=True, zero_division=ZERO_DIVISION)
    return {
        "n": int(len(y)),
        "n_purchases": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, s, pos_label=True)),
        "roc_auc": float(roc_auc_score(y, s)),
        "brier": float(brier_score_loss(y, s, pos_label=True)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, **pos)),
        "recall": float(recall_score(y, predicted, **pos)),
        "f1": float(f1_score(y, predicted, **pos)),
        "predicted_purchases": int(predicted.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_f1_threshold(y_true, scores) -> float:
    """Protocol threshold: maximize purchase F1 over distinct scores plus 0/0.5/1.

    Equal-F1 ties go to the largest threshold (fewer purchase predictions).
    """
    y, s = _as_arrays(y_true, scores)
    candidates = np.unique(np.concatenate([s, [0.0, REFERENCE_THRESHOLD, 1.0]]))
    f1s = np.array(
        [
            f1_score(y, s >= t, pos_label=True, zero_division=ZERO_DIVISION)
            for t in candidates
        ]
    )
    best = f1s.max()
    return float(candidates[np.flatnonzero(f1s >= best - TIE_TOLERANCE)].max())


def confusion_long(metrics: dict) -> list[dict]:
    """Confusion matrix as rows of (actual, predicted, count)."""
    cells = {
        ("non_purchase", "non_purchase"): "tn",
        ("non_purchase", "purchase"): "fp",
        ("purchase", "non_purchase"): "fn",
        ("purchase", "purchase"): "tp",
    }
    return [
        {"actual": actual, "predicted": predicted, "count": metrics[key]}
        for (actual, predicted), key in cells.items()
    ]


def split_id(split_dir: str | Path) -> str:
    """Content ID of a frozen split bundle (metadata plus the three manifests)."""
    directory = Path(split_dir)
    digest = hashlib.sha256()
    for name in ("metadata.json", "train.csv", "validation.csv", "test.csv"):
        digest.update(name.encode() + b"\0" + (directory / name).read_bytes() + b"\0")
    return "split-" + digest.hexdigest()[:16]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_id(prefix: str, config: dict) -> str:
    """Deterministic run ID derived from the run configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:12]}"


def write_outputs(
    output_dir: str | Path,
    run: dict,
    metrics_rows: list[dict],
    predictions: pd.DataFrame,
    checks: dict,
    notes: dict | None = None,
) -> dict[str, Path]:
    """Write the standard machine-readable result bundle for one run.

    Files: ``metrics.csv`` (result table), ``predictions.csv``,
    ``confusion_matrices.csv`` and ``results.json`` (run configuration,
    metrics, confusion matrices, checks). Every CSV row carries the run ID,
    split ID and seed. Output is deterministic (no timestamps).
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    keys = {"run_id": run["run_id"], "split_id": run["split_id"], "seed": run["seed"]}
    id_columns = list(keys)

    metrics = pd.DataFrame([{**keys, **row} for row in metrics_rows])
    confusion = pd.DataFrame(
        [
            {
                **keys,
                "model": row["model"],
                "partition": row["partition"],
                "threshold_rule": row["threshold_rule"],
                "threshold": row["threshold"],
                **cell,
            }
            for row in metrics_rows
            for cell in confusion_long(row)
        ]
    )
    predictions = predictions.copy()
    for offset, (column, value) in enumerate(keys.items()):
        predictions.insert(offset, column, value)

    paths = {
        "metrics": directory / "metrics.csv",
        "predictions": directory / "predictions.csv",
        "confusion_matrices": directory / "confusion_matrices.csv",
        "results": directory / "results.json",
    }
    metrics.to_csv(paths["metrics"], index=False, lineterminator="\n")
    predictions.to_csv(paths["predictions"], index=False, lineterminator="\n")
    confusion.to_csv(paths["confusion_matrices"], index=False, lineterminator="\n")
    payload = {
        "run": run,
        "conventions": {
            "positive_label": "Revenue=True",
            "hard_class_rule": "purchase when score >= threshold",
            "zero_division": ZERO_DIVISION,
            "zero_division_meaning": "undefined precision/recall/F1 reported as 0",
            "score_metrics": ["average_precision", "roc_auc", "brier"],
            "average_precision": "sklearn average_precision_score (non-interpolated)",
            "confusion_matrix_layout": "rows actual, columns predicted; "
            "[[tn, fp], [fn, tp]] ordered non_purchase, purchase",
            "id_columns": id_columns,
        },
        "metrics": metrics_rows,
        "checks": checks,
        "notes": notes or {},
    }
    paths["results"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths
