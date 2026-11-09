"""Fit training-only smoke pipelines, save them and verify raw-row inference.

Run from the repo root: python -m src.check_preprocessing
These small development artifacts do not select or evaluate a final model.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.feature_sets import FEATURE_SETS
from src.preprocessing import make_model_pipeline, split_features_target
from src.split_data import ROOT, load_frozen_splits


def verify_pipelines(output_dir: str | Path) -> dict[str, object]:
    data, assignments, metadata = load_frozen_splits()
    rows = assignments.loc[
        assignments["partition"].eq("train"), "row_index"
    ].sort_values()
    train = data.iloc[rows.to_numpy()]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    evidence: dict[str, object] = {
        "source_sha256": metadata["source"]["sha256"],
        "fit_partition": "train",
        "fit_rows": len(train),
        "inference_source_row": int(rows.iloc[0]),
        "validation_or_test_evaluated": False,
        "checks": checks,
    }
    for feature_set in FEATURE_SETS:
        X, y = split_features_target(train, feature_set)
        # Raw values, including the original Boolean, go through the saved validator.
        raw = train.loc[:, X.columns].iloc[:1].copy()
        unseen = raw.copy()
        unseen["Month"] = "Jan"
        unseen["Browser"] = 999
        for name, estimator, scale in [
            ("logistic", LogisticRegression(max_iter=2000, random_state=42), True),
            (
                "random_forest",
                RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42),
                False,
            ),
        ]:
            pipeline = make_model_pipeline(estimator, feature_set, scale_numeric=scale)
            pipeline.fit(X, y)
            expected = pipeline.predict_proba(raw)
            expected_unseen = pipeline.predict_proba(unseen)
            path = directory / f"{feature_set}_{name}.joblib"
            joblib.dump(pipeline, path)
            restored = joblib.load(path)
            np.testing.assert_allclose(
                restored.predict_proba(raw), expected, rtol=0, atol=0
            )
            np.testing.assert_allclose(
                restored.predict_proba(unseen), expected_unseen, rtol=0, atol=0
            )
            checks.append(
                {
                    "feature_set": feature_set,
                    "model": name,
                    "artifact": str(path),
                    "encoded_features": len(
                        restored.named_steps["preprocess"].get_feature_names_out()
                    ),
                    "prediction": int(restored.predict(raw)[0]),
                    "purchase_probability": float(expected[0, 1]),
                    "unseen_category_prediction": int(restored.predict(unseen)[0]),
                    "saved_reload_predictions_identical": True,
                }
            )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "models/preprocessing_smoke"
    )
    args = parser.parse_args()
    print(json.dumps(verify_pipelines(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
