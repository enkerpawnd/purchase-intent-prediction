"""Create or check the frozen, source-bound PIP-03 split (no model fitting)."""

import argparse
import hashlib
import io
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data/raw/online_shoppers_intention.csv"
DEFAULT_SPLIT_DIR = ROOT / "data/processed/splits"
GROUP_CONTRACT = "predictor-json-sha256-v1"
SEED = 42
CV_FOLDS = 5
PARTITIONS = ("train", "validation", "test")
MANIFEST_COLUMNS = ["row_index", "predictor_group", "cv_fold"]
SETTINGS = {
    "seed": SEED,
    "group_contract": GROUP_CONTRACT,
    "holdout_algorithm": "StratifiedGroupKFold",
    "holdout_folds": 20,
    "shuffle": True,
    "partition_folds": {
        "train": list(range(14)),
        "validation": list(range(14, 17)),
        "test": list(range(17, 20)),
    },
    "target_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "cv_algorithm": "StratifiedGroupKFold",
    "cv_folds": CV_FOLDS,
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_source(data_path: str | Path) -> tuple[pd.DataFrame, str]:
    # Record which CSV the saved row positions belong to; trust the committed source.
    payload = Path(data_path).expanduser().read_bytes()
    data = pd.read_csv(io.BytesIO(payload))
    if data.shape != (12330, 18) or data["Revenue"].dtype != bool:
        raise ValueError("Expected the audited 12,330-row source with Boolean Revenue.")
    if data.isna().any().any():
        raise ValueError("The audited source must not contain missing values.")
    return data, _sha256(payload)


def predictor_groups(data: pd.DataFrame) -> pd.Series:
    predictors = data.drop(columns="Revenue")
    columns = list(predictors.columns)
    seen = {}
    groups = []
    for row in predictors.itertuples(index=False, name=None):
        payload = json.dumps(
            {"columns": columns, "values": list(row)},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        group_id = _sha256(payload)
        if group_id in seen and seen[group_id] != row:
            raise ValueError("Predictor group hash collision.")
        seen[group_id] = row
        groups.append(group_id)
    return pd.Series(groups, index=data.index, name="predictor_group")


def generate_assignments(data: pd.DataFrame) -> pd.DataFrame:
    groups = predictor_groups(data)
    assignments = pd.DataFrame(
        {
            "row_index": np.arange(len(data)),
            "partition": "",
            "predictor_group": groups.to_numpy(),
            "cv_fold": -1,
        }
    )
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    for fold, (_, held_out) in enumerate(splitter.split(data, data["Revenue"], groups)):
        partition = next(
            name for name in PARTITIONS if fold in SETTINGS["partition_folds"][name]
        )
        assignments.loc[held_out, "partition"] = partition

    train = assignments.loc[
        assignments["partition"].eq("train"), "row_index"
    ].to_numpy()
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, held_out) in enumerate(
        cv.split(data.iloc[train], data["Revenue"].iloc[train], groups.iloc[train])
    ):
        assignments.loc[train[held_out], "cv_fold"] = fold
    validate_assignments(data, assignments)
    return assignments


def _counts(data: pd.DataFrame, rows: pd.DataFrame) -> dict:
    labels = data["Revenue"].iloc[rows["row_index"].to_numpy()]
    group_ids = cast(pd.Series, rows["predictor_group"])
    purchases = int(labels.sum())
    if purchases == 0 or purchases == len(labels):
        raise ValueError(
            "Every partition and CV fit/holdout must contain both classes."
        )
    return {
        "rows": len(rows),
        "groups": group_ids.nunique(),
        "non_purchases": len(rows) - purchases,
        "purchases": purchases,
        "purchase_fraction": purchases / len(rows),
    }


def validate_assignments(data: pd.DataFrame, assignments: pd.DataFrame) -> dict:
    # Check complete row coverage, source group IDs, separation and class counts.
    if list(assignments.columns) != [
        "row_index",
        "partition",
        "predictor_group",
        "cv_fold",
    ]:
        raise ValueError("Unexpected assignment columns.")
    for column in ("row_index", "cv_fold"):
        if not pd.api.types.is_integer_dtype(assignments[column]):
            raise ValueError(f"{column} must contain integers.")
    if not np.array_equal(np.sort(assignments["row_index"]), np.arange(len(data))):
        raise ValueError("Row indices must cover every source row exactly once.")
    if set(assignments["partition"]) != set(PARTITIONS):
        raise ValueError("Expected train, validation and test partitions only.")
    ordered = assignments.sort_values("row_index")
    if not np.array_equal(ordered["predictor_group"], predictor_groups(data)):
        raise ValueError("Manifest group IDs do not match source predictors.")
    partition_counts = cast(
        pd.Series, assignments.groupby("predictor_group")["partition"].nunique()
    )
    if partition_counts.gt(1).any():
        raise ValueError("A predictor group crosses partitions.")

    train = assignments.loc[assignments["partition"].eq("train")]
    held_out = assignments.loc[~assignments["partition"].eq("train")]
    if not held_out["cv_fold"].eq(-1).all():
        raise ValueError("Validation/test rows must not have training CV folds.")
    if set(train["cv_fold"]) != set(range(CV_FOLDS)):
        raise ValueError("Training rows must have exactly the five frozen CV folds.")
    fold_counts = cast(pd.Series, train.groupby("predictor_group")["cv_fold"].nunique())
    if fold_counts.gt(1).any():
        raise ValueError("A predictor group crosses training CV folds.")

    splits = {
        name: _counts(data, assignments.loc[assignments["partition"].eq(name)])
        for name in PARTITIONS
    }
    revenue = cast(pd.Series, data["Revenue"])
    prevalence = float(revenue.mean())
    for name, counts in splits.items():
        if abs(counts["rows"] / len(data) - SETTINGS["target_ratios"][name]) > 0.01:
            raise ValueError(f"{name} size differs by more than one percentage point.")
        if abs(counts["purchase_fraction"] - prevalence) > 0.01:
            raise ValueError(
                f"{name} purchase rate differs by more than one percentage point."
            )
    cv_counts = []
    for fold in range(CV_FOLDS):
        cv_counts.append(
            {
                "fold": fold,
                "fit": _counts(data, train.loc[train["cv_fold"].ne(fold)]),
                "holdout": _counts(data, train.loc[train["cv_fold"].eq(fold)]),
            }
        )
    return {
        "partitions": splits,
        "training_cv": cv_counts,
        "checks": {
            "all_source_rows_once": True,
            "source_group_ids_match": True,
            "no_groups_cross_partitions": True,
            "no_groups_cross_training_cv": True,
            "cv_excludes_validation_and_test": True,
            "both_classes_in_all_partitions_and_cv_sides": True,
        },
    }


def iter_training_cv(assignments: pd.DataFrame):
    """Yield fit/holdout POSITIONS into training data sorted by source row_index.

    Use validated assignments from load_frozen_splits. These are local indices
    for sklearn's cv= argument, not indices into the complete source DataFrame.
    """
    train = assignments.loc[assignments["partition"].eq("train")].sort_values(
        "row_index"
    )
    folds = train["cv_fold"].to_numpy()
    for fold in range(CV_FOLDS):
        yield np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)


def load_frozen_splits(
    data_path: str | Path = DEFAULT_DATA_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read and validate saved manifests before selecting rows with data.iloc."""
    data, source_checksum = _read_source(data_path)
    directory = Path(split_dir).expanduser()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    expected_source = {
        "sha256": source_checksum,
        "rows": len(data),
        "predictor_columns": list(data.drop(columns="Revenue").columns),
    }
    if metadata["source"]["sha256"] != source_checksum:
        raise ValueError(
            "Source checksum does not match the CSV used to create these splits."
        )
    if (
        metadata["schema_version"] != 1
        or metadata["settings"] != SETTINGS
        or metadata["source"] != expected_source
    ):
        raise ValueError("Manifest metadata does not match the frozen source/protocol.")
    frames = []
    for name in PARTITIONS:
        filename = f"{name}.csv"
        frame = pd.read_csv(directory / filename)
        if list(frame.columns) != MANIFEST_COLUMNS:
            raise ValueError(f"Unexpected columns in {filename}.")
        frame.insert(1, "partition", name)
        frames.append(frame)
    assignments = (
        pd.concat(frames, ignore_index=True)
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    if validate_assignments(data, assignments) != metadata["evidence"]:
        raise ValueError("Recorded split evidence does not match the manifests.")
    return data, assignments, metadata


def freeze_splits(
    data_path: str | Path = DEFAULT_DATA_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
) -> dict:
    """Create a complete bundle, or check an existing one without overwriting it."""
    directory = Path(split_dir).expanduser()
    if directory.exists():
        return load_frozen_splits(data_path, directory)[2]
    data, source_checksum = _read_source(data_path)
    assignments = generate_assignments(data)
    payloads = {
        f"{name}.csv": assignments.loc[
            assignments["partition"].eq(name), MANIFEST_COLUMNS
        ]
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
        for name in PARTITIONS
    }
    metadata = {
        "schema_version": 1,
        "source": {
            "sha256": source_checksum,
            "rows": len(data),
            "predictor_columns": list(data.drop(columns="Revenue").columns),
        },
        "row_index_definition": "Zero-based CSV data-row position, excluding header; use iloc.",
        "positive_label": "Revenue=True",
        "settings": SETTINGS,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "evidence": validate_assignments(data, assignments),
    }
    payloads["metadata.json"] = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Stage all files together, so a failed write cannot leave a partial bundle.
    with TemporaryDirectory(prefix=".split-", dir=directory.parent) as staging:
        for filename, payload in payloads.items():
            (Path(staging) / filename).write_bytes(payload)
        load_frozen_splits(data_path, staging)
        if directory.exists():
            raise FileExistsError(
                "Split directory appeared during generation; refusing replacement."
            )
        Path(staging).rename(directory)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check existing manifests; never generate them.",
    )
    args = parser.parse_args()
    try:
        metadata = (
            load_frozen_splits(args.data_path, args.split_dir)[2]
            if args.check
            else freeze_splits(args.data_path, args.split_dir)
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.exit(1, f"Split failed: {exc}\n")
    print(
        "Frozen split verified; source bytes unchanged. No model/test evaluation performed."
    )
    print(json.dumps(metadata["evidence"], indent=2))


if __name__ == "__main__":
    main()
