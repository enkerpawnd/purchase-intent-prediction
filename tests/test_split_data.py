# check if the training split happened properly

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.split_data import (
    DEFAULT_DATA_PATH,
    freeze_splits,
    generate_assignments,
    iter_training_cv,
    load_frozen_splits,
    predictor_groups,
    validate_assignments,
)


class FrozenSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name) / "splits"
        freeze_splits(split_dir=cls.directory)
        cls.data, cls.assignments, cls.metadata = load_frozen_splits(
            split_dir=cls.directory
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_generation_is_deterministic(self):
        regenerated = generate_assignments(self.data)
        pd.testing.assert_frame_equal(regenerated, self.assignments)

    def test_all_rows_and_duplicate_groups_are_retained(self):
        self.assertEqual(len(self.assignments), 12330)
        self.assertEqual(self.assignments["row_index"].nunique(), 12330)
        groups = self.assignments["predictor_group"]
        self.assertEqual(groups.nunique(), 12205)
        self.assertEqual(int(groups.duplicated(keep=False).sum()), 201)
        expected = {
            "train": (8632, 1336),
            "validation": (1849, 286),
            "test": (1849, 286),
        }
        for name, (rows, purchases) in expected.items():
            counts = self.metadata["evidence"]["partitions"][name]
            self.assertEqual((counts["rows"], counts["purchases"]), (rows, purchases))

    def test_group_contract_ignores_target_and_row_order(self):
        sample = self.data.iloc[:100].copy()
        groups = predictor_groups(sample)
        sample["Revenue"] = ~sample["Revenue"]
        pd.testing.assert_series_equal(groups, predictor_groups(sample))
        pd.testing.assert_series_equal(
            groups.iloc[::-1], predictor_groups(sample.iloc[::-1])
        )
        predictors = self.data.drop(columns="Revenue")
        self.assertEqual(
            predictors.duplicated().tolist(),
            self.assignments["predictor_group"].duplicated().tolist(),
        )

    def test_cv_indices_address_only_training_rows(self):
        train = self.assignments.loc[
            self.assignments["partition"].eq("train")
        ].reset_index(drop=True)
        holdout_positions = []
        for fold, (fit, holdout) in enumerate(iter_training_cv(self.assignments)):
            self.assertFalse(set(fit) & set(holdout))
            self.assertEqual(set(fit) | set(holdout), set(range(len(train))))
            self.assertFalse(
                set(train.iloc[fit]["predictor_group"])
                & set(train.iloc[holdout]["predictor_group"])
            )
            self.assertTrue(train.iloc[holdout]["cv_fold"].eq(fold).all())
            purchases = int(
                self.data.iloc[train.iloc[holdout]["row_index"]]["Revenue"].sum()
            )
            self.assertIn(purchases, (267, 268))
            holdout_positions.extend(holdout)
        self.assertEqual(sorted(holdout_positions), list(range(len(train))))
        # The helper preserves its local-index contract even if input is reordered.
        for expected, actual in zip(
            iter_training_cv(self.assignments),
            iter_training_cv(self.assignments.iloc[::-1]),
        ):
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(left, right)

    def test_rejects_missing_or_repeated_source_rows(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_assignments(self.data, self.assignments.iloc[1:])
        repeated = self.assignments.copy()
        repeated.loc[0, "row_index"] = repeated.loc[1, "row_index"]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_assignments(self.data, repeated)
        fractional = self.assignments.astype({"row_index": float})
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_assignments(self.data, fractional)

    def test_rejects_predictor_group_crossing_partitions(self):
        changed = self.assignments.copy()
        train = changed.loc[changed["partition"].eq("train")]
        row = train.loc[train["predictor_group"].duplicated(keep=False)].index[0]
        changed.loc[row, ["partition", "cv_fold"]] = ["validation", -1]
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            validate_assignments(self.data, changed)

    def test_rejects_predictor_group_crossing_cv_folds(self):
        changed = self.assignments.copy()
        train = changed.loc[changed["partition"].eq("train")]
        row = train.loc[train["predictor_group"].duplicated(keep=False)].index[0]
        changed.loc[row, "cv_fold"] = (changed.loc[row, "cv_fold"] + 1) % 5
        with self.assertRaisesRegex(ValueError, "crosses training CV"):
            validate_assignments(self.data, changed)

    def test_rejects_source_group_mismatch(self):
        changed = self.assignments.copy()
        changed.loc[0, "predictor_group"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "do not match source"):
            validate_assignments(self.data, changed)

    def test_rejects_heldout_rows_with_cv_assignment(self):
        changed = self.assignments.copy()
        row = changed.loc[changed["partition"].eq("test")].index[0]
        changed.loc[row, "cv_fold"] = 0
        with self.assertRaisesRegex(ValueError, "must not have training CV"):
            validate_assignments(self.data, changed)

    def test_rejects_single_class_evaluation(self):
        data = self.data.copy()
        data["Revenue"] = False
        with self.assertRaisesRegex(ValueError, "both classes"):
            validate_assignments(data, self.assignments)

    def test_existing_bundle_is_never_overwritten(self):
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        freeze_splits(split_dir=self.directory)
        self.assertEqual(
            before, {path.name: path.read_bytes() for path in self.directory.iterdir()}
        )

    def test_rejects_modified_source_and_preserves_original(self):
        before = DEFAULT_DATA_PATH.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            changed_source = Path(temp) / "changed.csv"
            changed_source.write_bytes(before + b"\n")
            with self.assertRaisesRegex(ValueError, "Source checksum"):
                load_frozen_splits(changed_source, self.directory)
        self.assertEqual(DEFAULT_DATA_PATH.read_bytes(), before)

    def test_rejects_duplicate_rows_in_saved_manifest_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "splits"
            directory.mkdir()
            for source in self.directory.iterdir():
                (directory / source.name).write_bytes(source.read_bytes())
            manifest = directory / "train.csv"
            rows = pd.read_csv(manifest)
            pd.concat([rows, rows.iloc[:1]], ignore_index=True).to_csv(
                manifest, index=False
            )
            before = manifest.read_bytes()
            with self.assertRaisesRegex(ValueError, "exactly once"):
                load_frozen_splits(split_dir=directory)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                freeze_splits(split_dir=directory)
            self.assertEqual(manifest.read_bytes(), before)

    def test_creation_records_source_without_a_hardcoded_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.csv"
            source.write_bytes(DEFAULT_DATA_PATH.read_bytes() + b"\n")
            directory = Path(temp) / "splits"
            metadata = freeze_splits(source, directory)
            self.assertNotEqual(
                metadata["source"]["sha256"], self.metadata["source"]["sha256"]
            )
            _, assignments, _ = load_frozen_splits(source, directory)
            pd.testing.assert_frame_equal(assignments, self.assignments)


if __name__ == "__main__":
    unittest.main()
