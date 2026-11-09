# check the common evaluation output and the dummy baseline run

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import run_baselines
from src.evaluation import evaluate_scores, select_f1_threshold
from src.split_data import DEFAULT_SPLIT_DIR


class EvaluationTests(unittest.TestCase):
    def test_constant_score_ap_equals_prevalence_for_any_constant(self):
        y = np.array([True] * 3 + [False] * 17)
        for constant in (0.0, 0.15, 0.5, 1.0):
            metrics = evaluate_scores(y, np.full(len(y), constant))
            self.assertAlmostEqual(metrics["average_precision"], 0.15, places=12)
            self.assertEqual(metrics["roc_auc"], 0.5)

    def test_zero_division_is_reported_as_zero(self):
        y = np.array([True, False, False, False])
        metrics = evaluate_scores(y, np.zeros(4), threshold=0.5)
        self.assertEqual(metrics["predicted_purchases"], 0)
        self.assertEqual((metrics["precision"], metrics["recall"]), (0.0, 0.0))
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_threshold_is_inclusive_and_confusion_layout(self):
        y = np.array([True, True, False, False])
        metrics = evaluate_scores(y, np.array([0.5, 0.2, 0.5, 0.1]), threshold=0.5)
        self.assertEqual(
            (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 1, 1)
        )

    def test_scores_not_hard_classes_drive_ranking_metrics(self):
        y = np.array([True, False, True, False])
        metrics = evaluate_scores(y, np.array([0.4, 0.1, 0.3, 0.2]), threshold=0.5)
        self.assertEqual(metrics["predicted_purchases"], 0)  # no hard purchases
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)

    def test_f1_threshold_ties_go_to_largest(self):
        y = np.array([True, False, False])
        # Thresholds 0 and 0.2 both predict everything as purchase (same F1).
        self.assertEqual(select_f1_threshold(y, np.full(3, 0.2)), 0.2)
        self.assertEqual(select_f1_threshold(y, np.array([0.9, 0.1, 0.1])), 0.9)

    def test_rejects_single_class_or_invalid_scores(self):
        with self.assertRaisesRegex(ValueError, "both classes"):
            evaluate_scores(np.array([False, False]), np.array([0.1, 0.2]))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            evaluate_scores(np.array([True, False]), np.array([1.2, 0.2]))
        with self.assertRaisesRegex(ValueError, "Boolean"):
            evaluate_scores(np.array([1, 0]), np.array([0.5, 0.2]))


class BaselineRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.first = Path(cls.temp.name) / "first"
        cls.result = run_baselines(output_dir=cls.first)
        cls.metrics = pd.read_csv(cls.first / "metrics.csv")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def row(self, model, rule):
        rows = self.metrics.loc[
            self.metrics["model"].eq(model) & self.metrics["threshold_rule"].eq(rule)
        ]
        self.assertEqual(len(rows), 1)
        return rows.iloc[0]

    def test_run_is_byte_for_byte_repeatable(self):
        second = Path(self.temp.name) / "second"
        run_baselines(output_dir=second)
        for path in self.first.iterdir():
            self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())

    def test_validation_values(self):
        prevalence = 286 / 1849
        for model in ("majority_class", "class_prior"):
            reference = self.row(model, "reference_0.5")
            self.assertAlmostEqual(reference["accuracy"], 1563 / 1849, places=12)
            self.assertEqual(reference["recall"], 0.0)
            self.assertEqual(reference["precision"], 0.0)
            self.assertAlmostEqual(reference["average_precision"], prevalence, 12)
            self.assertEqual(reference["roc_auc"], 0.5)
            selected = self.row(model, "validation_max_f1")
            self.assertEqual(selected["predicted_purchases"], 1849)
            self.assertAlmostEqual(selected["f1"], 2 * 286 / (1849 + 286), places=12)
        prior_threshold = self.row("class_prior", "validation_max_f1")["threshold"]
        self.assertEqual(prior_threshold, 1336 / 8632)

    def test_prior_comes_from_training_labels_only(self):
        predictions = pd.read_csv(self.first / "predictions.csv")
        prior = predictions.loc[predictions["model"].eq("class_prior")]
        self.assertTrue(np.allclose(prior["purchase_score"], 1336 / 8632))
        self.assertFalse(np.isclose(1336 / 8632, 286 / 1849, rtol=0, atol=1e-6))

    def test_only_validation_rows_are_scored(self):
        predictions = pd.read_csv(self.first / "predictions.csv")
        validation = pd.read_csv(
            DEFAULT_SPLIT_DIR / "validation.csv", usecols=["row_index"]
        )
        self.assertEqual(set(predictions["row_index"]), set(validation["row_index"]))
        self.assertEqual(set(predictions["partition"]), {"validation"})

    def test_exports_carry_run_split_and_seed(self):
        run = self.result["run"]
        for name in ("metrics.csv", "predictions.csv", "confusion_matrices.csv"):
            frame = pd.read_csv(self.first / name)
            self.assertEqual(list(frame.columns[:3]), ["run_id", "split_id", "seed"])
            self.assertEqual(set(frame["run_id"]), {run["run_id"]})
            self.assertEqual(set(frame["split_id"]), {run["split_id"]})
            self.assertEqual(set(frame["seed"]), {42})
        payload = json.loads((self.first / "results.json").read_text())
        self.assertEqual(payload["run"]["seed"], 42)
        self.assertEqual(payload["conventions"]["zero_division"], 0)
        full = payload["notes"]["full_csv_always_no_purchase"]
        self.assertAlmostEqual(full["accuracy"], 10422 / 12330, places=12)
        self.assertEqual(full["purchase_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
