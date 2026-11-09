# check PIP-07 logistic regression training, validation outputs and save/reload

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.split_data import DEFAULT_SPLIT_DIR, ROOT
from src.train_logistic import C_VALUES, CLASS_WEIGHTS, train_and_evaluate


class LogisticRegressionRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        base = Path(cls.temp.name)
        cls.out = base / "results"
        cls.model_path = base / "model.joblib"
        cls.result = train_and_evaluate(output_dir=cls.out, model_path=cls.model_path)
        cls.metrics = pd.read_csv(cls.out / "metrics.csv")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_bounded_grid_on_training_folds_selects_best_cv_ap(self):
        summary = pd.read_csv(self.out / "cv_summary.csv")
        folds = pd.read_csv(self.out / "cv_folds.csv")
        self.assertEqual(len(summary), len(C_VALUES) * len(CLASS_WEIGHTS))
        self.assertEqual(len(folds), len(summary) * 5)
        self.assertEqual(int(summary["selected"].sum()), 1)
        selected = summary.loc[summary["selected"]].iloc[0]
        self.assertEqual(
            selected["mean_average_precision"], summary["mean_average_precision"].max()
        )
        self.assertTrue(folds["converged"].all())

    def test_records_setting_solver_objective_seed_and_runtime(self):
        run = json.loads((self.out / "results.json").read_text())["run"]
        self.assertEqual(run["fixed_params"]["solver"], "lbfgs")
        self.assertEqual(run["fixed_params"]["l1_ratio"], 0.0)
        self.assertIn("log-loss", run["objective"])
        self.assertEqual(run["seed"], 42)
        self.assertIn(run["selected"]["class_weight"], ("none", "balanced"))
        log = json.loads((self.out / "run_log.json").read_text())
        self.assertEqual(log["run_id"], run["run_id"])
        self.assertGreater(log["total_seconds"], 0)

    def test_only_validation_rows_are_scored(self):
        predictions = pd.read_csv(self.out / "predictions.csv")
        validation = pd.read_csv(DEFAULT_SPLIT_DIR / "validation.csv")
        self.assertEqual(set(predictions["row_index"]), set(validation["row_index"]))
        self.assertEqual(set(predictions["partition"]), {"validation"})

    def test_metrics_come_from_scores_and_beat_dummies(self):
        predictions = pd.read_csv(self.out / "predictions.csv")
        reference = predictions.loc[predictions["threshold_rule"].eq("reference_0.5")]
        ap = average_precision_score(reference["y_true"], reference["purchase_score"])
        row = self.metrics.loc[self.metrics["threshold_rule"].eq("reference_0.5")].iloc[0]
        self.assertAlmostEqual(row["average_precision"], ap, places=12)
        comparison = pd.read_csv(self.out / "comparison.csv")
        self.assertEqual(
            set(comparison["model"]),
            {"logistic_regression", "majority_class", "class_prior"},
        )
        dummy_ap = comparison.loc[
            ~comparison["model"].eq("logistic_regression"), "average_precision"
        ].max()
        self.assertGreater(row["average_precision"], dummy_ap)

    def test_curve_and_reliability_outputs(self):
        reliability = pd.read_csv(self.out / "reliability.csv")
        self.assertEqual(len(reliability), 10)
        self.assertEqual(int(reliability["count"].sum()), 1849)
        self.assertEqual(int(reliability["purchases"].sum()), 286)
        curve = pd.read_csv(self.out / "pr_curve.csv")
        self.assertEqual(curve["recall"].iloc[0], 1.0)
        self.assertEqual(curve["recall"].iloc[-1], 0.0)
        for name in ("pr_curve.png", "reliability.png", "confusion_matrix.png"):
            self.assertGreater((self.out / name).stat().st_size, 0)

    def test_exports_carry_run_split_and_seed(self):
        run_id = self.result["run"]["run_id"]
        for name in ("metrics.csv", "predictions.csv", "confusion_matrices.csv",
                     "cv_summary.csv", "pr_curve.csv", "reliability.csv",
                     "comparison.csv"):
            frame = pd.read_csv(self.out / name)
            self.assertEqual(list(frame.columns[:3]), ["run_id", "split_id", "seed"])
            self.assertEqual(set(frame["run_id"]), {run_id})
            self.assertEqual(set(frame["seed"]), {42})

    def test_saved_pipeline_reproduces_prediction_in_fresh_process(self):
        self.assertTrue(self.result["checks"]["reloaded_predictions_identical"])
        example = json.loads((self.out / "results.json").read_text())["notes"][
            "saved_pipeline"
        ]["reload_example"]
        code = (
            "import joblib, sys\n"
            "from src.load_data import load_data\n"
            "row = int(sys.argv[2])\n"
            "pipe = joblib.load(sys.argv[1])\n"
            "raw = load_data().drop(columns='Revenue').iloc[[row]]\n"
            "print(repr(float(pipe.predict_proba(raw)[0, 1])))\n"
        )
        output = subprocess.run(
            [sys.executable, "-c", code, str(self.model_path),
             str(example["source_row_index"])],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(float(output), example["purchase_probability"])

    def test_rejects_baselines_from_another_split(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = pd.read_csv(ROOT / "results/baselines/metrics.csv")
            baseline["split_id"] = "split-other"
            path = Path(temp) / "metrics.csv"
            baseline.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "different split"):
                train_and_evaluate(output_dir=Path(temp) / "out",
                                   model_path=Path(temp) / "m.joblib",
                                   baseline_metrics=path)

    def test_probabilities_are_valid(self):
        scores = pd.read_csv(self.out / "predictions.csv")["purchase_score"]
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))


if __name__ == "__main__":
    unittest.main()
