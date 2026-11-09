"""Schema, encoding, fold isolation and persisted raw-input prediction checks."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.feature_sets import ALL_17, FEATURE_SETS, NOMINAL_CATEGORICAL, NUMERIC
from src.preprocessing import (
    SessionInputValidator,
    make_model_pipeline,
    make_preprocessor,
    split_features_target,
)
from src.split_data import ROOT, iter_training_cv, load_frozen_splits


def sessions():
    """Small valid raw fixture with independently varying predictors/target."""
    data = pd.DataFrame(0.0, index=range(8), columns=ALL_17)
    for column in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        data[column] = [1, 2, 3, 1, 2, 3, 1, 2]
    data["Month"] = ["Feb", "Mar"] * 4
    data["VisitorType"] = ["Returning_Visitor", "New_Visitor"] * 4
    data["Weekend"] = [False, True] * 4
    data["ProductRelated"] = [0, 2, 4, 6, 8, 10, 12, 14]
    data["ProductRelated_Duration"] = np.arange(8) * 10.5
    data["PageValues"] = np.arange(8) * 2.0
    data["Revenue"] = [False, False, True, True] * 2
    return data


def feature_names(preprocessor: Pipeline) -> list[str]:
    """Narrow sklearn's inferred optional return for our nonempty pipelines."""
    names = preprocessor.get_feature_names_out()
    assert names is not None
    return list(names)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        # unittest creates these per-test fixtures through setUp, not __init__.
        self.data = sessions()  # pyright: ignore[reportUninitializedInstanceVariable]
        self.X, self.y = split_features_target(self.data)  # pyright: ignore[reportUninitializedInstanceVariable]

    def test_separates_and_validates_target_without_mutation(self):
        before = self.data.copy(deep=True)
        self.assertEqual(list(self.X), ALL_17)
        self.assertEqual(self.y.tolist(), [0, 0, 1, 1] * 2)
        self.assertEqual(self.y.name, "Revenue")
        pd.testing.assert_frame_equal(self.data, before)
        for value in [None, 2, "yes", "", np.nan]:
            bad = self.data.astype({"Revenue": object})
            bad.loc[0, "Revenue"] = value
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "Revenue"),
            ):
                split_features_target(bad)
        with self.assertRaisesRegex(ValueError, "Revenue is required"):
            split_features_target(self.X)

    def test_variants_select_only_required_predictors_and_normalize_column_order(self):
        for name, columns in FEATURE_SETS.items():
            with self.subTest(name=name):
                X, _ = split_features_target(self.data, name)
                self.assertEqual(list(X), columns)
                prep = make_preprocessor(name)
                expected = prep.fit_transform(X)
                reordered = self.X.loc[:, ALL_17[::-1]].copy()
                # Excluded measurements cannot influence this variant, even when invalid.
                for col in set(ALL_17) - set(columns):
                    reordered[col] = np.nan
                np.testing.assert_allclose(prep.transform(reordered), expected)
                names = feature_names(prep)
                self.assertNotIn("Revenue", " ".join(names))
                for col in set(ALL_17) - set(columns):
                    self.assertFalse(any(col in s for s in names))

    def test_schema_errors_are_actionable_at_fit_and_inference(self):
        prep = make_preprocessor().fit(self.X)
        cases = [
            (self.X.drop(columns="Browser"), "missing=.*Browser"),
            (self.X.assign(typo=1), "unexpected=.*typo"),
            (self.data, "Revenue is the target"),
            (pd.concat([self.X, self.X[["Browser"]]], axis=1), "Duplicate"),
            (self.X.to_numpy(), "DataFrame"),
            (self.X.iloc[:0], "at least one"),
        ]
        for bad, message in cases:
            with self.subTest(message=message):
                for operation in [prep.transform, make_preprocessor().fit]:
                    with self.assertRaisesRegex(ValueError, message):
                        operation(bad)
        with self.assertRaisesRegex(ValueError, "feature_set"):
            make_preprocessor("typo")

    def test_missing_and_invalid_inputs_are_rejected_without_imputation(self):
        prep = make_preprocessor().fit(self.X)
        cases = [(column, None) for column in ALL_17] + [
            ("Administrative", -1),
            ("Administrative", 1.5),
            ("Administrative", True),
            ("Administrative", "2"),
            ("ProductRelated_Duration", -0.1),
            ("PageValues", np.inf),
            ("PageValues", -np.inf),
            ("BounceRates", 20),
            ("ExitRates", -0.01),
            ("SpecialDay", 1.1),
            ("Month", " "),
            ("VisitorType", 12),
            ("Browser", "2"),
            ("Browser", 1.5),
            ("Browser", False),
            ("Region", 0),
            ("TrafficType", -1),
            ("Weekend", "False-ish"),
            ("Weekend", "0"),
            ("Weekend", 2),
        ]
        for column, value in cases:
            bad = self.X.astype({column: object})
            bad.loc[0, column] = value
            with self.subTest(column=column, value=value):
                for operation in [prep.transform, make_preprocessor().fit]:
                    with self.assertRaisesRegex(ValueError, column):
                        operation(bad)

    def test_boolean_representations_share_identical_meaning(self):
        raw = self.X.copy()
        raw["Weekend"] = [False, True, 0, 1, "false", "true", " FALSE ", "True"]
        raw["Revenue"] = raw["Weekend"]
        X, y = split_features_target(raw)
        self.assertEqual(X["Weekend"].tolist(), [0, 1] * 4)
        self.assertEqual(y.tolist(), [0, 1] * 4)
        prep = make_preprocessor().fit(self.X)
        out = prep.transform(raw.drop(columns="Revenue"))
        position = feature_names(prep).index("boolean__Weekend")
        np.testing.assert_array_equal(out[:, position], [0, 1] * 4)

    def test_scaling_only_affects_numeric_measurements(self):
        linear = make_preprocessor().fit(self.X)
        tree = make_preprocessor(scale_numeric=False).fit(self.X)
        names = feature_names(linear)
        linear_out, tree_out = linear.transform(self.X), tree.transform(self.X)
        for col in NUMERIC:
            index = names.index(f"numeric__{col}")
            self.assertAlmostEqual(linear_out[:, index].mean(), 0)
            np.testing.assert_array_equal(tree_out[:, index], self.X[col])
        np.testing.assert_array_equal(
            linear_out[:, len(NUMERIC) :], tree_out[:, len(NUMERIC) :]
        )
        for col in NOMINAL_CATEGORICAL:
            indices = [
                i
                for i, name in enumerate(names)
                if name.startswith(f"categorical__{col}_")
            ]
            self.assertGreater(len(indices), 1)
            np.testing.assert_array_equal(linear_out[:, indices].sum(axis=1), 1)
        # Rates above the observed CSV maximum (0.2) and large valid durations
        # stay intact; validation does not silently clip to training support.
        raw = self.X.iloc[:1].copy()
        raw["BounceRates"] = 0.9
        raw["ProductRelated_Duration"] = 1e8
        transformed = tree.transform(raw)
        self.assertEqual(transformed[0, names.index("numeric__BounceRates")], 0.9)
        self.assertEqual(
            transformed[0, names.index("numeric__ProductRelated_Duration")], 1e8
        )

    def test_unseen_levels_are_all_zero_only_in_their_own_category_block(self):
        prep = make_preprocessor().fit(self.X)
        before = prep.transform(self.X.iloc[:1])
        raw = self.X.iloc[:1].copy()
        for column in NOMINAL_CATEGORICAL:
            raw[column] = (
                999
                if column in ["Browser", "OperatingSystems", "Region", "TrafficType"]
                else "Unseen"
            )
        after = prep.transform(raw)
        names = feature_names(prep)
        indices = [
            i for i, name in enumerate(names) if name.startswith("categorical__")
        ]
        np.testing.assert_array_equal(after[:, indices], 0)
        remaining = [i for i in range(len(names)) if i not in indices]
        np.testing.assert_array_equal(after[:, remaining], before[:, remaining])
        self.assertEqual(after.shape, before.shape)

    def test_factories_are_unfitted_and_cloneable(self):
        estimator = LogisticRegression().fit(self.X[NUMERIC], self.y)
        pipeline = clone(make_model_pipeline(estimator))
        assert isinstance(pipeline, Pipeline)
        self.assertFalse(hasattr(pipeline.named_steps["model"], "coef_"))
        self.assertFalse(
            hasattr(pipeline.named_steps["preprocess"]["schema"], "output_columns_")
        )
        self.assertIsNot(pipeline.named_steps["model"], estimator)
        self.assertEqual(pipeline.get_params()["model__C"], 1.0)
        validator = clone(SessionInputValidator())
        assert isinstance(validator, SessionInputValidator)
        self.assertEqual(validator.feature_set, "all_17")

    def test_model_variants_reload_in_a_fresh_python_process(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in FEATURE_SETS:
                for family, estimator, scale in [
                    ("logistic", LogisticRegression(max_iter=2000), True),
                    (
                        "forest",
                        RandomForestClassifier(n_estimators=3, random_state=42),
                        False,
                    ),
                ]:
                    with self.subTest(name=name, family=family):
                        X, y = split_features_target(self.data, name)
                        pipeline = make_model_pipeline(
                            estimator, name, scale_numeric=scale
                        ).fit(X, y)
                        raw = self.data.drop(columns="Revenue").iloc[:2].copy()
                        raw.loc[1, "Month"] = "Jan"
                        raw.loc[1, "Browser"] = 999
                        path = directory / f"{name}_{family}.joblib"
                        joblib.dump(pipeline, path)
                        np.testing.assert_array_equal(
                            joblib.load(path).predict(raw), pipeline.predict(raw)
                        )
                        expected = pipeline.predict_proba(raw)
                        input_path = directory / "input.json"
                        raw.to_json(input_path, orient="records")
                        output = directory / "probabilities.npy"
                        subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                "import sys, joblib, numpy as np, pandas as pd; "
                                "pipeline = joblib.load(sys.argv[1]); "
                                "raw = pd.read_json(sys.argv[2]); "
                                "np.save(sys.argv[3], pipeline.predict_proba(raw))",
                                str(path),
                                str(input_path),
                                str(output),
                            ],
                            cwd=ROOT,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        np.testing.assert_allclose(
                            np.load(output), expected, rtol=0, atol=0
                        )


class FoldIsolationTests(unittest.TestCase):
    def test_frozen_cv_learns_scaler_and_categories_only_from_each_fit_fold(self):
        data, assignments, _ = load_frozen_splits()
        rows = assignments.loc[
            assignments["partition"].eq("train"), "row_index"
        ].sort_values()
        X, y = split_features_target(data.iloc[rows.to_numpy()])
        folds = list(iter_training_cv(assignments))
        X.loc[X.index[folds[0][1]], "VisitorType"] = "Only_in_fold_zero_holdout"
        result = cross_validate(
            make_model_pipeline(DummyClassifier(strategy="prior")),
            X,
            y,
            cv=folds,
            return_estimator=True,
            # sklearn accepts "raise" at runtime; its inferred type uses the float default.
            error_score="raise",  # pyright: ignore[reportArgumentType]
        )
        for fold, ((fit, _), pipeline) in enumerate(zip(folds, result["estimator"])):
            columns = pipeline.named_steps["preprocess"]["columns"]
            scaler = columns.named_transformers_["numeric"]
            self.assertEqual(scaler.n_samples_seen_, len(fit))
            np.testing.assert_allclose(
                scaler.mean_, X.iloc[fit][NUMERIC].mean().to_numpy()
            )
            np.testing.assert_allclose(
                scaler.var_, X.iloc[fit][NUMERIC].var(ddof=0).to_numpy()
            )
            encoder = columns.named_transformers_["categorical"]
            for col, categories in zip(NOMINAL_CATEGORICAL, encoder.categories_):
                self.assertEqual(set(categories), set(X.iloc[fit][col]))
            self.assertEqual(
                "Only_in_fold_zero_holdout" in encoder.categories_[-1], fold != 0
            )
            means = scaler.mean_.copy()
            categories = [levels.copy() for levels in encoder.categories_]
            # Inference on future extremes/unseen levels cannot refit any state.
            future = X.iloc[:1].copy()
            future["ProductRelated_Duration"] = 1e12
            future["Month"] = "New_month"
            pipeline.predict(future)
            np.testing.assert_array_equal(scaler.mean_, means)
            for before, after in zip(categories, encoder.categories_):
                np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
