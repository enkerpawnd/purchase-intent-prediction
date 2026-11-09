"""Check the analysis boundary and quantities used to interpret the figures."""

import unittest
from unittest.mock import patch

import pandas as pd

from src.explore_training_data import build_tables, load_training_data
from src.feature_sets import ALL_17, FEATURE_SETS, NUMERIC
from src.split_data import load_frozen_splits


class TrainingExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.assignments, cls.metadata = load_frozen_splits()

    def test_only_training_rows_reach_analysis(self):
        # Poison all held-out measurements/labels; integrity validation itself
        # belongs to split_data, so mock only that loader's validated return.
        poisoned = self.data.copy()
        held_out = self.assignments.loc[
            self.assignments["partition"].ne("train"), "row_index"
        ]
        poisoned.loc[held_out, NUMERIC] = -999
        poisoned.loc[held_out, "Revenue"] = True
        with patch(
            "src.explore_training_data.load_frozen_splits",
            return_value=(poisoned, self.assignments.iloc[::-1], self.metadata),
        ):
            train, provenance = load_training_data()
        rows = self.assignments.loc[
            self.assignments["partition"].eq("train"), "row_index"
        ].sort_values()
        pd.testing.assert_frame_equal(train, self.data.iloc[rows.to_numpy()])
        self.assertEqual((provenance["rows"], provenance["purchases"]), (8632, 1336))
        self.assertFalse(train.isna().any().any())
        self.assertTrue((train[NUMERIC].to_numpy() >= 0).all())

    def test_feature_sets_are_exact_raw_column_subsets(self):
        self.assertEqual(ALL_17, self.metadata["source"]["predictor_columns"])
        for name, excluded, size in [
            ("all_17", set(), 17),
            ("without_pagevalues_16", {"PageValues"}, 16),
            ("without_analytics_14", {"PageValues", "BounceRates", "ExitRates"}, 14),
        ]:
            columns = FEATURE_SETS[name]
            self.assertEqual(columns, [col for col in ALL_17 if col not in excluded])
            self.assertEqual(len(set(columns)), size)
            self.assertNotIn("Revenue", columns)

    def test_training_summaries_match_report_and_keep_absent_months_missing(self):
        train, _ = load_training_data()
        tables = build_tables(train)
        for name in ["class_balance", "month", "visitor_type", "pagevalues_zero"]:
            self.assertEqual(tables[name]["sessions"].sum(), 8632)
            self.assertEqual(tables[name]["purchases"].sum(), 1336)
        self.assertTrue(tables["month"].loc[["Jan", "Apr"]].isna().all().all())
        self.assertEqual(tables["pagevalues_zero"].loc["Zero", "purchases"], 264)
        self.assertEqual(tables["pagevalues_zero"].loc["Positive", "sessions"], 1929)
        self.assertEqual(
            tables["numeric_summary"].loc["Informational_Duration", "zeros"], 6952
        )

    def test_synthetic_flags_preserve_zero_and_exclude_category_codes_from_correlations(
        self,
    ):
        sample = self.data.iloc[:4].copy()
        sample["Administrative"] = [0, 1, 2, 0]
        sample["Administrative_Duration"] = [0.0, 0.0, 5.0, 3.0]
        sample["PageValues"] = [0.0, 2.0, 0.0, 4.0]
        sample["Revenue"] = [False, True, True, False]
        tables = build_tables(sample)
        flags = (
            tables["count_duration_flags"].set_index("page_type").loc["Administrative"]
        )
        self.assertEqual(flags["positive_count_zero_duration"], 1)
        self.assertEqual(flags["zero_count_positive_duration"], 1)
        self.assertEqual(
            tables["pagevalues_zero"]["purchase_rate"].tolist(), [0.5, 0.5]
        )
        self.assertEqual(list(tables["spearman"].columns), [*NUMERIC, "Revenue"])
        self.assertNotIn("Browser", tables["spearman"])


if __name__ == "__main__":
    unittest.main()
