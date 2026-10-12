"""Method and artifact checks for frozen-model SHAP and authentic SAFE RGE."""
import inspect
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from safeaipackage import check_explainability, core
from safeaipackage.util.utils import manipulate_testdata

from src.baseline import ROOT, DATA_PATH, FEATURES, CATEGORICAL, NUMERIC, load_split, sha256
from src.explainability_evaluation import (TABLE_DIR, MODELS, GROUPS, background_positions,
    safe_frames, evaluate_rge, explain_shap, select_local_cases, protected_hashes)


class ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test, _, cls.y, cls.meta_train, cls.meta_test = load_split(pd.read_csv(DATA_PATH))
        cls.models = {k: joblib.load(ROOT / f"models/baseline_{k}.joblib") for k in MODELS}

    def read(self, name):
        return pd.read_csv(TABLE_DIR / f"{name}.csv", float_precision="round_trip")

    def test_training_background_and_model_specific_additivity(self):
        pos = background_positions(self.meta_train)
        np.testing.assert_array_equal(pos, background_positions(self.meta_train))
        self.assertEqual(len(pos), 100)
        self.assertFalse(set(self.meta_train.iloc[pos].ID) & set(self.meta_test.ID))
        for kind, model in self.models.items():
            with patch.object(model, "fit", side_effect=AssertionError("No refitting")):
                values, raw, base, mapping, error = explain_shap(model, kind, self.test.iloc[:24], self.train.iloc[pos])
            self.assertEqual(values.shape, (24, 22))
            self.assertEqual(raw.shape[1], len(mapping))
            self.assertLess(error, 2e-6)
            for j, feature in enumerate(FEATURES):
                np.testing.assert_array_equal(values[:, j], raw[:, mapping.feature.eq(feature)].sum(axis=1))
            np.testing.assert_allclose(expit(base + values.sum(axis=1)), model.predict_proba(self.test.iloc[:24])[:, 1], atol=5e-7, rtol=0)

    def test_authentic_rge_calls_categories_references_and_reproduction(self):
        self.assertIn("site-packages", inspect.getfile(check_explainability.compute_rge_values))
        self.assertEqual(sorted(f for group in GROUPS.values() for f in group), sorted(FEATURES))
        for kind, model in self.models.items():
            train, test = safe_frames(self.train, self.test.iloc[:80], model)
            for f in CATEGORICAL:
                self.assertIsInstance(train[f].dtype, pd.CategoricalDtype)
                replaced = manipulate_testdata(train, test, model, f)
                self.assertTrue(replaced[f].eq(train[f].mode().iloc[0]).all())
            for f in NUMERIC:
                replaced = manipulate_testdata(train, test, model, f)
                np.testing.assert_array_equal(replaced[f], np.repeat(train[f].mean(), len(test)))
            with patch.object(model, "fit", side_effect=AssertionError("No fitting")):
                with patch.object(check_explainability, "compute_rge_values", wraps=check_explainability.compute_rge_values) as actual:
                    first = evaluate_rge(model, self.train, self.test.iloc[:80])
                    self.assertEqual(actual.call_count, 1 + len(GROUPS))
                    self.assertEqual(actual.call_args_list[0].args[4], FEATURES)
                second = evaluate_rge(model, self.train, self.test.iloc[:80])
            for one, two in zip(first, second):
                pd.testing.assert_frame_equal(one, two, check_exact=True)
            # Check official wrapper output against the actual core, not a substitute metric.
            replacement = manipulate_testdata(train, test, model, "PAY_0")
            expected = 1 - core.rga(model.predict_proba(test)[:, 1].tolist(), model.predict_proba(replacement)[:, 1].tolist())
            observed = first[0].set_index("feature").loc["PAY_0", "RGE"]
            self.assertEqual(expected, observed)

    def test_local_rule_is_lowest_id_and_shared_union(self):
        predictions = self.read("audit_predictions")
        selected = select_local_cases(predictions)
        self.assertEqual(set(selected.selection_outcome), {"TP", "TN", "FP", "FN"})
        outcomes = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}
        for row in selected.itertuples():
            y, p = outcomes[row.selection_outcome]
            eligible = predictions.query("model == @row.selection_model and actual == @y and predicted == @p")
            self.assertEqual(row.ID, eligible.ID.min())
        local = self.read("local_shap_contributions")
        for kind in MODELS:
            self.assertEqual(set(local.query("model == @kind").ID), set(selected.ID))

    def test_complete_saved_shap_group_counts_and_finite_outputs(self):
        predictions = self.read("audit_predictions")
        groups = self.read("shap_importance_by_sex")
        reference = self.read("background_metadata")
        np.testing.assert_array_equal(reference.ID, self.meta_train.iloc[background_positions(self.meta_train)].ID)
        self.assertFalse(set(reference.ID) & set(self.meta_test.ID))
        fairness = pd.read_csv(ROOT / "outputs/tables/fairness_evaluation/group_summary.csv")
        for kind, model in self.models.items():
            values = self.read(f"shap_values_{kind}")
            self.assertEqual(values[FEATURES].shape, (5996, 22))
            np.testing.assert_array_equal(values.ID, self.meta_test.ID)
            np.testing.assert_array_equal(values.SEX, self.meta_test.SEX)
            p = predictions.query("model == @kind")
            np.testing.assert_array_equal(p.probability_default, model.predict_proba(self.test)[:, 1])
            np.testing.assert_array_equal(p.predicted, model.predict(self.test))
            np.testing.assert_allclose(values.base_value + values[FEATURES].sum(axis=1), p.margin, atol=2e-6, rtol=0)
            for row in fairness.itertuples():
                part = groups.query("model == @kind and SEX == @row.SEX").set_index("feature").loc[FEATURES]
                self.assertTrue(part.records.eq(row.records).all())
                # Different NumPy/pandas reduction orders accumulate rounding over thousands of rows.
                np.testing.assert_allclose(part.mean_abs_shap, values.loc[values.SEX == row.SEX, FEATURES].abs().mean(), atol=1e-12, rtol=0)
        for path in TABLE_DIR.glob("*.csv"):
            frame = pd.read_csv(path, float_precision="round_trip")
            self.assertFalse(frame.isna().any().any(), str(path))
            self.assertTrue(np.isfinite(frame.select_dtypes(include="number").to_numpy()).all(), str(path))

    def test_manifest_and_protected_artifacts(self):
        manifest = json.loads((TABLE_DIR / "run_manifest.json").read_text())
        self.assertEqual(manifest["protected_artifact_hashes"], protected_hashes())
        for category in ("source_hashes", "output_hashes"):
            for path, expected in manifest[category].items():
                self.assertEqual(sha256(ROOT / path), expected, path)
        self.assertEqual(manifest["threshold"], .5)
        self.assertEqual(manifest["holdout_rows"], 5996)
        self.assertEqual(set(manifest["predictors"]), set(FEATURES))


if __name__ == "__main__":
    unittest.main()
