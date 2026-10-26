import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

from research.model_selection_v2 import pilot as p


def fixture():
    X = pd.DataFrame({name: np.arange(12, dtype=float) + i for i, name in enumerate(p.FEATURE_ORDER())})
    for name in p.CATEGORICAL:
        X[name] = np.tile([1, 2], 6)
    X["LIMIT_BAL"] = np.arange(12, dtype=float)
    y = pd.Series(np.tile([0, 1, 1, 0], 3))
    metadata = pd.DataFrame({"ID": np.arange(101, 113), "SEX": np.tile([1, 2], 6),
                             "default": y, "duplicate_group_id": np.arange(101, 113), "split": "train"})
    assignment = metadata.drop(columns="split").copy()
    assignment["original_partition"] = "train"
    assignment["development_role"] = ["fitting"] * 8 + ["selection"] * 4
    assignment["fold"] = [2] * 8 + [1] * 4
    assignment["training_position"] = np.arange(12)
    return X, y, metadata, assignment


class PilotTests(unittest.TestCase):
    def test_saved_development_membership(self):
        X, y, metadata, assignment = fixture()
        positions = p.validate_assignments(X, y, metadata, assignment)
        np.testing.assert_array_equal(positions["fitting"], np.arange(8))
        np.testing.assert_array_equal(positions["selection"], np.arange(8, 12))
        for column in ("ID", "SEX", "default", "duplicate_group_id", "training_position"):
            broken = assignment.copy()
            broken.loc[0, column] = -100
            with self.subTest(column=column), self.assertRaises(ValueError):
                p.validate_assignments(X, y, metadata, broken)

    def test_original_group_leakage_rejected(self):
        X, y, metadata, assignment = fixture()
        X.loc[8] = X.loc[0]
        metadata.loc[8, "duplicate_group_id"] = metadata.loc[0, "ID"]
        assignment.loc[8, "duplicate_group_id"] = metadata.loc[0, "ID"]
        with self.assertRaisesRegex(ValueError, "crosses"):
            p.validate_assignments(X, y, metadata, assignment)

    def test_metadata_and_order_rejected(self):
        X, y, metadata, assignment = fixture()
        for broken in (X.assign(SEX=metadata.SEX), X.assign(ID=metadata.ID),
                       X.assign(default=y), X.assign(duplicate_group_id=metadata.duplicate_group_id),
                       X.loc[:, list(reversed(X.columns))]):
            with self.assertRaises(ValueError):
                p.validate_assignments(broken, y, metadata, assignment)

    def test_missing_and_nonfinite_rejected(self):
        X, *_ = fixture()
        for invalid in (np.nan, np.inf, -np.inf):
            broken = X.copy()
            broken.iloc[0, 0] = invalid
            with self.assertRaises(ValueError):
                p.finite_inputs(broken)

    def test_cleaning_is_fixed_and_row_local(self):
        X, y, metadata, _ = fixture()
        raw = X.assign(ID=metadata.ID, SEX=metadata.SEX)
        raw["default payment next month"] = y
        raw["EDUCATION"] = np.tile([0, 5, 6, 2], 3)
        raw["MARRIAGE"] = np.tile([0, 1, 2], 4)
        cleaned = p.clean_data(raw)
        expected = pd.concat([p.clean_data(raw.iloc[:8]), p.clean_data(raw.iloc[8:])])
        pd.testing.assert_frame_equal(cleaned, expected)
        pd.testing.assert_frame_equal(cleaned, p.clean_data(raw.iloc[::-1]).iloc[::-1])
        self.assertEqual(cleaned.EDUCATION.tolist(), [4, 4, 4, 2] * 3)
        self.assertEqual(cleaned.MARRIAGE.tolist(), [3, 1, 2] * 4)
        np.testing.assert_array_equal(cleaned.default, y)

    def fit_fixture(self):
        X, y, metadata, _ = fixture()
        selection = X.iloc[8:].copy()
        selection.loc[:, "LIMIT_BAL"] = 1000000
        selection.loc[:, "EDUCATION"] = 99
        model = p.build_candidate("decision_tree").fit(X.iloc[:8], y.iloc[:8])
        return model, X.iloc[:8], selection, metadata.iloc[8:]

    def test_fitting_only_preprocessing_mapping_and_unseen(self):
        model, fitting, selection, _ = self.fit_fixture()
        audit = p.preprocessing_audit(model, fitting, selection)
        self.assertEqual(audit["unseen_selection_categories"]["EDUCATION"]["records"], 4)
        for name in p.CATEGORICAL[1:]:
            self.assertEqual(audit["unseen_selection_categories"][name]["records"], 0)
        mapping = audit["original_predictor_to_transformed_columns"]
        indices = [item["index"] for entries in mapping.values() for item in entries]
        self.assertEqual(sorted(indices), list(range(len(audit["transformed_columns"]))))
        self.assertEqual({v["category"] for v in mapping["EDUCATION"]}, {1, 2})
        # Deliberately contaminated fit must fail this audit.
        contaminated = p.build_candidate("decision_tree").fit(pd.concat([fitting, selection]),
                                                              np.tile([0, 1], 6))
        with self.assertRaises(ValueError):
            p.preprocessing_audit(contaminated, fitting, selection)

    def test_positive_column_is_looked_up(self):
        class Estimator:
            classes_ = np.array([1, 0])
        class Model:
            classes_ = np.array([1, 0])
            named_steps = {"classifier": Estimator()}
            def predict_proba(self, X):
                return np.tile([.7, .3], (len(X), 1))
        np.testing.assert_array_equal(p.positive_probability(Model(), [0, 0]), [.7, .7])
        for classes in ([0], [1], [0, 2]):
            model = Model()
            model.classes_ = np.array(classes)
            with self.assertRaises(ValueError):
                p.positive_probability(model, [0])

    def test_probability_validation(self):
        class Estimator:
            classes_ = np.array([0, 1])
        class Model:
            classes_ = np.array([0, 1])
            named_steps = {"classifier": Estimator()}
            def predict_proba(self, X):
                return np.array([[0., self.value]])
        for invalid in (np.nan, np.inf, -0.1, 1.1):
            model = Model()
            model.value = invalid
            with self.assertRaises(ValueError):
                p.positive_probability(model, [0])

    def test_strict_threshold_tie(self):
        np.testing.assert_array_equal(p.decisions([.0, .4999, .5, .50001, 1.]), [0, 0, 0, 1, 1])

    def test_reload_prediction_alignment_and_metrics(self):
        model, _, selection, metadata = self.fit_fixture()
        probability = p.positive_probability(model, selection)
        table = p.prediction_table("decision_tree", probability, metadata)
        expected = p.metrics(metadata.default, probability)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            joblib.dump(model, path / "model.joblib")
            p.write_csv(path / "predictions.csv", table)
            restored = joblib.load(path / "model.joblib")
            np.testing.assert_array_equal(p.positive_probability(restored, selection), probability)
            saved = pd.read_csv(path / "predictions.csv", float_precision="round_trip")
            p.verify_predictions(restored, selection, metadata, saved, expected)
            with self.assertRaises(AssertionError):
                p.verify_predictions(restored, selection, metadata, saved.iloc[::-1], expected)
            broken = dict(expected, brier=100.)
            with self.assertRaises(AssertionError):
                p.verify_predictions(restored, selection, metadata, saved, broken)

    def test_metrics_against_independent_sklearn_functions(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        probability = np.array([.1, .5, .7, .4, .8, .9])
        predicted = p.decisions(probability)
        measured = p.metrics(y, probability)
        for name, function in (("accuracy", accuracy_score), ("balanced_accuracy", balanced_accuracy_score),
                               ("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
            self.assertAlmostEqual(measured[name], function(y, predicted))
        np.testing.assert_array_equal([measured[k] for k in ("tn", "fp", "fn", "tp")],
                                      confusion_matrix(y, predicted).ravel())
        self.assertAlmostEqual(measured["brier"], np.mean((probability - y) ** 2))
        self.assertAlmostEqual(measured["log_loss"], -np.mean(y*np.log(probability)+(1-y)*np.log(1-probability)))
        self.assertAlmostEqual(measured["roc_auc"], 2/3)
        self.assertAlmostEqual(measured["average_precision"], (1 + 2/3 + 3/5)/3)

    def test_undefined_metrics_are_explicit(self):
        values = p.metrics([0, 0], [.1, .2])
        for name in ("precision", "recall", "f1", "roc_auc", "average_precision", "balanced_accuracy"):
            self.assertIsNone(values[name])

    def test_existing_destinations_and_files_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = p.create_run(Path(folder) / "run")
            with self.assertRaises(FileExistsError):
                p.create_run(destination)
            path = destination / "value.json"
            p.write_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                p.write_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_incomplete_and_tampered_runs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = p.create_run(Path(folder) / "run")
            with self.assertRaises(FileNotFoundError):
                p.verify_run(path)
            p.write_json(path / "VERIFIED.json", {"status": "verified", "artifact_hashes": {
                         "RUN_STARTED.json": p.sha256(path / "RUN_STARTED.json")}})
            p.verify_run(path)
            (path / "partial.txt").write_text("unexpected output")
            with self.assertRaises(ValueError):
                p.verify_run(path)

    def test_interruption_is_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(p, "RUNS", Path(folder)):
            with patch.object(p, "execute", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                p.run("interrupted")
            self.assertEqual(json.loads((Path(folder) / "interrupted/FAILURE.json").read_text())["status"], "incomplete")
            self.assertFalse((Path(folder) / "interrupted/VERIFIED.json").exists())


if __name__ == "__main__":
    unittest.main()
