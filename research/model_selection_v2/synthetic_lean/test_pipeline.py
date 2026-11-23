"""Small fixtures for paired truth scoring and the visited gate path."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.model_selection_v2.confirmation_v1.schema import Schema
from research.model_selection_v2.confirmation_v1.selection import select_arms
from research.model_selection_v2.safe2026 import rank_candidates

from . import pipeline as subject


def _gate_inputs():
    ids = pd.Index(np.arange(1, 121), name="ID")
    female = ids.to_numpy() <= 60
    meta = pd.DataFrame({"ID": ids, "SEX": np.where(female, 2, 1),
                         "default": (ids % 4 == 0).astype(int), "duplicate_group_id": ids}, index=ids)
    frame = pd.DataFrame({"x": ids}, index=ids)
    truth = pd.Series(0.2, index=ids)
    rows = pd.DataFrame({
        "SAFE_score": (.8, .7, .6, .5),
        "Score2": (.5, .7, .8, .9),
        "clean_brier": (.1, .2, .3, .4),
        "S_bar": (.6, .61, .55, .57),
        "order_mild_score": (.8, .7, .6, .5),
        "accuracy_full_score": (.8, .7, .6, .5),
    }, index=subject.CANDIDATES)
    evidence = {}
    for candidate, probability in zip(subject.CANDIDATES, (.2, .3, .225, .95)):
        clean = np.full(len(ids), .2)
        positive_nodes = np.tile(np.where(female, probability, .2), (5, 1))
        evidence[candidate] = {"clean": clean, "swap_mild": np.vstack((clean, positive_nodes)),
                               "ID": ids.to_numpy()}
    return rows, evidence, frame, meta, truth


class SyntheticPipelineTests(unittest.TestCase):
    def test_truth_by_id_and_judge_averaging(self):
        frame = pd.DataFrame({"x": [.2, .8]}, index=pd.Index([17, 11], name="ID"))
        meta = pd.DataFrame({"ID": frame.index, "SEX": [2, 1], "default": [1, 0]}, index=frame.index)
        truth = pd.Series([.8, .2], index=[11, 17])
        families = {"A": ("a",), "B": ("b",), "C": ("c",)}
        model_set = {"candidate": object()}
        calls = []

        def fake_fault(original, fitting, schema, families, *, dataset_id, family, repetition, level, reference):
            calls.append((dataset_id, level, family, repetition))
            changed = original.copy()
            if level == "J1":
                changed.loc[17, "x"] = .4
            else:
                changed.loc[11, "x"] = .5
            return {"10": original, "01": original, "11": changed}, {}, {}

        def predict(model, data):
            return data.x.to_numpy()

        with patch.object(subject, "references"), patch.object(subject, "paired_transforms", fake_fault), \
                patch.object(subject, "positive_probability", predict):
            result = subject._judge_truth(frame, frame, meta, model_set, None, families, "fixture:1", ("J1", "J3"), truth)
        self.assertEqual(result["clean"]["candidate"], {"all": 0.0, "female": 0.0, "male": 0.0})
        self.assertAlmostEqual(result["J1"]["candidate"]["female"], (.4 - .2)**2 / 3)
        self.assertAlmostEqual(result["J1"]["candidate"]["all"], (.4 - .2)**2 / 6)
        self.assertAlmostEqual(result["J3"]["candidate"]["male"], (.5 - .8)**2 / 3)
        self.assertEqual(calls, [("fixture:1", level, family, realization)
                                 for level in ("J1", "J3") for family in families for realization in range(1, 31)])
        with self.assertRaisesRegex(ValueError, "aligned borrower IDs"):
            subject.align_truth(frame, meta.iloc[::-1], truth)
        with self.assertRaisesRegex(ValueError, "Missing or invalid original probabilities"):
            subject.align_truth(frame, meta, truth.iloc[:1])

    def test_frozen_choice_and_harmful_first_veto_with_harmful_final_choice(self):
        rows, evidence, frame, meta, truth = _gate_inputs()
        choices, ranked, checks = subject._choices(rows, evidence, meta, repeat=7)
        with TemporaryDirectory() as path:
            folder = Path(path)
            frozen = select_arms(folder, rows, evidence, meta, repeat=7)
            audit = json.loads((folder / "fairness_check.json").read_text())
        for arm, candidate in frozen.items():
            if arm in choices:
                self.assertEqual(choices[arm], candidate)
        self.assertEqual(list(checks), audit["visited_in_choice_rule"])
        self.assertEqual(checks, {candidate: audit["candidate_checks"][candidate] for candidate in checks})
        trace = subject._group_gate_trace(choices, ranked, checks, evidence, frame, meta, truth)
        self.assertEqual(trace["visited"], ["gradient_boosting", "random_forest"])
        self.assertEqual(choices["selector2"], "random_forest")
        self.assertTrue(trace["first_attempt"]["blocked"])
        self.assertEqual(trace["first_attempt"]["triggering_groups"], ["female"])
        self.assertEqual(trace["first_attempt"]["harmful_groups"], ["female"])
        self.assertFalse(trace["later_attempts"][0]["blocked"])
        self.assertTrue(trace["final_harmful"])
        self.assertFalse(trace["prevented_harm"])
        self.assertAlmostEqual(trace["final_truth_delta"]["female"], .000625)
        self.assertEqual(choices[subject.DIRECT_ARM], "decision_tree")
        self.assertNotIn(choices[subject.DIRECT_ARM], trace["visited"])
        self.assertAlmostEqual(trace["direct_truth_delta"]["female"], .01)
        self.assertEqual(trace["direct_harmful_groups"], ["female"])
        self.assertEqual(trace["direct_material_harm_groups"], ["female"])

    def test_started_at_safe_has_no_attempt_and_exact_equal_choice_contrasts(self):
        rows, evidence, frame, meta, truth = _gate_inputs()
        rows["Score2"] = rows.SAFE_score
        rows["S_bar"] = rows.SAFE_score
        choices, ranked, checks = subject._choices(rows, evidence, meta, repeat=1)
        trace = subject._group_gate_trace(choices, ranked, checks, evidence, frame, meta, truth)
        self.assertTrue(trace["started_at_safe"])
        self.assertEqual(trace["visited"], ["logistic"])
        self.assertIsNone(trace["first_attempt"])
        self.assertEqual(trace["later_attempts"], [])
        self.assertEqual(trace["final_truth_delta"], {"female": 0.0, "male": 0.0})
        self.assertFalse(trace["prevented_harm"])
        losses = {"J1": {"logistic": {"all": .1123, "female": .0834, "male": .1412}}}
        for setting in ("R", "F"):
            for values in subject._contrasts(setting, choices, losses).values():
                self.assertEqual(values["J1"], {"all": 0.0, "female": 0.0, "male": 0.0})

    def test_optimized_judge_predictions_match_full_frame_for_frozen_candidates(self):
        rng = np.random.default_rng(825)
        fitting = pd.DataFrame({"x": rng.normal(size=150), "o": rng.integers(1, 4, 150),
                                "u": rng.integers(0, 2, 150)})
        frame = fitting.iloc[::3].copy()
        schema = Schema(("x", "o", "u"), ("x",), {"o": (1, 2, 3)}, ("u",))
        families = {"A": ("x",), "B": ("o",), "C": ("u",)}
        models = subject._fit_candidates(fitting, (fitting.x > .2).astype(int), schema)
        reference = subject.references(fitting, schema)
        for level in ("J1", "J3"):
            for family in families:
                frames, _, _ = subject.paired_transforms(frame, fitting, schema, families, dataset_id="fixture:4",
                                                         family=family, repetition=8, level=level, reference=reference)
                for model in models.values():
                    clean = subject.positive_probability(model, frame)
                    for changed in frames.values():
                        _, audit = subject.predict_changed_rows(model, frame, changed, clean,
                                                                subject.positive_probability, audit_full=True)
                        self.assertGreater(audit["clean_rows_reused"], 0)

    def test_identical_predictors_group_across_labels_and_sex(self):
        from types import SimpleNamespace

        ids = np.array([19, 13, 21])
        frame = pd.DataFrame({"x": [0., 1., 0.]}, index=ids)
        population = SimpleNamespace(X=frame, ids=ids, y=np.array([1, 1, 0]), sex=np.array([1, 1, 2]))
        metadata = subject._metadata(population)
        self.assertEqual(metadata.duplicate_group_id.tolist(), [19, 13, 19])
        self.assertEqual(metadata.ID.tolist(), [19, 13, 21])


if __name__ == "__main__":
    unittest.main()
