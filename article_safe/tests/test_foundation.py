"""Selection boundaries and training-only data separation on synthetic fixtures."""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from article_safe.data import build_development_partition
from article_safe.protocol import build_protocol, canonical_digest
from article_safe.run import run_directory
from article_safe.selection import forward_step, choose_forward_prefix, choose_backward_removal
from src.baseline import FEATURES


class SelectionTests(unittest.TestCase):
    def test_significant_deterioration_cannot_be_accepted(self):
        self.assertFalse(forward_step(-.01, .001)["accept"])
        self.assertFalse(forward_step(0, .001)["accept"])

    def test_alpha_boundary_and_invalid_tests(self):
        self.assertFalse(forward_step(.01, .05)["accept"])
        self.assertTrue(forward_step(.01, .049)["accept"])
        self.assertIsNone(forward_step(.01, None, "degenerate_variance")["accept"])

    def test_first_stop_cannot_be_undone_by_later_recovery(self):
        result = choose_forward_prefix([
            {"k": 2, "gain": .01, "p_value": .01},
            {"k": 3, "gain": .001, "p_value": .10},
            {"k": 4, "gain": .10, "p_value": .0001},
        ])
        self.assertEqual(result["selected_k"], 2)
        self.assertEqual(result["stopped_at_k"], 3)

    def test_invalid_forward_path_does_not_return_an_accepted_model(self):
        result = choose_forward_prefix([{"k": 2, "gain": .01, "p_value": None, "status": "degenerate_variance"}])
        self.assertIsNone(result["selected_k"])
        self.assertEqual(result["status"], "not_estimable")
        self.assertEqual(result["last_valid_k"], 1)

    def test_backward_largest_p_and_deterministic_tie_order(self):
        candidates = [{"feature": "AGE", "p_value": .9}, {"feature": "EDUCATION", "p_value": .9}]
        result = choose_backward_removal(candidates, ["EDUCATION", "AGE"])
        self.assertEqual(result["feature"], "EDUCATION")
        self.assertEqual(result["action"], "remove")

    def test_backward_alpha_boundary_and_retaining_significant_terms(self):
        self.assertEqual(choose_backward_removal([{"feature": "AGE", "p_value": .05}], ["AGE"])["action"], "remove")
        self.assertEqual(choose_backward_removal([{"feature": "AGE", "p_value": .049}], ["AGE"])["action"], "stop")
        self.assertEqual(choose_backward_removal([], [])["reason"], "intercept_only")

    def test_invalid_removal_is_not_silently_ignored(self):
        result = choose_backward_removal([
            {"feature": "AGE", "p_value": .9},
            {"feature": "EDUCATION", "p_value": None, "status": "not_estimable"},
        ], ["AGE", "EDUCATION"])
        self.assertEqual(result["status"], "not_estimable")
        self.assertIsNone(result["feature"])
        with self.assertRaises(ValueError):
            choose_backward_removal([{"feature": "AGE", "p_value": .9}], ["AGE", "EDUCATION"])


class DevelopmentDataTests(unittest.TestCase):
    def setUp(self):
        n = 80
        self.features = list(FEATURES)
        self.X = pd.DataFrame(np.zeros((n, 22)), columns=self.features)
        self.X["LIMIT_BAL"] = np.repeat(np.arange(n // 2), 2)
        self.y = pd.Series(np.tile([0, 1], n // 2), name="default")
        self.metadata = pd.DataFrame({"ID": np.arange(1, n + 1), "SEX": np.tile([1, 2], n // 2),
                                      "default": self.y, "split": "train",
                                      "duplicate_group_id": np.repeat(np.arange(1, n + 1, 2), 2)})

    def build(self, X=None, y=None, metadata=None):
        return build_development_partition(self.X if X is None else X, self.y if y is None else y,
                                           self.metadata if metadata is None else metadata, self.features)

    def test_groups_crossing_sex_and_outcome_stay_whole(self):
        result = self.build()
        self.assertFalse(set(result.groups[result.fit_positions]) & set(result.groups[result.selection_positions]))
        table = result.assignment_table()
        self.assertTrue(table.groupby("duplicate_group_id").development_role.nunique().eq(1).all())
        self.assertEqual(set(result.X.columns), set(FEATURES))
        self.assertFalse({"ID", "SEX", "default"} & set(result.X.columns))
        self.assertEqual(len(result.fit_positions) + len(result.selection_positions), len(self.X))

    def test_same_seed_and_metadata_sex_changes_preserve_assignments(self):
        first, second = self.build(), self.build()
        altered = self.build(metadata=self.metadata.assign(SEX=3 - self.metadata.SEX))
        np.testing.assert_array_equal(first.fold_assignments, second.fold_assignments)
        np.testing.assert_array_equal(first.fold_assignments, altered.fold_assignments)

    def test_no_holdout_interface_or_role(self):
        self.assertNotIn("holdout", inspect.signature(build_development_partition).parameters)
        result = self.build()
        self.assertFalse(hasattr(result, "holdout"))
        with self.assertRaises(ValueError):
            self.build(metadata=self.metadata.assign(split="test"))

    def test_grouping_requires_all_original_inputs(self):
        with self.assertRaises(ValueError):
            self.build(X=self.X.drop(columns="AGE"))
        with self.assertRaises(ValueError):
            self.build(metadata=self.metadata.assign(duplicate_group_id=np.arange(len(self.X))))

    def test_order_alignment_and_outcome_mismatch_are_rejected(self):
        with self.assertRaises(ValueError):
            self.build(X=self.X.loc[:, self.features[::-1]])
        with self.assertRaises(ValueError):
            self.build(metadata=self.metadata.assign(default=1 - self.y))
        with self.assertRaises(ValueError):
            self.build(y=self.y.set_axis(np.arange(100, 180)))

    def test_input_and_returned_views_cannot_mutate_partition(self):
        result = self.build()
        fitting_X, _, _ = result.fitting()
        fitting_X.iloc[0, 0] = -999
        self.assertNotEqual(result.X.iloc[result.fit_positions[0], 0], -999)
        self.assertTrue((self.X.LIMIT_BAL >= 0).all())
        with self.assertRaises(ValueError):
            result.fit_positions[0] = -1


class ProtocolAndRunTests(unittest.TestCase):
    def test_protocol_records_estimator_and_package_scope_distinction(self):
        protocol = build_protocol()
        self.assertEqual(len(protocol["features"]), 22)
        self.assertFalse(protocol["lr"]["regularized"])
        self.assertEqual(protocol["development"]["selection_fold"], 1)
        fidelity = {item["component"]: item["status"] for item in protocol["fidelity"]}
        self.assertEqual(fidelity["lr_coefficient_ks"], "not_identifiable_from_article")
        changed = build_protocol()
        changed["features"].reverse()
        self.assertNotEqual(canonical_digest(changed), canonical_digest(protocol))
        self.assertEqual(build_protocol()["features"], protocol["features"])

    def test_run_destinations_cannot_escape_or_replace_existing_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("article_safe.run.ROOT", root):
                for invalid in ("../escape", "C:/elsewhere", "bad/name", "CON", "LPT1", ""):
                    with self.assertRaises(ValueError):
                        run_directory(invalid)
                target = run_directory("checkpoint_01")
                self.assertEqual(target.parent, root / "article_safe/runs")
                target.mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    run_directory("checkpoint_01")