"""Synthetic protocol and runner checks; no empirical execution or model loading."""
import builtins
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from research.model_selection_v2.judge_core import BootstrapPlan, PredictionTable, pareto_relation
from src.baseline import CATEGORICAL, FEATURES
from src.robustness_protocol import child_seed
from research.model_selection_v2.selection_exploration_v1 import analysis as a
from research.model_selection_v2.selection_exploration_v1 import runner as r
from research.model_selection_v2.selection_exploration_v1 import transforms as t
from research.model_selection_v2.selection_exploration_v1.protocol import FAMILIES, RISK_AXES, scenario_name

ORDER = ("logistic", "decision_tree", "random_forest", "gradient_boosting")


def synthetic_frame(n=40):
    index = np.arange(n)
    columns = {}
    for position, name in enumerate(FEATURES):
        if name.startswith("PAY_"):
            columns[name] = np.resize([-2, -1, 0, 1, 2, 3, 4, 5], n)
        elif name == "EDUCATION":
            columns[name] = np.resize([1, 2, 3, 4], n)
        elif name == "MARRIAGE":
            columns[name] = np.resize([1, 2, 3], n)
        elif name == "AGE":
            columns[name] = 20 + index % 21
        else:
            columns[name] = 1000 + position*100 + index + (index % 3 == 0)
    return pd.DataFrame(columns)[FEATURES]


def reference(p=(.2, .8, .3, .7), *, ids=(1, 2, 3, 4), groups=("cross", "cross", "cross", "single")):
    return PredictionTable(tuple(ids), np.array([0, 1, 0, 1]), np.array([1, 1, 2, 2]), tuple(groups), np.asarray(p), "synthetic")


def prediction_fixture():
    predictions = {}
    for position, candidate in enumerate(ORDER):
        clean = np.array([.2, .8, .3, .7]) + position * np.array([.04, -.04, .02, -.02])
        predictions[candidate] = {"clean": clean[None, :]}
        for index, scenario in enumerate(a.SCENARIOS[1:]):
            shift = .01*(index+1)
            predictions[candidate][scenario] = np.stack((np.clip(clean+shift, 0, 1), np.clip(clean-shift*2, 0, 1)))
    return predictions


def safe_scores():
    return {name: {"arithmetic": .9-i*.1, "geometric": .8-i*.1, "quadratic": .95-i*.1}
            for i, name in enumerate(ORDER)}


def means_from_axes(axes):
    result = {}
    for candidate, values in axes.items():
        result[candidate] = {"clean": {"brier": values[0], "log_loss": values[1]}}
        for scenario in a.MARGINALS:
            result[candidate][scenario] = {"brier": values[2], "log_loss": values[3]}
        for scenario in a.JOINTS:
            result[candidate][scenario] = {"brier": values[4], "log_loss": values[5]}
    return result


class FrozenToy:
    def fit(self, *args, **kwargs):
        raise AssertionError("Fitting is forbidden")

    def predict(self, frame):
        score = .0001*frame.LIMIT_BAL.to_numpy()+.02*frame.AGE.to_numpy()
        score = score + .04*frame.PAY_0.to_numpy()+.00003*frame.BILL_AMT1.to_numpy()
        return 1/(1+np.exp(-score))


class TransformTests(unittest.TestCase):
    def test_fitting_only_references_means_modes_and_rules(self):
        fitting = synthetic_frame(20)
        refs = t.reference_bundle(fitting)
        self.assertEqual(refs["records"], 20)
        for name in FEATURES:
            expected = fitting[name].mode().min() if name in CATEGORICAL else fitting[name].mean()
            self.assertEqual(refs["mask_values"][name], expected)
        self.assertEqual(refs["mask_values"]["EDUCATION"], 1)
        self.assertEqual(refs["semantic_rules"]["AGE"]["lower"], fitting.AGE.min())
        self.assertEqual(refs["semantic_rules"]["LIMIT_BAL"]["iqr"], fitting.LIMIT_BAL.quantile(.75)-fitting.LIMIT_BAL.quantile(.25))

    def test_seed_scope_matches_independent_sha256_and_nests_exposure(self):
        text = "20261007|exploration_v1:repayment|1|mask"
        expected = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        self.assertEqual(child_seed("exploration_v1:repayment", 1, "mask"), expected)
        for family in FAMILIES:
            small = t.selected_rows(40, family, 1, .10)
            large = t.selected_rows(40, family, 1, .20)
            self.assertEqual(small.sum(), 4)
            self.assertEqual(large.sum(), 8)
            self.assertTrue((~small | large).all())

    def test_paired_masks_commutation_valid_categories_and_unchanged_rows(self):
        fitting, selection = synthetic_frame(), synthetic_frame()
        before = selection.copy(deep=True)
        refs = t.reference_bundle(fitting)
        for family, q in itertools.product(FAMILIES, (.10, .20)):
            frames, registry, streams = t.paired_transforms(selection, fitting, refs, family=family, repetition=1, q=q)
            selected = t.selected_rows(len(selection), family, 1, q)
            for condition, transformed in frames.items():
                np.testing.assert_array_equal(transformed.loc[~selected].to_numpy(), selection.loc[~selected].to_numpy())
                self.assertEqual(registry[condition]["input_sha256"], t.frame_hash(transformed))
                self.assertEqual(registry[condition]["exposure"]["selected_rows"], int(q*len(selection)))
                self.assertTrue(registry[condition]["commutation_checked"])
                for name in CATEGORICAL:
                    self.assertTrue(set(transformed[name]) <= set(fitting[name]) | set(selection[name]))
            for name in FAMILIES[family]:
                np.testing.assert_array_equal(frames["10"].loc[selected, name], np.full(selected.sum(), refs["mask_values"][name]))
                np.testing.assert_array_equal(frames["11"][name], frames["10"][name])
                np.testing.assert_array_equal(frames["01"][name], selection[name])
            self.assertEqual(streams["scope"], "exploration_v1:"+family)
        pd.testing.assert_frame_equal(selection, before, check_exact=True)

    def test_severity_independent_streams_shared_signs_and_category_uniforms(self):
        fitting = synthetic_frame()
        refs = t.reference_bundle(fitting)
        small, _, seed_small = t.paired_transforms(fitting, fitting, refs, family="repayment", repetition=2, q=.10)
        large, _, seed_large = t.paired_transforms(fitting, fitting, refs, family="repayment", repetition=2, q=.20)
        self.assertEqual(seed_small, seed_large)
        selected = t.selected_rows(len(fitting), "repayment", 2, .10)
        # Categories use identical uniforms; magnitude does not depend on q.
        np.testing.assert_array_equal(small["01"].loc[selected, "EDUCATION"], large["01"].loc[selected, "EDUCATION"])
        difference_small = small["01"].loc[selected, "BILL_AMT1"]-fitting.loc[selected, "BILL_AMT1"]
        difference_large = large["01"].loc[selected, "BILL_AMT1"]-fitting.loc[selected, "BILL_AMT1"]
        np.testing.assert_array_equal(np.sign(difference_small), np.sign(difference_large))

    def test_clipping_zero_scale_noops_and_support_are_preserved(self):
        fitting = synthetic_frame()
        fitting["LIMIT_BAL"] = 1000
        selection = fitting.copy(deep=True)
        refs = t.reference_bundle(fitting)
        selected = t.selected_rows(len(selection), "billing", 1, .20)
        signs = np.where(np.random.default_rng(child_seed("exploration_v1:billing", 1, "sign:AGE")).random(len(selection)) < .5, -1, 1)
        selection.loc[selected, "AGE"] = np.where(signs[selected] < 0, fitting.AGE.min(), fitting.AGE.max())
        _, registry, _ = t.paired_transforms(selection, fitting, refs, family="billing", repetition=1, q=.20)
        exposure = registry["01"]["exposure"]
        self.assertEqual(exposure["per_feature"]["AGE"]["clipped"], selected.sum())
        self.assertTrue(exposure["per_feature"]["LIMIT_BAL"]["zero_scale"])
        self.assertEqual(exposure["per_feature"]["LIMIT_BAL"]["no_op"], selected.sum())
        self.assertIn("support", exposure)

    def test_reference_source_and_feature_order_mismatch_are_rejected(self):
        fitting = synthetic_frame()
        refs = t.reference_bundle(fitting)
        changed = fitting.copy()
        changed.loc[0, "AGE"] += 1
        with self.assertRaisesRegex(ValueError, "source changed"):
            t.paired_transforms(fitting, changed, refs, family="payment", repetition=1, q=.10)
        with self.assertRaisesRegex(ValueError, "order/source"):
            t.paired_transforms(fitting[FEATURES[::-1]], fitting, refs, family="payment", repetition=1, q=.10)

    def test_frozen_predictor_unchanged_row_reuse_and_full_frame_audit(self):
        fitting = synthetic_frame()
        refs = t.reference_bundle(fitting)
        frames, _, _ = t.paired_transforms(fitting, fitting, refs, family="billing", repetition=1, q=.10)
        model = FrozenToy()
        before = joblib.hash(model)
        clean = model.predict(fitting)
        for frame in frames.values():
            probability, evidence = t.predict_changed_rows(model, fitting, frame, clean, lambda m, x: m.predict(x), audit_full=True)
            np.testing.assert_array_equal(probability, model.predict(frame))
            unchanged = ~(fitting.to_numpy() != frame.to_numpy()).any(axis=1)
            np.testing.assert_array_equal(probability[unchanged], clean[unchanged])
            self.assertEqual(evidence["maximum_absolute_error"], 0.)
        self.assertEqual(joblib.hash(model), before)

    def test_batch_dependent_prediction_fails_reuse_audit(self):
        original = synthetic_frame()
        transformed = original.copy()
        transformed.loc[0, "AGE"] += 1
        predict = lambda model, frame: np.full(len(frame), .2+len(frame)*.001)
        clean = predict(None, original)
        with self.assertRaises(AssertionError):
            t.predict_changed_rows(None, original, transformed, clean, predict, audit_full=True)

    def test_changed_row_predictions_cannot_silently_broadcast(self):
        original = synthetic_frame()
        transformed = original.copy()
        transformed.loc[[0, 1], "AGE"] += 1
        with self.assertRaisesRegex(ValueError, "Changed-row prediction alignment"):
            t.predict_changed_rows(None, original, transformed, np.full(len(original), .5), lambda m, x: [.5])


class AnalysisTests(unittest.TestCase):
    def test_endpoint_counts_and_independent_hand_losses(self):
        p = np.array([0., 1., np.finfo(float).eps/2, 1-np.finfo(float).eps/2])
        counts = a.endpoint_counts(p)
        self.assertEqual(counts["probability_zero"], 1)
        self.assertEqual(counts["probability_one"], 1)
        self.assertEqual(counts["clipped_below_epsilon"], 2)
        self.assertEqual(counts["clipped_above_one_minus_epsilon"], 2)
        values = a.pointwise_losses([0, 1, 0, 1], [.25, .75, .25, .75])
        np.testing.assert_array_equal(values["brier"], [.0625]*4)
        np.testing.assert_allclose(values["log_loss"], [-math.log(.75)]*4, rtol=0, atol=1e-15)

    def test_maximum_after_mean_and_never_loss_of_mean_probability(self):
        predictions = {"A": {scenario: np.full((1 if scenario == "clean" else 2, 4), .25) for scenario in a.SCENARIOS}}
        predictions["A"][a.MARGINALS[0]] = np.array([[0.]*4, [1.]*4])
        predictions["A"][a.MARGINALS[1]] = np.array([[1.]*4, [0.]*4])
        ref = PredictionTable((1, 2, 3, 4), np.zeros(4), np.array([1, 1, 2, 2]), (1, 1, 2, 2), np.full(4, .25), "synthetic")
        cache = a.loss_cache(predictions, ref, ("A",))
        means = a.mean_losses(cache)
        vectors, maxima = a.risk_vectors(means, ("A",))
        self.assertEqual(vectors["A"]["marginal_max_brier"], .5)
        self.assertEqual(maxima["A"]["marginal_max_brier"]["attaining_scenarios"], list(a.MARGINALS[:2]))
        self.assertNotEqual(vectors["A"]["marginal_max_brier"], 1.)  # mean of per-repetition maxima
        self.assertNotEqual(means["A"][a.MARGINALS[0]]["brier"], .25)  # loss of mean p

    def test_all_selectors_and_joint_ablation_are_preserved(self):
        axes = {"A": (.3, .6, .4, .8, .4, .8), "B": (.2, .5, .3, .7, .5, .9)}
        scores = {"A": {name: .9 for name in ("arithmetic", "geometric", "quadratic")},
                  "B": {name: .8 for name in ("arithmetic", "geometric", "quadratic")}}
        result = a.selector_decisions(means_from_axes(axes), scores, ("A", "B"))
        self.assertEqual(set(result["decisions"]), set(a.SELECTORS))
        self.assertEqual(result["decisions"]["safe_arithmetic"]["selected_id"], "A")
        self.assertEqual(result["decisions"]["screened_safe_clean"]["selected_id"], "B")
        self.assertEqual(result["decisions"]["screened_safe_marginal"]["selected_id"], "B")
        self.assertEqual(result["decisions"]["screened_safe_outcome_v0"]["selected_id"], "A")
        self.assertIsNone(result["primary_evaluation"])

    def test_nested_retention_and_score_monotonicity_when_old_vectors_distinct(self):
        axes = {"A": (.1, .1, .8, .8, .9, .9), "B": (.2, .2, .2, .2, .8, .8), "C": (.3, .3, .3, .3, .1, .1)}
        scores = {name: {aggregation: value for aggregation in ("arithmetic", "geometric", "quadratic")}
                  for name, value in (("A", .2), ("B", .5), ("C", .9))}
        decisions = a.selector_decisions(means_from_axes(axes), scores, ("A", "B", "C"))["decisions"]
        retained = [set(decisions[name]["retained_ids"]) for name in
                    ("screened_safe_clean", "screened_safe_marginal", "screened_safe_outcome_v0")]
        self.assertTrue(retained[0] <= retained[1] <= retained[2])
        chosen_scores = [scores[decisions[name]["selected_id"]]["arithmetic"] for name in
                         ("screened_safe_clean", "screened_safe_marginal", "screened_safe_outcome_v0")]
        self.assertEqual(chosen_scores, [.2, .5, .9])

    def test_exact_old_vector_ties_can_remove_frontier_member_and_reduce_safe_score(self):
        axes = {"A": (.4, .4, .5, .5, .5, .5), "B": (.4, .4, .6, .6, .7, .7)}
        scores = {name: {aggregation: value for aggregation in ("arithmetic", "geometric", "quadratic")}
                  for name, value in (("A", .7), ("B", .9))}
        decisions = a.selector_decisions(means_from_axes(axes), scores, ("A", "B"))["decisions"]
        clean, marginal = decisions["screened_safe_clean"], decisions["screened_safe_marginal"]
        self.assertEqual(clean["retained_ids"], ["A", "B"])
        self.assertEqual(marginal["retained_ids"], ["A"])
        self.assertEqual(clean["selected_id"], "B")
        self.assertEqual(marginal["selected_id"], "A")
        self.assertLess(scores[marginal["selected_id"]]["arithmetic"], scores[clean["selected_id"]]["arithmetic"])

    def test_frontier_replacement_need_not_dominate_original_safe_winner(self):
        vectors = {"original": {"x": .4, "y": .4}, "A": {"x": .3, "y": .3}, "B": {"x": .5, "y": .2}}
        scores = {"original": .9, "A": .7, "B": .8}
        chosen = a.screen(vectors, ("original", "A", "B"), ("x", "y"), scores)
        self.assertEqual(chosen["selected_id"], "B")
        self.assertEqual(chosen["retained_ids"], ["A", "B"])
        relation = pareto_relation(vectors["B"], vectors["original"], metric_directions={"x": "min", "y": "min"})
        self.assertEqual(relation["status"], "incomparable")

    def test_loss_minima_exact_ties_and_safe_declared_tolerance(self):
        near = np.nextafter(.5, 1.)
        chosen = a.choose_scores({"A": near, "B": .5}, ("A", "B"), minimize=True)
        self.assertEqual(chosen["selected_id"], "B")
        tied = a.choose_scores({"A": .5, "B": .5}, ("B", "A"), minimize=True)
        self.assertEqual(tied["tied_ids"], ["B", "A"])
        safe = a.choose_scores({"A": .8, "B": .8-5e-13}, ("B", "A"), minimize=False, tolerance=1e-12)
        self.assertEqual(safe["selected_id"], "B")

    def test_required_missing_coordinate_unresolved_without_axis_or_candidate_drop(self):
        vectors = {"A": dict(zip(RISK_AXES, [.1]*6)), "B": dict(zip(RISK_AXES, [.2]*6))}
        vectors["B"][RISK_AXES[-1]] = None
        result = a.screen(vectors, ("A", "B"), RISK_AXES, {"A": .9, "B": .8})
        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(result["selected_id"])
        result = a.choose_scores({"A": .1, "B": np.nan}, ("A", "B"), minimize=True)
        self.assertEqual(result["status"], "unresolved")

    def test_brier_interaction_can_be_nonzero_with_additive_probability_response(self):
        y, probabilities = np.array([0]), (.2, .3, .4, .5)
        loss = [a.pointwise_losses(y, [p])["brier"][0] for p in probabilities]
        self.assertAlmostEqual(probabilities[3]-probabilities[1]-probabilities[2]+probabilities[0], 0.)
        self.assertAlmostEqual(loss[3]-loss[1]-loss[2]+loss[0], 2*.1*.2)


class BootstrapAndBindingTests(unittest.TestCase):
    def test_record_mean_bootstrap_equals_direct_per_realization_recomputation(self):
        predictions = prediction_fixture()
        cache = a.loss_cache(predictions, reference(), ORDER)
        plan = BootstrapPlan(reference(), np.array([[2, 0], [1, 1], [0, 2]]), None)
        keys, matrix = a.bootstrap_loss_matrix(cache, plan)
        for draw in range(plan.n_draws):
            weights = plan.row_multiplicities(draw)
            means = a.mean_losses(cache, weights)
            for column, (candidate, scenario, loss) in enumerate(keys):
                direct = np.mean([np.average(a.pointwise_losses(reference().outcome, p)[loss], weights=weights)
                                  for p in predictions[candidate][scenario]])
                self.assertAlmostEqual(direct, matrix[draw, column], places=14)
                self.assertAlmostEqual(direct, means[candidate][scenario][loss], places=14)

    def test_identity_binding_rejects_same_length_wrong_ids_or_groups(self):
        cache = a.loss_cache(prediction_fixture(), reference(), ORDER)
        for ref in (reference(ids=(4, 3, 2, 1)), reference(groups=(1, 1, 1, 2))):
            plan = BootstrapPlan(ref, np.array([[1, 1]]), None)
            with self.assertRaises(ValueError):
                a.bootstrap_loss_matrix(cache, plan)
        wrong = prediction_fixture()
        wrong[ORDER[0]]["clean"][0, 0] += .01
        with self.assertRaises(AssertionError):
            a.loss_cache(wrong, reference(), ORDER)

    def test_conditional_fixed_safe_frequency_pairs_and_intervals(self):
        cache = a.loss_cache(prediction_fixture(), reference(), ORDER)
        plan = BootstrapPlan(reference(), np.array([[2, 0], [1, 1], [0, 2]]), None)
        result = a.conditional_bootstrap(cache, plan, safe_scores(), percentile_bounds=(0, 100))
        self.assertTrue(result["safe_scores_held_fixed"])
        self.assertFalse(result["joint_confidence_certification"])
        self.assertIn("not complete selector uncertainty", result["scope"])
        self.assertEqual(result["selector_frequencies"]["safe_arithmetic"]["selected_counts"]["logistic"], 3)
        pairs = result["paired_fixed_candidate_intervals"]
        self.assertEqual(len(pairs), 6)
        pair = pairs["gradient_boosting_minus_logistic"]
        direct = []
        interactions = []
        for draw in range(3):
            means = a.mean_losses(cache, plan.row_multiplicities(draw))
            direct.append(means["gradient_boosting"]["clean"]["brier"]-means["logistic"]["clean"]["brier"])
            interactions.append(a.interactions(means, ORDER)["logistic"]["repayment"]["brier"])
        interval = pair["loss_intervals"]["clean"]["brier"]
        self.assertAlmostEqual(interval["lower"], min(direct), places=14)
        self.assertAlmostEqual(interval["upper"], max(direct), places=14)
        self.assertEqual(interval["valid_draws"], 3)
        candidate = result["candidate_interaction_intervals"]["logistic"]["repayment"]["brier"]
        self.assertAlmostEqual(candidate["lower"], min(interactions), places=14)
        self.assertAlmostEqual(candidate["upper"], max(interactions), places=14)

    def test_bootstrap_records_all_tie_memberships_without_changing_fixed_order_choice(self):
        predictions = prediction_fixture()
        for candidate in ORDER[1:]:
            predictions[candidate] = {scenario: values.copy() for scenario, values in predictions[ORDER[0]].items()}
        scores = {candidate: {aggregation: .8 for aggregation in ("arithmetic", "geometric", "quadratic")} for candidate in ORDER}
        cache = a.loss_cache(predictions, reference(), ORDER)
        plan = BootstrapPlan(reference(), np.array([[2, 0], [0, 2]]), None)
        result = a.conditional_bootstrap(cache, plan, scores, percentile_bounds=(0, 100))
        for frequency in result["selector_frequencies"].values():
            self.assertEqual(frequency["selected_counts"][ORDER[0]], 2)
            self.assertEqual(frequency["multiple_choice_tie_draws"], 2)
            self.assertEqual(frequency["tied_choice_membership_counts"], dict.fromkeys(ORDER, 2))

    def test_repetition_binding_saves_shared_input_and_all_row_id_sets(self):
        fitting = synthetic_frame()
        frames, registry, _ = t.paired_transforms(fitting, fitting, t.reference_bundle(fitting), family="repayment", repetition=1, q=.10)
        metadata = pd.DataFrame({"ID": np.arange(100, 140)})
        records = [r.registry_bindings(metadata, fitting, frames[condition], registry[condition]) for condition in frames]
        for record in records:
            self.assertEqual(record["ordered_row_ids"], list(range(100, 140)))
            self.assertEqual(len(record["selected_row_ids"]), 4)
            self.assertEqual(set(record["changed_row_ids"]) | set(record["clean_reused_row_ids"]), set(range(100, 140)))
            self.assertTrue(set(record["changed_row_ids"]) <= set(record["selected_row_ids"]))
        self.assertEqual(len({record["repetition_id"] for record in records}), 1)
        self.assertEqual(len({tuple(record["selected_row_ids"]) for record in records}), 1)

    def test_no_empirical_load_fit_or_file_access_in_pure_analysis(self):
        fitting = synthetic_frame()
        refs = t.reference_bundle(fitting)
        predictions = prediction_fixture()
        with patch.object(r.safe2026_run, "load_contract", side_effect=AssertionError("Empirical loader")), \
             patch.object(r, "verify_safe_inputs", side_effect=AssertionError("Empirical inputs")), \
             patch.object(builtins, "open", side_effect=AssertionError("File access")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("File access")):
            t.paired_transforms(fitting, fitting, refs, family="payment", repetition=1, q=.10)
            cache = a.loss_cache(predictions, reference(), ORDER)
            plan = BootstrapPlan(reference(), np.array([[1, 1]]), None)
            a.conditional_bootstrap(cache, plan, safe_scores(), percentile_bounds=(0, 100))


class ArtifactSafetyTests(unittest.TestCase):
    def test_exclusive_artifact_writes_and_read_only_receipt_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            r.write_json(folder / "payload.json", {"result": "synthetic only"})
            before = (folder / "payload.json").read_bytes()
            with self.assertRaises(FileExistsError):
                r.write_json(folder / "payload.json", {"result": "overwrite"})
            self.assertEqual((folder / "payload.json").read_bytes(), before)
            r.write_npz(folder / "predictions.npz", probability=np.array([.2, .8]))
            with self.assertRaises(FileExistsError):
                r.write_npz(folder / "predictions.npz", probability=np.array([.1, .9]))
            r.write_json(folder / "VERIFIED.json", {"status": "verified", "artifact_hashes": {
                path.name: r.sha256(path) for path in folder.iterdir()}})
            self.assertEqual(r.verify_run(folder)["status"], "verified")
            (folder / "payload.json").write_text("changed")
            with self.assertRaises(ValueError):
                r.verify_run(folder)

    def test_specification_snapshot_precedes_any_input_loading_or_prediction(self):
        class StopBeforeEmpiricalAccess(RuntimeError):
            pass
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "source.md"
            source.write_bytes(b"synthetic source\r\nexact bytes\r\n")
            output = folder / "output"
            output.mkdir()
            def stop():
                self.assertEqual((output / "specification.md").read_bytes(), source.read_bytes())
                protocol = json.loads((output / "protocol.json").read_text())
                self.assertEqual(protocol["specification_sha256"], r.sha256(source))
                raise StopBeforeEmpiricalAccess()
            fake_test = type("TestResult", (), {"args": ["synthetic"], "returncode": 0, "stdout": "", "stderr": ""})()
            with patch.object(r, "SPECIFICATION", source), patch.object(r, "source_hashes", return_value={}), \
                 patch.object(r.subprocess, "run", return_value=fake_test), patch.object(r, "verify_safe_inputs", side_effect=stop):
                with self.assertRaises(StopBeforeEmpiricalAccess):
                    r.execute(output, {})


if __name__ == "__main__":
    unittest.main()
