"""Synthetic judge checks; never read empirical data or current selection results."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from research.model_selection_v2.judge_core import BootstrapPlan
from research.model_selection_v2.selection_exploration_v1.analysis import SCENARIOS
from src.baseline import CATEGORICAL, FEATURES
from src.robustness_protocol import child_seed
from research.model_selection_v2.judge_v1 import analysis as a, data as d, freeze as f, io, reconstruction as rec, runner as r, transforms as t
from research.model_selection_v2.judge_v1.protocol import CANDIDATES, COORDINATES, DIRECTIONS, FAMILIES, METHODS, ROLE, scenario_name, settings


def metadata():
    return pd.DataFrame({"ID": [101, 102, 103, 104], "SEX": [1, 1, 2, 2], "default": [0, 1, 0, 1],
                         "duplicate_group_id": [101, 101, 101, 104], "partition_order": range(4)})


def choices(*, same=False):
    result = dict.fromkeys(METHODS, CANDIDATES[0])
    if not same:
        result["screened_safe_outcome_v0"] = CANDIDATES[1]
    return result


def predictions():
    output = {}
    for index, candidate in enumerate(CANDIDATES):
        clean = np.array([.2, .8, .3, .7]) + index*np.array([.02, -.02, .01, -.01])
        if candidate == "decision_tree":
            clean = np.array([0., .5, 1., .75])
        output[candidate] = {"clean": clean[None, :]}
        for position, scenario in enumerate(SCENARIOS[1:]):
            output[candidate][scenario] = np.stack((np.clip(clean+.01*(position+1), 0, 1), np.clip(clean-.01*(position+1), 0, 1)))
    # Preserve a different-candidate equality case independently of identity.
    output["random_forest"] = {key: value.copy() for key, value in output["logistic"].items()}
    return output


def frame(n=40):
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
            columns[name] = 20+index % 21
        else:
            columns[name] = 1000+position*100+index+(index % 3 == 0)
    return pd.DataFrame(columns)[FEATURES]


def fake_frozen_payload(folder):
    folder = Path(folder)
    sources = {}
    for relative in f.required_sources():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic frozen source\n")
        sources[relative] = io.sha256(path)
    model_hashes = {}
    receipt_hashes, artifact_hashes = {}, {}
    for relative in f.RUN_RELATIVES:
        run_folder = folder / relative
        run_folder.mkdir(parents=True, exist_ok=True)
        if relative == f.RUN_RELATIVES[0]:
            (run_folder / "models").mkdir()
            for candidate in CANDIDATES:
                model = run_folder / f"models/{candidate}.joblib"
                model.write_bytes(b"synthetic model identity only")
                model_hashes[relative+f"/models/{candidate}.joblib"] = io.sha256(model)
        io.write_json(run_folder / "synthetic.json", {"synthetic": True})
        marker = io.seal(run_folder)
        receipt_hashes[relative+"/VERIFIED.json"] = io.sha256(run_folder / "VERIFIED.json")
        artifact_hashes[relative] = marker["artifact_hashes"]
    reference = {method: {"status": "selected", "selected_id": candidate, "tied_ids": [candidate],
                         "retained_ids": list(CANDIDATES), "reason": None} for method, candidate in choices().items()}
    diagnostic = copy.deepcopy(reference)
    return {"status": "frozen", "primary_evaluation": None, "candidate_order": list(CANDIDATES),
            "reference_choices": reference, "q20_diagnostic_choices": diagnostic,
            "reference_method_to_candidate": choices(), "q20_diagnostic_method_to_candidate": choices(),
            "judge_settings": f.canonical(settings()), "exploration_settings": f.canonical(f.exploration_settings()),
            "source_hashes": sources, "model_hashes": model_hashes,
            "source_run_receipt_hashes": receipt_hashes, "source_run_artifact_hashes": artifact_hashes,
            "recorded_data_input_hashes": dict.fromkeys(f.DATA_RELATIVES, "recorded-only"),
            "judge_specification_sha256": sources[f.JUDGE_SPEC_RELATIVE], "exploration_specification_sha256": sources[f.EXPLORATION_SPEC_RELATIVE],
            "environment": f.environment_versions()}


class FrozenToy:
    def fit(self, *args):
        raise AssertionError("Fitting forbidden")


class TransformTests(unittest.TestCase):
    def test_new_judge_scope_independent_sha256_and_different_from_exploration(self):
        expected = int.from_bytes(hashlib.sha256(b"20261007|judge_v1:repayment|1|mask").digest()[:8], "little")
        self.assertEqual(child_seed("judge_v1:repayment", 1, "mask"), expected)
        self.assertNotEqual(expected, child_seed("exploration_v1:repayment", 1, "mask"))
        selected = t.selected_rows(40, "repayment", 1, .10)
        large = t.selected_rows(40, "repayment", 1, .20)
        self.assertEqual(selected.sum(), 4)
        self.assertTrue((~selected | large).all())

    def test_adapter_pairs_masks_commutes_preserves_categories_and_fitting_references(self):
        fitting = frame()
        original = fitting.copy(deep=True)
        refs = t.reference_bundle(fitting)
        for family in FAMILIES:
            transformed, registry, seeds = t.paired_transforms(fitting, fitting, refs, family=family, repetition=1, q=.10)
            selected = t.selected_rows(len(fitting), family, 1, .10)
            self.assertEqual(seeds["scope"], "judge_v1:"+family)
            for condition, changed in transformed.items():
                self.assertTrue(registry[condition]["commutation_checked"])
                np.testing.assert_array_equal(changed.loc[~selected].to_numpy(), fitting.loc[~selected].to_numpy())
                for name in CATEGORICAL:
                    self.assertTrue(set(changed[name]) <= set(fitting[name]))
            for name in FAMILIES[family]:
                np.testing.assert_array_equal(transformed["10"].loc[selected, name], np.full(selected.sum(), refs["mask_values"][name]))
                np.testing.assert_array_equal(transformed["10"][name], transformed["11"][name])
        pd.testing.assert_frame_equal(fitting, original)

    def test_adapter_severity_stream_identity_and_reference_mismatch(self):
        fitting = frame()
        refs = t.reference_bundle(fitting)
        small, _, s1 = t.paired_transforms(fitting, fitting, refs, family="billing", repetition=2, q=.10)
        large, _, s2 = t.paired_transforms(fitting, fitting, refs, family="billing", repetition=2, q=.20)
        self.assertEqual(s1, s2)
        selected = t.selected_rows(len(fitting), "billing", 2, .10)
        np.testing.assert_array_equal(small["01"].loc[selected, "EDUCATION"], large["01"].loc[selected, "EDUCATION"])
        changed = fitting.copy()
        changed.loc[0, "AGE"] += 1
        with self.assertRaisesRegex(ValueError, "source changed"):
            t.paired_transforms(fitting, changed, refs, family="billing", repetition=1, q=.10)

    def test_prediction_reuse_and_frozen_model_behavior(self):
        model = FrozenToy()
        before = joblib.hash(model)
        original = frame()
        changed = original.copy()
        changed.loc[[1, 3], "AGE"] += 1
        predict = lambda m, X: .1+.01*X.AGE.to_numpy()
        clean = predict(model, original)
        result, audit = t.predict_changed_rows(model, original, changed, clean, predict, audit_full=True)
        np.testing.assert_array_equal(result, predict(model, changed))
        self.assertEqual(audit["maximum_absolute_error"], 0)
        self.assertEqual(joblib.hash(model), before)


class AssessmentTests(unittest.TestCase):
    def test_six_axes_hand_values_strict_threshold_and_no_extra_primary(self):
        point, _, _ = a.point_assessment(predictions(), metadata(), choices())
        dt = point["coordinates"]["decision_tree"]
        self.assertEqual(set(dt), set(COORDINATES))
        self.assertEqual(dt["clean_brier"], (0+.25+1+.0625)/4)
        self.assertEqual(dt["absolute_tpr_gap"], 1.)
        self.assertEqual(dt["absolute_fpr_gap"], 1.)
        self.assertEqual(point["signed_gaps"]["decision_tree"]["signed_tpr_gap"], 1.)
        self.assertIsNone(point["primary_evaluation"])
        self.assertTrue(all(value == "min" for value in DIRECTIONS.values()))

    def test_same_choice_separate_from_different_candidates_equal_measures(self):
        point, _, _ = a.point_assessment(predictions(), metadata(), choices(same=True))
        self.assertEqual(point["comparisons"]["principal_contrast"]["status"], "same_choice")
        pair = point["comparisons"]["pairwise_matrix"]["random_forest"]["logistic"]
        self.assertFalse(pair["same_candidate"])
        self.assertEqual(pair["status"], "equal")
        self.assertEqual(len(point["comparisons"]["fixed_candidate_pairs"]), 6)
        self.assertIn("gradient_boosting_minus_logistic", point["comparisons"]["fixed_candidate_pairs"])

    def test_reliability_endpoints_bins_and_clean_descriptive_counts(self):
        rows = a.reliability_bins([0, 1, 0, 1], [0., .1, .5, 1.])
        self.assertEqual(len(rows), 10)
        self.assertEqual([rows[i]["count"] for i in (0, 1, 5, 9)], [1, 1, 1, 1])
        self.assertEqual(sum(row["count"] for row in rows), 4)
        self.assertIsNone(rows[2]["observed_rate"])
        point, _, _ = a.point_assessment(predictions(), metadata(), choices())
        diagnostics = point["clean_diagnostics"]["logistic"]
        self.assertEqual(diagnostics["auc"], 1.)
        self.assertEqual(diagnostics["average_precision"], 1.)
        self.assertEqual(diagnostics["recall"], 1.)
        self.assertEqual(diagnostics["fpr"], 0.)

    def test_joint_max_after_repetition_mean_never_marginal_or_record_max(self):
        p = predictions()
        for candidate in CANDIDATES:
            for scenario in SCENARIOS[1:]:
                p[candidate][scenario] = np.full((2, 4), .5)
            p[candidate][scenario_name("repayment", "11")] = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=float)
            p[candidate][scenario_name("billing", "11")] = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=float)
            p[candidate][scenario_name("payment", "10")] = np.array([[1, 0, 1, 0]]*2, dtype=float)
        point, _, _ = a.point_assessment(p, metadata(), choices())
        self.assertEqual(point["coordinates"]["logistic"]["worst_joint_brier"], .5)
        self.assertEqual(point["maxima"]["logistic"]["worst_joint_brier"]["attaining_scenarios"],
                         [scenario_name("repayment", "11"), scenario_name("billing", "11")])

    def test_loss_scale_interaction_and_endpoint_diagnostics_visible(self):
        point, _, _ = a.point_assessment(predictions(), metadata(), choices())
        endpoint = point["endpoint_counts"]["decision_tree"]["clean"]["aggregate"]
        self.assertEqual(endpoint["probability_zero"], 1)
        self.assertEqual(endpoint["probability_one"], 1)
        self.assertTrue(np.isfinite(point["coordinates"]["decision_tree"]["clean_logloss"]))
        self.assertIn("loss scale", point["interactions"]["interpretation"])

    def test_undefined_sex_rates_leave_full_comparison_unresolved(self):
        meta = metadata()
        meta["SEX"] = 1
        point, _, _ = a.point_assessment(predictions(), meta, choices())
        for candidate in CANDIDATES:
            self.assertIsNone(point["coordinates"][candidate]["absolute_tpr_gap"])
            self.assertIsNone(point["coordinates"][candidate]["absolute_fpr_gap"])
        self.assertEqual(point["comparisons"]["principal_contrast"]["status"], "unresolved")
        self.assertTrue(point["comparisons"]["frontier"]["incomplete_pool"])
        self.assertEqual(point["comparisons"]["frontier"]["frontier_ids"], [])


class BootstrapTests(unittest.TestCase):
    def test_grouped_cache_direct_repetition_reconstruction_crossing_unequal_groups(self):
        point, cache, clean = a.point_assessment(predictions(), metadata(), choices())
        plan = BootstrapPlan(clean[CANDIDATES[0]], np.array([[2, 0], [1, 1], [0, 2]]), None)
        summary, evidence = a.bootstrap_assessment(cache, clean, choices(), plan, percentile_bounds=(0, 100))
        self.assertFalse(summary["dynamic_reselection"])
        self.assertFalse(summary["joint_confidence_certification"])
        direct_keys = list(zip(evidence["loss_key_candidate"], evidence["loss_key_scenario"], evidence["loss_key_name"]))
        for draw in range(3):
            weights = plan.row_multiplicities(draw)
            self.assertTrue(weights[0] == weights[1] == weights[2])
            for column, (candidate, scenario, loss) in enumerate(direct_keys):
                expected = np.mean([rec.direct_loss(metadata().default.to_numpy(), p, weights)[loss] for p in predictions()[candidate][scenario]])
                self.assertAlmostEqual(evidence["loss_matrix"][draw, column], expected, places=14)
        interval = summary["candidate_coordinate_and_signed_gap_intervals"]["logistic"]["signed_tpr_gap"]
        self.assertEqual(interval["valid_draws"], 1)
        self.assertEqual(interval["undefined_draws"], 2)
        self.assertEqual(summary["frontier_stability"]["complete_pool_draws"], 1)

    def test_same_choice_identity_does_not_hide_undefined_pareto_draws(self):
        _, cache, clean = a.point_assessment(predictions(), metadata(), choices(same=True))
        plan = BootstrapPlan(clean[CANDIDATES[0]], np.array([[2, 0], [1, 1], [0, 2]]), None)
        summary, _ = a.bootstrap_assessment(cache, clean, choices(same=True), plan, percentile_bounds=(0, 100))
        principal = summary["principal_contrast"]
        self.assertTrue(principal["same_choice"])
        self.assertEqual(principal["relation_stability"]["identity_status_counts"], {"same_choice": 3})
        self.assertEqual(principal["relation_stability"]["unresolved_draws"], 2)
        self.assertEqual(principal["relation_stability"]["resolved_draws"], 1)

    def test_abs_gap_recomputed_each_draw_not_absolute_mean_signed_gap(self):
        y = np.tile([1, 1, 0, 0], 3)
        sex = np.tile([1, 2, 1, 2], 3)
        clean_p = np.array([.8, .2, .1, .1]+[.2, .8, .1, .1]*2)
        meta = pd.DataFrame({"ID": np.arange(1, 13), "SEX": sex, "default": y, "duplicate_group_id": [1]*4+[5]*8})
        p = {candidate: {scenario: np.tile(clean_p, (1 if scenario == "clean" else 2, 1)) for scenario in SCENARIOS} for candidate in CANDIDATES}
        _, cache, clean = a.point_assessment(p, meta, choices())
        plan = BootstrapPlan(clean[CANDIDATES[0]], np.array([[2, 0], [0, 2], [1, 1]]), None)
        _, evidence = a.bootstrap_assessment(cache, clean, choices(), plan, percentile_bounds=(0, 100))
        absolute = evidence["logistic___absolute_tpr_gap"]
        signed = evidence["logistic___signed_tpr_gap"]
        np.testing.assert_allclose(absolute, [1., 1., 1/3], rtol=0, atol=1e-15)
        self.assertNotAlmostEqual(absolute.mean(), abs(signed.mean()))

    def test_worst_joint_scenario_recomputed_after_group_reweighting(self):
        p = predictions()
        for candidate in CANDIDATES:
            p[candidate][scenario_name("repayment", "11")] = np.array([[.9, .1, .9, .7]]*2)
            p[candidate][scenario_name("billing", "11")] = np.array([[.2, .8, .3, .1]]*2)
            p[candidate][scenario_name("payment", "11")] = np.array([[.5]*4]*2)
        _, cache, clean = a.point_assessment(p, metadata(), choices())
        plan = BootstrapPlan(clean[CANDIDATES[0]], np.array([[2, 0], [0, 2]]), None)
        _, evidence = a.bootstrap_assessment(cache, clean, choices(), plan, percentile_bounds=(0, 100))
        masks = evidence["logistic___worst_joint_brier___maximizers"]
        np.testing.assert_array_equal(masks, [[True, False, False], [False, True, False]])

    def test_no_selector_called_and_roles_and_ids_cannot_switch(self):
        _, cache, clean = a.point_assessment(predictions(), metadata(), choices())
        plan = BootstrapPlan(clean[CANDIDATES[0]], np.array([[1, 1]]), None)
        with patch.object(f, "selector_decisions", side_effect=AssertionError("Dynamic reselection")), \
             patch.object(builtins, "open", side_effect=AssertionError("Unexpected file access")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("Unexpected data access")):
            a.bootstrap_assessment(cache, clean, choices(), plan, percentile_bounds=(0, 100))
        other = metadata()
        other.ID = [104, 103, 102, 101]
        wrong = BootstrapPlan(a.table(other, predictions()["logistic"]["clean"][0]), np.array([[1, 1]]), None)
        with self.assertRaises(ValueError):
            a.bootstrap_assessment(cache, clean, choices(), wrong, percentile_bounds=(0, 100))


class FreezeAndLoaderTests(unittest.TestCase):
    def test_freeze_immutable_snapshot_and_no_evaluation_file_or_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = fake_frozen_payload(root)
            destination = root / "freeze"
            with patch.object(f, "ROOT", root), patch.object(f, "collect_freeze_payload", return_value=payload), \
                 patch.object(pd, "read_csv", side_effect=AssertionError("Evaluation data read")), \
                 patch.object(joblib, "load", side_effect=AssertionError("Model load")):
                receipt = f.create_freeze(destination)
                self.assertEqual(f.verify_freeze(receipt)["reference_method_to_candidate"], choices())
                example = next(iter(payload["source_hashes"]))
                self.assertEqual((destination / "source_snapshots" / example).read_bytes(), (root / example).read_bytes())
                with self.assertRaises(ValueError):
                    f.create_freeze(destination)
            self.assertFalse((root / "data/processed/test_metadata.csv").exists())

    def test_freeze_rejects_source_model_settings_version_and_choice_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = fake_frozen_payload(root)
            f.verify_current_bindings(payload, root=root)
            for mutate in (
                lambda x: x["source_hashes"].update({next(iter(x["source_hashes"])): "bad"}),
                lambda x: x["model_hashes"].update({next(iter(x["model_hashes"])): "bad"}),
                lambda x: x["source_hashes"].pop(next(iter(x["source_hashes"]))),
                lambda x: x["model_hashes"].pop(next(iter(x["model_hashes"]))),
                lambda x: x["environment"].update({"python": "changed"}),
                lambda x: x["judge_settings"].update({"primary_evaluation": "brier"}),
                lambda x: x["reference_method_to_candidate"].update({"clean_brier": "changed"}),
            ):
                changed = copy.deepcopy(payload)
                mutate(changed)
                with self.assertRaises(ValueError):
                    f.verify_current_bindings(changed, root=root)

    def test_loader_and_runner_verify_freeze_before_any_data_or_preservation_read(self):
        class GuardFailure(RuntimeError):
            pass
        with patch.object(d, "verify_freeze", side_effect=GuardFailure), \
             patch.object(pd, "read_csv", side_effect=AssertionError("Data read before freeze")), \
             patch.object(d.safe2026_run, "load_contract", side_effect=AssertionError("Model/data load")):
            with self.assertRaises(GuardFailure):
                d.load_evaluation_contract("missing/FREEZE.json")
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(r, "verify_freeze", side_effect=GuardFailure), \
                 patch.object(r, "capture_preservation", side_effect=AssertionError("Input hashing before freeze")):
                with self.assertRaises(GuardFailure):
                    r.run(Path(temporary) / "new_run", "missing/FREEZE.json")
                self.assertFalse((Path(temporary) / "new_run").exists())

    def test_data_hash_mismatch_stops_loader_before_parsing_or_model_loading(self):
        frozen = {"recorded_data_input_hashes": {"data/processed/test_metadata.csv": "expected"}}
        with patch.object(d, "verify_freeze", return_value=frozen), patch.object(d, "sha256", return_value="different"), \
             patch.object(pd, "read_csv", side_effect=AssertionError("Data parsing before identity check")), \
             patch.object(d.safe2026_run, "load_contract", side_effect=AssertionError("Model loading before identity check")):
            with self.assertRaisesRegex(ValueError, "data-input identity changed"):
                d.load_evaluation_contract("synthetic/FREEZE.json")

    def test_role_membership_original_group_reconstruction_and_crossings(self):
        def role(values, ids, groups):
            X = pd.DataFrame({name: values for name in FEATURES})
            meta = pd.DataFrame({"ID": ids, "SEX": [1, 2], "default": [0, 1], "duplicate_group_id": groups})
            return X, meta
        fit, fm = role([1, 1], [1, 2], [1, 1])
        sel, sm = role([10, 11], [10, 11], [10, 11])
        eva, em = role([20, 21], [20, 21], [20, 21])
        audit = d.validate_role_separation(fit, sel, eva, fm, sm, em)
        self.assertTrue(audit["cross_SEX_outcome_groups_preserved"])
        changed = em.copy()
        changed.loc[0, "duplicate_group_id"] = 1
        with self.assertRaises(ValueError):
            d.validate_role_separation(fit, sel, eva, fm, sm, changed)
        with self.assertRaises(ValueError):
            d.validate_role_separation(fit, sel, eva.drop(columns=[FEATURES[0]]), fm, sm, em)


class ReconstructionTests(unittest.TestCase):
    def test_saved_synthetic_artifacts_reconstruct_without_original_data_or_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "freeze").mkdir()
            io.write_json(folder / "freeze/FREEZE.json", {"reference_method_to_candidate": choices()})
            io.write_csv(folder / "evaluation_identity.csv", metadata())
            p = predictions()
            _, _, tables = a.point_assessment(p, metadata(), choices())
            plan = BootstrapPlan(tables[CANDIDATES[0]], np.array([[2, 0], [1, 1], [0, 2]]), None)
            io.write_npz(folder / "bootstrap_multiplicities.npz", group_multiplicities=plan.group_multiplicities,
                row_group_positions=plan.row_group_positions, group_order=np.asarray(plan.group_order), ID=np.asarray(plan.reference.ids))
            for q in (.10, .20):
                sub = folder / f"q{q:.2f}"
                (sub / "predictions").mkdir(parents=True)
                for candidate in CANDIDATES:
                    io.write_npz(sub / f"predictions/{candidate}.npz", ID=metadata().ID.to_numpy(), **p[candidate])
                point, cache, tables = a.point_assessment(p, metadata(), choices())
                summary, evidence = a.bootstrap_assessment(cache, tables, choices(), plan, percentile_bounds=(0, 100))
                io.write_json(sub / "point.json", point)
                io.write_json(sub / "bootstrap_summary.json", summary)
                io.write_npz(sub / "bootstrap_draw_evidence.npz", **evidence)
            with patch.object(joblib, "load", side_effect=AssertionError("Model loading")), \
                 patch.object(d, "load_evaluation_contract", side_effect=AssertionError("Original dataset access")):
                result = rec.reconstruct_saved_run(folder)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["saved_prediction_scenarios_checked"], 80)
            self.assertFalse(result["original_data_or_models_read"])
            target = folder / "q0.10/bootstrap_summary.json"
            original = target.read_bytes()
            changes = (
                lambda summary: summary["frontier_stability"].update({"complete_pool_draws": 100}),
                lambda summary: summary["principal_contrast"]["relation_stability"].update({"resolved_draws": 100}),
                lambda summary: summary["candidate_scenario_loss_intervals"]["logistic"]["clean"]["brier"].update({"lower": 100.}),
                lambda summary: summary["candidate_interaction_intervals"]["logistic"]["repayment"]["brier"].update({"lower": 100.}),
            )
            for change in changes:
                altered = json.loads(original)
                change(altered)
                target.write_text(json.dumps(altered))
                with self.assertRaises((ValueError, AssertionError)):
                    rec.reconstruct_saved_run(folder)
                target.write_bytes(original)

    def test_exclusive_writes_integrity_and_source_bindings_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            io.write_json(folder / "payload.json", {"synthetic": True})
            with self.assertRaises(FileExistsError):
                io.write_json(folder / "payload.json", {})
            io.seal(folder)
            self.assertEqual(io.verify_artifacts(folder)["status"], "verified")
            (folder / "payload.json").write_text("tampered")
            with self.assertRaises(ValueError):
                io.verify_artifacts(folder)
        with self.assertRaises(ValueError):
            f.merge_bindings({"x": "a"}, {"x": "b"})

    def test_nested_verified_receipts_are_hashed_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            nested = folder / "source_runs/example"
            nested.mkdir(parents=True)
            io.write_json(nested / "VERIFIED.json", {"status": "synthetic nested marker"})
            io.seal(folder)
            self.assertIn("source_runs/example/VERIFIED.json", io.read_json(folder / "VERIFIED.json")["artifact_hashes"])
            (nested / "VERIFIED.json").write_text("tampered nested receipt")
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                io.verify_artifacts(folder)


if __name__ == "__main__":
    unittest.main()
