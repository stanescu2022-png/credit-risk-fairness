"""Prepared selection-only exploration runner; never invoke historical holdout runners."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.model_selection_v2 import pilot
from research.model_selection_v2 import safe2026_run
from research.model_selection_v2.judge_core import PredictionTable, fairness_gaps, group_bootstrap_plan
from .analysis import (LOSSES, SCENARIOS, conditional_bootstrap, interactions, loss_cache, mean_losses,
                       repetition_interactions, selector_decisions)
from .protocol import (BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CONDITIONS, DEFAULT_RUN, FAMILIES,
                       PERCENTILE_BOUNDS, REPETITIONS, ROOT, SAFE_RUN, SEVERITIES, SPECIFICATION,
                       scenario_name, settings)
from .transforms import exposure_summary, frame_hash, paired_transforms, predict_changed_rows, reference_bundle, require, selected_rows

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported artifact value {type(item)}")
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=encode)
        stream.write("\n")


def write_npz(path, **arrays):
    require(all(np.asarray(value).dtype.kind != "O" for value in arrays.values()), "NPZ artifacts cannot contain pickled object arrays.")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def write_csv(path, frame):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def source_hashes():
    paths = [*HERE.glob("*.py"), HERE / "README.md", SPECIFICATION,
             *(HERE.parent / "judge_core").glob("*.py"),
             ROOT / "src/robustness_protocol.py", ROOT / "src/baseline.py",
             HERE.parent / "pilot.py", HERE.parent / "safe2026.py", HERE.parent / "safe2026_run.py"]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(set(paths))}


def verify_safe_inputs():
    """Only identity reads; must complete before train-only model loading."""
    pilot.verify_run(SAFE_RUN)
    manifest = json.loads((SAFE_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    for section in ("source_hashes", "input_hashes"):
        for relative, expected in manifest[section].items():
            path = (ROOT / relative).resolve()
            require(path.is_relative_to(ROOT.resolve()) and sha256(path) == expected,
                    f"Saved SAFE {section} identity mismatch: {relative}")
    candidate_manifest = safe2026_run.verify_candidate_identities()
    candidate_receipt = json.loads((safe2026_run.CANDIDATE_RUN / "VERIFIED.json").read_text(encoding="utf-8"))
    require(manifest["candidate_artifact_hashes"] == candidate_receipt["artifact_hashes"], "SAFE/pilot candidate identities disagree.")
    scores = json.loads((SAFE_RUN / "integrated_scores.json").read_text(encoding="utf-8"))
    require(set(scores) == set(pilot.CANDIDATES), "Saved SAFE pool changed.")
    for candidate in pilot.CANDIDATES:
        require(scores[candidate]["status"] == "defined" and all(np.isfinite(scores[candidate][name])
                for name in ("arithmetic", "geometric", "quadratic")), "Saved SAFE scores unavailable.")
    return scores, manifest, candidate_manifest


def capture_preservation():
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    excluded = {"PROJECT_CONTEXT.md", "README.md", "AGENTS.md"}
    return {name.replace("\\", "/"): sha256(ROOT / name) for name in names
            if name and name not in excluded and (ROOT / name).is_file()}


def check_preservation(baseline):
    require(all((ROOT / name).is_file() and sha256(ROOT / name) == expected for name, expected in baseline.items()),
            "A protected pre-existing file changed; run cannot be verified.")


def prediction_table(metadata, probability):
    return PredictionTable.from_columns({"ID": metadata.ID, "outcome": metadata.default, "SEX": metadata.SEX,
        "original_group": metadata.duplicate_group_id, "probability": probability}, role_label="selection")


def registry_bindings(metadata, selection, transformed, record):
    """Bind repetition, common input and ordered record identities to outputs."""
    selected = selected_rows(len(selection), record["family"], record["repetition"], record["q"])
    changed = (selection.to_numpy() != transformed.to_numpy()).any(axis=1)
    return {**record, "ordered_row_ids": metadata.ID.tolist(), "selected_row_ids": metadata.ID[selected].tolist(),
            "changed_row_ids": metadata.ID[changed].tolist(), "clean_reused_row_ids": metadata.ID[~changed].tolist(),
            "candidate_order": list(pilot.CANDIDATES), "models_share_exact_input": True,
            "source_role": "selection", "input_sha256": frame_hash(transformed)}


def fixed_model_predict(model, frame):
    # Every candidate sees the same already-constructed frame values. Protect
    # the common frame from any estimator/preprocessor mutation by prediction.
    before = frame_hash(frame)
    probability = pilot.positive_probability(model, frame.copy(deep=True))
    require(frame_hash(frame) == before, "Prediction mutated the shared input frame.")
    return probability


def scenario_rows(means, cache, q):
    rows = []
    for candidate in cache["candidate_order"]:
        for scenario in SCENARIOS:
            for loss in LOSSES:
                repetition = cache["repetitions"][candidate][scenario][loss]
                rows.append({"q": q, "design": "reference" if q == .10 else "sensitivity",
                    "candidate_id": candidate, "scenario": scenario, "loss": loss,
                    "mean_loss": means[candidate][scenario][loss], "mc_sd": repetition["mc_sd"],
                    "repetitions": len(repetition["values"])})
    return pd.DataFrame(rows)


def report_text(results):
    lines = ["# Selection-only exploration v1 — 30 October 2026", "",
        "Exploratory results on the saved selection role, conditional on four frozen pilot "
        "candidates and the declared synthetic measurement-error scenarios. The independent "
        "primary evaluation remains undecided. This is hypothesis development, not independent "
        "validation or evidence of selector superiority.", "",
        "| Selector | q=.10 reference choice | q=.20 sensitivity choice |",
        "| --- | --- | --- |"]
    reference = results["q0.10"]["selectors"]["decisions"]
    sensitivity = results["q0.20"]["selectors"]["decisions"]
    for name in reference:
        lines.append(f"| {name} | {reference[name]['selected_id'] or 'unresolved'} | {sensitivity[name]['selected_id'] or 'unresolved'} |")
    lines += ["", "All selector definitions, baselines and ablations are reported. Same choices "
        "mean methods coincide in this pool; they are not independent replications of candidate "
        "performance. Scenario maxima are taken after repetition-loss means, with every exact "
        "maximizer retained. The q=.20 design increases exposure and magnitude and remains a "
        "labelled sensitivity; it never replaces the q=.10 reference choice.", "",
        "Interaction contrasts are signed L11-L10-L01+L00 under matched record exposure. They "
        "describe interaction on the chosen loss scale, not causal or nonlinear model/feature "
        "interactions. Brier's quadratic scale can produce a nonzero contrast even when "
        "the joint probability response is exactly additive. A larger joint loss alone "
        "does not establish an interaction beyond marginal loss effects.", "",
        "Pareto screening uses exact unrounded losses, with no dominance tolerance or practical "
        "margins. The final SAFE tie convention alone uses 1e-12. Adding coordinates expands "
        "the nondominated set only when old vectors are distinct; the chosen retained SAFE "
        "score then cannot decrease, apart from its declared score-tie policy. Exact old-vector "
        "ties can instead split on added axes, remove former frontier members and decrease "
        "the chosen SAFE score. Both cases are retained in synthetic tests. Selecting the "
        "highest-SAFE frontier candidate does not guarantee that a "
        "replacement dominates the original SAFE choice; the synthetic counterexample in "
        "the tests preserves this limitation.", "",
        "The bootstrap shares whole-original-group multiplicities across candidates, scenarios "
        "and saved repetitions, retaining SEX/outcome crossings. It recomputes means, maxima, "
        "Pareto sets and the added outcome-layer choices while holding saved SAFE scores "
        "fixed. Frequencies therefore do not describe complete SAFE/selector uncertainty. "
        "Percentile intervals concern fixed candidates and all six fixed candidate pairs, "
        "later pilot-order ID minus earlier ID, plus candidate interaction contrasts. No "
        "dynamic-winning-model interval or joint confidence certification is claimed.", "",
        "Monte Carlo repetition SD is saved separately from group sampling intervals. LogLoss "
        "uses float64 epsilon clipping only; exact zero/one and clipping counts are retained "
        "by candidate/scenario/repetition. Brier uses original probabilities. Clean signed "
        "sex gaps remain descriptive and explainability remains unresolved.", "",
        "The experiment uses fitting-only references and the train-only contract, with no "
        "holdout predictors/outcomes/predictions, fitting, calibration or tuning. Per-condition "
        "registries bind every repetition to ordered IDs, selected/changed/reused row IDs, "
        "common input hashes, actual child seeds and support/exposure diagnostics. Fixed "
        "repetition-1 cases compare changed-row prediction reuse with full-frame predictions "
        "for every family/severity/condition/candidate at absolute 1e-15; this numerical "
        "reproduction tolerance never enters Pareto dominance.", "",
        "Preserve negative, same-choice and inconclusive findings. Do not optimize on the "
        "historical holdout or revise a judge to favor a selector. Independent validation "
        "and future full-procedure fitting/selection resampling are distinct later stages. "
        "No real-world error frequency, deployment recommendation, practical equivalence "
        "or new-method superiority is established.", ""]
    return "\n".join(lines)


def verify_run(destination):
    destination = Path(destination)
    receipt = json.loads((destination / "VERIFIED.json").read_text(encoding="utf-8"))
    actual = {path.relative_to(destination).as_posix(): sha256(path) for path in destination.rglob("*")
              if path.is_file() and path.name != "VERIFIED.json"}
    require(receipt.get("status") == "verified" and receipt["artifact_hashes"] == actual, "Exploration receipt mismatch.")
    return receipt


def execute(destination, preservation):
    started = now()
    sources = source_hashes()
    # Save immutable settings/specification before any model loading/prediction.
    with (destination / "specification.md").open("xb") as stream:
        stream.write(SPECIFICATION.read_bytes())
    write_json(destination / "protocol.json", {**settings(), "specification_sha256": sha256(SPECIFICATION),
               "source_hashes": sources, "specified_before_new_scenario_predictions": True})
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(HERE),
                            "-p", "test_exploration.py", "-v"], cwd=ROOT, capture_output=True, text=True)
    write_json(destination / "focused_tests.json", {"command": tests.args, "exit_code": tests.returncode,
                                                   "stdout": tests.stdout, "stderr": tests.stderr})
    require(tests.returncode == 0, "Synthetic exploration tests failed before empirical access.")
    scores, safe_manifest, candidate_manifest = verify_safe_inputs()
    fitting, selection, metadata, models, clean, loaded_manifest = safe2026_run.load_contract()
    require(loaded_manifest == candidate_manifest, "Train-only candidate contract changed.")
    require(len(fitting) == 19203 and len(selection) == len(metadata) == 4801, "Saved role sizes changed.")
    fitting_hash, selection_hash = frame_hash(fitting), frame_hash(selection)
    model_states = {name: joblib.hash(model) for name, model in models.items()}
    references = reference_bundle(fitting)
    write_json(destination / "fitting_references.json", references)
    write_json(destination / "saved_safe_scores.json", scores)
    write_csv(destination / "selection_identity.csv", metadata[["ID", "SEX", "default", "duplicate_group_id"]])
    write_json(destination / "clean_input_registry.json", {"condition": "00", "mode": "clean", "source_role": "selection",
        "input_sha256": selection_hash, "ordered_row_ids": metadata.ID.tolist(), "candidate_order": list(pilot.CANDIDATES),
        "shared_across_all_families_severities_realizations": True,
        "exposure": exposure_summary(selection, selection, np.zeros(len(selection), dtype=bool), (), {}, fitting)})
    reference = prediction_table(metadata, clean[pilot.CANDIDATES[0]])
    plan = group_bootstrap_plan(reference, n_draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    write_npz(destination / "bootstrap_multiplicities.npz", group_multiplicities=plan.group_multiplicities,
              row_group_positions=plan.row_group_positions, group_order=np.asarray(plan.group_order), ID=np.asarray(reference.ids))
    write_json(destination / "bootstrap_plan.json", {key: value for key, value in plan.evidence().items() if key not in {"group_multiplicities", "row_group_positions"}})
    write_json(destination / "clean_fairness_descriptive.json", {name: fairness_gaps(prediction_table(metadata, clean[name])) for name in pilot.CANDIDATES})
    results, audit_cases = {}, []
    with threadpool_limits(limits=1):
        for q in SEVERITIES:
            qname = f"q{q:.2f}"
            folder = destination / qname
            folder.mkdir()
            (folder / "registry").mkdir()
            (folder / "predictions").mkdir()
            predictions = {name: {"clean": np.asarray(clean[name])[None, :].copy(),
                                 **{scenario: np.empty((REPETITIONS, len(selection)), dtype=float) for scenario in SCENARIOS if scenario != "clean"}}
                           for name in pilot.CANDIDATES}
            for family in FAMILIES:
                for repetition in range(1, REPETITIONS+1):
                    frames, registry, streams = paired_transforms(selection, fitting, references, family=family, repetition=repetition, q=q)
                    for condition, transformed in frames.items():
                        binding = registry_bindings(metadata, selection, transformed, registry[condition])
                        binding["actual_child_seeds"] = streams
                        for name in pilot.CANDIDATES:
                            probability, evidence = predict_changed_rows(models[name], selection, transformed, clean[name],
                                                                         fixed_model_predict, audit_full=repetition == 1)
                            predictions[name][scenario_name(family, condition)][repetition-1] = probability
                            if repetition == 1:
                                audit_cases.append({"candidate_id": name, "q": q, "family": family, "condition": condition,
                                    "repetition": repetition, "input_sha256": binding["input_sha256"],
                                    "ordered_row_ids": binding["ordered_row_ids"], "changed_row_ids": binding["changed_row_ids"],
                                    "clean_reused_row_ids": binding["clean_reused_row_ids"], **evidence})
                        write_json(folder / f"registry/{family}_r{repetition:02d}_{condition}.json", binding)
                print(f"Prepared {qname} {family}: all 30 paired realizations", flush=True)
            cache = loss_cache(predictions, reference, pilot.CANDIDATES)
            means = mean_losses(cache)
            selected = selector_decisions(means, scores, pilot.CANDIDATES)
            # Check saved selector choices agree with the unchanged SAFE receipt.
            rankings = json.loads((SAFE_RUN / "rankings.json").read_text(encoding="utf-8"))
            for aggregation in ("arithmetic", "geometric", "quadratic"):
                decision = selected["decisions"]["safe_"+aggregation]
                require(decision["selected_id"] == rankings[aggregation]["selected_id"] and
                        decision["tied_ids"] == rankings[aggregation]["tied_selected_ids"], "Saved SAFE choice reproduction failed.")
            for name in pilot.CANDIDATES:
                write_npz(folder / f"predictions/{name}.npz", ID=metadata.ID.to_numpy(), **predictions[name])
            write_npz(folder / "mean_record_losses.npz", ID=metadata.ID.to_numpy(), **{
                f"{name}___{scenario}___{loss}": values for name, scenarios in cache["record_means"].items()
                for scenario, metrics in scenarios.items() for loss, values in metrics.items()})
            write_json(folder / "per_realization_losses.json", cache["repetitions"])
            write_json(folder / "logloss_endpoint_counts.json", cache["endpoint_counts"])
            write_json(folder / "scenario_matrix.json", means)
            write_csv(folder / "scenario_matrix.csv", scenario_rows(means, cache, q))
            write_json(folder / "selectors.json", selected)
            write_json(folder / "interactions.json", {"mean": interactions(means, pilot.CANDIDATES), "per_realization": repetition_interactions(cache)})
            bootstrap = conditional_bootstrap(cache, plan, scores, percentile_bounds=PERCENTILE_BOUNDS)
            write_json(folder / "conditional_bootstrap.json", bootstrap)
            results[qname] = {"selectors": selected, "scenario_matrix": means}
            require(frame_hash(fitting) == fitting_hash and frame_hash(selection) == selection_hash, "Source frames changed.")
    write_json(destination / "prediction_reuse_audits.json", audit_cases)
    require(len(audit_cases) == len(pilot.CANDIDATES)*len(SEVERITIES)*len(FAMILIES)*len(CONDITIONS), "Incomplete fixed probability-reuse audits.")
    require({name: joblib.hash(model) for name, model in models.items()} == model_states, "Frozen model state changed during prediction.")
    require(source_hashes() == sources, "Source/specification changed during exploration.")
    verify_safe_inputs()
    check_preservation(preservation)
    write_json(destination / "preservation.json", {"before_hashes": preservation, "files_checked": len(preservation), "after_matches_before": True,
              "in_memory_model_states_before_after": model_states, "source_frame_hashes": {"fitting": fitting_hash, "selection": selection_hash}})
    with (destination / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report_text(results))
    write_json(destination / "run_manifest.json", {"started_at_utc": started, "completed_at_utc": now(),
        "python": platform.python_version(), "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        "source_hashes": sources, "input_hashes": safe_manifest["input_hashes"],
        "source_run_receipts": {str(path.relative_to(ROOT)): sha256(path) for path in
             (SAFE_RUN / "VERIFIED.json", safe2026_run.CANDIDATE_RUN / "VERIFIED.json")},
        "candidate_artifact_hashes": safe_manifest["candidate_artifact_hashes"], "specification_sha256": sha256(SPECIFICATION),
        "primary_evaluation": None, "role": "selection", "holdout_access": False, "models_fitted": 0,
        "completion_rule": "Only a hash-valid VERIFIED.json establishes completion"})
    write_json(destination / "VERIFIED.json", {"status": "verified", "verified_at_utc": now(),
        "artifact_hashes": {path.relative_to(destination).as_posix(): sha256(path) for path in destination.rglob("*") if path.is_file()}})
    verify_run(destination)
    return destination


def run(destination=DEFAULT_RUN):
    destination = Path(destination)
    require(destination.resolve().parent == DEFAULT_RUN.resolve().parent and destination.name.startswith("selection_exploration_v1_"),
            "Use a fresh selection_exploration_v1 directory under runs/model_selection_v2.")
    preservation = capture_preservation()
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "RUN_STARTED.json", {"status": "incomplete", "started_at_utc": now()})
    try:
        return execute(destination, preservation)
    except BaseException as error:
        write_json(destination / "FAILURE.json", {"status": "incomplete", "at_utc": now(), "error_type": type(error).__name__,
                                                "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-dir", type=Path, help="Execute only after code/protocol review; existing directories are refused")
    action.add_argument("--verify-run", type=Path, help="Read-only receipt check; no predictions or regeneration")
    args = parser.parse_args()
    print(run(args.run_dir) if args.run_dir else verify_run(args.verify_run)["status"])
