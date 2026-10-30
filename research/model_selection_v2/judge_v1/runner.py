"""Prepared judge freeze/assessment entry points; empirical execution requires root review."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.model_selection_v2 import pilot
from research.model_selection_v2.judge_core import group_bootstrap_plan
from research.model_selection_v2.selection_exploration_v1.analysis import SCENARIOS
from .analysis import bootstrap_assessment, point_assessment, table
from .data import load_evaluation_contract
from .freeze import create_freeze, environment_versions, verify_freeze
from .io import (copy_exact, now, read_json, require, seal, sha256, verify_artifacts, write_csv, write_json, write_npz)
from .plots import make_plots
from .protocol import (AUDIT_ATOL, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CANDIDATES, CONDITIONS, COORDINATES, DEFAULT_RUN,
                       FAMILIES, HERE, METHODS, PERCENTILE_BOUNDS, REPETITIONS, ROOT, SEVERITIES, SPECIFICATION, scenario_name, settings)
from .reconstruction import reconstruct_saved_run
from .transforms import (exposure_summary, frame_hash, paired_transforms, predict_changed_rows, reference_bundle, selected_rows)


def focused_tests():
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_judge_v1.py", "-v"],
                            cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, "Synthetic judge tests failed before empirical access.")
    return {"command": result.args, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def capture_preservation():
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    return {name.replace("\\", "/"): sha256(ROOT / name) for name in names if name and
            name not in {"PROJECT_CONTEXT.md", "README.md", "AGENTS.md"} and (ROOT / name).is_file()}


def check_preservation(before):
    require(all((ROOT / name).is_file() and sha256(ROOT / name) == expected for name, expected in before.items()), "Protected pre-existing artifact changed.")


def copy_freeze(freeze_receipt, destination):
    folder = Path(freeze_receipt).parent
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            copy_exact(path, Path(destination) / path.relative_to(folder))


def fixed_predict(model, frame):
    before = frame_hash(frame)
    probability = pilot.positive_probability(model, frame.copy(deep=True))
    require(frame_hash(frame) == before, "Prediction mutated shared inputs.")
    return probability


def bind_registry(metadata, original, transformed, record):
    selected = selected_rows(len(original), record["family"], record["repetition"], record["q"])
    changed = (original.to_numpy() != transformed.to_numpy()).any(axis=1)
    return {**record, "source_role": "historical_evaluation", "ordered_row_ids": metadata.ID.tolist(),
            "selected_row_ids": metadata.ID[selected].tolist(), "changed_row_ids": metadata.ID[changed].tolist(),
            "clean_reused_row_ids": metadata.ID[~changed].tolist(), "input_sha256": frame_hash(transformed),
            "candidate_order": CANDIDATES, "models_share_exact_input": True}


def coordinate_table(point, q):
    groups = point["comparisons"]["same_choice_method_groups"]
    return pd.DataFrame([{"q": q, "design": "reference" if q == .10 else "sensitivity", "candidate_id": candidate,
                         **point["coordinates"][candidate], **point["signed_gaps"][candidate],
                         "frozen_methods": ";".join(groups[candidate])} for candidate in CANDIDATES])


def report_text(points, frozen):
    principal = points["q0.10"]["comparisons"]["principal_contrast"]
    lines = ["# Predeclared Pareto judge v1 — 30 October 2026", "",
        "Historical Taiwan out-of-sample exploration on the saved 5,996 records. These records "
        "were excluded from the current candidate fitting/selection but were historically inspected; "
        "this is not pristine confirmation or sufficient independent evidence of a new superiority claim.", "",
        f"The displayed q=.10 contrast is {principal['our_choice_id']} (full-screen choice) minus "
        f"{principal['SAFE_choice_id']} (arithmetic-SAFE choice): **{principal['status']}**. "
        "Primary evaluation remains null; no scalar primary metric or total-order winner is introduced.", "",
        "| Method | Frozen q=.10 candidate used in both judge designs | q=.20 development choice (diagnostic only) |",
        "| --- | --- | --- |"]
    for method in METHODS:
        lines.append(f"| {method} | {frozen['reference_method_to_candidate'][method]} | {frozen['q20_diagnostic_method_to_candidate'][method]} |")
    lines += ["", "Identical method choices share one pipeline and predictions and are recorded as same_choice, "
        "not independent wins. Different candidates may have equal measured vectors. Every candidate was "
        "evaluated once on each common input. The q=.20 assessment changes stress severity only; it keeps "
        "q=.10 frozen identities and cannot replace the reference result.", ""]
    for qname in ("q0.10", "q0.20"):
        point = points[qname]
        lines += [f"## {qname}: {'reference' if qname == 'q0.10' else 'separate stress sensitivity'}", "",
                  "| Candidate | " + " | ".join(COORDINATES) + " |", "| --- | " + " | ".join(["---:"]*6) + " |"]
        for candidate in CANDIDATES:
            values = ["undefined" if point["coordinates"][candidate][name] is None else f"{point['coordinates'][candidate][name]:.8f}" for name in COORDINATES]
            lines.append("| "+candidate+" | "+" | ".join(values)+" |")
        frontier = point["comparisons"]["frontier"]
        lines += ["", f"Reported frontier ({frontier['frontier_scope']}): {', '.join(frontier['frontier_ids']) or 'unresolved'}. "
                  +(frontier["warning"] or ""), ""]
    lines += ["All six unrounded coordinates are lower-is-better peers. Dominance uses no numerical "
        "or practical margin. Display rounding and the 1e-15 prediction-reproduction tolerance never "
        "enter dominance. Full candidate pair matrices and all six fixed pair contrasts, later "
        "pilot order minus earlier (including GBM-minus-LR), accompany the principal contrast. "
        "A small sex gap is not a fairness guarantee; signed female-minus-male gaps, subgroup "
        "rates, denominators/confusion counts, clean AUC/sklearn AP and reliability bins remain visible.", "",
        "Worst-joint coordinates maximize the three scenario means after averaging repeated losses. "
        "Every exact maximizing scenario is retained. LogLoss alone clips float64 probabilities "
        "to [2^-52,1-2^-52]; exact-zero/one and clipping counts are saved. Supporting diagnostics "
        "include all nine scenario losses, signed deterioration, mean absolute probability movement, "
        "strict p>.5 classification flips, and paired factorial loss contrasts L11-L10-L01+L00. "
        "These contrasts describe interaction on the chosen loss scale: Brier can show a contrast "
        "under additive probability responses. They do not establish causal synergy or nonlinear "
        "model/feature interactions. Explainability is unassessed; this is not complete trustworthiness.", "",
        "Shared 2,000 whole-original-group bootstrap draws (seed 42) retain unequal group sizes "
        "and crossing SEX/outcome groups. Rates, signed/absolute gaps, scenario loss means, maxima "
        "and full-vector relations are recomputed; chosen IDs, fitted models and scenario realizations "
        "stay fixed. Percentile intervals include valid/undefined counts. Dominance/frontier "
        "frequencies describe exploratory conditional stability, not true-superiority probabilities "
        "or joint confidence certification. Monte Carlo repetition SD remains separate. Fitting "
        "and selection variability are omitted; boundary/tie behavior is retained.", "",
        "Plots are labelled two-coordinate projections and cannot establish full-vector dominance. "
        "Source/input/choice identities are frozen before evaluation access. The guarded loader "
        "checks exact saved membership/order and original-group separation, without repartition, "
        "refitting, calibration, tuning or threshold change. Saved predictions support independent "
        "reconstruction of losses, gaps, maxima, contrasts and relations; preservation and hash "
        "receipts establish artifact completion, not publication readiness.", "",
        "Preserve negative, same-choice, trade-off, uncertain and failure-to-improve findings. "
        "There is no post-evaluation replacement selection. No practical equivalence, legal "
        "discrimination, causal mechanism, real fault frequency, future-population safety, universal "
        "winner or novelty claim follows. A stronger claim requires a separately frozen design "
        "and evidence not used to develop it. Future dataset relevance/access criteria and "
        "full-procedure resampling are separate decisions.", ""]
    return "\n".join(lines)


def execute(destination, freeze_receipt, frozen, preservation):
    copy_freeze(freeze_receipt, destination / "freeze")
    copy_exact(SPECIFICATION, destination / "judge_specification.md")
    write_json(destination / "protocol.json", {**settings(), "freeze_sha256": sha256(freeze_receipt), "environment": environment_versions()})
    write_json(destination / "focused_tests.json", focused_tests())
    fitting, evaluation, metadata, models, data_audit = load_evaluation_contract(freeze_receipt)
    write_json(destination / "data_identity_audit.json", data_audit)
    write_csv(destination / "evaluation_identity.csv", metadata[["ID", "SEX", "default", "duplicate_group_id", "partition_order"]])
    fitting_hash, evaluation_hash = frame_hash(fitting), frame_hash(evaluation)
    model_states = {name: joblib.hash(model) for name, model in models.items()}
    references = reference_bundle(fitting)
    write_json(destination / "fitting_references.json", references)
    clean = {candidate: fixed_predict(models[candidate], evaluation) for candidate in CANDIDATES}
    reference = table(metadata, clean[CANDIDATES[0]])
    plan = group_bootstrap_plan(reference, n_draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    write_npz(destination / "bootstrap_multiplicities.npz", group_multiplicities=plan.group_multiplicities,
              row_group_positions=plan.row_group_positions, group_order=np.asarray(plan.group_order), ID=np.asarray(reference.ids))
    write_json(destination / "bootstrap_plan.json", {key: value for key, value in plan.evidence().items() if key not in {"group_multiplicities", "row_group_positions"}})
    write_json(destination / "clean_input_registry.json", {"condition": "00", "input_sha256": evaluation_hash,
        "ordered_row_ids": metadata.ID.tolist(), "shared_across_every_family_severity_realization": True,
        "exposure": exposure_summary(evaluation, evaluation, np.zeros(len(evaluation), dtype=bool), (), {}, fitting)})
    choices = frozen["reference_method_to_candidate"]
    points, audits = {}, []
    with threadpool_limits(limits=1):
        for q in SEVERITIES:
            folder = destination / f"q{q:.2f}"
            folder.mkdir()
            (folder / "registry").mkdir()
            (folder / "predictions").mkdir()
            predictions = {candidate: {"clean": clean[candidate][None, :].copy(),
                **{scenario: np.empty((REPETITIONS, len(evaluation))) for scenario in SCENARIOS if scenario != "clean"}}
                for candidate in CANDIDATES}
            for family in FAMILIES:
                for repetition in range(1, REPETITIONS+1):
                    frames, registry, streams = paired_transforms(evaluation, fitting, references, family=family, repetition=repetition, q=q)
                    for condition, transformed in frames.items():
                        binding = bind_registry(metadata, evaluation, transformed, registry[condition])
                        binding["actual_child_seeds"] = streams
                        for candidate in CANDIDATES:
                            probability, audit = predict_changed_rows(models[candidate], evaluation, transformed, clean[candidate], fixed_predict, audit_full=repetition == 1)
                            predictions[candidate][scenario_name(family, condition)][repetition-1] = probability
                            if repetition == 1:
                                audits.append({"candidate_id": candidate, "q": q, "family": family, "condition": condition,
                                    "input_sha256": binding["input_sha256"], "ordered_row_ids": binding["ordered_row_ids"],
                                    "changed_row_ids": binding["changed_row_ids"], "clean_reused_row_ids": binding["clean_reused_row_ids"], **audit})
                        write_json(folder / f"registry/{family}_r{repetition:02d}_{condition}.json", binding)
                print(f"Judge q={q:.2f}: {family}, 30 paired realizations", flush=True)
            for candidate in CANDIDATES:
                write_npz(folder / f"predictions/{candidate}.npz", ID=metadata.ID.to_numpy(), **predictions[candidate])
            point, cache, clean_tables = point_assessment(predictions, metadata, choices)
            write_json(folder / "point.json", point)
            write_csv(folder / "coordinates.csv", coordinate_table(point, q))
            write_npz(folder / "mean_record_losses.npz", ID=metadata.ID.to_numpy(), **{
                f"{candidate}___{scenario}___{loss}": vector for candidate, scenarios in cache["record_means"].items()
                for scenario, metrics in scenarios.items() for loss, vector in metrics.items()})
            summary, evidence = bootstrap_assessment(cache, clean_tables, choices, plan, percentile_bounds=PERCENTILE_BOUNDS)
            write_json(folder / "bootstrap_summary.json", summary)
            write_npz(folder / "bootstrap_draw_evidence.npz", **evidence)
            make_plots(folder, point, q)
            points[f"q{q:.2f}"] = point
    write_json(destination / "prediction_reuse_audits.json", audits)
    require(len(audits) == 72 and all(audit["maximum_absolute_error"] <= AUDIT_ATOL for audit in audits), "Fixed full-frame audits incomplete/failed.")
    require(frame_hash(fitting) == fitting_hash and frame_hash(evaluation) == evaluation_hash, "Source frame changed.")
    require({name: joblib.hash(model) for name, model in models.items()} == model_states, "Frozen fitted model state changed.")
    reconstruction = reconstruct_saved_run(destination)
    write_json(destination / "independent_reconstruction.json", reconstruction)
    verify_freeze(freeze_receipt)
    check_preservation(preservation)
    write_json(destination / "preservation.json", {"before_hashes": preservation, "files_checked": len(preservation), "after_matches_before": True,
        "model_states_before_after": model_states, "source_frame_hashes": {"fitting": fitting_hash, "historical_evaluation": evaluation_hash}})
    with (destination / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report_text(points, frozen))
    write_json(destination / "run_manifest.json", {"completed_at_utc": now(), "environment": environment_versions(),
        "freeze_receipt_sha256": sha256(freeze_receipt), "source_hashes": frozen["source_hashes"], "input_hashes": frozen["recorded_data_input_hashes"],
        "model_hashes": frozen["model_hashes"], "primary_evaluation": None, "models_fitted_refitted": 0,
        "evidence": "historically inspected out-of-sample exploration; not pristine validation", "reconstruction": reconstruction})
    seal(destination)
    return destination


def run(destination, freeze_receipt):
    destination = Path(destination)
    require(not destination.exists(), "Assessment destination already exists; outputs cannot be overwritten.")
    # No global preservation/data reads or model loads precede this guard.
    frozen = verify_freeze(freeze_receipt)
    preservation = capture_preservation()
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "RUN_STARTED.json", {"status": "incomplete", "started_at_utc": now(), "freeze_sha256": sha256(freeze_receipt)})
    try:
        return execute(destination, freeze_receipt, frozen, preservation)
    except BaseException as error:
        write_json(destination / "FAILURE.json", {"status": "incomplete", "at_utc": now(), "error_type": type(error).__name__,
                                                "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", type=Path, help="Create a fresh immutable choice/source freeze after review; no evaluation-data access")
    action.add_argument("--run", type=Path, help="Execute reviewed historical assessment into a fresh directory")
    action.add_argument("--verify-run", type=Path, help="Read-only artifact integrity check; no regeneration")
    action.add_argument("--reconstruct-run", type=Path, help="Read-only independent reconstruction from saved predictions")
    parser.add_argument("--freeze-receipt", type=Path, help="Required with --run; explicit FREEZE.json path")
    args = parser.parse_args()
    if args.freeze:
        focused_tests()
        print(create_freeze(args.freeze))
    elif args.run:
        parser.error("--run requires --freeze-receipt") if not args.freeze_receipt else print(run(args.run, args.freeze_receipt))
    elif args.reconstruct_run:
        verify_artifacts(args.reconstruct_run)
        print(reconstruct_saved_run(args.reconstruct_run))
    else:
        print(verify_artifacts(args.verify_run)["status"])
