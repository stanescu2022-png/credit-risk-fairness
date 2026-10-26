"""Versioned selection-only SAFE assessment; frozen candidates are never fitted."""
import argparse
import ast
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import traceback
import zipfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from research.model_selection_v2 import pilot as p
from research.model_selection_v2 import safe2026 as s

ROOT = p.ROOT
SOURCE = ROOT / "research/model_selection_v2"
CANDIDATE_RUN = ROOT / "runs/model_selection_v2/pilot_candidates_2026-10-26"
PAPER = ROOT / "docs/context/sources/literature_2026-10-09/1-s2.0-S266682702500204X-main.pdf"
PAPER_HASH = "fbc611acf0cf21463b8a63e7febc9d989b4ef651a2b0e08c7b0d9f4aa7814bc4"
WHEEL = SOURCE / "safe2026_sources/safe_ai_metrics-0.2.0-py3-none-any.whl"
WHEEL_HASH = "454d724b448a941345b94016d90e048b7d7c15fda0f18feb48434832ac9e6303"


def verify_candidate_identities():
    p.verify_run(CANDIDATE_RUN)  # Must precede loading any joblib object.
    manifest = json.loads((CANDIDATE_RUN / "run_manifest.json").read_text())
    for category in ("input_hashes", "implementation_hashes"):
        for relative, expected in manifest[category].items():
            p.require(p.sha256(ROOT / relative) == expected, f"Candidate {category} mismatch: {relative}")
    return manifest


def load_contract():
    manifest = verify_candidate_identities()
    protocol = json.loads((CANDIDATE_RUN / "protocol.json").read_text())
    order = protocol["feature_order"]
    assignment = pd.read_csv(CANDIDATE_RUN / "development_assignments.csv")
    pd.testing.assert_frame_equal(assignment, pd.read_csv(p.ARTICLE / "development_assignments.csv"))
    train_metadata = pd.read_csv(p.DATA_PATH.parent / "train_metadata.csv")
    # Identify source rows using membership only. Do not parse holdout outcomes
    # or expose holdout predictors to the assessment. Only train rows survive
    # the CSV reader's skiprows filter, and no outcome column is requested.
    membership = pd.read_csv(p.DATA_PATH.parent / "split_metadata.csv", usecols=["ID", "split"])
    source_rows = set(np.flatnonzero(membership.split.eq("train")))
    training = pd.read_csv(p.DATA_PATH, usecols=["ID"] + order,
                           skiprows=lambda row: row > 0 and row - 1 not in source_rows)
    p.require(set(training.ID) == set(train_metadata.ID) and len(training) == 24004,
              "Training-only CSV membership changed.")
    X = training.set_index("ID").loc[train_metadata.ID, order].reset_index(drop=True)
    positions = p.validate_assignments(X, train_metadata.default, train_metadata, assignment)
    p.require([len(positions[r]) for r in ("fitting", "selection")] == [19203, 4801], "Role sizes changed.")
    fit = X.iloc[positions["fitting"]].reset_index(drop=True)
    selection = X.iloc[positions["selection"]].reset_index(drop=True)
    meta = assignment.iloc[positions["selection"]].reset_index(drop=True)
    models, probabilities = {}, {}
    for candidate in p.CANDIDATES:
        table = pd.read_csv(CANDIDATE_RUN / f"predictions/{candidate}.csv", float_precision="round_trip")
        for name in ("ID", "SEX", "default", "duplicate_group_id"):
            np.testing.assert_array_equal(meta[name], table[name])
        p.require(set(table.candidate_id) == {candidate}, "Saved candidate ID mismatch.")
        probability = table.probability_default.to_numpy()
        model = joblib.load(CANDIDATE_RUN / f"models/{candidate}.joblib")
        p.require(list(model.feature_names_in_) == order, "Primary feature order changed.")
        np.testing.assert_array_equal(p.positive_probability(model, selection), probability)
        np.testing.assert_array_equal(p.decisions(probability), table.prediction)
        models[candidate], probabilities[candidate] = model, probability
    return fit, selection, meta, models, probabilities, manifest


def inspect_later_source():
    p.require(p.sha256(WHEEL) == WHEEL_HASH, "Downloaded source identity changed.")
    p.require(p.sha256(PAPER) == PAPER_HASH, "Published source identity changed.")
    # Execute only these short, inspected pure numerical definitions in isolation;
    # do not import/install the package or its torch/image/adversarial dependencies.
    names = {"lorenz_curve", "gini_via_lorenz", "cvm1_concordance_weighted",
             "_clean_binary_inputs", "_binary_rga_score"}
    definitions = []
    for file in ("cramer.py", "rga.py"):
        tree = ast.parse((SOURCE / "safe2026_sources" / file).read_text())
        definitions.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)
    namespace = {"np": np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "inspected_pure_source", "exec"), namespace)
    fixtures = []
    for label, y, score in (("reversed_binary", [0, 0, 1, 1], [4, 3, 2, 1]),
                             ("binary_tie", [0, 1, 0, 1], [.1, .2, .2, .4]),
                             ("constant_comparison_row_order_A", [0, 1, 0, 1], [1, 1, 1, 1]),
                             ("constant_comparison_row_order_B", [1, 1, 0, 0], [1, 1, 1, 1])):
        fixtures.append({"fixture": label, "reference": y, "ordering": score,
                         "equation_4_with_average_ties": s.rank_graduation(y, score).value,
                         "independent_roc_auc": float(roc_auc_score(y, score)),
                         "later_package_binary_rga": namespace["_binary_rga_score"](y, score)})
    return {"paper_sha256": PAPER_HASH, "paper_available_online": "2025-12-15",
            "package": "safe-ai-metrics", "version": "0.2.0", "released": "2026-06-10",
            "source_url": "https://pypi.org/project/safe-ai-metrics/0.2.0/",
            "wheel_sha256": WHEEL_HASH, "installed": False,
            "claim_that_package_generated_paper_tables": False,
            "numerical_counterexamples": fixtures,
            "differences": ["weighted absolute CvM scalar differs from paper Eq.4",
                            "stable score sorting can depend on row order within ties",
                            "local-slice partials are rescaled and undefined slices replaced",
                            "perfect-reference area normalization defaults on",
                            "greedy fixed-model masking, zero or input-mean references",
                            "optional RGA rescaling of E/R curves and nonfinite-to-zero",
                            "no percentile-swap implementation identified in inspected rgr.py"]}


def support_diagnostics(changed, fitting, original):
    result = {}
    for name in p.CATEGORICAL:
        p.require(set(changed[name]) <= set(original[name]), "Swapping created unseen category levels.")
    combinations = pd.MultiIndex.from_frame(changed[p.CATEGORICAL])
    for name, source in (("fitting", fitting), ("original_selection", original)):
        support = pd.MultiIndex.from_frame(source[p.CATEGORICAL].drop_duplicates())
        result[f"records_with_categorical_combination_absent_from_{name}"] = int((~combinations.isin(support)).sum())
    result["new_invalid_category_levels"] = 0
    result["interpretation"] = "Empirical combination novelty, not a known-invalid-combination test."
    return result


def save_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def independently_rank(reference, ordering):
    reference, ordering = np.asarray(reference, float), np.asarray(ordering, float)
    p.require(reference.shape == ordering.shape and reference.ndim == 1, "Evidence row alignment changed.")
    p.require(np.isfinite(reference).all() and np.isfinite(ordering).all(), "Nonfinite evidence.")
    if np.ptp(reference) == 0:
        return None
    reference = reference - reference.min()
    ascending = np.sort(reference)
    weights = np.arange(1, len(reference) + 1)
    denominator = np.dot(weights, ascending - ascending[::-1])
    return float((np.dot(reference, rankdata(ordering, method="average")) -
                  np.dot(weights, ascending[::-1])) / denominator)


def independent_evidence_check(destination):
    """Reconstruct from saved evidence; no assessment kernel or integrate call."""
    destination = Path(destination)
    meta = pd.read_csv(destination / "selection_identity.csv")
    results = json.loads((destination / "integrated_scores.json").read_text())
    scores, checked = {}, 0
    for candidate in p.CANDIDATES:
        curves = json.loads((destination / f"curves/{candidate}.json").read_text())
        saved = pd.read_csv(CANDIDATE_RUN / f"predictions/{candidate}.csv", float_precision="round_trip")
        for field in ("ID", "SEX", "default", "duplicate_group_id"):
            np.testing.assert_array_equal(meta[field], saved[field])
        full = saved.probability_default.to_numpy()
        accuracy = curves["accuracy"]
        np.testing.assert_allclose(accuracy["scalar_rga"], roc_auc_score(meta.default, full), rtol=0, atol=s.TOL)
        np.testing.assert_allclose(accuracy["scalar_rga"], independently_rank(meta.default, full), rtol=0, atol=s.TOL)
        np.testing.assert_allclose(sum(accuracy["partial_contributions"]), accuracy["scalar_rga"], rtol=0, atol=s.TOL)
        np.testing.assert_allclose(accuracy["values"][-1], 0, rtol=0, atol=s.TOL)
        expected_area = sum((b-a)*v for a, b, v in zip(accuracy["x"], accuracy["x"][1:], accuracy["values"]))
        np.testing.assert_allclose(expected_area, accuracy["raw_step_area"], rtol=0, atol=s.TOL)
        with np.load(destination / f"predictions/{candidate}.npz", allow_pickle=False) as evidence:
            np.testing.assert_array_equal(evidence["ID"], meta.ID)
            np.testing.assert_array_equal(evidence["rgr_probabilities"][0], full)
            np.testing.assert_array_equal(evidence["rge_probabilities"][0], full)
            for key, name in (("rgr_probabilities", "robustness"), ("rge_probabilities", "explainability")):
                probabilities = evidence[key]
                p.require(probabilities.shape == (23, len(meta)), "Evidence dimensions changed.")
                p.require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), "Invalid evidence probabilities.")
                for index, probability in enumerate(probabilities):
                    value = independently_rank(full, probability)
                    np.testing.assert_allclose(value, curves[name]["values"][index], rtol=0, atol=s.TOL)
                    checked += 1
                if name == "explainability":
                    p.require(np.ptp(probabilities[-1]) == 0, "All-removal predictions are not constant.")
                    np.testing.assert_allclose(curves[name]["original_rge_values"],
                                               1 - np.array(curves[name]["values"]), rtol=0, atol=s.TOL)
                    path_prediction_hashes = [s.prediction_hash(v) for v in probabilities[1:]]
        for name in ("robustness", "explainability"):
            curve = curves[name]
            area = sum((b-a)*(u+v)/2 for a, b, u, v in zip(curve["x"], curve["x"][1:], curve["values"], curve["values"][1:]))
            np.testing.assert_allclose(area, curve["summary"]["trapezoidal_area"], rtol=0, atol=s.TOL)
        path = json.loads((destination / f"removal/{candidate}.json").read_text())
        p.require(len(path["alternatives"]) == 253 and len(path["path"]) == 22, "Incomplete greedy search.")
        removed = []
        for chosen in path["path"]:
            alternatives = [v for v in path["alternatives"] if v["step"] == chosen["step"]]
            p.require(set(v["alternative"] for v in alternatives) == set(p.FEATURE_ORDER()) - set(removed), "Missing greedy alternative.")
            maximum = max(v["reversed_rge"] for v in alternatives)
            tied = [name for name in p.FEATURE_ORDER() if any(v["alternative"] == name and maximum-v["reversed_rge"] <= s.TOL for v in alternatives)]
            p.require(chosen["alternative"] == tied[0] and chosen["tied_alternatives"] == tied, "Greedy choice/tie rule mismatch.")
            p.require(chosen["removed_before"] == removed, "Non-nested path.")
            p.require(chosen["prediction_sha256"] == path_prediction_hashes[chosen["step"] - 1], "Chosen path prediction hash mismatch.")
            for alternative in alternatives:
                p.require(alternative["removed_subset"] == removed + [alternative["alternative"]], "Alternative subset mismatch.")
            removed.append(chosen["alternative"])
        A, E, R = [curves[name]["values"] for name in ("accuracy", "explainability", "robustness")]
        cells = list(itertools.product(A, E, R))
        independent = {"arithmetic": math.fsum((a+e+r)/3 for a, e, r in cells)/len(cells),
                       "geometric": math.fsum((a*e*r)**(1/3) for a, e, r in cells)/len(cells),
                       "quadratic": math.fsum(math.sqrt((a*a+e*e+r*r)/3) for a, e, r in cells)/len(cells)}
        for name, value in independent.items():
            np.testing.assert_allclose(value, results[candidate][name], rtol=0, atol=s.TOL)
        np.testing.assert_allclose(independent["arithmetic"], (np.mean(A)+np.mean(E)+np.mean(R))/3, rtol=0, atol=s.TOL)
        scores[candidate] = independent
    rankings = json.loads((destination / "rankings.json").read_text())
    csv_rows = pd.read_csv(destination / "scores_and_rankings.csv", float_precision="round_trip")
    for name in ("arithmetic", "geometric", "quadratic"):
        remaining = list(p.CANDIDATES)
        ranked = []
        while remaining:
            maximum = max(scores[candidate][name] for candidate in remaining)
            tied = [candidate for candidate in remaining if maximum - scores[candidate][name] <= s.TOL]
            ranked.extend(tied)
            remaining = [candidate for candidate in remaining if candidate not in tied]
        p.require([v["candidate_id"] for v in rankings[name]["rows"]] == ranked, "Ranking mismatch.")
        p.require(rankings[name]["selected_id"] == ranked[0], "Selection mismatch.")
        for candidate in p.CANDIDATES:
            row = csv_rows.loc[csv_rows.aggregation.eq(name) & csv_rows.candidate_id.eq(candidate)]
            p.require(len(row) == 1, "Score CSV row mismatch.")
            np.testing.assert_allclose(row.score.iloc[0], scores[candidate][name], rtol=0, atol=s.TOL)
    return {"status": "passed", "independent_kernel": "scipy average ranks / rank-weighted Eq.4 expression",
            "saved_rgr_rge_scores_reconstructed": checked, "integrated_scores_reconstructed": 12,
            "full_greedy_alternatives_checked": 1012, "absolute_tolerance": s.TOL,
            "all_prediction_IDs_and_bounds": True, "areas_recomputed": True}


def source_hashes():
    files = [SOURCE / "safe2026.py", SOURCE / "safe2026_run.py", SOURCE / "safe2026_specification.md",
             *list((SOURCE / "safe2026_tests").glob("*.py")), *list((SOURCE / "safe2026_sources").glob("*"))]
    return p.hashes([path for path in files if path.is_file()])


def protocol():
    return {"version": "safe2026_taiwan_v1", "resolved_at_date": "2026-10-26",
            "accepted_by_user": ["full-sample additive RGA contributions", "fixed-model mean/mode masking",
                                 "all-original-predictor swapping", "defer Gaussian"],
            "paper_doi": "10.1016/j.mlwa.2025.100821", "paper_sha256": PAPER_HASH,
            "specification_path": "research/model_selection_v2/safe2026_specification.md",
            "specification_sha256": p.sha256(SOURCE / "safe2026_specification.md"),
            "candidate_run": CANDIDATE_RUN.relative_to(ROOT).as_posix(),
            "feature_order": p.FEATURE_ORDER(), "removal_units": 22, "grid_nodes": 23,
            "normalized_grid": np.linspace(0, 1, 23).tolist(),
            "raw_swapping_grid": (np.linspace(0, 1, 23)*.5).tolist(),
            "RGA_grid": "actual fraction of full-sample rank-area cells removed in 22 nearly equal slices",
            "cartesian_node_weights": [1/23]*23, "grid_cells": 23**3,
            "component_curves": "raw; no perfect-reference, RGA or candidate min-max rescaling",
            "area_quadrature": {"A": "left-node step", "E": "trapezoidal", "R": "trapezoidal"},
            "ties": {"score": "tie-block mean / expected concordance", "tail_pairing": "original ID secondary sort",
                     "greedy": "maximum preservation; ties within 1e-12 -> saved predictor order",
                     "candidate": "higher score; ties within 1e-12 -> fixed pilot candidate order"},
            "primary_selector": "arithmetic_percentile_swap", "sensitivity": ["geometric_percentile_swap", "quadratic_percentile_swap"],
            "random_seed": None, "repetitions": 1, "randomness": "none; deterministic swaps and masking",
            "prediction_and_frame_hash_format": "SHA256 of row-major little-endian float64 bytes; ID order saved separately",
            "Gaussian": {"status": "deferred", "reason": "user decision: categorical treatment unresolved"},
            "TOPSIS": {"status": "unsupported", "reason": "published raw matrix/normalization/weights/ideals insufficiently specified"},
            "fitting": "none; primary models frozen; fitting records supply reference values only",
            "assessment_role": "selection only (4801); no holdout outcomes/predictors exposed to assessment",
            "reference_role": "fitting only (19203)", "threshold": "unchanged p(default) > .5; not used by SAFE kernels",
            "undefined": "explicit null/unrankable; no record dropping, zero replacement or curve monotonicization",
            "tolerance": s.TOL, "final_judge_or_extended_method": False}


def make_plots(destination, curves):
    colors = {"logistic": "#3366a0", "decision_tree": "#c78128", "random_forest": "#318568", "gradient_boosting": "#995492"}
    labels = {"logistic": "Logistic regression", "decision_tree": "Decision tree", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for axis, component, title, xlabel in zip(axes, ("accuracy", "explainability", "robustness"),
             ("Accuracy: cumulative RGA", "Explainability: reversed RGE", "Robustness: percentile swapping"),
             ("Fraction of rank-area cells removed", "Fraction of original predictors masked", "Normalized severity (raw severity / 0.5)")):
        for candidate in p.CANDIDATES:
            curve = curves[candidate][component]
            kwargs = {"where": "post"} if component == "accuracy" else {}
            plot = axis.step if component == "accuracy" else axis.plot
            plot(curve["x"], curve["values"], color=colors[candidate], label=labels[candidate], linewidth=1.8, **kwargs)
        axis.set(title=title, xlabel=xlabel, ylim=(-.02, 1.03), xlim=(0, 1), ylabel="Raw component value")
        axis.grid(alpha=.2)
        if component != "accuracy":
            axis.axhline(.5, linestyle="--", color="#999999", linewidth=.8)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Selection-set SAFE assessment | fixed Taiwan pilot candidates", fontsize=13)
    target = destination / "component_curves.png"
    p.require(not target.exists(), "Existing plot cannot be overwritten.")
    fig.savefig(target, dpi=170)
    plt.close(fig)


def report_text(scores, rankings, curves, preservation_count):
    text = "# Selection-set integrated SAFE assessment — 26 October 2026\n\n"
    text += "Taiwan application of Giudici & Kolesnikov (2026), DOI 10.1016/j.mlwa.2025.100821. "
    text += "Four frozen, untuned primary candidates; 4,801 selection records. Fitting records supply references only.\n\n"
    text += "## Scores and rankings\n\nHigher is preferred. Arithmetic percentile swapping is the declared primary selector. "
    text += "Geometric and quadratic are cellwise sensitivity variants, not means of three curve averages.\n\n"
    text += "| Candidate | Arithmetic | Rank | Geometric | Rank | Quadratic | Rank |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for candidate in p.CANDIDATES:
        cells = []
        for method in ("arithmetic", "geometric", "quadratic"):
            row = next(v for v in rankings[method]["rows"] if v["candidate_id"] == candidate)
            cells += [f"{scores[candidate][method]:.8f}", str(row["rank"])]
        text += "| " + " | ".join([candidate] + cells) + " |\n"
    primary = rankings["arithmetic"]
    text += f"\nPrimary selected candidate ID: **{primary['selected_id']}**; status: {primary['status']}; tied top IDs: {primary['tied_selected_ids']}.\n\n"
    text += "## Component areas and sampled means\n\n"
    text += "AURGA is the raw step area; reversed AURGE and swapping AURGR use trapezoidal areas. "
    text += "These areas are not the uniformly sampled node means entering Eqs. (13)-(17).\n\n"
    text += "| Candidate | Scalar RGA | AURGA | Reversed AURGE | Swap AURGR | Mean A | Mean E | Mean R |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for c in p.CANDIDATES:
        a, e, r = (curves[c][key] for key in ("accuracy", "explainability", "robustness"))
        values = [a["scalar_rga"], a["raw_step_area"], e["summary"]["trapezoidal_area"], r["summary"]["trapezoidal_area"],
                  a["uniform_node_mean"], e["summary"]["uniform_node_mean"], r["summary"]["uniform_node_mean"]]
        text += "| " + " | ".join([c] + [f"{v:.8f}" for v in values]) + " |\n"
    text += "\n![Component curves](component_curves.png)\n\n## Specification and fidelity\n\n"
    text += "1. **Paper-prescribed:** Lorenz/concordance ratios (Eqs. 4, 8, 10, 12), cumulative RGA accounting "
    text += "(Eqs. 5-7), raw swapping severity 0-.5 normalized to 0-1, reversed RGE, and the three cellwise means "
    text += "uniformly averaged over the Cartesian grid (Eqs. 13-17). No fairness addition or cross-candidate normalization.\n"
    text += "2. **User-approved Taiwan operational choices:** full-sample additive area slices; expected concordance "
    text += "within score ties; whole-original-predictor fixed-model masking at fitting-only numeric means/category modes; "
    text += "greedy maximum preservation against the full model; deterministic mirror tail pairing with ID tie breaks; "
    text += "all 22 original predictors swapped; 23 endpoint-inclusive grid nodes; explicit quadrature and tie rules. "
    text += "See protocol.json and the preserved specification/fidelity table. These are declared adaptations, not proof "
    text += "of the authors' exact HMDA implementation.\n"
    text += "3. **Unresolved/unsupported reproduction:** Gaussian is deferred by explicit user decision because "
    text += "categorical treatment is unresolved. TOPSIS is unsupported because the exact matrix, normalization, "
    text += "directions, weights and ideal points are not identified. Later safe-ai-metrics 0.2.0 was inspected and "
    text += "hash pinned without installation; its different scalar divergence, local-slice rescaling, masking baselines, "
    text += "optional normalizations and nonfinite replacements were not silently adopted. source_inspection.json "
    text += "contains numerical counterexamples and provenance.\n\n"
    text += "## Verification and diagnostics\n\n"
    text += "18 synthetic tests passed before credit scoring; the unchanged 15 pilot tests also passed. "
    text += "Independent rank-weighted calculations reconstructed all 184 saved RGR/RGE values; explicit Cartesian "
    text += "products reconstructed all 12 integrated scores. Binary scalar RGA agrees with sklearn AUROC. "
    text += "All 1,012 greedy alternatives, nested paths, exact tie choices, selection IDs, class-1 mappings and "
    text += "finite probability bounds checked. Necessary chosen-path/perturbation predictions are saved compactly; "
    text += "alternative prediction hashes and scalar scores support deterministic regeneration.\n\n"
    for c in p.CANDIDATES:
        e, r = curves[c]["explainability"], curves[c]["robustness"]
        text += f"- {c}: no/all-removal preservation {e['values'][0]:.12g}/{e['values'][-1]:.12g}; "
        text += f"reversed-RGE increases at steps {e['summary']['increases']}; RGR increases at intervals {r['summary']['increases']}.\n"
    text += "\nAll actual increases are retained. Undefined values are never replaced with zero, records are never "
    text += "dropped, and curves are not forced monotone. Swapping exposures and novel categorical combinations "
    text += "are reported separately in perturbation_diagnostics.json; preserved marginal categories do not imply "
    text += "semantically valid cross-feature combinations. No unsupported business constraints are invented. "
    text += "There is no stochastic repetition: no Monte Carlo or sampling interval is claimed.\n\n"
    text += f"All {preservation_count} pre-task protected files (including uncommitted pilot work and the entire "
    text += "candidate run) retain their hashes. All 410 historical manifest references passed before/after. "
    text += "Candidate completion and its recorded input/implementation identities passed before/after loading and assessment. "
    text += "No primary candidate or preprocessing object was fitted, modified or replaced; no auxiliary fitting occurred. "
    text += "No holdout predictions, performance or model-selection access. Only a hash-valid VERIFIED.json establishes completion.\n\n"
    text += "## Limits and next decision\n\n"
    text += "These are selection-set SAFE assessments conditional on this fixed candidate pool. They establish neither "
    text += "independent predictive superiority, human explanation quality, legal compliance, real-world robustness, "
    text += "nor a winner against our future methodology. The constructed grid is not a sampling distribution or "
    text += "measured simultaneous real-world faults. No threshold tuning, extended methodology or independent judge "
    text += "was implemented.\n\nNext decision: specify the independent judge and practical comparison margins, "
    text += "including whether proposed Brier is primary, LogLoss secondary/co-primary, and how any independent "
    text += "explainability requirement is defined. Prior holdout inspection still makes Taiwan exploratory.\n"
    return text


def execute(destination, baseline):
    starting_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    starting_status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True)
    sources_before = source_hashes()
    p.require(p.hashes([ROOT / name for name in baseline]) == baseline, "Pre-task preservation baseline changed.")
    historical_before = p.check_evaluation_artifacts()
    inspection = inspect_later_source()
    config = protocol()
    p.write_json(destination / "protocol.json", config)
    p.write_json(destination / "source_inspection.json", inspection)
    with (destination / "specification.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write((SOURCE / "safe2026_specification.md").read_text(encoding="utf-8"))
    tests = []
    for folder in ("safe2026_tests", "tests"):
        result = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "unittest", "discover", "-s", str(SOURCE / folder), "-v"],
                                cwd=ROOT, capture_output=True, text=True)
        tests.append({"command": result.args, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    p.write_json(destination / "tests.json", tests)
    p.require(all(result["exit_code"] == 0 for result in tests), "Focused tests failed before credit scoring.")
    fit, selection, meta, models, full, candidate_manifest = load_contract()
    references = s.fitting_references(fit)
    p.write_json(destination / "masking_references.json", {"source_role": "fitting", "records": len(fit),
                 "values": references, "numeric": "arithmetic mean", "categorical": "mode; smallest level on ties",
                 "original_predictor_to_transformed_columns": {
                     c: json.loads((CANDIDATE_RUN / f"diagnostics/{c}.json").read_text())["original_predictor_to_transformed_columns"]
                     for c in p.CANDIDATES}})
    p.write_csv(destination / "selection_identity.csv", meta[["ID", "SEX", "default", "duplicate_group_id"]])
    for name in ("curves", "predictions", "removal"):
        (destination / name).mkdir()
    curves = {c: {"accuracy": s.accuracy_curve(meta.default, full[c])} for c in p.CANDIDATES}
    robust_values, robust_predictions = {c: [] for c in p.CANDIDATES}, {c: [] for c in p.CANDIDATES}
    perturbation = []
    for index, strength in enumerate(config["raw_swapping_grid"]):
        changed, exposure = s.tail_swap(selection, meta.ID, strength)
        exposure["support"] = support_diagnostics(changed, fit, selection)
        exposure["predictor_frame_sha256"] = s.prediction_hash(changed.to_numpy().ravel())
        perturbation.append(exposure)
        for candidate, model in models.items():
            probability = p.positive_probability(model, changed)
            score = s.rank_graduation(full[candidate], probability)
            robust_values[candidate].append(score.value)
            robust_predictions[candidate].append(probability)
        if index % 5 == 0:
            print(f"Swapping grid: {index + 1}/23, shared realization verified", flush=True)
    p.write_json(destination / "perturbation_diagnostics.json", perturbation)
    scores, area_rows = {}, []
    for candidate, model in models.items():
        calls = 0
        def predict_removed(removed):
            nonlocal calls
            calls += 1
            if calls % 50 == 0:
                print(f"{candidate}: evaluated {calls}/253 greedy alternatives", flush=True)
            return p.positive_probability(model, s.mask_original(selection, removed, references))
        removal = s.greedy_preservation(selection.columns, full[candidate], predict_removed)
        p.require(removal["status"] == "defined", f"Unavailable removal path: {candidate}: {removal['reason']}")
        p.require(len(removal["alternatives"]) == 253, "Incomplete greedy search.")
        np.testing.assert_allclose(removal["values"][0], 1, rtol=0, atol=s.TOL)
        np.testing.assert_allclose(removal["values"][-1], .5, rtol=0, atol=s.TOL)
        p.require(np.ptp(removal["predictions"][-1]) == 0, "All-removal input should give constant predictions.")
        e = {"x": removal["x"], "values": removal["values"],
             "original_rge_values": [1-v for v in removal["values"]],
             "summary": s.curve_summary(removal["x"], removal["values"]),
             "original_rge_area": float(np.trapezoid(1-np.array(removal["values"]), removal["x"]))}
        r = {"x": config["normalized_grid"], "raw_x": config["raw_swapping_grid"], "values": robust_values[candidate],
             "summary": s.curve_summary(config["normalized_grid"], robust_values[candidate])}
        curves[candidate].update(explainability=e, robustness=r)
        save_npz(destination / f"predictions/{candidate}.npz", ID=meta.ID.to_numpy(),
                 rgr_probabilities=np.array(robust_predictions[candidate]), rge_probabilities=np.array(removal["predictions"]))
        p.write_json(destination / f"removal/{candidate}.json", {key: value for key, value in removal.items() if key != "predictions"})
        p.write_json(destination / f"curves/{candidate}.json", curves[candidate])
        a = curves[candidate]["accuracy"]
        p.write_csv(destination / f"curves/{candidate}.csv", pd.DataFrame({"node": np.arange(23),
                    "accuracy_x": a["x"], "accuracy": a["values"], "explainability_x": e["x"],
                    "reversed_rge": e["values"], "original_rge": e["original_rge_values"],
                    "robustness_normalized_x": r["x"], "robustness_raw_x": r["raw_x"], "robustness": r["values"]}))
        scores[candidate] = s.integrate(a["values"], e["values"], r["values"])
        area_rows.append({"candidate_id": candidate, "scalar_RGA": a["scalar_rga"], "raw_AURGA_step": a["raw_step_area"],
                "reversed_AURGE_trapezoid": e["summary"]["trapezoidal_area"], "original_RGE_area": e["original_rge_area"],
                "swap_AURGR_trapezoid": r["summary"]["trapezoidal_area"], "mean_A": a["uniform_node_mean"],
                "mean_E": e["summary"]["uniform_node_mean"], "mean_R": r["summary"]["uniform_node_mean"]})
        print(f"{candidate}: full removal path complete; primary SAFE score={scores[candidate]['arithmetic']:.8f}", flush=True)
    p.write_json(destination / "integrated_scores.json", scores)
    rankings = {name: s.rank_candidates({c: scores[c][name] for c in p.CANDIDATES}, p.CANDIDATES)
                for name in ("arithmetic", "geometric", "quadratic")}
    p.write_json(destination / "rankings.json", rankings)
    p.write_json(destination / "primary_selection.json", {"specification": "arithmetic_percentile_swap", **rankings["arithmetic"]})
    p.write_csv(destination / "scores_and_rankings.csv", pd.DataFrame([
        {"aggregation": method, "perturbation": "percentile_swap", **row,
         "tied_ids": ";".join(row["tied_ids"])} for method, table in rankings.items() for row in table["rows"]]))
    p.write_csv(destination / "component_areas.csv", pd.DataFrame(area_rows))
    independent = independent_evidence_check(destination)
    p.write_json(destination / "independent_verification.json", independent)
    make_plots(destination, curves)
    with (destination / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report_text(scores, rankings, curves, len(baseline)))
    p.require(p.hashes([ROOT / name for name in baseline]) == baseline, "Historical or primary artifacts changed.")
    p.require(source_hashes() == sources_before, "SAFE implementation/specification changed during run.")
    verify_candidate_identities()
    historical_after = p.check_evaluation_artifacts()
    p.write_json(destination / "preservation.json", {"before_hashes": baseline, "after_matches_before": True,
                 "files_checked": len(baseline), "historical_before": historical_before, "historical_after": historical_after,
                 "candidate_completion_and_recorded_identities_before_after": "passed"})
    p.write_json(destination / "run_manifest.json", {"run_id": destination.name, "completed_at_utc": p.now(),
                 "starting_git_commit": starting_commit, "starting_working_tree_status": starting_status,
                 "python": platform.python_version(), "versions": {name: version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "safeaipackage")},
                 "source_hashes": sources_before, "input_hashes": {**candidate_manifest["input_hashes"],
                         **p.hashes([CANDIDATE_RUN / "VERIFIED.json", CANDIDATE_RUN / "run_manifest.json", PAPER, WHEEL])},
                 "candidate_artifact_hashes": json.loads((CANDIDATE_RUN / "VERIFIED.json").read_text())["artifact_hashes"],
                 "protocol_sha256": p.sha256(destination / "protocol.json"),
                 "completion_rule": "Require hash-valid VERIFIED.json; manifest alone is insufficient",
                 "verification": independent, "primary_or_auxiliary_models_fitted": 0,
                 "holdout_predictions_or_assessment": False})
    p.write_json(destination / "VERIFIED.json", {"status": "verified", "verified_at_utc": p.now(),
                 "artifact_hashes": {path.relative_to(destination).as_posix(): p.sha256(path)
                                     for path in destination.rglob("*") if path.is_file()}})
    p.verify_run(destination)
    return destination


def capture_preservation():
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    return p.hashes([ROOT / name for name in names if name and name not in
                    {"PROJECT_CONTEXT.md", "README.md", "AGENTS.md"}])


def run(run_id, preservation_baseline=None):
    p.require(Path(run_id).name == run_id and run_id not in {".", ".."} and "/" not in run_id and "\\" not in run_id,
              "Run ID must be a directory name.")
    baseline = json.loads(Path(preservation_baseline).read_text()) if preservation_baseline else capture_preservation()
    destination = p.create_run(p.RUNS / run_id)
    try:
        return execute(destination, baseline)
    except BaseException as error:
        p.write_json(destination / "FAILURE.json", {"status": "incomplete", "error": str(error),
                     "error_type": type(error).__name__, "at_utc": p.now(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--preservation-baseline", type=Path)
    parser.add_argument("--verify-run", type=Path)
    args = parser.parse_args()
    if args.verify_run:
        p.verify_run(args.verify_run)
        verify_candidate_identities()
        print(independent_evidence_check(args.verify_run))
    elif args.run_id:
        print(run(args.run_id, args.preservation_baseline))
    else:
        parser.error("use --verify-run or --run-id")
