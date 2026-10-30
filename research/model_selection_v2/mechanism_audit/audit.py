"""Deterministic synthetic SAFE mechanism audit; never load empirical data."""
import argparse
from fractions import Fraction
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from research.model_selection_v2 import safe2026 as safe
from research.model_selection_v2.mechanism_audit.joint_information import joint_fixture

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs/model_selection_v2/mechanism_audit_2026-10-30"
FEATURES = ("a", "b", "c")
Y = np.array([0, 1, 0, 0, 1, 1, 0, 1])
TOL = 1e-12
TRANSFORMS = {
    "identity": (1., 0.),
    "compression": (.5, .25),
    "shift_compression": (.75, .125),
    "near_constant_nonzero": (2. ** -40, .5),
}
PERMUTATIONS = {
    ("a",): [0, 2, 1, 3, 5, 4, 6, 7],
    ("b",): [2, 1, 0, 3, 4, 7, 6, 5],
    ("c",): [7, 6, 5, 4, 3, 2, 1, 0],
    ("a", "b"): [0, 2, 3, 1, 6, 4, 5, 7],
    ("a", "c"): [3, 2, 1, 0, 4, 5, 6, 7],
    ("b", "c"): [4, 5, 6, 7, 0, 1, 2, 3],
}


def fraction(value):
    """Use the exact rational represented by a Python/numpy numeric value."""
    return value if isinstance(value, Fraction) else Fraction(float(value))


def pairwise_reference(reference, ordering):
    """Exact weighted concordance oracle; no ranks or cumulative areas."""
    r, q = tuple(map(fraction, reference)), tuple(map(fraction, ordering))
    if len(r) != len(q) or not r:
        raise ValueError("Require aligned nonempty vectors.")
    numerator = denominator = Fraction(0)
    for i, j in itertools.combinations(range(len(r)), 2):
        difference = r[i] - r[j]
        weight = abs(difference)
        denominator += weight
        if q[i] == q[j]:
            numerator += weight / 2
        elif difference * (q[i] - q[j]) > 0:
            numerator += weight
    return numerator / denominator if denominator else None


def accuracy_reference(outcome, probability, segments=3):
    """Enumerate all tied-score orders and average exact area contributions."""
    y, q = tuple(map(fraction, outcome)), tuple(map(fraction, probability))
    if not set(y) <= {0, 1}:
        raise ValueError("Tie-enumeration accuracy oracle is specified for binary outcomes.")
    groups = [tuple(i for i, value in enumerate(q) if value == score)
              for score in sorted(set(q))]
    descending, ascending = sorted(y, reverse=True), sorted(y)
    denominator = sum(sum(descending[:k]) - sum(ascending[:k])
                      for k in range(1, len(y) + 1))
    if not denominator:
        return None
    terms = [Fraction(0)] * len(y)
    orders = 0
    # Within a tie, each distinct binary outcome arrangement has the same
    # number of record permutations. Enumerate those arrangements exactly.
    patterns = []
    for group in groups:
        patterns.append([tuple(Fraction(int(i in ones)) for i in range(len(group)))
                         for ones in itertools.combinations(range(len(group)), int(sum(y[i] for i in group)))])
    for block_order in itertools.product(*patterns):
        ordered = tuple(itertools.chain.from_iterable(block_order))
        for k in range(1, len(y) + 1):
            terms[k - 1] += (sum(descending[:k]) - sum(ordered[:k])) / denominator
        orders += 1
    terms = [term / orders for term in terms]
    # Construct the segment lengths without numpy's production array_split.
    lengths = [len(y) // segments + int(i < len(y) % segments) for i in range(segments)]
    reverse = terms[::-1]
    removed, values = 0, [sum(terms)]
    for length in lengths:
        removed += length
        values.append(sum(reverse[removed:]))
    assert values[0] == pairwise_reference(y, q)
    return {"values": values, "terms": terms, "distinct_tied_outcome_orders_enumerated": orders}


def integrated_reference(A, E, R):
    """Independent explicit Cartesian loops, rather than array broadcasting."""
    arithmetic, geometric, quadratic = [], [], []
    for a, e, r in itertools.product(A, E, R):
        arithmetic.append(float((a + e + r) / 3))
        geometric.append(float(a * e * r) ** (1 / 3))
        quadratic.append(math.sqrt(float((a*a + e*e + r*r) / 3)))
    return {name: math.fsum(values) / len(values) for name, values in
            (("arithmetic", arithmetic), ("geometric", geometric), ("quadratic", quadratic))}


def proper_losses(outcome, probability):
    y, p = tuple(map(fraction, outcome)), tuple(map(fraction, probability))
    if len(y) != len(p) or not p or any(value <= 0 or value >= 1 for value in p):
        raise ValueError("Synthetic direct-loss calculation requires aligned interior probabilities.")
    return {"brier": float(sum((a-b)**2 for a, b in zip(y, p)) / len(y)),
            "log_loss": math.fsum(-math.log(float(b if a else 1-b))
                                  for a, b in zip(y, p)) / len(y)}


def fixture(name):
    levels = {"strict": [1, 3, 5, 7, 9, 11, 13, 15],
              "ties": [2, 2, 5, 5, 10, 10, 14, 14]}
    clean = np.asarray(levels[name], float) / 16
    masked = {key: clean[index] for key, index in PERMUTATIONS.items()}
    masked[FEATURES] = np.full(len(clean), .5)
    perturbed = [clean.copy(), clean[[1, 0, 2, 4, 3, 5, 7, 6]],
                 clean[[4, 1, 6, 3, 0, 5, 2, 7]], np.full(len(clean), .5)]
    return {"name": name, "outcome": Y.copy(), "clean": clean,
            "masked": masked, "perturbed": perturbed}


def map_outputs(original, transform):
    """Wrap every output of the fixed provider, including all alternatives."""
    def checked(values):
        result = np.asarray(transform(values.copy()), float)
        if result.shape != values.shape or not np.isfinite(result).all():
            raise ValueError("Transformed outputs must be finite and aligned.")
        if not ((result >= 0) & (result <= 1)).all():
            raise ValueError("Transformed outputs must be valid probabilities.")
        return result
    return {"name": original["name"], "outcome": original["outcome"].copy(),
            "clean": checked(original["clean"]),
            "masked": {key: checked(value) for key, value in original["masked"].items()},
            "perturbed": [checked(value) for value in original["perturbed"]]}


def subset_key(removed):
    return tuple(name for name in FEATURES if name in removed)


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=TOL)


def evaluate(values):
    """Production curves checked against independent exact arithmetic oracles."""
    clean, y = values["clean"], values["outcome"]
    accuracy = safe.accuracy_curve(y, clean, segments=len(FEATURES))
    independent_accuracy = accuracy_reference(y, clean)
    assert_close(accuracy["values"], list(map(float, independent_accuracy["values"])))
    assert_close(accuracy["full_sample_area_terms_low_to_high_rank"],
                 list(map(float, independent_accuracy["terms"])))
    assert_close(accuracy["scalar_rga"], float(pairwise_reference(y, clean)))
    preservation = safe.greedy_preservation(FEATURES, clean,
                                          lambda removed: values["masked"][subset_key(removed)])
    robustness = [safe.rank_graduation(clean, perturbed).value for perturbed in values["perturbed"]]
    independent_robustness = [pairwise_reference(clean, perturbed) for perturbed in values["perturbed"]]
    independent_alternatives = []
    for row in preservation["alternatives"]:
        expected = pairwise_reference(clean, values["masked"][subset_key(row["removed_subset"])])
        if expected is None:
            assert row["reversed_rge"] is None
        else:
            assert_close(row["reversed_rge"], float(expected))
        independent_alternatives.append({"step": row["step"], "alternative": row["alternative"],
                                         "exact_score": str(expected) if expected is not None else None})
    if np.ptp(clean) == 0:
        assert preservation["status"] == "undefined"
        assert all(value is None for value in robustness)
        return {"status": "undefined", "reason": preservation["reason"],
                "accuracy": accuracy["values"], "scalar_rga": accuracy["scalar_rga"],
                "preservation": preservation["values"], "robustness": robustness,
                "scores": safe.integrate(accuracy["values"], [None], robustness),
                "losses": proper_losses(y, clean), "float64_clean_range": float(np.ptp(clean))}
    assert_close(robustness, list(map(float, independent_robustness)))
    # Reconstruct the greedy selection independently from all subset scores.
    removed, remaining, expected_path = [], list(FEATURES), []
    independent_E = [pairwise_reference(clean, clean)]
    for _ in FEATURES:
        scores = {name: pairwise_reference(clean, values["masked"][subset_key(removed + [name])])
                  for name in remaining}
        maximum = max(scores.values())
        tied = [name for name in remaining if float(maximum - scores[name]) <= TOL]
        chosen = tied[0]
        expected_path.append({"alternative": chosen, "tied_alternatives": tied})
        independent_E.append(scores[chosen])
        removed.append(chosen)
        remaining.remove(chosen)
    actual_path = [{key: row[key] for key in ("alternative", "tied_alternatives")}
                   for row in preservation["path"]]
    assert actual_path == expected_path
    assert_close(preservation["values"], list(map(float, independent_E)))
    integrated = safe.integrate(accuracy["values"], preservation["values"], robustness)
    independent_integrated = integrated_reference(independent_accuracy["values"], independent_E,
                                                  independent_robustness)
    for name, expected in independent_integrated.items():
        assert_close(integrated[name], expected)
    return {"status": "defined", "scalar_rga": accuracy["scalar_rga"],
            "accuracy": accuracy["values"], "accuracy_area_terms": independent_accuracy["terms"],
            "accuracy_distinct_tied_outcome_orders_enumerated": independent_accuracy["distinct_tied_outcome_orders_enumerated"],
            "preservation": preservation["values"], "robustness": robustness,
            "path": actual_path, "alternatives_checked": independent_alternatives,
            "scores": integrated, "independent_scores": independent_integrated,
            "losses": proper_losses(y, clean), "float64_clean_range": float(np.ptp(clean)),
            "independent_robustness_exact": list(map(str, independent_robustness)),
            "independent_preservation_exact": list(map(str, independent_E))}


def assert_invariant(baseline, transformed):
    assert baseline["status"] == transformed["status"] == "defined"
    for field in ("accuracy", "preservation", "robustness", "scalar_rga"):
        assert_close(baseline[field], transformed[field])
    for name in ("arithmetic", "geometric", "quadratic"):
        assert_close(baseline["scores"][name], transformed["scores"][name])
    assert baseline["path"] == transformed["path"]
    assert baseline["alternatives_checked"] == transformed["alternatives_checked"]


def serializable(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported output type {type(value)}")


def output_evidence(values):
    return {"outcome": values["outcome"], "clean": values["clean"],
            "perturbed": values["perturbed"],
            "masked": {"+".join(key): value for key, value in values["masked"].items()}}


def make_audit():
    rows = {}
    for name in ("strict", "ties"):
        original = fixture(name)
        results, evidence = {}, {}
        for transform_name, (a, b) in TRANSFORMS.items():
            transformed = map_outputs(original, lambda values: a * values + b)
            results[transform_name] = evaluate(transformed)
            evidence[transform_name] = output_evidence(transformed)
            if transform_name != "identity":
                assert_invariant(results["identity"], results[transform_name])
        for transform_name, function in (("square_nonlinear_control", np.square),
                ("exact_constant_boundary", lambda values: values * 0 + .5),
                ("float64_collapse_positive_a", lambda values: values * (2. ** -60) + .5)):
            transformed = map_outputs(original, function)
            results[transform_name] = evaluate(transformed)
            evidence[transform_name] = output_evidence(transformed)
        square = results["square_nonlinear_control"]
        assert_close(results["identity"]["accuracy"], square["accuracy"])
        assert any(abs(a-b) > TOL for a, b in zip(results["identity"]["robustness"], square["robustness"]))
        assert results["exact_constant_boundary"]["status"] == "undefined"
        assert results["float64_collapse_positive_a"]["status"] == "undefined"
        # Loss differences are diagnostics, not selection criteria for transforms.
        for transform_name in TRANSFORMS:
            if transform_name != "identity":
                for loss in ("brier", "log_loss"):
                    assert abs(results["identity"]["losses"][loss] - results[transform_name]["losses"][loss]) > TOL
        rows[name] = {"results": results, "output_evidence": evidence}
    joint = joint_fixture()
    for scenario in ("clean", "stress_only", "mask_only"):
        np.testing.assert_array_equal(joint["candidate_A"][scenario], joint["candidate_B"][scenario])
    joint_results = {name: {scenario: proper_losses(joint["outcome"], probability)
                           for scenario, probability in joint[name].items()}
                     for name in ("candidate_A", "candidate_B")}
    for loss in ("brier", "log_loss"):
        assert joint_results["candidate_A"]["joint"][loss] < joint_results["candidate_B"]["joint"][loss]
    return {"protocol_version": "synthetic_affine_mechanism_v1", "specified_date": "2026-10-30",
            "scope": "Operational SAFE kernel; diagnostic wrappers, not empirical candidates",
            "transforms": {name: {"a": a, "b": b} for name, (a, b) in TRANSFORMS.items()},
            "tolerance": TOL, "fixtures": rows,
            "joint_information_counterexample": {"fixture": joint, "losses": joint_results,
                "claim": "Identical separate input vectors cannot identify joint-intervention loss",
                "scope": "Finite response-table information counterexample; not full SAFE or Taiwan evidence"},
            "conclusions": {"positive_affine_exact_arithmetic": "proved for this operationalization",
                "finite_fixture_float64_verification": "passed, including nonzero near-constant scale",
                "arbitrary_monotone_invariance": "refuted by square-output control for K with probability reference",
                "proper_loss_invariance": "refuted by supplied affine examples",
                "constant_reference": "undefined; positive a may numerically collapse in float64",
                "new_empirical_data_or_predictions_accessed": False,
                "article_formula_freshly_verified": False,
                "new_empirical_candidates": False}}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = list(HERE.glob("*.py")) + [HERE / "PROTOCOL.md", HERE.parent / "safe2026.py",
            HERE.parent / "pilot.py", HERE.parent / "safe2026_specification.md"]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(paths)}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, default=serializable, allow_nan=False)
        stream.write("\n")


def report_markdown(audit):
    lines = ["# Synthetic SAFE mechanism audit — 30 October 2026", "",
        "Executed result: positive affine output invariance is supported by the algebra and "
        "all fixed synthetic checks for our Taiwan operationalization. Each of the three "
        "integrated means is unchanged; proper probability losses change. This is a mechanism "
        "diagnostic, not an empirical model comparison or a proposed addition to the four candidates.", "",
        "The independent oracle uses exact rational, absolute-reference-gap-weighted pairwise "
        "concordance. Positive affine maps scale every gap equally and preserve comparison "
        "order/ties. Accuracy's binary reference stays unchanged. Every greedy alternative, "
        "path/tie set, curve node and Cartesian aggregate was checked independently. "
        "Tie-order enumeration also checked the full additive accuracy curve.", "",
        "| Fixture | Output transform | Arithmetic | Geometric | Quadratic | Brier | LogLoss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for fixture_name, fixture_rows in audit["fixtures"].items():
        for name, row in fixture_rows["results"].items():
            if row["status"] == "defined":
                numbers = [row["scores"][metric] for metric in ("arithmetic", "geometric", "quadratic")]
                numbers += [row["losses"][loss] for loss in ("brier", "log_loss")]
                lines.append(f"| {fixture_name} | {name} | " + " | ".join(f"{x:.10f}" for x in numbers) + " |")
    lines += ["", "Near-constant affine scale is 2^-40 with offset 1/2; all clean probability "
        "levels remain distinct where originally distinct. Exact constant scale zero is "
        "undefined for K with the probability as reference. A positive 2^-60 scale at the "
        "same offset collapses all levels in float64 and is also undefined. Neither result "
        "is silently assigned a zero score. Constant comparison outputs remain defined "
        "at K=.5 when the clean reference is nonconstant.", "",
        "Squaring outputs preserves accuracy order/ties but changes probability-reference "
        "pair weights and the robustness curve in both fixtures. It refutes a general "
        "nonlinear monotone invariance claim. These fixtures also show affine changes "
        "to Brier and LogLoss; they do not assert that every affine map changes every loss.", "",
        "## Operational proof versus article fidelity", "",
        "The proof applies to the centered cumulative-area / weighted-pairwise rank kernel, "
        "average exact-score ties, additive accuracy contributions, fixed-model greedy masking "
        "and raw Cartesian means currently specified in safe2026_specification.md. Its "
        "documented equation correspondence is Eqs. (4), (8), (10), (12)–(17). This audit "
        "does not newly inspect the printed article, establish the authors' software used "
        "that kernel, resolve alternate slice normalizations or prove TOPSIS invariance. "
        "Publication wording about published SAFE requires checking the printed equations "
        "and reporting all operational adaptations. The later author-credited package has "
        "documented kernel/normalization differences and is outside this proof.", "",
        "## Separate marginals and joint loss", "",
        "A second, finite response-table counterexample gives candidates A/B identical clean, "
        "stress-only and mask-only predictions record by record. Therefore every deterministic "
        "function given only those identical vectors (and the same outcomes/metadata) must tie. "
        "Their joint stress+mask outputs are different: candidate A has joint Brier .0625 "
        "and LogLoss -log(.75), while candidate B has joint Brier .5625 and LogLoss -log(.25). "
        "The separate tables therefore do not identify joint loss. This is an information "
        "insufficiency counterexample, not an end-to-end SAFE experiment, evidence of such "
        "interactions on Taiwan, or a claim that no method could use joint inputs. The lookup "
        "responses can be represented by finite functions of record identity and two binary "
        "intervention flags; no fitted estimator or empirical data is invoked.", "",
        "Improvement remains a hypothesis. Both the invariance result and the nonlinear "
        "counterexample are preserved. These diagnostics belong to hypothesis generation; "
        "they provide no independent superiority evidence. Any later extension must avoid "
        "optimization on the historical holdout, and the independent judge must not be "
        "revised in response to a favored selector. Exploratory experiments, protocol "
        "freezing and independent validation remain distinct; a superiority claim requires "
        "evidence not used to develop that claim.", "",
        "## Later judge interface recommendation (inspection only)", "",
        "Keep pure assessment functions separate from identity-checked data and model access. "
        "Pass aligned `(ID, outcome, SEX, original_22_predictor_group, probability)` tables "
        "to loss/fairness functions, and paired clean/stressed tables to stress summaries. "
        "Carry the saved whole-original-predictor group ID; never reconstruct groups from "
        "a reduced candidate input set. Define undefined denominators explicitly.", "",
        "pilot.load_data validates partitions but reads the full processed/raw data and "
        "holdout identity metadata. safe2026_run.load_contract is a cleaner train-only "
        "precedent: verify candidate receipts/input hashes before joblib loading, filter "
        "source rows by membership before parsing predictors, restore saved feature/ID "
        "order, and validate exact saved probabilities. Neither loader was executed here. "
        "A later explicit assessment-only loader should expose only the authorized role, "
        "frozen models and aligned metadata, refuse fitting/tuning, and run only after "
        "selector choices and the judge protocol are fixed. Identity/hash verification "
        "must stay separate from a method that creates empirical predictions. No new "
        "holdout access, prediction or metric occurred in this audit.", "",
        "protocol.json records source identities and focused-test results; audit.json "
        "contains every synthetic output and independent score. VERIFIED.json binds all "
        "run files. Existing destinations are refused. No existing scientific files were edited.", ""]
    return "\n".join(lines)


def verify_run(destination):
    destination = Path(destination)
    receipt = json.loads((destination / "VERIFIED.json").read_text(encoding="utf-8"))
    actual = {path.relative_to(destination).as_posix(): sha256(path)
              for path in destination.rglob("*") if path.is_file() and path.name != "VERIFIED.json"}
    if receipt.get("status") != "verified" or receipt["artifact_hashes"] != actual:
        raise ValueError("Synthetic run receipt mismatch.")
    return receipt


def run(destination=DEFAULT_RUN):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    command = [sys.executable, "-m", "unittest", "discover", "-s",
               "research/model_selection_v2/mechanism_audit", "-p", "test_audit.py", "-v"]
    tests = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    write_json(destination / "focused_tests.json", {"command": command, "exit_code": tests.returncode,
                                                   "stdout": tests.stdout, "stderr": tests.stderr})
    if tests.returncode:
        raise ValueError("Synthetic tests failed; no verification receipt created.")
    audit = make_audit()
    if source_hashes() != before:
        raise ValueError("Input implementation/protocol changed during execution.")
    write_json(destination / "audit.json", audit)
    write_json(destination / "protocol.json", {"protocol": "research/model_selection_v2/mechanism_audit/PROTOCOL.md",
        "protocol_version": "synthetic_affine_mechanism_v1", "date": "2026-10-30", "source_hashes": before,
        "python": platform.python_version(), "versions": {name: version(name) for name in ("numpy", "scikit-learn")},
        "data_loaders_executed": [], "empirical_predictions_computed": False,
        "existing_implementation_source_hashes_unchanged": True})
    with (destination / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report_markdown(audit))
    write_json(destination / "VERIFIED.json", {"status": "verified", "artifact_hashes": {
        path.relative_to(destination).as_posix(): sha256(path) for path in destination.rglob("*") if path.is_file()}})
    verify_run(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--verify-run", type=Path)
    arguments = parser.parse_args()
    if bool(arguments.run_dir) == bool(arguments.verify_run):
        parser.error("Specify exactly one of --run-dir or --verify-run.")
    print(verify_run(arguments.verify_run)["status"] if arguments.verify_run else run(arguments.run_dir))
