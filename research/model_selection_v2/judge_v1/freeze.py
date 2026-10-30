"""Immutable development-choice freeze; no evaluation-data reads or predictions."""
from importlib.metadata import version
import json
from pathlib import Path
import platform

from research.model_selection_v2.selection_exploration_v1.analysis import selector_decisions
from research.model_selection_v2.selection_exploration_v1.protocol import settings as exploration_settings
from .protocol import (CANDIDATES, EXPLORATION_RUN, EXPLORATION_SPECIFICATION, HERE, METHODS, PILOT_RUN,
                       ROOT, SAFE_RUN, SPECIFICATION, settings)
from .io import copy_exact, now, read_json, require, seal, sha256, verify_artifacts, write_json

JUDGE_SPEC_RELATIVE = "research/model_selection_v2/judge_v1_specification.md"
EXPLORATION_SPEC_RELATIVE = "research/model_selection_v2/exploration_v1_specification.md"
RUN_RELATIVES = ("runs/model_selection_v2/pilot_candidates_2026-10-26", "runs/model_selection_v2/safe2026_swapping_2026-10-26",
                 "runs/model_selection_v2/selection_exploration_v1_2026-10-30")
DATA_RELATIVES = {"data/processed/taiwan_credit_clean.csv", "data/processed/test_metadata.csv", "data/processed/train_metadata.csv",
                  "data/processed/split_metadata.csv", "data/processed/preparation_manifest.json"}


def required_sources():
    project_root = HERE.parents[2]
    paths = [*HERE.glob("*.py"), HERE / "README.md", *(HERE.parent / "judge_core").glob("*.py"),
             *(HERE.parent / "selection_exploration_v1").glob("*.py"), project_root / "src/robustness_protocol.py",
             project_root / "src/baseline.py", project_root / "src/performance_evaluation.py"]
    return {path.relative_to(project_root).as_posix() for path in paths} | {JUDGE_SPEC_RELATIVE, EXPLORATION_SPEC_RELATIVE}


def environment_versions():
    return {"python": platform.python_version(), **{name: version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "threadpoolctl")}}


def canonical(value):
    return json.loads(json.dumps(value))


def merge_bindings(*mappings):
    result = {}
    for mapping in mappings:
        for key, value in mapping.items():
            require(key not in result or result[key] == value, f"Conflicting identity binding: {key}")
            result[key] = value
    return result


def validate_choices(reference, sensitivity):
    for label, choices in (("q=.10 reference", reference), ("q=.20 diagnostic", sensitivity)):
        require(set(choices) == set(METHODS), f"Incomplete {label} method/choice mapping.")
        for method, choice in choices.items():
            require(choice.get("status") == "selected" and choice.get("selected_id") in CANDIDATES,
                    f"Unresolved frozen choice: {label} {method}")
            tied = choice.get("tied_ids")
            require(isinstance(tied, list) and bool(tied) and len(set(tied)) == len(tied) and
                    set(tied) <= set(CANDIDATES) and choice["selected_id"] == tied[0], "Invalid frozen choice/tie identities.")
            require(tied == [candidate for candidate in CANDIDATES if candidate in tied], "Frozen choice ties must follow pilot order.")


def collect_freeze_payload():
    """Read only development artifacts, source files and model bytes, never data."""
    runs = {"pilot": PILOT_RUN, "safe": SAFE_RUN, "exploration": EXPLORATION_RUN}
    receipts = {name: verify_artifacts(path) for name, path in runs.items()}
    manifests = {name: read_json(path / "run_manifest.json") for name, path in runs.items()}
    reference = read_json(EXPLORATION_RUN / "q0.10/selectors.json")["decisions"]
    sensitivity = read_json(EXPLORATION_RUN / "q0.20/selectors.json")["decisions"]
    validate_choices(reference, sensitivity)
    scores = read_json(SAFE_RUN / "integrated_scores.json")
    # Verify the choice definitions against saved DEVELOPMENT summaries only.
    for qname, choices in (("q0.10", reference), ("q0.20", sensitivity)):
        reconstructed = selector_decisions(read_json(EXPLORATION_RUN / qname / "scenario_matrix.json"), scores, CANDIDATES)
        require(reconstructed["decisions"] == choices, "Saved development selector definitions/choices mismatch.")
    own_paths = [*HERE.glob("*.py"), HERE / "README.md", SPECIFICATION, EXPLORATION_SPECIFICATION]
    own = {path.relative_to(ROOT).as_posix(): sha256(path) for path in own_paths}
    sources = merge_bindings(own, manifests["pilot"]["implementation_hashes"], manifests["safe"]["source_hashes"],
                             manifests["exploration"]["source_hashes"])
    for relative, expected in sources.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.parts[len(ROOT.parts)] not in {"data", "models", "runs"},
                "Source bindings cannot cause evaluation-data reads during freeze.")
        require(sha256(path) == expected, f"Bound source changed: {relative}")
    model_hashes = {f"runs/model_selection_v2/pilot_candidates_2026-10-26/models/{candidate}.joblib":
                    receipts["pilot"]["artifact_hashes"][f"models/{candidate}.joblib"] for candidate in CANDIDATES}
    # These are recorded hashes only: actual data files are NOT opened here.
    data_bindings = merge_bindings(*(manifest.get("input_hashes", {}) for manifest in manifests.values()))
    receipt_hashes = {path.relative_to(ROOT).as_posix()+"/VERIFIED.json": sha256(path / "VERIFIED.json") for path in runs.values()}
    return {"status": "frozen", "frozen_at_utc": now(), "primary_evaluation": None,
            "judge_settings": canonical(settings()), "exploration_settings": canonical(exploration_settings()), "candidate_order": CANDIDATES,
            "reference_choices": reference, "q20_diagnostic_choices": sensitivity,
            "reference_method_to_candidate": {name: choice["selected_id"] for name, choice in reference.items()},
            "q20_diagnostic_method_to_candidate": {name: choice["selected_id"] for name, choice in sensitivity.items()},
            "source_hashes": sources, "model_hashes": model_hashes, "recorded_data_input_hashes": data_bindings,
            "source_run_receipt_hashes": receipt_hashes,
            "source_run_artifact_hashes": {path.relative_to(ROOT).as_posix(): receipts[name]["artifact_hashes"] for name, path in runs.items()},
            "environment": environment_versions(), "judge_specification_sha256": sha256(SPECIFICATION),
            "exploration_specification_sha256": sha256(EXPLORATION_SPECIFICATION),
            "evaluation_data_read": False, "models_loaded_or_predicted": False,
            "evidence_scope": "historically inspected out-of-sample exploration; not pristine validation"}


def create_freeze(destination):
    destination = Path(destination)
    require(not destination.exists(), "Freeze destination already exists; immutable freezes cannot be overwritten.")
    payload = collect_freeze_payload()
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "FREEZE.json", payload)
    for relative in payload["source_hashes"]:
        copy_exact(ROOT / relative, destination / "source_snapshots" / relative)
    for relative in payload["source_run_receipt_hashes"]:
        copy_exact(ROOT / relative, destination / "source_run_receipts" / relative)
    seal(destination)
    verify_freeze(destination / "FREEZE.json")
    return destination / "FREEZE.json"


def verify_current_bindings(payload, *, root=None):
    root = Path(ROOT if root is None else root)
    require(payload.get("status") == "frozen" and payload.get("primary_evaluation") is None, "Invalid freeze status/primary evaluation.")
    require(tuple(payload.get("candidate_order", ())) == CANDIDATES, "Frozen candidate order changed.")
    require(required_sources() <= set(payload["source_hashes"]), "Freeze omits a required source/specification binding.")
    expected_models = {RUN_RELATIVES[0]+f"/models/{candidate}.joblib" for candidate in CANDIDATES}
    require(set(payload["model_hashes"]) == expected_models, "Freeze must bind all four exact pilot models.")
    require(set(payload["source_run_receipt_hashes"]) == {name+"/VERIFIED.json" for name in RUN_RELATIVES} and
            set(payload["source_run_artifact_hashes"]) == set(RUN_RELATIVES), "Freeze omits a required source run receipt.")
    require(DATA_RELATIVES <= set(payload["recorded_data_input_hashes"]), "Freeze omits required recorded data-input identities.")
    require(payload["judge_specification_sha256"] == payload["source_hashes"][JUDGE_SPEC_RELATIVE] and
            payload["exploration_specification_sha256"] == payload["source_hashes"][EXPLORATION_SPEC_RELATIVE], "Frozen specification hash annotations mismatch.")
    validate_choices(payload["reference_choices"], payload["q20_diagnostic_choices"])
    require(payload["reference_method_to_candidate"] == {name: choice["selected_id"] for name, choice in payload["reference_choices"].items()},
            "Frozen reference choice mapping changed.")
    require(payload["q20_diagnostic_method_to_candidate"] == {name: choice["selected_id"] for name, choice in payload["q20_diagnostic_choices"].items()},
            "Frozen sensitivity diagnostic choice mapping changed.")
    require(payload["judge_settings"] == canonical(settings()) and payload["exploration_settings"] == canonical(exploration_settings()), "Frozen method/judge settings changed.")
    require(payload["environment"] == environment_versions(), "Frozen software versions changed.")
    for section in ("source_hashes", "model_hashes", "source_run_receipt_hashes"):
        for relative, expected in payload[section].items():
            path = (root / relative).resolve()
            require(path.is_relative_to(root.resolve()) and sha256(path) == expected, f"Freeze {section} mismatch: {relative}")
    for relative, expected in payload["source_run_artifact_hashes"].items():
        receipt = verify_artifacts(root / relative)
        require(receipt["artifact_hashes"] == expected, "Frozen source run artifacts/choices changed.")
    return payload


def verify_freeze(receipt_path):
    receipt_path = Path(receipt_path)
    require(receipt_path.name == "FREEZE.json", "Supply the explicit FREEZE.json receipt.")
    verify_artifacts(receipt_path.parent)
    return verify_current_bindings(read_json(receipt_path))
