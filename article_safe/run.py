"""Checkpoint entry point. Model-fitting stages are added after milestone review."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import platform

from article_safe import __version__
from article_safe.data import check_evaluation_artifacts, load_development_data
from article_safe.protocol import ROOT, build_protocol, canonical_digest, sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_directory(run_id):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("Run IDs must be 1-80 letters/digits, underscores or hyphens.")
    if run_id.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError("A Windows device name cannot be a run ID.")
    base = (ROOT / "article_safe/runs").resolve()
    if not base.is_relative_to((ROOT / "article_safe").resolve()):
        raise ValueError("Run base escapes the isolated package directory.")
    path = base / run_id
    if path.resolve().parent != base or path.is_symlink():
        raise ValueError("Run destination escapes the isolated run directory.")
    if path.exists():
        raise FileExistsError(f"Run already exists; use a new ID: {path}")
    return path


def package_reference():
    """Check the installed SAFE against existing inspected-source provenance."""
    from src.safe_adapter import provenance
    from safeaipackage import core
    result = provenance()
    package_root = Path(core.__file__).parent
    inventory = {}
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        inventory[path.relative_to(package_root).as_posix()] = {
            "sha256": sha256(path),
            "functions": [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))],
        }
    result.update(repository="https://github.com/GolnooshBabaei/safeaipackage",
                  role="reference and previously implemented native-metric baseline",
                  installed_python_inventory=inventory,
                  direct_metric_calls_in_this_checkpoint=False,
                  article_procedures_supplied_by_installed_python_package=[],
                  distinction="Installed Python functions implement rank graduation; the article's DeLong/LR selection and Gini/KS require the separate specified procedures.")
    return result


def prepare(run_id):
    """Freeze source/configuration and audit data integrity without fitting models."""
    path = run_directory(run_id)
    before = check_evaluation_artifacts()
    protocol = build_protocol()
    source_files = sorted((ROOT / "article_safe").glob("*.py"))
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files}
    # Check all prerequisites before creating the new run directory.
    development = load_development_data(protocol)
    reference = package_reference()
    path.mkdir(parents=True)
    write_json(path / "protocol.json", protocol)
    write_json(path / "package_reference.json", reference)
    development.assignment_table().to_csv(path / "development_assignments.csv", index=False)
    summary = development.structural_summary()
    summary.update(holdout_records=5996, crossing_original_train_holdout_groups=0,
                   stage="prepare", credit_models_fitted=0, credit_predictions_generated=0)
    write_json(path / "data_audit.json", summary)
    input_files = [*sorted((ROOT / "data/processed").glob("*")), *sorted((ROOT / "models").glob("baseline_*")),
                   ROOT / "data/raw/default of credit card clients.xls", ROOT / "requirements.txt",
                   ROOT / "requirements-fairness.txt", ROOT / "requirements-explainability.txt",
                   ROOT / "article_safe/requirements.txt"]
    input_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in input_files if p.is_file()}
    outputs = {p.name: sha256(p) for p in sorted(path.iterdir()) if p.is_file()}
    after = check_evaluation_artifacts()
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "package_version": __version__, "run_id": run_id, "stage": "prepare",
        "checkpoint": "1 of 4 work packages; waiting for user review",
        "protocol_sha256": canonical_digest(protocol), "source_hashes": source_hashes,
        "input_hashes": input_hashes, "output_hashes": outputs, "prior_artifact_checks_before": before,
        "prior_artifact_checks_after": after, "credit_models_fitted": 0,
        "model_selection_executed": False, "heldout_performance_evaluated": False,
    }
    write_json(path / "run_manifest.json", manifest)
    print(json.dumps({"run_directory": str(path), "data_audit": summary,
                      "prior_evaluation_hash_checks": after["checks"]}, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "select", "evaluate"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run_id)
    elif args.stage == "select":
        from article_safe.paths import select
        select(args.run_id)
    else:
        from article_safe.evaluation import evaluate
        evaluate(args.run_id)


if __name__ == "__main__":
    main()