"""Strict artifacts and stage identity checks; existing runs are never overwritten."""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import numpy as np
from article_safe.protocol import ROOT, canonical_digest, sha256


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def existing_run(run_id):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run ID.")
    base = (ROOT / "article_safe/runs").resolve()
    path = base / run_id
    if not base.is_relative_to((ROOT / "article_safe").resolve()) or path.resolve().parent != base or path.is_symlink():
        raise ValueError("Run path escapes the isolated package.")
    if not path.is_dir():
        raise FileNotFoundError("Prepare the run before selecting or evaluating.")
    return path


def verify_preparation(path):
    manifest = json.loads((path / "run_manifest.json").read_text())
    protocol = json.loads((path / "protocol.json").read_text())
    if canonical_digest(protocol) != manifest["protocol_sha256"]:
        raise ValueError("The prepared protocol changed.")
    for category in ("source_hashes", "input_hashes"):
        for relative, expected in manifest[category].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Prepared {category} identity changed: {relative}")
    for relative, expected in manifest["output_hashes"].items():
        if sha256(path / relative) != expected:
            raise ValueError(f"Prepared output identity changed: {relative}")
    return protocol


def stage_manifest(stage_dir, stage, protocol, extra=None):
    output_hashes = {p.relative_to(stage_dir).as_posix(): sha256(p) for p in stage_dir.rglob("*") if p.is_file()}
    source_paths = list((ROOT / "article_safe").glob("*.py"))
    if stage == "selection":
        used = {"__init__.py", "protocol.py", "data.py", "inference.py", "selection.py", "models.py", "explainability.py", "artifacts.py", "paths.py", "run.py"}
        source_paths = [p for p in source_paths if p.name in used]
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_paths}
    manifest = {"stage": stage, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": canonical_digest(protocol), "source_hashes": source_hashes,
                "output_hashes": output_hashes, **(extra or {})}
    write_json(stage_dir / "stage_manifest.json", manifest)
    return manifest


def verify_stage(stage_dir, protocol):
    manifest = json.loads((stage_dir / "stage_manifest.json").read_text())
    if manifest["protocol_sha256"] != canonical_digest(protocol):
        raise ValueError("Stage protocol identity changed.")
    for relative, expected in manifest["output_hashes"].items():
        if sha256(stage_dir / relative) != expected:
            raise ValueError(f"Frozen stage result changed: {relative}")
    # Preserve the exact selection decisions; report code additions separately.
    for relative, expected in manifest["source_hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Code used by the completed stage changed: {relative}")
    return manifest