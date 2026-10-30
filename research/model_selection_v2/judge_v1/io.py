"""Exclusive artifact writes and read-only integrity checks, without regeneration."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported artifact type {type(item)}")
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=encode)
        stream.write("\n")


def copy_exact(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(Path(source).read_bytes())


def write_npz(path, **arrays):
    require(all(np.asarray(array).dtype.kind != "O" for array in arrays.values()), "NPZ object/pickle arrays are forbidden.")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def write_csv(path, frame):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def artifact_hashes(folder):
    folder = Path(folder)
    return {path.relative_to(folder).as_posix(): sha256(path) for path in folder.rglob("*")
            if path.is_file() and path != folder / "VERIFIED.json"}


def verify_artifacts(folder):
    folder = Path(folder)
    marker = read_json(folder / "VERIFIED.json")
    require(marker.get("status") == "verified" and marker["artifact_hashes"] == artifact_hashes(folder), "Artifact receipt mismatch.")
    return marker


def seal(folder):
    write_json(Path(folder) / "VERIFIED.json", {"status": "verified", "verified_at_utc": now(), "artifact_hashes": artifact_hashes(folder)})
    return verify_artifacts(folder)
