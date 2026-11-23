"""Manually launch the registered Brazil jobs on the local study machine."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
TAG = "prereg-v2-2026-11"
PUBLIC = "research/model_selection_v2/prereg_v2_manifest.json"
PRIVATE = "docs/frozen_inputs/2026-11-23_prereg/private_manifest.json"
DATASET = "brazil_pakdd2010"
WORKER = "research.model_selection_v2.confirmation_v2_jobs.registered"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_freeze():
    ref = f"refs/tags/{TAG}"
    tagged = subprocess.check_output(["git", "show", f"{ref}:{PUBLIC}"], cwd=ROOT)
    if tagged != (ROOT / PUBLIC).read_bytes():
        raise ValueError("Local public manifest differs from the tagged manifest")
    manifest = json.loads(tagged)
    if digest(ROOT / PRIVATE) != manifest["private_manifest_sha256"]:
        raise ValueError("Private protocol and timing manifest differs from the registration")
    private_files = json.loads((ROOT / PRIVATE).read_bytes())
    for name, expected in {**manifest["code_sha256"], **private_files}.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Registered input changed: {name}")
    commit = subprocess.check_output(["git", "rev-parse", ref + "^{commit}"], cwd=ROOT, text=True).strip()
    return {"tag": TAG, "commit": commit, "manifest_sha256": digest(ROOT / PUBLIC)}


def run_worker(spec):
    import joblib
    from threadpoolctl import threadpool_limits
    from ..confirmation_v1 import brazil
    from ..confirmation_v2.timing import FAMILIES
    from .artifacts import read_json, sha, verify
    from .jobs import check_threads, fit_and_score_A, score_B

    repeat = spec["repeat"]
    if type(repeat) is not int or repeat not in range(50):
        raise ValueError("Expected a registered repeat from 0 to 49")
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        check_threads()
        if spec["job"] == "A":
            data = brazil.load(ROOT, include_outcome=True)
            source = dict(data=data, purpose="registered", families=FAMILIES,
                          dataset_id=DATASET, data_seed=None)
            fit_and_score_A(source, Path(spec["output"]), repeat, started)
        elif spec["job"] == "B":
            source = Path(spec["source_A"])
            receipt = read_json(source / "COMPLETE.json")
            if (receipt["job"], receipt["repeat"], receipt["complete"]) != ("A", repeat, True):
                raise ValueError("Wrong upstream Job A receipt")
            if sha(source / "artifact_manifest.json") != receipt["artifact_manifest_sha256"]:
                raise ValueError("Job A manifest differs from its receipt")
            verify(source)
            payload = joblib.load(source / "inputs.joblib")
            if (payload["dataset_id"], payload["purpose"], payload["repeat"]) != (DATASET, "registered", repeat):
                raise ValueError("Wrong upstream study or repeat")
            score_B(source, Path(spec["output"]), payload, started)
        else:
            raise ValueError("Expected Job A or B")


def run_brazil(output, freeze):
    from .artifacts import private_folder, write_json
    from .parallel import run_batch
    from .section_6_5 import collect

    output = private_folder(output.resolve())
    write_json(output / "registration.json", dict(freeze, dataset=DATASET, repeats=50,
               workers=3, threads_per_worker=2, hours_per_job=12))
    for job in ("A", "B"):
        specs = []
        folders = []
        for repeat in range(50):
            dest = output / f"{job}_{repeat}"
            spec = {"job": job, "output": str(dest), "repeat": repeat}
            if job == "B":
                spec["source_A"] = str(output / f"A_{repeat}")
            specs.append(spec)
            folders.append(dest)
        run_batch(specs, output / f"workers_{job}", processes=3, limit_hours=12,
                  progress=True, worker_module=WORKER)
        collect(job, folders, output / f"report_{job}")
    write_json(output / "COMPLETE.json", dict(freeze, jobs=["A", "B"], repeats_per_job=50))


def main():
    if not getattr(sys, "_credit_risk_jobs_offline", False):
        raise RuntimeError("Use the local offline launcher")
    # Workers are started by the existing capped subprocess controller.
    if len(sys.argv) == 2 and Path(sys.argv[1]).suffix == ".json":
        run_worker(json.loads(Path(sys.argv[1]).read_bytes()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "brazil", "synthetic"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    freeze = verify_freeze()
    from ..synthetic_lean import run as synthetic
    timing = ROOT / "runs/model_selection_v2/synthetic_lean_timing_01"
    if args.stage == "check":
        synthetic._accepted_timing(timing)
        print(json.dumps(freeze))
    elif args.stage == "brazil":
        run_brazil(args.output or ROOT / "runs/model_selection_v2/brazil_registered_01", freeze)
    else:
        synthetic._configure_threads()
        output = args.output or ROOT / "runs/model_selection_v2/synthetic_registered_01"
        url = f"https://github.com/stanescu2022-png/credit-risk-fairness/commit/{freeze['commit']}"
        synthetic.production(output, timing, url)
        synthetic.report(output.with_name(output.name + "_replay"), output)


if __name__ == "__main__":
    main()
