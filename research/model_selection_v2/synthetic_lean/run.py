"""Run blinded timing, the frozen synthetic study, or a saved-results report."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout, suppress
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
import traceback


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
SETTINGS = ("R", "F")
REPEAT_COUNTS = (20, 30, 40, 50)
PRODUCTION_COUNTS = {"R": 150, "F": 300}
ORIGINAL_TIMING_RECEIPT = "d70d1e4fe3bf0185906c2dcc451d533140dd5320a6e450d06960f767ad16fa98"
TIMING_REPEATS = 3
WORKERS = 3
THREADS = 2
BORROWERS = 30_000
CONTROLS = 1_000
BUDGET_HOURS = 7.5
MAX_HOURS = 12
RESERVE = .20
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _source_hashes():
    selection = PACKAGE.parent
    files = [
        *(PACKAGE / name for name in (
            "__init__.py", "run.py", "protocol.md", "generators_spec.md",
            "generators.py", "pipeline.py", "control.py", "summaries.py",
        )),
        *(selection / name for name in (
            "__init__.py", "pilot.py", "safe2026.py",
            "step0_v1/__init__.py", "step0_v1/core.py",
            "step1_v1/__init__.py", "step1_v1/core.py", "step1_v1/runner.py",
            "confirmation_v1/__init__.py", "confirmation_v1/judge.py",
            "confirmation_v1/schema.py", "confirmation_v1/selection.py",
            "confirmation_v1/tables.py", "confirmation_v2/__init__.py",
            "confirmation_v2/arms.py", "confirmation_v2/judge.py",
            "confirmation_v2/reporting.py", "confirmation_v2_jobs/__init__.py",
            "confirmation_v2_jobs/artifacts.py", "confirmation_v2_jobs/collect.py",
            "confirmation_v2_jobs/names.py",
            "confirmation_v2_jobs/section_6_5.py",
            "judge_core/__init__.py", "judge_core/bootstrap.py", "judge_core/metrics.py",
            "judge_core/pareto.py", "judge_core/tables.py",
            "selection_exploration_v1/__init__.py",
            "selection_exploration_v1/protocol.py", "selection_exploration_v1/transforms.py",
        )),
        *(ROOT / name for name in (
            "article_safe/__init__.py", "article_safe/data.py", "article_safe/protocol.py",
            "src/baseline.py", "src/performance_evaluation.py", "src/robustness_protocol.py",
            "docs/frozen_inputs/2026-11-17_v2/version2_amendments.md",
            "docs/frozen_inputs/2026-11-18_v2_1/section_6_5.md",
        )),
    ]
    return {path.relative_to(ROOT).as_posix(): _hash(path) for path in files}


def _configuration():
    # On Windows the offline socket guard also blocks the local gethostname call.
    node = platform.node() or os.environ.get("COMPUTERNAME", "")
    machine = (node, platform.machine(), platform.platform(), os.cpu_count())
    return {
        "settings": list(SETTINGS),
        "repeat_counts": list(REPEAT_COUNTS),
        "timing_repeats_per_setting": TIMING_REPEATS,
        "workers": WORKERS,
        "threads_per_worker": THREADS,
        "borrowers_per_repeat": BORROWERS,
        "control_trials": CONTROLS,
        "standard_budget_hours": BUDGET_HOURS,
        "maximum_budget_hours": MAX_HOURS,
        "reserve_fraction": RESERVE,
        "machine_sha256": hashlib.sha256(repr(machine).encode()).hexdigest(),
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in (
                "joblib", "numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl",
            )},
        },
        "source_sha256": _source_hashes(),
    }


def _configure_threads():
    for name in THREAD_VARIABLES:
        os.environ[name] = str(THREADS)


def _json_bytes(value):
    return (json.dumps(value, allow_nan=False, indent=2) + "\n").encode("utf-8")


def _write_json(path, value):
    with path.open("xb") as output:
        output.write(_json_bytes(value))


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fresh_folder(path):
    path = Path(path).resolve()
    if any((parent / "COMPLETE.json").is_file() for parent in path.parents):
        raise ValueError("Choose a destination outside any completed output")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _artifact_hashes(folder):
    return {
        path.relative_to(folder).as_posix(): _hash(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path != folder / "COMPLETE.json"
    }


def _complete(folder, stage, deadline=None):
    receipt = {"stage": stage, "sha256": _artifact_hashes(folder)}
    if deadline is not None:
        _time_left(deadline)
    _write_json(folder / "COMPLETE.json", receipt)


def _verify(folder, stage):
    folder = Path(folder).resolve()
    if not (folder / "COMPLETE.json").is_file() or (folder / "FAILED.json").exists():
        raise ValueError(f"No completed {stage} run in {folder}")
    receipt = _read_json(folder / "COMPLETE.json")
    if receipt != {"stage": stage, "sha256": _artifact_hashes(folder)}:
        raise ValueError(f"Invalid or changed {stage} artifacts in {folder}")
    return folder


class _BudgetExceeded(TimeoutError):
    pass


def _time_left(deadline):
    remaining = deadline - perf_counter()
    if remaining <= 0:
        raise _BudgetExceeded("Combined timing and production reached the 12-hour limit")
    return remaining


@contextmanager
def _record_failure(folder, stage, started, prior_seconds=0):
    try:
        yield
    except (Exception, KeyboardInterrupt) as error:
        elapsed = perf_counter() - started
        with suppress(OSError):
            _write_json(folder / "FAILED.json", {
                "stage": stage,
                "error_type": type(error).__name__,
                "budget_exceeded": isinstance(error, _BudgetExceeded),
                "elapsed_seconds": elapsed,
                "combined_elapsed_seconds": prior_seconds + elapsed,
            })
        raise


def budget_decision(measurements):
    """Project both settings in whole batches, including the timing already spent."""
    elapsed = measurements["timing_elapsed_seconds"]
    setup = measurements["timing_setup_and_calibration_seconds"]
    additional = (
        measurements["production_setup_seconds"],
        measurements["control_seconds"],
        measurements["reporting_seconds_at_50"],
    )
    batches = measurements["batch_seconds"]
    if set(batches) != set(SETTINGS):
        raise ValueError("Timing requires one complete three-repeat batch per setting")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (elapsed, setup, *additional, *batches.values())):
        raise ValueError("All elapsed-time measurements must be finite and positive")
    if elapsed < setup + sum(batches.values()) + sum(additional):
        raise ValueError("Total timing elapsed must cover the measured stages")

    projections = {
        str(count): {
            "batches_per_setting": math.ceil(count / WORKERS),
            "hours_with_reserve": (
                (elapsed + sum(additional) + math.ceil(count / WORKERS) * sum(batches.values()))
                * (1 + RESERVE) / 3600
            ),
        }
        for count in REPEAT_COUNTS
    }
    repeats = next((count for count in reversed(REPEAT_COUNTS)
                    if projections[str(count)]["hours_with_reserve"] <= BUDGET_HOURS), None)
    limit = BUDGET_HOURS
    if repeats is None:
        limit = MAX_HOURS
        if projections[str(REPEAT_COUNTS[0])]["hours_with_reserve"] <= limit:
            repeats = REPEAT_COUNTS[0]
    return {
        "projections": projections,
        "repeats_per_setting": repeats,
        "budget_hours": limit,
        "stop": repeats is None,
    }


def _worker(setting, purpose, repeat, calibration):
    from threadpoolctl import threadpool_info, threadpool_limits
    from .pipeline import run_repeat

    with threadpool_limits(limits=THREADS):
        if any(pool["num_threads"] > THREADS for pool in threadpool_info()):
            raise RuntimeError("A numerical library exceeded the worker thread cap")
        return run_repeat(setting, purpose, repeat, calibration, n=BORROWERS)


def _blinded(label, target, *args):
    # Worker messages and errors can contain timing-stage scientific results.
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                return target(*args)
    except (Exception, SystemExit) as error:
        raise RuntimeError(f"{label} failed ({type(error).__name__})") from None


def _timing_worker(setting, purpose, repeat, calibration):
    return _blinded(f"Timing repeat {setting}/{repeat}", _worker, setting, purpose, repeat, calibration)


def _process_tasks(target, tasks, connection, scratch):
    _configure_threads()
    tempfile.tempdir = scratch
    with connection:
        for index, args in tasks:
            try:
                connection.send((index, target(*args), None))
            except BaseException:
                connection.send((index, None, traceback.format_exc()))
                return


def _run_tasks(target, arguments, deadline, on_result=None):
    """Run ordered tasks in spawned workers that can be stopped on Windows."""
    context = multiprocessing.get_context("spawn")
    results = [None] * len(arguments)
    tasks = list(enumerate(arguments))
    worker_count = min(WORKERS, len(tasks))
    children = {}
    with tempfile.TemporaryDirectory(prefix="synthetic_lean_workers_") as scratch:
        try:
            for i in range(worker_count):
                _time_left(deadline)
                assigned = tasks[i::worker_count]
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=_process_tasks, args=(target, assigned, sender, scratch))
                with sender:
                    try:
                        process.start()
                    except BaseException:
                        receiver.close()
                        raise
                children[receiver] = process

            pending = {receiver: len(tasks[i::worker_count]) for i, receiver in enumerate(children)}
            while pending:
                ready = wait(pending, timeout=_time_left(deadline))
                _time_left(deadline)
                for receiver in ready:
                    try:
                        index, result, error = receiver.recv()
                    except EOFError as failure:
                        raise RuntimeError("A synthetic worker exited before returning its results") from failure
                    if error is not None:
                        raise RuntimeError(f"A synthetic worker failed\n{error}")
                    if on_result is not None:
                        on_result(index, result)
                    results[index] = result
                    pending[receiver] -= 1
                    if pending[receiver] == 0:
                        process = children[receiver]
                        process.join(_time_left(deadline))
                        _time_left(deadline)
                        if process.exitcode != 0:
                            raise RuntimeError("A synthetic worker failed to exit normally")
                        del pending[receiver]
        finally:
            for receiver, process in children.items():
                receiver.close()
                if process.is_alive():
                    process.kill()
            for process in children.values():
                process.join()
                process.close()
    return results


def _task(target, deadline, *args):
    return _run_tasks(target, (args,), deadline)[0]


def _batch(setting, purpose, repeats, calibration, deadline, folder=None):
    worker = _timing_worker if purpose == "timing" else _worker
    completed = 0

    def accept(repeat, result):
        nonlocal completed
        if (result["setting"], result["purpose"], result["repeat"]) != (setting, purpose, repeat):
            raise ValueError("Worker returned a mismatched repeat identity")
        if folder is not None:
            _write_json(folder / f"{setting}_{repeat:02d}.json", result)
            completed += 1
            print(f"{setting} completed {completed}/{repeats}", flush=True)

    return _run_tasks(worker, [(setting, purpose, repeat, calibration) for repeat in range(repeats)],
                      deadline, on_result=accept)


def _prepare():
    from threadpoolctl import threadpool_limits
    from .generators import calibrate, r_reference_curve

    with threadpool_limits(limits=THREADS):
        calibrations = {setting: calibrate(setting) for setting in SETTINGS}
        return {
            "calibration": {setting: value.as_dict() for setting, value in calibrations.items()},
            "R_truth_reference": r_reference_curve(calibrations["R"]),
        }


def _load_preparation(path):
    from .generators import Calibration

    preparation = _read_json(path)
    calibrations = {
        setting: Calibration.from_dict(preparation["calibration"][setting]) for setting in SETTINGS
    }
    return preparation, calibrations


def _control(purpose):
    from threadpoolctl import threadpool_limits
    from .control import control_trials

    with threadpool_limits(limits=THREADS):
        return control_trials(range(CONTROLS), purpose=purpose)


def _report(results, controls):
    from .control import summarize_control
    from .summaries import summarize

    return {
        "R": summarize(results["R"]),
        "F": summarize(results["F"]),
        "control": summarize_control(controls),
    }


def _reporting_probe(results, controls, folder):
    # Size the reporting work at the full 50 repeats using timing results in memory.
    # Only zero bytes ever reach the temporary file used to measure write overhead.
    count = REPEAT_COUNTS[-1]
    sample = {}
    for setting in SETTINGS:
        sample[setting] = [
            dict(results[setting][repeat % TIMING_REPEATS], repeat=repeat,
                 dataset_id=f"timing-report-probe:{setting}:{repeat}")
            for repeat in range(count)
        ]
    report = _report(sample, controls)
    artifacts = [
        _json_bytes(result) for setting in SETTINGS for result in sample[setting]
    ] + [_json_bytes(controls), _json_bytes(report)]
    with tempfile.TemporaryFile(dir=folder) as output:
        for payload in artifacts:
            output.write(bytes(len(payload)))
        output.flush()


def timing(output):
    started = perf_counter()
    deadline = started + MAX_HOURS * 3600
    folder = _fresh_folder(output)
    with _record_failure(folder, "timing", started):
        configuration = _configuration()
        stage_started = perf_counter()
        _write_json(folder / "preparation.json", _task(_prepare, deadline))
        setup_seconds = perf_counter() - stage_started

        stage_started = perf_counter()
        _configuration()
        _, calibrations = _load_preparation(folder / "preparation.json")
        production_setup_seconds = perf_counter() - stage_started

        stages = {}
        results = {}
        for setting in SETTINGS:
            print(f"Timing {setting} with {WORKERS} workers", flush=True)
            stage_started = perf_counter()
            results[setting] = _batch(setting, "timing", TIMING_REPEATS, calibrations[setting], deadline)
            stages[setting] = perf_counter() - stage_started

        stage_started = perf_counter()
        controls = _task(_blinded, deadline, "Timing control", _control, "timing")
        control_seconds = perf_counter() - stage_started

        stage_started = perf_counter()
        _task(_blinded, deadline, "Timing report probe", _reporting_probe, results, controls, folder)
        reporting_seconds = perf_counter() - stage_started
        del results, controls

        if configuration["source_sha256"] != _source_hashes():
            raise ValueError("Study sources changed during timing")
        measurements = {
            "timing_elapsed_seconds": perf_counter() - started,
            "timing_setup_and_calibration_seconds": setup_seconds,
            "production_setup_seconds": production_setup_seconds,
            "batch_seconds": stages,
            "control_seconds": control_seconds,
            "reporting_seconds_at_50": reporting_seconds,
        }
        decision = budget_decision(measurements)
        _write_json(folder / "timing.json", {
            "configuration": configuration, "measurements": measurements, "decision": decision,
        })
        _complete(folder, "timing", deadline)
    print(json.dumps(decision), flush=True)
    return decision


def _accepted_timing(path):
    folder = _verify(path, "timing")
    receipt = _read_json(folder / "timing.json")
    expected = _configuration()
    if _hash(folder / "COMPLETE.json") == ORIGINAL_TIMING_RECEIPT:
        # Keep the authenticated timing receipt tied to its original driver and amendment.
        original_sources = dict(expected["source_sha256"])
        original_sources["research/model_selection_v2/synthetic_lean/run.py"] = (
            "8f7c4ac6f43fe94938e401bf818068b9bbd1f1b53b4578b38209720af9cc043b"
        )
        original_sources["docs/frozen_inputs/2026-11-17_v2/version2_amendments.md"] = (
            "ba2b756c92da2d666cf9c5a51c3da393e09bf027b41f92ea114b3dfd5a0455a2"
        )
        expected = dict(expected, source_sha256=original_sources)
    if receipt["configuration"] != expected:
        raise ValueError("Timing configuration differs from the current code, host, or software")
    decision = budget_decision(receipt["measurements"])
    if decision != receipt["decision"] or decision["stop"]:
        raise ValueError("A valid within-budget timing decision is required for production")
    return folder, receipt


def production(output, timing_path, freeze_reference):
    started = perf_counter()
    if not freeze_reference.strip():
        raise ValueError("Supply the public freeze reference for the specification, code, and timing")
    timing_folder, timing_receipt = _accepted_timing(timing_path)
    production_sources = _configuration()["source_sha256"]
    measurements = timing_receipt["measurements"]
    prior_seconds = measurements["timing_elapsed_seconds"]
    batch_seconds = sum(math.ceil(count / WORKERS) * measurements["batch_seconds"][setting]
                        for setting, count in PRODUCTION_COUNTS.items())
    reporting_seconds = measurements["reporting_seconds_at_50"] * sum(PRODUCTION_COUNTS.values()) / 100
    projected = (prior_seconds + measurements["production_setup_seconds"] + batch_seconds
                 + measurements["control_seconds"] + reporting_seconds) * (1 + RESERVE)
    if projected > BUDGET_HOURS * 3600:
        raise ValueError("Amended production counts exceed the registered timing budget")
    deadline = started + MAX_HOURS * 3600 - prior_seconds
    folder = _fresh_folder(output)
    with _record_failure(folder, "production", started, prior_seconds):
        _time_left(deadline)
        preparation, calibrations = _load_preparation(timing_folder / "preparation.json")
        _write_json(folder / "inputs.json", {
            "public_freeze_reference": freeze_reference.strip(),
            "timing_receipt_sha256": _hash(timing_folder / "COMPLETE.json"),
            "timing": timing_receipt,
            "production_counts": PRODUCTION_COUNTS,
            "production_source_sha256": production_sources,
            "projected_combined_seconds_with_reserve": projected,
        })
        _write_json(folder / "preparation.json", preparation)
        repeat_folder = folder / "repeats"
        repeat_folder.mkdir()
        results = {
            setting: _batch(setting, "production", PRODUCTION_COUNTS[setting], calibrations[setting], deadline, repeat_folder)
            for setting in SETTINGS
        }
        controls = _task(_control, deadline, "control")
        _write_json(folder / "control_trials.json", controls)
        _write_json(folder / "report.json", _task(_report, deadline, results, controls))
        if production_sources != _source_hashes():
            raise ValueError("Study sources changed during production")
        elapsed = perf_counter() - started
        _write_json(folder / "runtime.json", {
            "production_elapsed_seconds": elapsed,
            "combined_elapsed_seconds": prior_seconds + elapsed,
            "maximum_combined_hours": MAX_HOURS,
        })
        _complete(folder, "production", deadline)
    return folder


def report(output, production_path):
    started = perf_counter()
    source = _verify(production_path, "production")
    inputs = _read_json(source / "inputs.json")
    sources = inputs.get("production_source_sha256", inputs["timing"]["configuration"]["source_sha256"])
    if sources != _source_hashes():
        raise ValueError("Reporting code or protocol differs from the production run")
    counts = inputs.get("production_counts", dict.fromkeys(SETTINGS, inputs["timing"]["decision"]["repeats_per_setting"]))
    results = {
        setting: [_read_json(source / "repeats" / f"{setting}_{repeat:02d}.json")
                  for repeat in range(counts[setting])]
        for setting in SETTINGS
    }
    controls = _read_json(source / "control_trials.json")
    result = _report(results, controls)
    if result != _read_json(source / "report.json"):
        raise ValueError("Recomputed report differs from the saved report")
    folder = _fresh_folder(output)
    with _record_failure(folder, "report", started):
        _write_json(folder / "report.json", result)
        _write_json(folder / "inputs.json", {"production_receipt_sha256": _hash(source / "COMPLETE.json")})
        _complete(folder, "report")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="stage", required=True)
    timing_parser = subcommands.add_parser("timing", help="Time six discarded fitted repeats")
    production_parser = subcommands.add_parser("production", help="Run the frozen study")
    report_parser = subcommands.add_parser("report", help="Replay a completed study's report")
    for command in (timing_parser, production_parser, report_parser):
        command.add_argument("--output", required=True, type=Path)
    production_parser.add_argument("--timing", required=True, type=Path)
    production_parser.add_argument("--freeze-reference", required=True)
    report_parser.add_argument("--production", required=True, type=Path)
    args = parser.parse_args(argv)
    _configure_threads()
    if args.stage == "timing":
        return 2 if timing(args.output)["stop"] else 0
    if args.stage == "production":
        print(production(args.output, args.timing, args.freeze_reference), flush=True)
    else:
        print(report(args.output, args.production), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
