"""Driver tests with mocked scientific work and lightweight spawned workers."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter, sleep
import unittest
from unittest.mock import patch

from . import run as subject


ARMS = ("safe", "plain_loss", "direct_small_swap_brier", "selector2_before_check", "selector2")
GROUPS = ("all", "female", "male")


def _repeat(setting, purpose, index, calibration):
    conditions = ("clean", "J1", "J3") if setting == "R" else ("clean", "J1")
    choices = dict.fromkeys(ARMS, "logistic")
    result = {
        "setting": setting, "purpose": purpose, "repeat": index,
        "dataset_id": f"fixture:{purpose}:{setting}:{index}",
        "choices": choices,
        "judge_truth": {condition: {"logistic": dict.fromkeys(GROUPS, .1)} for condition in conditions},
    }
    if setting == "F":
        result["gate"] = {
            "started_at_safe": True, "first_attempt": None, "later_attempts": [],
            "final_truth_delta": dict.fromkeys(GROUPS[1:], 0.0), "final_harmful": False,
            "prevented_harm": False, "direct_harmful_groups": [], "direct_material_harm_groups": [],
        }
    return result


def _spawned_fixture(index):
    return {
        "index": index, "pid": os.getpid(),
        "threads": tuple(os.environ[name] for name in subject.THREAD_VARIABLES),
        "scratch": tempfile.gettempdir(),
    }


def _slow_fixture(folder, index):
    scratch = Path(tempfile.gettempdir())
    (scratch / f"unfinished-{index}.txt").write_text("fixture", encoding="utf-8")
    (folder / f"started-{index}.txt").write_text(str(scratch), encoding="utf-8")
    sleep(30)


def _abrupt_fixture():
    os._exit(7)


def _measurements(batch_total):
    return {
        "timing_elapsed_seconds": batch_total + 100,
        "timing_setup_and_calibration_seconds": 10,
        "production_setup_seconds": 11,
        "batch_seconds": {"R": batch_total * .6, "F": batch_total * .4},
        "control_seconds": 12,
        "reporting_seconds_at_50": 13,
    }


class BudgetTests(unittest.TestCase):
    def test_standard_and_fallback_counts(self):
        for batch_total, count, budget in (
            (1200, 50, 7.5), (1300, 40, 7.5), (1600, 30, 7.5),
            (2250, 20, 7.5), (3000, 20, 12), (4600, None, 12),
        ):
            with self.subTest(batch_total=batch_total):
                decision = subject.budget_decision(_measurements(batch_total))
                self.assertEqual(decision["repeats_per_setting"], count)
                self.assertEqual(decision["budget_hours"], budget)
                self.assertEqual(decision["stop"], count is None)
                projected = decision["projections"]["20"]
                self.assertEqual(projected["batches_per_setting"], 7)
                self.assertAlmostEqual(projected["hours_with_reserve"], (8 * batch_total + 136) / 3000)

    def test_invalid_measurements(self):
        for patch_fields in (
            {"control_seconds": float("nan")}, {"production_setup_seconds": False},
            {"timing_setup_and_calibration_seconds": -1}, {"batch_seconds": {"F": 10}},
            {"timing_elapsed_seconds": 10},
        ):
            with self.subTest(patch_fields=patch_fields), self.assertRaises(ValueError):
                subject.budget_decision(dict(_measurements(1), **patch_fields))


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.timing_path = self.root / "timing"
        self.production_path = self.root / "production"
        self.report_path = self.root / "report"
        self.now = 0
        self.batch_clock = 0
        self.preparation = {"calibration": {"R": {}, "F": {}}, "R_truth_reference": {"fixture": 1}}
        self.config = {
            "machine_sha256": "test-host", "software": {"python": "test-version"},
            "source_sha256": {"test-source": "test-digest"},
        }
        self.enterContext(patch.object(subject, "perf_counter", side_effect=self._clock))
        self.enterContext(patch.object(subject, "_configuration", return_value=self.config))
        self.enterContext(patch.object(subject, "_source_hashes", return_value=self.config["source_sha256"]))
        self.enterContext(patch.object(subject, "_prepare", return_value=self.preparation))
        self.enterContext(patch.object(subject, "_load_preparation", side_effect=self._load_preparation))
        self.enterContext(patch.object(subject, "_run_tasks", side_effect=self._run_inline))
        self.enterContext(patch.object(subject, "_configure_threads"))
        self.worker = self.enterContext(patch.object(subject, "_worker", side_effect=_repeat))
        self.control = self.enterContext(patch.object(subject, "_control", side_effect=self._controls))
        self.stdout = self.enterContext(redirect_stdout(StringIO()))
        self.stderr = self.enterContext(redirect_stderr(StringIO()))

    def _clock(self):
        self.now += 1
        return self.now

    def _load_preparation(self, path):
        preparation = subject._read_json(path)
        return preparation, preparation["calibration"]

    def _run_inline(self, target, arguments, deadline, on_result=None):
        results = []
        if target is subject._timing_worker:
            self.now += self.batch_clock
        for index, args in enumerate(arguments):
            subject._time_left(deadline)
            result = target(*args)
            if on_result is not None:
                on_result(index, result)
            results.append(result)
        return results

    def _controls(self, purpose):
        if purpose == "timing":
            print("sensitive control estimate")
        return [{"purpose": purpose, "repeat": 0, "blocked": False}]

    def _timing(self):
        self.assertEqual(subject.main(["timing", "--output", str(self.timing_path)]), 0)
        return subject._read_json(self.timing_path / "timing.json")["decision"]["repeats_per_setting"]

    def _production(self):
        self.assertEqual(subject.main([
            "production", "--output", str(self.production_path), "--timing", str(self.timing_path),
            "--freeze-reference", " fixture-freeze ",
        ]), 0)

    def test_timing_blindness_production_records_and_report_replay(self):
        def noisy_repeat(*args):
            if args[1] == "timing":
                print("sensitive fitted choice")
                print("sensitive fitted warning", file=sys.stderr)
            return _repeat(*args)

        self.worker.side_effect = noisy_repeat
        count = self._timing()
        self.assertEqual(count, 50)
        self.assertEqual({p.name for p in self.timing_path.iterdir()},
                         {"COMPLETE.json", "preparation.json", "timing.json"})
        self.assertEqual(subject._read_json(self.timing_path / "preparation.json"), self.preparation)
        self.assertNotIn("sensitive", self.stdout.getvalue())
        self.assertNotIn("sensitive", self.stderr.getvalue())
        calls = [(c.args[0], c.args[1], c.args[2]) for c in self.worker.call_args_list]
        self.assertEqual(calls, [(setting, "timing", repeat) for setting in ("R", "F") for repeat in range(3)])
        self.assertNotIn("fixture:timing", (self.timing_path / "timing.json").read_text(encoding="utf-8"))

        self._production()
        subject._verify(self.production_path, "production")
        self.assertEqual(self.worker.call_count, 6 + 150 + 300)
        self.assertEqual([c.args for c in self.control.call_args_list], [("timing",), ("control",)])
        inputs = subject._read_json(self.production_path / "inputs.json")
        self.assertEqual(inputs["public_freeze_reference"], "fixture-freeze")
        self.assertEqual(inputs["timing_receipt_sha256"], subject._hash(self.timing_path / "COMPLETE.json"))
        runtime = subject._read_json(self.production_path / "runtime.json")
        self.assertEqual(runtime["combined_elapsed_seconds"],
                         runtime["production_elapsed_seconds"] + inputs["timing"]["measurements"]["timing_elapsed_seconds"])
        repeats = self.production_path / "repeats"
        self.assertEqual(inputs["production_counts"], {"R": 150, "F": 300})
        self.assertEqual({p.name for p in repeats.iterdir()}, {
            f"{setting}_{r:02d}.json" for setting, n in (("R", 150), ("F", 300)) for r in range(n)
        })
        saved_report = subject._read_json(self.production_path / "report.json")
        self.assertEqual([saved_report[s]["repeats"] for s in ("R", "F")], [150, 300])
        self.assertEqual(subject._read_json(repeats / "F_49.json")["dataset_id"], "fixture:production:F:49")

        self.worker.reset_mock()
        self.control.reset_mock()
        self.assertEqual(subject.main([
            "report", "--output", str(self.report_path), "--production", str(self.production_path),
        ]), 0)
        subject._verify(self.report_path, "report")
        self.assertEqual((self.production_path / "report.json").read_bytes(),
                         (self.report_path / "report.json").read_bytes())
        self.worker.assert_not_called()
        self.control.assert_not_called()

    def test_stop_writes_projection_and_blocks_production(self):
        self.batch_clock = 2300
        self.assertEqual(subject.main(["timing", "--output", str(self.timing_path)]), 2)
        subject._verify(self.timing_path, "timing")
        self.assertTrue(subject._read_json(self.timing_path / "timing.json")["decision"]["stop"])
        with self.assertRaisesRegex(ValueError, "within-budget timing decision"):
            self._production()
        self.assertFalse(self.production_path.exists())
        self.assertEqual(self.worker.call_count, 6)

    def test_requires_matching_configuration_and_hash_valid_receipt(self):
        self._timing()
        for field in self.config:
            with self.subTest(field=field), patch.object(
                subject, "_configuration", return_value=dict(self.config, **{field: "changed"})
            ), self.assertRaisesRegex(ValueError, "current code, host, or software"):
                self._production()
        with self.assertRaisesRegex(ValueError, "public freeze reference"):
            subject.production(self.production_path, self.timing_path, "  ")
        with self.assertRaises(FileExistsError):
            self._timing()
        with self.assertRaisesRegex(ValueError, "outside any completed output"):
            subject._fresh_folder(self.timing_path / "nested")
        (self.timing_path / "injected.txt").write_text("changed files", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid or changed timing"):
            self._production()
        self.assertFalse(self.production_path.exists())

    def test_timing_failure_is_blinded_and_cannot_be_used(self):
        self.worker.side_effect = ValueError("sensitive fitted choices")
        with self.assertRaisesRegex(RuntimeError, "Timing repeat R/0 failed") as raised:
            self._timing()
        self.assertNotIn("sensitive", str(raised.exception))
        self.assertNotIn("sensitive", self.stdout.getvalue())
        self.assertFalse((self.timing_path / "COMPLETE.json").exists())
        self.assertEqual(subject._read_json(self.timing_path / "FAILED.json")["stage"], "timing")
        self.assertNotIn("sensitive", (self.timing_path / "FAILED.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "No completed timing"):
            self._production()

    def test_production_expiry_charges_timing_and_preserves_partial_repeats(self):
        self._timing()
        timing_elapsed = subject._read_json(self.timing_path / "timing.json")["measurements"]["timing_elapsed_seconds"]

        def overrun_second_repeat(*args):
            if args[2] == 0:
                self.now += 12 * 3600 - timing_elapsed
            return _repeat(*args)

        self.worker.side_effect = overrun_second_repeat
        with self.assertRaisesRegex(subject._BudgetExceeded, "12-hour limit"):
            self._production()
        self.assertEqual([p.name for p in (self.production_path / "repeats").iterdir()], ["R_00.json"])
        receipt = subject._read_json(self.production_path / "FAILED.json")
        self.assertTrue(receipt["budget_exceeded"])
        self.assertGreaterEqual(receipt["combined_elapsed_seconds"], 12 * 3600)
        self.assertFalse((self.production_path / "COMPLETE.json").exists())
        with self.assertRaisesRegex(ValueError, "No completed production"):
            subject.report(self.report_path, self.production_path)

    def test_report_rejects_changed_sources_or_artifacts(self):
        self._timing()
        self._production()
        with patch.object(subject, "_source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "Reporting code or protocol differs"):
                subject.report(self.report_path, self.production_path)
        with patch.object(subject, "_report", return_value={}):
            with self.assertRaisesRegex(ValueError, "Recomputed report differs"):
                subject.report(self.report_path, self.production_path)
        (self.production_path / "report.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid or changed production"):
            subject.report(self.report_path, self.production_path)
        self.assertFalse(self.report_path.exists())

    def test_cannot_seal_study_after_sources_change_during_compute(self):
        self._timing()
        # Timing compatibility succeeds at startup, then final source validation fails.
        with patch.object(subject, "_source_hashes", return_value={"changed-source": "changed-digest"}):
            with self.assertRaisesRegex(ValueError, "sources changed during production"):
                self._production()
        self.assertFalse((self.production_path / "COMPLETE.json").exists())
        self.assertEqual(subject._read_json(self.production_path / "FAILED.json")["stage"], "production")


class WorkerTests(unittest.TestCase):
    def test_spawn_uses_three_processes_and_two_threads(self):
        results = subject._run_tasks(_spawned_fixture, [(i,) for i in range(6)], perf_counter() + 15)
        self.assertEqual([result["index"] for result in results], list(range(6)))
        self.assertEqual(len({result["pid"] for result in results}), 3)
        self.assertTrue(all(result["pid"] != os.getpid() for result in results))
        self.assertTrue(all(set(result["threads"]) == {"2"} for result in results))
        self.assertTrue(all(not Path(result["scratch"]).exists() for result in results))

    def test_timeout_kills_active_workers_and_removes_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            children = {process.pid for process in multiprocessing.active_children()}
            with self.assertRaises(subject._BudgetExceeded):
                subject._run_tasks(_slow_fixture, [(folder, 0), (folder, 1)], perf_counter() + 4)
            markers = list(folder.glob("started-*.txt"))
            self.assertTrue(markers)
            self.assertTrue(all(not Path(marker.read_text(encoding="utf-8")).exists() for marker in markers))
            self.assertEqual({process.pid for process in multiprocessing.active_children()}, children)

    def test_abrupt_worker_exit_is_detected(self):
        with self.assertRaisesRegex(RuntimeError, "exited before returning"):
            subject._task(_abrupt_fixture, perf_counter() + 15)

    def test_worker_checks_actual_thread_cap_before_fitting(self):
        from . import pipeline
        from threadpoolctl import threadpool_info

        with patch("threadpoolctl.threadpool_info", return_value=[{"num_threads": 3}]):
            with patch.object(pipeline, "run_repeat") as fitted:
                with self.assertRaisesRegex(RuntimeError, "exceeded the worker thread cap"):
                    subject._worker("R", "timing", 0, None)
                fitted.assert_not_called()
        with patch.object(pipeline, "run_repeat", side_effect=lambda *args, **kwargs: threadpool_info()) as fitted:
            self.assertTrue(all(pool["num_threads"] <= 2 for pool in subject._worker("R", "timing", 0, None)))
            fitted.assert_called_once_with("R", "timing", 0, None, n=30_000)


if __name__ == "__main__":
    unittest.main()
