"""Guarded historical evaluation loader; no repartition or fitting operation."""
import numpy as np
import pandas as pd

from research.model_selection_v2 import pilot, safe2026_run
from src.baseline import FEATURES
from src.performance_evaluation import predictor_groups
from .freeze import verify_freeze
from .io import require, sha256
from .protocol import CANDIDATES, ROOT


def validate_role_separation(fitting, selection, evaluation, fitting_meta, selection_meta, evaluation_meta):
    frames, metadata = (fitting, selection, evaluation), (fitting_meta, selection_meta, evaluation_meta)
    for frame, meta in zip(frames, metadata):
        require(len(frame) == len(meta) and frame.index.equals(meta.index), "Role predictor/metadata row alignment mismatch.")
        require(len(frame.columns) == 22 and set(frame.columns) == set(FEATURES) and list(frame.columns) == list(fitting.columns),
                "Every role must preserve the same original 22-predictor order.")
        require(np.isfinite(frame.to_numpy(dtype=float)).all(), "Nonfinite original predictors cannot define verified groups.")
    ids = [set(meta.ID) for meta in metadata]
    groups = [set(meta.duplicate_group_id) for meta in metadata]
    require(all(not ids[i] & ids[j] and not groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)),
            "Fitting/selection/evaluation IDs or original groups overlap.")
    all_X = pd.concat(frames, ignore_index=True)
    all_meta = pd.concat([meta[["ID", "SEX", "default", "duplicate_group_id"]] for meta in metadata], ignore_index=True)
    require(all_meta.ID.is_unique, "Role accounting contains duplicate IDs.")
    np.testing.assert_array_equal(predictor_groups(all_X, all_meta), all_meta.duplicate_group_id)
    return {"records": [len(frame) for frame in frames], "IDs_separate": True, "original_groups_separate": True,
            "whole_original_22_predictor_group_reconstruction": True, "cross_SEX_outcome_groups_preserved": True}


def load_evaluation_contract(freeze_receipt):
    # This guard precedes EVERY dataset read, including fitting/metadata reads.
    frozen = verify_freeze(freeze_receipt)
    for relative, expected in frozen["recorded_data_input_hashes"].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()) and sha256(path) == expected, f"Frozen data-input identity changed: {relative}")
    fitting, selection, selection_meta, models, _, _ = safe2026_run.load_contract()
    assignment = pd.read_csv(safe2026_run.CANDIDATE_RUN / "development_assignments.csv")
    fitting_meta = assignment[assignment.development_role.eq("fitting")].reset_index(drop=True)
    membership = pd.read_csv(pilot.DATA_PATH.parent / "split_metadata.csv")
    saved = pd.read_csv(pilot.DATA_PATH.parent / "test_metadata.csv")
    canonical = membership[membership.split.eq("test")].sort_values("partition_order").reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, canonical, check_exact=True)
    require(len(saved) == 5996 and saved.ID.is_unique and set(saved.split) == {"test"}, "Historical evaluation membership changed.")
    np.testing.assert_array_equal(saved.partition_order, np.arange(5996))
    order = list(fitting.columns)
    source_rows = set(np.flatnonzero(membership.split.eq("test")))
    source = pd.read_csv(pilot.DATA_PATH, usecols=["ID", "SEX", "default"]+order,
                         skiprows=lambda row: row > 0 and row-1 not in source_rows)
    require(len(source) == 5996 and set(source.ID) == set(saved.ID), "Historical evaluation source-row membership changed.")
    source = source.set_index("ID").loc[saved.ID].reset_index()
    np.testing.assert_array_equal(source.SEX, saved.SEX)
    np.testing.assert_array_equal(source.default, saved.default)
    evaluation = source[order].reset_index(drop=True)
    pilot.finite_inputs(evaluation)
    require(len(fitting) == 19203 and len(selection) == 4801 and set(models) == set(CANDIDATES), "Frozen candidate/role contract changed.")
    audit = validate_role_separation(fitting, selection, evaluation, fitting_meta, selection_meta, saved)
    require(sum(audit["records"]) == 30000, "Original record accounting changed.")
    return fitting, evaluation, saved, models, {**audit, "evidence": "historically inspected out-of-sample exploration",
        "repartition_refit_calibration_tuning": False, "freeze_verified_before_any_data_read": True}
