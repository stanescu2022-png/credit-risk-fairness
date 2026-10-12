"""Read-only access to the saved split and training-only development partition."""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from article_safe.protocol import ROOT, sha256
from src.baseline import DATA_PATH, FEATURES, load_split
from src.performance_evaluation import predictor_groups, training_folds


@dataclass(frozen=True)
class DevelopmentPartition:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    fit_positions: np.ndarray
    selection_positions: np.ndarray
    fold_assignments: np.ndarray
    groups: np.ndarray
    feature_order: tuple

    def fitting(self):
        positions = self.fit_positions
        return self.X.iloc[positions].copy(), self.y.iloc[positions].copy(), self.metadata.iloc[positions].copy()

    def selection(self):
        positions = self.selection_positions
        return self.X.iloc[positions].copy(), self.y.iloc[positions].copy(), self.metadata.iloc[positions].copy()

    def assignment_table(self):
        result = self.metadata[["ID", "SEX", "default", "duplicate_group_id"]].copy()
        result["original_partition"] = "train"
        result["development_role"] = "fitting"
        result.iloc[self.selection_positions, result.columns.get_loc("development_role")] = "selection"
        result["fold"] = self.fold_assignments
        result["training_position"] = np.arange(len(result))
        return result.reset_index(drop=True)

    def structural_summary(self):
        summary = {}
        for name, positions in (("fitting", self.fit_positions), ("selection", self.selection_positions)):
            summary[name] = {
                "records": int(len(positions)), "predictor_groups": int(len(np.unique(self.groups[positions]))),
                "defaults": int(self.y.iloc[positions].sum()),
            }
        summary["training_records"] = len(self.X)
        summary["training_predictor_groups"] = int(len(np.unique(self.groups)))
        summary["crossing_fitting_selection_groups"] = len(set(self.groups[self.fit_positions]) & set(self.groups[self.selection_positions]))
        summary["predictors"] = list(self.feature_order)
        summary["holdout_outcomes_used_for_selection"] = False
        return summary


def build_development_partition(X, y, metadata, feature_order):
    """No holdout argument is accepted: only the saved training rows enter here."""
    feature_order = tuple(feature_order)
    if len(feature_order) != 22 or set(feature_order) != set(FEATURES):
        raise ValueError("Development data require all 22 original predictors.")
    if list(X.columns) != list(feature_order):
        raise ValueError("Training columns do not follow the saved model input order.")
    if not X.index.equals(y.index) or not X.index.equals(metadata.index):
        raise ValueError("Training predictors, outcomes and metadata are not aligned.")
    if "split" not in metadata or set(metadata["split"]) != {"train"}:
        raise ValueError("Development interfaces accept only saved training-partition rows.")
    if metadata.ID.isna().any() or not metadata.ID.is_unique:
        raise ValueError("Training IDs must be unique and present.")
    if not np.array_equal(y.to_numpy(), metadata.default.to_numpy()):
        raise ValueError("Training outcomes differ from the saved metadata.")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Expected finite predictors and both binary outcome classes.")
    if set(metadata.SEX.unique()) - {1, 2}:
        raise ValueError("Unexpected SEX metadata values.")
    groups = predictor_groups(X, metadata)
    if not np.array_equal(groups, metadata.duplicate_group_id.to_numpy()):
        raise ValueError("Saved predictor groups do not match exact training inputs.")
    folds, assignments, checked_groups = training_folds(X, y, metadata)
    fit, selection = folds[0]
    if set(groups[fit]) & set(groups[selection]):
        raise ValueError("An original predictor group crosses fitting and selection.")
    for positions in (fit, selection):
        if set(np.unique(y.iloc[positions])) != {0, 1}:
            raise ValueError("A development partition lacks an outcome class.")
    arrays = [np.asarray(item).copy() for item in (fit, selection, assignments, checked_groups)]
    for item in arrays:
        item.flags.writeable = False
    return DevelopmentPartition(X.copy(), y.copy(), metadata.copy(), *arrays, feature_order)


def load_development_data(protocol):
    """Read the original partition; expose only training rows to selection code."""
    df = pd.read_csv(DATA_PATH)
    # load_split checks preparation identities and metadata alignment; it cannot refit/resplit.
    parts = load_split(df)
    X, y, metadata = parts[0], parts[2], parts[4]
    if len(X) != protocol["outer_partition"]["training"]:
        raise ValueError("The saved training partition size changed.")
    features = protocol["features"]
    result = build_development_partition(X.loc[:, features], y, metadata, features)
    # Only ID/group membership is required from H for structural overlap validation.
    holdout_metadata = pd.read_csv(ROOT / "data/processed/test_metadata.csv", usecols=["ID", "duplicate_group_id"])
    if len(holdout_metadata) != protocol["outer_partition"]["holdout"]:
        raise ValueError("The saved holdout partition size changed.")
    if set(metadata.ID) & set(holdout_metadata.ID):
        raise ValueError("A record crosses the original train/holdout split.")
    if set(result.groups) & set(holdout_metadata.duplicate_group_id):
        raise ValueError("An original predictor group crosses train/holdout.")
    return result


def check_evaluation_artifacts(root=ROOT):
    """Verify saved stage hashes without running any stage evaluation."""
    root = Path(root)
    checked = 0
    failures = []
    for path in sorted(root.glob("outputs/tables/*/run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for category in ("protected_artifact_hashes", "locked_input_hashes", "source_hashes", "output_hashes"):
            for relative, expected in manifest.get(category, {}).items():
                source = root / relative.replace("\\", "/")
                checked += 1
                if not source.is_file() or sha256(source) != expected:
                    failures.append({"manifest": str(path.relative_to(root)), "path": relative})
    if failures or checked != 410:
        raise ValueError(f"Prior evaluation integrity failed: {checked} checks; {failures}")
    return {"checks": checked, "failures": failures}