"""Judge-scoped adapter: same frozen faults, fresh judge_v1 seed streams."""
import numpy as np
import pandas as pd

from src.baseline import FEATURES
from src.robustness_protocol import child_seed, semantic_transform
from research.model_selection_v2.selection_exploration_v1.transforms import (
    exposure_summary, frame_hash, mask_selected, predict_changed_rows, reference_bundle,
)
from .protocol import CONDITIONS, FAMILIES, MASTER_SEED, REPETITIONS, SEVERITIES
from .io import require


def selected_rows(n, family, repetition, q):
    seed = child_seed(f"judge_v1:{family}", repetition, "mask")
    selected = np.zeros(n, dtype=bool)
    selected[np.random.default_rng(seed).permutation(n)[:int(np.floor(q*n))]] = True
    return selected


def stream_registry(family, repetition, rules, feature_order):
    scope = f"judge_v1:{family}"
    complement = [name for name in feature_order if name not in FAMILIES[family]]
    return {"scope": scope, "repetition": repetition, "master_seed": MASTER_SEED, "mask": child_seed(scope, repetition, "mask"),
            "features": {name: {"sign": child_seed(scope, repetition, "sign:"+name),
                         **({"category": child_seed(scope, repetition, "category:"+name)} if rules[name]["kind"] == "category" else {})}
                         for name in complement}}


def paired_transforms(evaluation, fitting, references, *, family, repetition, q):
    require(family in FAMILIES and q in SEVERITIES and isinstance(repetition, int) and 1 <= repetition <= REPETITIONS,
            "Use declared judge family/q/repetition.")
    require(list(evaluation.columns) == references["predictor_order"] == list(fitting.columns), "Predictor source/order mismatch.")
    require(frame_hash(fitting) == references["fitting_frame_sha256"], "Fitting reference source changed.")
    require(set(evaluation.columns) == set(FEATURES) and np.isfinite(evaluation.to_numpy(dtype=float)).all(), "Require finite original 22 predictors.")
    selected = selected_rows(len(evaluation), family, repetition, q)
    family_features = FAMILIES[family]
    complement = [name for name in evaluation.columns if name not in family_features]
    require(len(complement) == 16 and not set(complement) & set(family_features), "Fault units must be disjoint 6/16 original columns.")
    scenario = {"scope": f"judge_v1:{family}", "repetition": repetition, "q": q,
                "level": "primary" if q == .10 else "stronger", "features": complement}
    masked = mask_selected(evaluation, family_features, selected, references["mask_values"])
    corrupted, info = semantic_transform(evaluation, scenario, references["semantic_rules"])
    for item in info.values():
        np.testing.assert_array_equal(item["selected"], selected)
    joint = mask_selected(corrupted, family_features, selected, references["mask_values"])
    opposite, _ = semantic_transform(masked, scenario, references["semantic_rules"])
    pd.testing.assert_frame_equal(joint, opposite, check_exact=True)
    frames = {"10": masked, "01": corrupted, "11": joint}
    registry = {}
    for condition, frame in frames.items():
        pd.testing.assert_frame_equal(frame.loc[~selected], evaluation.loc[~selected], check_dtype=False, check_exact=True)
        active = family_features if condition == "10" else complement if condition == "01" else list(evaluation.columns)
        registry[condition] = {"family": family, "q": q, "repetition": repetition, "condition": condition,
            "mode": CONDITIONS[condition], "repetition_id": f"judge_v1:q{q:.2f}:{family}:r{repetition:02d}",
            "input_sha256": frame_hash(frame), "commutation_checked": True,
            "exposure": exposure_summary(evaluation, frame, selected, active, info if condition != "10" else {}, fitting)}
    return frames, registry, stream_registry(family, repetition, references["semantic_rules"], evaluation.columns)
