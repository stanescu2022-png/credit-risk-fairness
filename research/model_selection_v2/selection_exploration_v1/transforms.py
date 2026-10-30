"""Paired, fitting-referenced input transformations using frozen semantic rules."""
import hashlib
import json

import numpy as np
import pandas as pd

from research.model_selection_v2.safe2026 import fitting_references
from src.baseline import CATEGORICAL, FEATURES
from src.robustness_protocol import child_seed, semantic_rules, semantic_transform
from .protocol import (CONDITIONS, FAMILIES, MASTER_SEED, PROBABILITY_AUDIT_ATOL,
                       REPETITIONS, SEVERITIES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frame_hash(frame):
    """Canonical ordered schema/value hash, independent of column storage dtype."""
    values = np.asarray(frame.to_numpy(), dtype="<f8", order="C")
    header = json.dumps({"columns": list(frame.columns), "shape": values.shape}, separators=(",", ":"))
    return hashlib.sha256(header.encode("utf-8") + values.tobytes()).hexdigest()


def reference_bundle(fitting):
    require(len(fitting) > 0 and set(fitting.columns) == set(FEATURES) and len(fitting.columns) == 22,
            "Fitting references require the original 22 predictors, without metadata.")
    require(np.isfinite(fitting.to_numpy(dtype=float)).all(), "Fitting references must be finite.")
    return {"source_role": "fitting", "records": len(fitting), "predictor_order": list(fitting.columns),
            "fitting_frame_sha256": frame_hash(fitting), "mask_values": fitting_references(fitting),
            "semantic_rules": semantic_rules(fitting),
            "mask_rule": "fitting numeric arithmetic means; categorical modes, smallest mode on ties"}


def stream_registry(family, repetition, rules, feature_order):
    scope = f"exploration_v1:{family}"
    complementary = [name for name in feature_order if name not in FAMILIES[family]]
    return {"scope": scope, "repetition": repetition, "master_seed": MASTER_SEED,
            "mask": child_seed(scope, repetition, "mask"),
            "features": {name: {"sign": child_seed(scope, repetition, "sign:"+name),
                                 **({"category": child_seed(scope, repetition, "category:"+name)}
                                    if rules[name]["kind"] == "category" else {})} for name in complementary}}


def selected_rows(n, family, repetition, q):
    seed = child_seed(f"exploration_v1:{family}", repetition, "mask")
    selected = np.zeros(n, dtype=bool)
    selected[np.random.default_rng(seed).permutation(n)[:int(np.floor(q*n))]] = True
    return selected


def mask_selected(frame, family_features, selected, references):
    out = frame.copy(deep=True)
    for name in family_features:
        # Cast explicitly because fitting means need not be integral even when
        # the original monetary inputs are integer-valued.
        column = out[name].to_numpy(dtype=float, copy=True)
        column[selected] = references[name]
        out[name] = column
    return out


def support_summary(frame, fitting):
    details = {}
    for name in frame.columns:
        values = frame[name]
        if name in CATEGORICAL:
            details[name] = {"outside_fitting_levels": int((~values.isin(fitting[name].unique())).sum())}
        else:
            details[name] = {"below_fitting_minimum": int((values < fitting[name].min()).sum()),
                             "above_fitting_maximum": int((values > fitting[name].max()).sum())}
    categories = [name for name in frame.columns if name in CATEGORICAL]
    combinations = pd.MultiIndex.from_frame(frame[categories])
    fitted_combinations = pd.MultiIndex.from_frame(fitting[categories].drop_duplicates())
    return {"per_feature": details, "categorical_combinations_absent_from_fitting": int((~combinations.isin(fitted_combinations)).sum()),
            "interpretation": "empirical fitting support diagnostics, not proof of impossible records"}


def exposure_summary(original, transformed, selected, active_features, corruption_info, fitting):
    changes = original.to_numpy() != transformed.to_numpy()
    selected_cells = int(selected.sum()) * len(active_features)
    per_feature = {}
    for position, name in enumerate(original.columns):
        if name not in active_features:
            continue
        info = corruption_info.get(name)
        changed = changes[:, position]
        per_feature[name] = {"selected": int(selected.sum()), "changed": int(changed.sum()),
            "no_op": int((selected & ~changed).sum()),
            "clipped": int(info["clipped"].sum()) if info else 0,
            "outside_semantic_support": int(info["outside_support"].sum()) if info else 0,
            "eligible_selected": int((info["eligible"] & selected).sum()) if info else int(selected.sum()),
            "special_state_selected": int((info["special_state"] & selected).sum()) if info else 0,
            "zero_scale": bool(info["zero_scale"]) if info else False}
    return {"selected_rows": int(selected.sum()), "selected_cells": selected_cells,
            "changed_rows": int(changes.any(axis=1).sum()), "changed_cells": int(changes.sum()),
            "no_op_rows": int((selected & ~changes.any(axis=1)).sum()),
            "no_op_cells": selected_cells-int(changes.sum()),
            "clipped_cells": sum(v["clipped"] for v in per_feature.values()), "per_feature": per_feature,
            "support": support_summary(transformed, fitting)}


def paired_transforms(selection, fitting, references, *, family, repetition, q):
    require(family in FAMILIES and q in SEVERITIES and isinstance(repetition, int) and 1 <= repetition <= REPETITIONS,
            "Use a declared family, q=.10/.20 and repetition 1..30.")
    require(list(selection.columns) == references["predictor_order"] == list(fitting.columns), "Predictor order/source alignment changed.")
    require(frame_hash(fitting) == references["fitting_frame_sha256"], "Fitting reference source changed.")
    require(set(selection.columns) == set(FEATURES) and np.isfinite(selection.to_numpy(dtype=float)).all(),
            "Selection requires finite original predictors, no metadata.")
    family_features = FAMILIES[family]
    complement = [name for name in selection.columns if name not in family_features]
    require(len(family_features) == 6 and len(complement) == 16 and not set(family_features) & set(complement),
            "Family and corruption units must be disjoint 6/16 original predictors.")
    selected = selected_rows(len(selection), family, repetition, q)
    scenario = {"scope": f"exploration_v1:{family}", "repetition": repetition, "q": q,
                "level": "primary" if q == .10 else "stronger", "features": complement}
    masked = mask_selected(selection, family_features, selected, references["mask_values"])
    corrupted, info = semantic_transform(selection, scenario, references["semantic_rules"])
    for feature_info in info.values():
        np.testing.assert_array_equal(feature_info["selected"], selected)
    joint = mask_selected(corrupted, family_features, selected, references["mask_values"])
    opposite_order, _ = semantic_transform(masked, scenario, references["semantic_rules"])
    pd.testing.assert_frame_equal(joint, opposite_order, check_exact=True)
    frames = {"10": masked, "01": corrupted, "11": joint}
    registry = {}
    for condition, frame in frames.items():
        pd.testing.assert_frame_equal(frame.loc[~selected], selection.loc[~selected], check_dtype=False, check_exact=True)
        active = family_features if condition == "10" else complement if condition == "01" else list(selection.columns)
        registry[condition] = {"family": family, "q": q, "repetition": repetition, "condition": condition,
            "mode": CONDITIONS[condition], "repetition_id": f"q{q:.2f}:{family}:r{repetition:02d}",
            "input_sha256": frame_hash(frame), "commutation_checked": True,
            "exposure": exposure_summary(selection, frame, selected, active, info if condition != "10" else {}, fitting)}
    return frames, registry, stream_registry(family, repetition, references["semantic_rules"], selection.columns)


def predict_changed_rows(model, original, transformed, clean_probability, predict, *, audit_full=False):
    require(list(original.columns) == list(transformed.columns) and original.index.equals(transformed.index), "Prediction frame alignment changed.")
    clean = np.asarray(clean_probability, dtype=float)
    require(clean.shape == (len(original),) and np.isfinite(clean).all() and ((clean >= 0) & (clean <= 1)).all(),
            "Clean predictions must be aligned valid probabilities.")
    changed = (original.to_numpy() != transformed.to_numpy()).any(axis=1)
    result = clean.copy()
    if changed.any():
        partial = np.asarray(predict(model, transformed.loc[changed]), dtype=float)
        require(partial.shape == (int(changed.sum()),), "Changed-row prediction alignment changed.")
        result[changed] = partial
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), "Invalid frozen-model probabilities.")
    audit = {"changed_rows_predicted": int(changed.sum()), "clean_rows_reused": int((~changed).sum()),
             "full_frame_audit": audit_full, "absolute_tolerance": PROBABILITY_AUDIT_ATOL}
    if audit_full:
        full = np.asarray(predict(model, transformed), dtype=float)
        require(full.shape == clean.shape, "Full-frame audit prediction shape changed.")
        np.testing.assert_allclose(result, full, rtol=0, atol=PROBABILITY_AUDIT_ATOL)
        audit["maximum_absolute_error"] = float(np.max(np.abs(result-full)))
    return result, audit
