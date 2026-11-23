"""Run one lean synthetic repeat, with choices locked before truth scoring."""

from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from research.model_selection_v2.confirmation_v1.judge import references
from research.model_selection_v2.confirmation_v1.schema import build_candidate
from research.model_selection_v2.confirmation_v1.selection import bootstrap_plan, selection, split
from research.model_selection_v2.confirmation_v2.judge import MODES, paired_transforms
from research.model_selection_v2.confirmation_v2_jobs.section_6_5 import (
    DIRECT_ARM,
    choose_direct_small_swap,
)
from research.model_selection_v2.pilot import CANDIDATES, positive_probability
from research.model_selection_v2.selection_exploration_v1.transforms import predict_changed_rows
from research.model_selection_v2.step1_v1.core import (
    MILD,
    SEXES,
    choice,
    fairness,
    group_means,
    paired_difference,
    ranking,
)

from .generators import generate_population, study_families, study_schema


JUDGE_LEVELS = {"R": ("J1", "J3"), "F": ("J1",)}
MATERIAL_HARM = 0.0005


def _metadata(population):
    index = pd.Index(population.ids, name="ID")
    if not index.is_unique or not index.equals(population.X.index):
        raise ValueError("Synthetic predictors and borrower IDs must align")
    frame = population.X.assign(ID=population.ids)
    groups = frame.groupby(list(population.X), sort=False, dropna=False).ID.transform("min")
    return pd.DataFrame({"ID": population.ids, "SEX": population.sex,
                         "default": population.y, "duplicate_group_id": groups}, index=index)


def _fit_candidates(fitting, outcome, schema):
    models = {}
    for candidate in CANDIDATES:
        model = build_candidate(candidate, schema)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(fitting, outcome)
        models[candidate] = model
    return models


def align_truth(frame, metadata, probabilities_by_id):
    """Retrieve each original probability in the evaluated borrower order."""
    ids = pd.Index(metadata.ID, name="ID")
    if not ids.is_unique or not ids.equals(frame.index) or not probabilities_by_id.index.is_unique:
        raise ValueError("Truth scoring requires unique, aligned borrower IDs")
    truth = probabilities_by_id.reindex(ids).to_numpy(dtype=float)
    if not np.isfinite(truth).all() or not ((0 <= truth) & (truth <= 1)).all():
        raise ValueError("Missing or invalid original probabilities for evaluated borrowers")
    return truth


def _group_masks(meta):
    masks = {
        key: np.ones(len(meta), dtype=bool) if code is None else meta.SEX.eq(code).to_numpy()
        for key, code in SEXES
    }
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("Truth scoring needs borrowers from both groups")
    return masks


def _group_averages(values, masks):
    return {key: float(values[mask].mean()) for key, mask in masks.items()}


def _selection_predictions(evidence, meta, candidate):
    values = evidence[candidate]
    if not np.array_equal(values["ID"], meta.ID.to_numpy()):
        raise ValueError("Mild-swap predictions and selection borrower IDs must align")
    mild = np.asarray(values["swap_mild"])
    if mild.shape != (len(MILD), len(meta)):
        raise ValueError("Expected six aligned mild-swap nodes")
    return mild


class _VisitedChecks(dict):
    """Evaluate a comparison only when frozen choice requests it."""

    def __init__(self, safe, point, draws):
        super().__init__()
        self.safe = CANDIDATES.index(safe)
        self.point = point
        self.draws = draws
        self.results = {}

    def __missing__(self, candidate):
        i = CANDIDATES.index(candidate)
        check = fairness(self.point[i, :, 1] - self.point[self.safe, :, 1],
                         self.draws[:, i, :, 1] - self.draws[:, self.safe, :, 1])
        self.results[candidate] = check
        self[candidate] = check["blocked"]
        return self[candidate]


def _choices(rows, evidence, meta, repeat):
    safe = ranking(rows.SAFE_score.to_dict())[0]
    ranked = ranking(rows.Score2.to_dict())
    choices = {
        "safe": safe,
        "plain_loss": ranking((-rows.clean_brier).to_dict())[0],
        DIRECT_ARM: choose_direct_small_swap(rows.S_bar.to_dict()),
        "selector2_before_check": ranked[0],
    }
    if ranked[0] == safe:
        choices["selector2"] = safe
        return choices, ranked, {}

    plan = bootstrap_plan(meta, repeat)
    point = np.empty((len(CANDIDATES), 2, 3))
    draws = np.empty((len(plan.group_multiplicities), *point.shape))
    outcome = meta.default.to_numpy()
    for i, candidate in enumerate(CANDIDATES):
        mild = _selection_predictions(evidence, meta, candidate)
        clean = (evidence[candidate]["clean"] - outcome) ** 2
        stressed = ((mild[1:] - outcome) ** 2).mean(axis=0)
        values = np.column_stack((clean, stressed, stressed - clean))
        for j, (_, code) in enumerate(SEXES[1:]):
            mask = meta.SEX.eq(code).to_numpy()
            point[i, j] = values[mask].mean(axis=0)
            draws[:, i, j] = group_means(values, plan.group_multiplicities, plan.row_group_positions, mask)

    checks = _VisitedChecks(safe, point, draws)
    choices["selector2"] = choice(ranked, safe, checks)
    return choices, ranked, checks.results


def _group_gate_trace(choices, ranked, checks, evidence, frame, meta, probabilities_by_id):
    truth = align_truth(frame, meta, probabilities_by_id)
    masks = _group_masks(meta)
    masks.pop("all")
    safe, final = choices["safe"], choices["selector2"]
    truth_losses = {
        candidate: ((_selection_predictions(evidence, meta, candidate)[1:] - truth) ** 2).mean(axis=0)
        for candidate in dict.fromkeys((safe, choices[DIRECT_ARM], *checks))
    }
    deltas = {candidate: _group_averages(loss - truth_losses[safe], masks)
              for candidate, loss in truth_losses.items()}
    attempts = []
    for candidate, check in checks.items():
        delta = deltas[candidate]
        intervals = check["joint_intervals_female_male"]
        attempts.append({
            "candidate": candidate,
            "rank": ranked.index(candidate) + 1,
            "triggering_groups": [key for key, (lower, _) in zip(masks, intervals)
                                  if lower is not None and lower > 0],
            "truth_delta": delta,
            "harmful_groups": [key for key, value in delta.items() if value > 0],
            "material_harm_groups": [key for key, value in delta.items() if value >= MATERIAL_HARM],
            **check,
        })

    first = attempts[0] if attempts else None
    final_delta = deltas[final]
    direct_delta = deltas[choices[DIRECT_ARM]]
    final_harmful = any(value > 0 for value in final_delta.values())
    return {
        "started_at_safe": ranked[0] == safe,
        "visited": ranked[:ranked.index(final) + 1],
        "first_attempt": first,
        "later_attempts": attempts[1:],
        "final_truth_delta": final_delta,
        "final_harmful": final_harmful,
        "prevented_harm": bool(first and first["blocked"] and first["harmful_groups"] and not final_harmful),
        "direct_truth_delta": direct_delta,
        "direct_harmful_groups": [key for key, value in direct_delta.items() if value > 0],
        "direct_material_harm_groups": [key for key, value in direct_delta.items() if value >= MATERIAL_HARM],
    }


def _judge_truth(frame, fitting, meta, models, schema, families, dataset_id, levels, probabilities_by_id):
    truth = align_truth(frame, meta, probabilities_by_id)
    masks = _group_masks(meta)
    clean = {candidate: positive_probability(model, frame) for candidate, model in models.items()}
    result = {"clean": {
        candidate: _group_averages((probability - truth) ** 2, masks)
        for candidate, probability in clean.items()
    }}
    reference = references(fitting, schema)
    for level in levels:
        losses = {candidate: np.zeros(len(truth)) for candidate in models}
        for family in families:
            for realization in range(1, 31):
                frames, _, _ = paired_transforms(
                    frame, fitting, schema, families, dataset_id=dataset_id, family=family,
                    repetition=realization, level=level, reference=reference,
                )
                if frames.keys() != MODES.keys():
                    raise ValueError("A judge draw did not produce the three frozen modes")
                for changed in frames.values():
                    for candidate, model in models.items():
                        probability, _ = predict_changed_rows(
                            model, frame, changed, clean[candidate], positive_probability,
                        )
                        losses[candidate] += (probability - truth) ** 2
        draws = len(families) * 30 * len(MODES)
        result[level] = {
            candidate: _group_averages(values / draws, masks) for candidate, values in losses.items()
        }
    return result


def _contrasts(setting, choices, judge_truth):
    pairs = {
        "selector2_minus_safe": ("selector2", "safe"),
        "selector2_minus_direct": ("selector2", DIRECT_ARM),
    }
    if setting == "R":
        pairs["direct_minus_clean"] = (DIRECT_ARM, "plain_loss")
        pairs["selector2_minus_clean"] = ("selector2", "plain_loss")
    else:
        pairs["final_minus_pre"] = ("selector2", "selector2_before_check")
    return {
        contrast: {
            level: {
                key: paired_difference({candidate: by_group[key] for candidate, by_group in values.items()},
                                       choices[arm], choices[baseline])
                for key, _ in SEXES
            }
            for level, values in judge_truth.items()
        }
        for contrast, (arm, baseline) in pairs.items()
    }


def run_repeat(setting, purpose, repeat, calibration, *, n=30_000):
    """Compute a serializable fitted result from a fresh synthetic population."""
    if setting not in JUDGE_LEVELS or purpose not in {"fixture", "timing", "production"}:
        raise ValueError("Expected an R or F synthetic repeat with an explicit study purpose")
    if purpose != "fixture" and n != 30_000:
        raise ValueError("Timing and production require the frozen 30,000-borrower population")

    population = generate_population(setting, purpose, repeat, calibration, n=n)
    schema = study_schema(setting)
    schema.validate(population.X)
    metadata = _metadata(population)
    _, positions, audit = split(metadata, repeat, set())
    role_frames = {role: population.X.iloc[positions[role]] for role in positions}
    role_meta = {role: metadata.iloc[positions[role]] for role in positions}
    fitting, selected, judged = (role_frames[role] for role in ("fitting", "selection", "judge"))
    fitting_meta, selection_meta, judge_meta = (role_meta[role] for role in ("fitting", "selection", "judge"))
    models = _fit_candidates(fitting, fitting_meta.default, schema)
    # The frozen selection implementation writes its prediction evidence to a folder.
    with TemporaryDirectory(prefix="synthetic_lean_selection_") as temporary:
        rows, evidence = selection(Path(temporary), fitting, selected, selection_meta, models, schema)
    choices, ranked, checks = _choices(rows, evidence, selection_meta, repeat)

    original_probabilities = pd.Series(population.p, index=metadata.index)
    judge_truth = _judge_truth(judged, fitting, judge_meta, models, schema, study_families(setting),
                               population.dataset_id, JUDGE_LEVELS[setting], original_probabilities)
    contrasts = _contrasts(setting, choices, judge_truth)
    result = {
        "setting": setting, "purpose": purpose, "repeat": repeat, "dataset_id": population.dataset_id,
        "role_sizes": audit["sizes"], "choices": choices, "judge_truth": judge_truth, "contrasts": contrasts,
    }
    if setting == "R":
        delta = contrasts["direct_minus_clean"]
        result["reversals"] = {
            level: delta["clean"]["all"] > 0 and delta[level]["all"] < 0 for level in JUDGE_LEVELS[setting]
        }
    else:
        result["gate"] = _group_gate_trace(choices, ranked, checks, evidence, selected, selection_meta,
                                           original_probabilities)
    return result
