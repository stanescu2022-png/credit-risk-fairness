"""Settings fixed by the root-owned exploratory specification, not tuning knobs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPECIFICATION = ROOT / "research/model_selection_v2/exploration_v1_specification.md"
DEFAULT_RUN = ROOT / "runs/model_selection_v2/selection_exploration_v1_2026-10-30"
SAFE_RUN = ROOT / "runs/model_selection_v2/safe2026_swapping_2026-10-26"
FAMILIES = {
    "repayment": ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    "billing": tuple(f"BILL_AMT{i}" for i in range(1, 7)),
    "payment": tuple(f"PAY_AMT{i}" for i in range(1, 7)),
}
CONDITIONS = {"10": "mask_only", "01": "corruption_only", "11": "joint"}
SEVERITIES = (.10, .20)
REPETITIONS = 30
MASTER_SEED = 20261007
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42
PERCENTILE_BOUNDS = (2.5, 97.5)
PROBABILITY_AUDIT_ATOL = 1e-15
SAFE_TIE_TOLERANCE = 1e-12
RISK_AXES = ("clean_brier", "clean_logloss", "marginal_max_brier", "marginal_max_logloss",
             "joint_max_brier", "joint_max_logloss")


def scenario_name(family, condition):
    return f"{family}__{CONDITIONS[condition]}"


def settings():
    return {"version": "selection_exploration_v1", "status": "exploratory; not independent validation",
            "primary_evaluation": None, "severities": list(SEVERITIES), "reference_q": .10,
            "sensitivity_q": .20, "repetitions": REPETITIONS, "families": FAMILIES,
            "conditions": CONDITIONS, "master_seed": MASTER_SEED,
            "seed_scope": "exploration_v1:<family>", "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED, "percentile_bounds": PERCENTILE_BOUNDS,
            "bootstrap_scope": "added outcome layer conditional on fixed saved SAFE scores; not complete selector uncertainty",
            "probability_reuse_audit": {"cases": "repetition 1, every q/family/condition/candidate",
                                        "absolute_tolerance": PROBABILITY_AUDIT_ATOL},
            "risk_axes": RISK_AXES, "risk_direction": "min", "max_order": "maximum after scenario repetition-loss mean",
            "practical_dominance_tolerance": None, "safe_final_tie_tolerance": SAFE_TIE_TOLERANCE,
            "minimum_loss_ties": "exact unrounded equality; fixed pilot order",
            "pairwise_reporting": "later candidate in fixed pilot order minus earlier; all six fixed pairs",
            "holdout_access": False, "fitting_calibration_tuning": False}
