"""Fixed judge-v1 settings; the primary evaluation remains explicitly null."""
from pathlib import Path
from research.model_selection_v2.selection_exploration_v1.protocol import FAMILIES, CONDITIONS, scenario_name

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SPECIFICATION = HERE.parent / "judge_v1_specification.md"
EXPLORATION_SPECIFICATION = HERE.parent / "exploration_v1_specification.md"
EXPLORATION_RUN = ROOT / "runs/model_selection_v2/selection_exploration_v1_2026-10-30"
SAFE_RUN = ROOT / "runs/model_selection_v2/safe2026_swapping_2026-10-26"
PILOT_RUN = ROOT / "runs/model_selection_v2/pilot_candidates_2026-10-26"
DEFAULT_FREEZE = ROOT / "runs/model_selection_v2/judge_v1_freeze_2026-10-30"
DEFAULT_RUN = ROOT / "runs/model_selection_v2/judge_v1_2026-10-30"
CANDIDATES = ("logistic", "decision_tree", "random_forest", "gradient_boosting")
METHODS = ("safe_arithmetic", "safe_geometric", "safe_quadratic", "clean_brier", "clean_logloss", "worst_brier", "worst_logloss",
           "screened_safe_clean", "screened_safe_marginal", "screened_safe_outcome_v0")
COORDINATES = ("clean_brier", "clean_logloss", "absolute_tpr_gap", "absolute_fpr_gap", "worst_joint_brier", "worst_joint_logloss")
DIRECTIONS = {name: "min" for name in COORDINATES}
ROLE = "historical_evaluation"
SEVERITIES = (.10, .20)
REPETITIONS = 30
MASTER_SEED = 20261007
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42
PERCENTILE_BOUNDS = (2.5, 97.5)
AUDIT_ATOL = 1e-15


def settings():
    return {"judge_version": "judge_v1", "method_version": "screened_safe_outcome_v0", "primary_evaluation": None,
            "role": ROLE, "evidence": "historically inspected out-of-sample exploration; not pristine validation",
            "candidate_order": CANDIDATES, "method_ids": METHODS, "coordinates": COORDINATES, "directions": DIRECTIONS,
            "reference_q": .10, "sensitivity_q": .20, "selected_ids_for_both_severities": "frozen q=.10 development choices",
            "q20_development_choices": "diagnostic only, cannot replace reference identities",
            "families": FAMILIES, "conditions": CONDITIONS, "repetitions": REPETITIONS,
            "master_seed": MASTER_SEED, "scope": "judge_v1:<family>",
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED, "percentile_bounds": PERCENTILE_BOUNDS,
            "bootstrap": "fixed choices, models and saved scenarios; no reselection or complete-procedure uncertainty",
            "principal_contrast": "screened_safe_outcome_v0 choice minus safe_arithmetic choice",
            "fixed_pairs": "later pilot-order candidate minus earlier; preserves GBM-minus-LR",
            "prediction_reuse_audit": "repetition 1, every family/q/condition/candidate", "audit_atol": AUDIT_ATOL,
            "logloss_epsilon": 2.**-52, "brier_clipping": False, "decision_rule": "p > .5; ties class 0",
            "signed_gap": "female minus male", "dominance_tolerance": None, "fitting_refit_tuning_calibration": False}
