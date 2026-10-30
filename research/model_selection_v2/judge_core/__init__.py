"""Pure fixed-prediction Pareto judge foundation; no empirical loading or protocol."""
from .tables import PredictionTable, validate_alignment
from .metrics import CandidateTables, candidate_metrics, fairness_gaps, losses, scenario_losses
from .pareto import JudgeConfig, compare_choices, evaluate_candidates, pareto_relation, pool_frontier
from .bootstrap import BootstrapPlan, bootstrap_judge, bootstrap_with_plan, group_bootstrap_plan, percentile_summary

__all__ = ["PredictionTable", "validate_alignment", "CandidateTables", "candidate_metrics", "fairness_gaps",
           "losses", "scenario_losses", "JudgeConfig", "compare_choices", "evaluate_candidates", "pareto_relation",
           "pool_frontier", "BootstrapPlan", "bootstrap_judge", "bootstrap_with_plan", "group_bootstrap_plan", "percentile_summary"]
