"""Pure selection decisions; fitting and credit-model execution come next milestone."""
import numpy as np

VALID_TEST_STATUSES = {"ok", "identical_result"}


def forward_step(gain, p_value, status="ok", alpha=0.05):
    """A significant deterioration never counts as an accepted addition."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    if status not in VALID_TEST_STATUSES or p_value is None:
        return {"accept": None, "reason": "unestimable_comparison"}
    if not np.isfinite(gain) or not np.isfinite(p_value) or not 0 <= p_value <= 1:
        return {"accept": None, "reason": "invalid_comparison"}
    if gain <= 0:
        return {"accept": False, "reason": "nonpositive_auc_gain"}
    if p_value >= alpha:
        return {"accept": False, "reason": "nonsignificant_auc_gain"}
    return {"accept": True, "reason": "positive_significant_auc_gain"}


def choose_forward_prefix(steps, alpha=0.05):
    """Start with k=1; a later recovery cannot undo the first stopping decision.

    Each supplied step represents k versus k-1, starting at k=2. The model runner
    will retain the complete diagnostic path separately from this decision.
    """
    selected = 1
    for expected_k, step in enumerate(steps, start=2):
        if step["k"] != expected_k:
            raise ValueError("Forward comparisons must be consecutive prefixes starting at k=2.")
        decision = forward_step(step["gain"], step["p_value"], step.get("status", "ok"), alpha)
        if decision["accept"] is None:
            return {"selected_k": None, "last_valid_k": selected, "status": "not_estimable",
                    "stopped_at_k": expected_k, "reason": decision["reason"]}
        if not decision["accept"]:
            return {"selected_k": selected, "status": "complete", "stopped_at_k": expected_k,
                    "reason": decision["reason"]}
        selected = expected_k
    return {"selected_k": selected, "status": "complete", "stopped_at_k": None,
            "reason": "all_available_additions_accepted"}


def choose_backward_removal(candidates, feature_order, alpha=0.05):
    """Choose the largest whole-predictor removal p; an invalid comparison is visible."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    features = list(feature_order)
    if len(features) != len(set(features)):
        raise ValueError("Feature order must have no duplicates.")
    names = [item["feature"] for item in candidates]
    if len(names) != len(set(names)) or set(names) != set(features):
        raise ValueError("Exactly one whole-predictor comparison is required for each current feature.")
    if not candidates:
        return {"action": "stop", "feature": None, "status": "complete", "reason": "intercept_only"}
    invalid = [item["feature"] for item in candidates
               if item.get("status", "ok") != "ok" or item["p_value"] is None
               or not np.isfinite(item["p_value"]) or not 0 <= item["p_value"] <= 1]
    if invalid:
        return {"action": "stop", "feature": None, "status": "not_estimable",
                "reason": "unestimable_removal_candidates", "invalid_features": invalid}
    order = {feature: position for position, feature in enumerate(features)}
    chosen = min(candidates, key=lambda item: (-item["p_value"], order[item["feature"]]))
    if chosen["p_value"] >= alpha:
        return {"action": "remove", "feature": chosen["feature"], "p_value": chosen["p_value"],
                "status": "complete", "reason": "largest_removal_p_at_or_above_alpha"}
    return {"action": "stop", "feature": None, "status": "complete", "reason": "all_removal_p_below_alpha"}