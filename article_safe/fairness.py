"""Article-interpreted contribution comparisons and subgroup coefficient diagnostics."""
import numpy as np
import pandas as pd
from article_safe.inference import concentration_gini, contribution_ks, signed_share_audit, holm_adjust
from article_safe.models import FittedDesign, fit_lr


def contribution_table(values, sex, feature_order, active_features, family):
    values, sex = np.asarray(values), np.asarray(sex)
    if values.shape != (len(sex), len(feature_order)) or set(np.unique(sex)) != {1, 2}:
        raise ValueError("Expected aligned contributions and both SEX groups.")
    active = set(active_features)
    rows = []
    for j, feature in enumerate(feature_order):
        female, male = values[sex == 2, j], values[sex == 1, j]
        signed = [float(female.mean()), float(male.mean())]
        magnitudes = [float(np.abs(female).mean()), float(np.abs(male).mean())]
        gini = concentration_gini(magnitudes)
        audit = signed_share_audit(signed)
        ks = contribution_ks(female, male) if feature in active else {"statistic": None, "p_value": None, "pooled_tie_fraction": None}
        rows.append({"feature": feature, "active": feature in active,
                     "female_mean_signed": signed[0], "male_mean_signed": signed[1],
                     "signed_mean_gap": signed[0] - signed[1],
                     "female_mean_absolute": magnitudes[0], "male_mean_absolute": magnitudes[1],
                     "absolute_mean_gap": magnitudes[0] - magnitudes[1],
                     "gini": gini["gini"], "normalized_gini": gini["normalized_gini"],
                     "gini_status": gini["status"], "signed_denominator": audit["denominator"],
                     "signed_cancellation": audit["cancellation"], "negative_effect": audit["negative_effect"],
                     "negative_share": audit["negative_share"],
                     "signed_female_share": None if audit["shares"] is None else audit["shares"][0],
                     "signed_male_share": None if audit["shares"] is None else audit["shares"][1],
                     "ks_statistic": ks["statistic"], "ks_p_value": ks["p_value"],
                     "pooled_tie_fraction": ks["pooled_tie_fraction"],
                     "female_records": len(female), "male_records": len(male),
                     "ks_fidelity": "explicit_interpretation" if family == "gbm" else "supporting_diagnostic",
                     "ks_inference": "nominal; ties/clustering not calibrated", "verdict": "not assigned"})
    result = pd.DataFrame(rows)
    result["ks_p_value_holm"] = np.nan
    mask = result.active
    result.loc[mask, "ks_p_value_holm"] = holm_adjust(result.loc[mask, "ks_p_value"].tolist())
    return result


def subgroup_coefficients(model, y_train, metadata_train, protocol):
    """Subgroup GLMs share the corresponding pooled design and reference categories."""
    fitted, diagnostic = {}, {}
    for code in (1, 2):
        selected = metadata_train.SEX.to_numpy() == code
        design = FittedDesign(model.design.preprocessor, model.design.features, model.design.mapping,
                              model.design.matrix[selected])
        subgroup = fit_lr(design, np.asarray(y_train)[selected], model.features, protocol)
        diagnostic[str(code)] = subgroup.diagnostics
        fitted[code] = np.asarray(subgroup.estimator.params)[1:] if subgroup.estimable else None
    rows = []
    for j, term in model.active_mapping.iterrows():
        female = float(fitted[2][j]) if fitted[2] is not None else np.nan
        male = float(fitted[1][j]) if fitted[1] is not None else np.nan
        if np.isfinite([female, male]).all():
            gini = concentration_gini([abs(female), abs(male)])
            status = gini["status"]
        else:
            gini = {"gini": None, "normalized_gini": None}
            status = "not_estimable"
        rows.append({"term": term.term, "feature": term.feature, "reference": term.reference,
                     "female_coefficient": female, "male_coefficient": male,
                     "female_minus_male": female - male, "opposite_signs": bool(female * male < 0) if status != "not_estimable" else None,
                     "gini": gini["gini"], "normalized_gini": gini["normalized_gini"], "status": status,
                     "coefficient_ks_status": "not_identifiable_from_article"})
    return pd.DataFrame(rows), diagnostic