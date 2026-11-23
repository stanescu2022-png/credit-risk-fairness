"""Oracle loss and gate summaries across independent synthetic populations."""

from collections import Counter
from numbers import Integral

import numpy as np
from scipy.stats import norm, t

from research.model_selection_v2.pilot import CANDIDATES


GROUPS = ('all', 'female', 'male')
SEXES = GROUPS[1:]
ARMS = ('safe', 'plain_loss', 'direct_small_swap_brier', 'selector2_before_check', 'selector2')
CONTRASTS = {
    'selector2_minus_safe': ('selector2', 'safe'),
    'selector2_minus_direct': ('selector2', 'direct_small_swap_brier'),
}
FOCAL = {
    'R': ('direct_minus_clean', ('direct_small_swap_brier', 'plain_loss')),
    'F': ('final_minus_pre', ('selector2', 'selector2_before_check')),
}
CONDITIONS = {'R': ('clean', 'J1', 'J3'), 'F': ('clean', 'J1')}
MATERIAL_HARM = .0005


def wilson(numerator, denominator):
    """Pointwise 95% Wilson score interval for an event across independent trials."""
    if (not isinstance(numerator, Integral) or not isinstance(denominator, Integral)
            or not 0 <= numerator <= denominator):
        raise ValueError('Wilson counts must be integers with 0 <= numerator <= denominator')
    numerator, denominator = int(numerator), int(denominator)
    result = dict(numerator=numerator, denominator=denominator, estimate=None, lower=None, upper=None)
    if denominator == 0:
        return dict(result, reason='no opportunity')
    estimate = numerator / denominator
    z = norm.ppf(.975)
    divisor = 1 + z*z/denominator
    center = (estimate + z*z/(2*denominator)) / divisor
    half_width = z*np.sqrt(estimate*(1-estimate)/denominator + z*z/(4*denominator**2)) / divisor
    return dict(result, estimate=estimate,
                lower=0. if numerator == 0 else float(center-half_width),
                upper=1. if numerator == denominator else float(center+half_width))


def independent_t_interval(values):
    """Pointwise 95% t interval with actual sample size and rho = 0."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('An interval requires at least two finite independent-repeat values')
    n = len(values)
    # Identical rows have exact zero variance, including nonzero floating-point constants.
    constant = np.all(values == values[0])
    estimate = float(values[0] if constant else values.mean())
    sample_sd = 0. if constant else float(values.std(ddof=1))
    se = sample_sd/np.sqrt(n)
    half_width = float(t.ppf(.975, n-1)*se)
    return dict(estimate=estimate, lower=estimate-half_width, upper=estimate+half_width,
                sample_sd=sample_sd, standard_error=float(se), df=n-1, rho=0., repeats=n)


def _attempt_counts(attempts):
    return dict(
        attempts=len(attempts),
        harmful=sum(bool(attempt['harmful_groups']) for attempt in attempts),
        harmful_blocked=sum(bool(attempt['harmful_groups']) and attempt['blocked'] for attempt in attempts),
        harmful_allowed=sum(bool(attempt['harmful_groups']) and not attempt['blocked'] for attempt in attempts),
        harmless=sum(not attempt['harmful_groups'] for attempt in attempts),
        harmless_blocked=sum(not attempt['harmful_groups'] and attempt['blocked'] for attempt in attempts),
        material_harmful=sum(bool(attempt['material_harm_groups']) for attempt in attempts),
        material_harmful_blocked=sum(bool(attempt['material_harm_groups']) and attempt['blocked'] for attempt in attempts),
        material_harmful_allowed=sum(bool(attempt['material_harm_groups']) and not attempt['blocked'] for attempt in attempts),
        by_group={group: dict(
            harmful=sum(group in attempt['harmful_groups'] for attempt in attempts),
            material_harmful=sum(group in attempt['material_harm_groups'] for attempt in attempts),
            triggered_blocks=sum(group in attempt['triggering_groups'] for attempt in attempts),
        ) for group in SEXES},
    )


def _gate_summary(rows):
    gates = [row['gate'] for row in rows]
    first = [gate['first_attempt'] for gate in gates if gate['first_attempt'] is not None]
    later = [attempt for gate in gates for attempt in gate['later_attempts']]
    first_counts = _attempt_counts(first)
    later_counts = _attempt_counts(later)
    later_counts['by_rank'] = {
        str(rank): _attempt_counts([attempt for attempt in later if attempt['rank'] == rank])
        for rank in sorted({attempt['rank'] for attempt in later})
    }
    final_harm = sum(gate['final_harmful'] for gate in gates)
    final_material_harm = sum(any(delta >= MATERIAL_HARM for delta in gate['final_truth_delta'].values())
                              for gate in gates)
    prevented = sum(gate['prevented_harm'] for gate in gates)
    direct_harm = [bool(gate['direct_harmful_groups']) for gate in gates]
    return dict(
        started_at_safe=wilson(sum(gate['started_at_safe'] for gate in gates), len(rows)),
        first_proposals=first_counts,
        first_proposal_rates=dict(
            harmful_blocked=wilson(first_counts['harmful_blocked'], first_counts['harmful']),
            harmful_allowed=wilson(first_counts['harmful_allowed'], first_counts['harmful']),
            harmless_blocked=wilson(first_counts['harmless_blocked'], first_counts['harmless']),
        ),
        later_visited_attempts=later_counts,
        final_harm=wilson(final_harm, len(rows)),
        final_material_harm=wilson(final_material_harm, len(rows)),
        prevented_harm=wilson(prevented, first_counts['harmful']),
        harmful_first_blocked_but_final_harmful=sum(
            gate['first_attempt'] is not None and gate['first_attempt']['blocked']
            and bool(gate['first_attempt']['harmful_groups']) and gate['final_harmful']
            for gate in gates
        ),
        final_harm_by_group={group: wilson(sum(gate['final_truth_delta'][group] > 0 for gate in gates), len(rows))
                             for group in SEXES},
        direct_choice=dict(
            repeats=len(rows), harmful=sum(direct_harm),
            material_harmful=sum(bool(gate['direct_material_harm_groups']) for gate in gates),
            harmful_and_final_harmless=sum(harmful and not gate['final_harmful']
                                          for harmful, gate in zip(direct_harm, gates)),
            disagreements_with_final=sum(row['choices']['direct_small_swap_brier'] != row['choices']['selector2']
                                         for row in rows),
        ),
    )


def _truth(row, arm, condition, group):
    return row['judge_truth'][condition][row['choices'][arm]][group]


def _paired_delta(row, left, right, condition, group):
    if row['choices'][left] == row['choices'][right]:
        return 0.
    return float(_truth(row, left, condition, group)-_truth(row, right, condition, group))


def _contrast(rows, left, right, conditions):
    disagreements = sum(row['choices'][left] != row['choices'][right] for row in rows)
    summary, values = {}, {}
    for condition in conditions:
        summary[condition], values[condition] = {}, {}
        for group in GROUPS:
            deltas = [_paired_delta(row, left, right, condition, group) for row in rows]
            values[condition][group] = deltas
            summary[condition][group] = dict(
                independent_t_interval(deltas), disagreements=disagreements,
                exact_zero_differences=sum(delta == 0 for delta in deltas),
                negative_differences=sum(delta < 0 for delta in deltas),
                positive_differences=sum(delta > 0 for delta in deltas),
            )
    return summary, values


def summarize(results):
    """Summarize one fitted setting using every independent population exactly once."""
    rows = list(results)
    settings = {row['setting'] for row in rows}
    if len(rows) < 2 or len(settings) != 1 or not settings <= CONDITIONS.keys():
        raise ValueError('Expected two or more independent repeats from one fitted setting')
    if len({row['repeat'] for row in rows}) != len(rows):
        raise ValueError('Repeat numbers must be unique within a setting')
    setting = rows[0]['setting']
    conditions = CONDITIONS[setting]
    for row in rows:
        if any(row['choices'][arm] not in CANDIDATES for arm in ARMS):
            raise ValueError('All arm choices must come from the fixed candidate pool')

    choice_counts, arm_truth = {}, {}
    for arm in ARMS:
        counts = Counter(row['choices'][arm] for row in rows)
        choice_counts[arm] = {candidate: counts[candidate] for candidate in CANDIDATES}
        arm_truth[arm] = {
            condition: {group: independent_t_interval([_truth(row, arm, condition, group) for row in rows])
                        for group in GROUPS}
            for condition in conditions
        }

    focal_name, (left, right) = FOCAL[setting]
    focal_summary, focal_values = _contrast(rows, left, right, conditions)
    contrasts = {name: _contrast(rows, a, b, conditions)[0] for name, (a, b) in CONTRASTS.items()}
    contrasts[focal_name] = focal_summary
    summary = dict(setting=setting, repeats=len(rows), choice_counts=choice_counts,
                   arm_truth_errors=arm_truth, contrasts=contrasts)
    if setting == 'R':
        contrasts['selector2_minus_clean'] = _contrast(rows, 'selector2', 'plain_loss', conditions)[0]
        clean = focal_values['clean']['all']
        summary['same_repeat_reversals'] = {
            condition: sum(clean_delta > 0 and judge_delta < 0
                           for clean_delta, judge_delta in zip(clean, focal_values[condition]['all']))
            for condition in ('J1', 'J3')
        }
    else:
        summary['gate'] = _gate_summary(rows)
    return summary
