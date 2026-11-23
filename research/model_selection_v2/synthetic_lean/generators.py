"""Fixed synthetic populations and label-free calibration for the lean study."""
from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit, ndtri, roots_hermitenorm

from research.model_selection_v2.confirmation_v1.schema import Schema
from research.model_selection_v2.safe2026 import tail_swap
from research.model_selection_v2.step1_v1.core import MILD


MASTER_SEED = 20261117
POPULATION_SIZE = 30_000
REFERENCE_SIZE = 1_000_000
TARGET_PREVALENCE = .22
TARGET_AUC = .78
FEMALE_SHARE = .60
F_CURVATURE = .67
F_FEMALE_STRENGTH = .5
R_TAIL_MASS = .05
R_TAIL_THRESHOLD = float(ndtri(1 - R_TAIL_MASS))
R_CENTRAL_HALF_THRESHOLD = float(ndtri(.75))

R_FIELDS = tuple(f'x{i}' for i in range(1, 16))
R_FAMILIES = {
    'A': ('x1', 'x2', 'x5', 'x6', 'x7'),
    'B': ('x3', 'x8', 'x9', 'x10', 'x11'),
    'C': ('x4', 'x12', 'x13', 'x14', 'x15'),
}
F_FAMILIES = {
    'A': ('age', 'dependants', 'education', 'marital_status', 'residence_type'),
    'B': ('monthly_income', 'credit_limit_requested', 'existing_debt', 'months_in_job', 'open_credit_lines'),
    'C': ('worst_delay', 'months_with_delay', 'utilisation', 'product_type', 'past_restructuring'),
}
F_FIELDS = tuple(name for fields in F_FAMILIES.values() for name in fields)
F_ORDERED = {'education': (1, 2, 3, 4), 'worst_delay': (-1, 0, 1, 2, 3, 4)}
F_UNORDERED = ('marital_status', 'residence_type', 'product_type', 'past_restructuring')
F_LINEAR_TERMS = (
    ('age', -1), ('education', -1), ('monthly_income', -1), ('existing_debt', 1),
    ('months_in_job', -1), ('open_credit_lines', 1), ('worst_delay', 1),
    ('months_with_delay', 1), ('utilisation', 1), ('past_restructuring', 1),
)


def study_schema(setting):
    if setting == 'R':
        return Schema(R_FIELDS, R_FIELDS, {}, ())
    if setting == 'F':
        numeric = tuple(name for name in F_FIELDS if name not in F_ORDERED and name not in F_UNORDERED)
        return Schema(F_FIELDS, numeric, F_ORDERED.copy(), F_UNORDERED)
    raise ValueError('Expected synthetic setting R or F')


def study_families(setting):
    if setting == 'R':
        return R_FAMILIES.copy()
    if setting == 'F':
        return F_FAMILIES.copy()
    raise ValueError('Expected synthetic setting R or F')


def child_seed(purpose, setting, repeat, field):
    """First 8 SHA-256 bytes, little-endian, shared with the no-fit control."""
    if any(not isinstance(token, str) or not token or ':' in token for token in (purpose, setting, field)):
        raise ValueError('Seed purpose, setting and field must be nonempty and contain no colon')
    if isinstance(repeat, bool) or not isinstance(repeat, Integral) or repeat < 0:
        raise ValueError('Seed repeat must be a nonnegative integer')
    scope = f'{MASTER_SEED}:{purpose}:{setting}:{repeat}:{field}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(scope).digest()[:8], 'little')


def _rng(purpose, setting, repeat, field):
    return np.random.default_rng(child_seed(purpose, setting, repeat, field))


@dataclass(frozen=True)
class FReference:
    records: int
    median_income: float
    linear_means: tuple
    linear_scales: tuple
    nonlinear_means: tuple
    nonlinear_scales: tuple
    female_debt_mean: float
    female_debt_scale: float
    mixture_mean: float
    mixture_scale: float
    common_risk_mean: float
    common_risk_auc: float
    f_risk_mean: float
    f_risk_auc: float

    def __post_init__(self):
        if not isinstance(self.records, Integral) or self.records < 2:
            raise ValueError('Invalid F calibration size')
        for name, size in (('linear_means', len(F_LINEAR_TERMS)), ('linear_scales', len(F_LINEAR_TERMS)),
                           ('nonlinear_means', 5), ('nonlinear_scales', 5)):
            values = tuple(getattr(self, name))
            if len(values) != size or not np.isfinite(values).all():
                raise ValueError(f'Invalid F calibration values for {name}')
            object.__setattr__(self, name, values)
        if not all(np.isfinite(value) and value > 0 for value in (
            self.median_income, *self.linear_scales, *self.nonlinear_scales,
            self.female_debt_scale, self.mixture_scale,
        )):
            raise ValueError('F calibration requires positive finite scales and median income')
        if not np.isfinite([self.female_debt_mean, self.mixture_mean]).all():
            raise ValueError('F calibration requires finite centering values')
        if not all(0 < value < 1 for value in (
            self.common_risk_mean, self.common_risk_auc, self.f_risk_mean, self.f_risk_auc,
        )):
            raise ValueError('Invalid F calibration summary')
        if abs(self.common_risk_mean - TARGET_PREVALENCE) > 1e-10 or abs(self.common_risk_auc - TARGET_AUC) > 1e-10:
            raise ValueError('Common L2 reference did not reach its calibration targets')


@dataclass(frozen=True)
class Calibration:
    setting: str
    intercept: float
    scale: float
    f_reference: FReference | None = None

    def __post_init__(self):
        valid_r = self.setting == 'R' and self.scale == 1. and self.f_reference is None
        valid_f = self.setting == 'F' and isinstance(self.f_reference, FReference)
        if not (valid_r or valid_f) or not np.isfinite(self.intercept) or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid synthetic calibration')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        if values.get('f_reference') is not None:
            values['f_reference'] = FReference(**values['f_reference'])
        return cls(**values)


@dataclass(frozen=True)
class Population:
    X: pd.DataFrame
    ids: np.ndarray
    sex: np.ndarray
    y: np.ndarray
    p: np.ndarray
    dataset_id: str


def _row_ids(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError('Population size must be a positive integer')
    ids = np.arange(1, n + 1, dtype=np.int64)
    ids.flags.writeable = False
    return ids


def _sex(purpose, setting, repeat, n):
    return (1 + (_rng(purpose, setting, repeat, 'sex').random(n) < FEMALE_SHARE)).astype(np.int8)


def _r_features(purpose, repeat, ids):
    draws = {name: _rng(purpose, 'R', repeat, name).standard_normal(len(ids)) for name in R_FIELDS}
    return pd.DataFrame(draws, index=pd.Index(ids, name='ID'), dtype=np.float64)


def _f_features(purpose, repeat, ids, sex):
    n = len(ids)

    def rng(field):
        return _rng(purpose, 'F', repeat, field)

    age = rng('age').integers(21, 71, size=n)
    dependants = np.minimum(rng('dependants').poisson(1., n), 6)
    education = rng('education').choice(F_ORDERED['education'], n, p=(.15, .40, .30, .15))
    marital_status = rng('marital_status').choice((1, 2, 3), n, p=(.45, .45, .10))
    residence_type = rng('residence_type').choice((1, 2, 3, 4), n, p=(.40, .30, .20, .10))
    income = 3000. * np.exp(.15 * (education - 2) + .5 * rng('monthly_income').standard_normal(n))
    income *= np.where(sex == 2, .9, 1.)
    limit = 3. * income * np.exp((2 / 3) * rng('credit_limit_requested').standard_normal(n))
    debt = 18000. * (income / 3000.) ** .3 * np.exp(.477 * rng('existing_debt').standard_normal(n))
    months_in_job = rng('months_in_job').gamma(4., (60. + .6 * (age - 45.5)) / 4.)
    open_lines = np.minimum(rng('open_credit_lines').poisson(2., n), 10)
    delay = rng('worst_delay').choice(F_ORDERED['worst_delay'], n, p=(.30, .45, .12, .07, .04, .02))
    delayed = delay > 0
    months_with_delay = np.zeros(n)
    months_with_delay[delayed] = 1 + rng('months_with_delay').binomial(5, delay[delayed] / 5.)
    utilisation = rng('utilisation').beta(1.8, 2.2, n)
    product = rng('product_type').choice((1, 2, 3), n, p=(.60, .30, .10))
    restructuring = rng('past_restructuring').binomial(1, .05, n)
    columns = (age, dependants, education, marital_status, residence_type, income, limit, debt,
               months_in_job, open_lines, delay, months_with_delay, utilisation, product, restructuring)
    return pd.DataFrame(dict(zip(F_FIELDS, columns)), index=pd.Index(ids, name='ID'), dtype=np.float64)


def _r_components(frame):
    x1 = frame['x1'].to_numpy()
    x2 = frame['x2'].to_numpy()
    tail = (x1 > R_TAIL_THRESHOLD).astype(np.int8) - (x1 < -R_TAIL_THRESHOLD)
    half = np.where(np.abs(x2) > R_CENTRAL_HALF_THRESHOLD, 1, -1)
    background = .8 * (frame['x3'].to_numpy() + frame['x4'].to_numpy()) + .4 * half
    return tail, half, background


def _f_terms(frame, median_income):
    income = frame['monthly_income'].to_numpy()
    debt_excess = frame['existing_debt'].to_numpy() - 6. * income
    utilisation = frame['utilisation'].to_numpy()
    delay = frame['worst_delay'].to_numpy()
    linear = np.column_stack([sign * frame[name].to_numpy() for name, sign in F_LINEAR_TERMS])
    nonlinear = np.column_stack((
        utilisation > .9,
        (delay >= 2) & (utilisation > .6),
        (frame['age'].to_numpy() - 45.5) ** 2,
        -np.minimum(income, median_income),
        debt_excess > 0.,
    ))
    return linear, nonlinear, debt_excess


def _f_indices(linear, nonlinear, debt_excess, female, *, linear_stats, nonlinear_stats, excess_stats):
    linear_index = ((linear - linear_stats[0]) / linear_stats[1]).sum(axis=1)
    nonlinear_terms = (nonlinear - nonlinear_stats[0]) / nonlinear_stats[1]
    common = (1 - F_CURVATURE) * linear_index + F_CURVATURE * nonlinear_terms.sum(axis=1)
    nonlinear_terms[female, -1] = F_FEMALE_STRENGTH * (debt_excess[female] - excess_stats[0]) / excess_stats[1]
    actual = (1 - F_CURVATURE) * linear_index + F_CURVATURE * nonlinear_terms.sum(axis=1)
    return common, actual


def generate_population(setting, purpose, repeat, calibration, *, n=POPULATION_SIZE):
    """Use fixed calibration and independent streams; keep labels and true risks out of X."""
    if not isinstance(calibration, Calibration) or calibration.setting != setting:
        raise ValueError('Population setting and calibration must match')
    ids = _row_ids(n)
    sex = _sex(purpose, setting, repeat, n)
    if setting == 'R':
        frame = _r_features(purpose, repeat, ids)
        tail, half, background = _r_components(frame)
        risk = expit(calibration.intercept + background + .8 * tail * half)
    else:
        frame = _f_features(purpose, repeat, ids, sex)
        reference = calibration.f_reference
        linear, nonlinear, excess = _f_terms(frame, reference.median_income)
        _, actual = _f_indices(linear, nonlinear, excess, sex == 2,
                               linear_stats=(reference.linear_means, reference.linear_scales),
                               nonlinear_stats=(reference.nonlinear_means, reference.nonlinear_scales),
                               excess_stats=(reference.female_debt_mean, reference.female_debt_scale))
        risk = expit(calibration.intercept + calibration.scale * (actual - reference.mixture_mean) / reference.mixture_scale)
    y = _rng(purpose, setting, repeat, 'outcome').binomial(1, risk).astype(np.int8)
    for vector in (sex, y, risk):
        vector.flags.writeable = False
    return Population(frame, ids, sex, y, risk, f'synthetic_lean:{purpose}:{setting}:{repeat}')


def _weighted_auc(ranked_probability, tie_starts):
    """AUC with each row supplying positive weight p and negative weight 1-p."""
    positive = np.add.reduceat(ranked_probability, tie_starts)
    negative = np.diff(tie_starts, append=len(ranked_probability)) - positive
    negatives_below_or_tied = np.cumsum(negative) - .5 * negative
    return float(np.sum(positive * negatives_below_or_tied) / (positive.sum() * negative.sum()))


def _rank(scores):
    order = np.argsort(scores, kind='stable')
    ranked = scores[order]
    tie_starts = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    return order, tie_starts


def _prevalence_intercept(signal):
    target_logit = logit(TARGET_PREVALENCE)
    return brentq(lambda intercept: expit(intercept + signal).mean() - TARGET_PREVALENCE,
                  target_logit - signal.max() - 1., target_logit - signal.min() + 1., xtol=1e-13)


def _calibrate_f(n):
    ids = _row_ids(n)
    sex = _sex('reference', 'F', 0, n)
    frame = _f_features('reference', 0, ids, sex)
    median_income = float(frame['monthly_income'].median())
    linear, nonlinear, excess = _f_terms(frame, median_income)
    linear_stats = linear.mean(axis=0), linear.std(axis=0)
    nonlinear_stats = nonlinear.mean(axis=0), nonlinear.std(axis=0)
    excess_stats = float(excess.mean()), float(excess.std())
    if min(*linear_stats[1], *nonlinear_stats[1], excess_stats[1]) <= 0:
        raise ValueError('Degenerate risk term on F calibration reference')
    common, actual = _f_indices(linear, nonlinear, excess, sex == 2,
                                linear_stats=linear_stats, nonlinear_stats=nonlinear_stats, excess_stats=excess_stats)
    mixture_mean, mixture_scale = float(common.mean()), float(common.std())
    if not np.isfinite(mixture_scale) or mixture_scale <= 0:
        raise ValueError('Degenerate risk index on F calibration reference')
    common = (common - mixture_mean) / mixture_scale
    actual = (actual - mixture_mean) / mixture_scale
    order, tie_starts = _rank(common)
    ranked_common = common[order]

    def solve_auc(scale):
        signal = scale * ranked_common
        intercept = _prevalence_intercept(signal)
        probability = expit(intercept + signal)
        return _weighted_auc(probability, tie_starts) - TARGET_AUC

    scale = brentq(solve_auc, 0., 16., xtol=1e-12)
    intercept = _prevalence_intercept(scale * ranked_common)
    p_common = expit(intercept + scale * ranked_common)
    p_actual = expit(intercept + scale * actual)
    actual_order, actual_ties = _rank(actual)
    reference = FReference(
        records=n, median_income=median_income,
        linear_means=tuple(map(float, linear_stats[0])), linear_scales=tuple(map(float, linear_stats[1])),
        nonlinear_means=tuple(map(float, nonlinear_stats[0])), nonlinear_scales=tuple(map(float, nonlinear_stats[1])),
        female_debt_mean=excess_stats[0], female_debt_scale=excess_stats[1],
        mixture_mean=mixture_mean, mixture_scale=mixture_scale,
        common_risk_mean=float(p_common.mean()), common_risk_auc=_weighted_auc(p_common, tie_starts),
        f_risk_mean=float(p_actual.mean()), f_risk_auc=_weighted_auc(p_actual[actual_order], actual_ties),
    )
    return Calibration('F', float(intercept), float(scale), reference)


def calibrate(setting):
    """Calibrate once without labels or fitted estimators."""
    if setting == 'R':
        nodes, weights = roots_hermitenorm(64)
        weights /= np.sqrt(2 * np.pi)
        tail = np.array([-1, 0, 1])[:, None, None]
        half = np.array([-1, 1])[None, :, None]
        mixture_weight = np.array([R_TAIL_MASS, 1 - 2 * R_TAIL_MASS, R_TAIL_MASS])[:, None, None] * .5
        signal = .8 * np.sqrt(2) * nodes + (.4 + .8 * tail) * half

        def prevalence_difference(intercept):
            return np.sum(mixture_weight * weights * expit(intercept + signal)) - TARGET_PREVALENCE

        return Calibration('R', float(brentq(prevalence_difference, -10., 0., xtol=1e-13)), 1.)
    if setting == 'F':
        return _calibrate_f(REFERENCE_SIZE)
    raise ValueError('Expected synthetic setting R or F')


def _preferred(full, omitted):
    if full == omitted:
        return 'tie'
    return 'full' if full < omitted else 'omitted'


def _r_curve(calibration, ids):
    frame = _r_features('reference', 0, ids)
    tail, half, background = _r_components(frame)
    original_signal = tail * half
    original_p = expit(calibration.intercept + background + .8 * original_signal)
    results = []
    for q in MILD:
        transformed, exposure = tail_swap(frame, ids, q)
        swapped_tail, swapped_half, swapped_background = _r_components(transformed)
        swapped_signal = swapped_tail * swapped_half
        omitted = expit(calibration.intercept + swapped_background)
        full = expit(calibration.intercept + swapped_background + .8 * swapped_signal)
        full_loss = float(((full - original_p) ** 2).mean())
        omitted_loss = float(((omitted - original_p) ** 2).mean())
        results.append(dict(
            q=float(q), full_truth_mse=full_loss, omitted_truth_mse=omitted_loss,
            full_minus_omitted=full_loss - omitted_loss,
            ideal_full_mse_in_a_squared=8 * min(float(q), R_TAIL_MASS),
            ideal_omitted_mse_in_a_squared=2 * R_TAIL_MASS,
            tail_size_per_feature=exposure['tail_size_per_feature'],
            rows_with_any_input_change=exposure['records_with_any_change'],
            interaction_signal_changed=int(np.count_nonzero(swapped_signal != original_signal)),
            interaction_sign_reversed=int(np.count_nonzero((original_signal != 0) & (swapped_signal == -original_signal))),
        ))
    actual_full_mean = float(np.mean([row['full_truth_mse'] for row in results]))
    actual_omitted_mean = float(np.mean([row['omitted_truth_mse'] for row in results]))
    ideal_full_mean = float(np.mean([row['ideal_full_mse_in_a_squared'] for row in results]))
    ideal_omitted_mean = 2 * R_TAIL_MASS
    return dict(
        records=len(ids), original_mean_risk=float(original_p.mean()), population_tail_mass_per_side=R_TAIL_MASS,
        tail_counts=dict(lower=int(np.count_nonzero(tail == -1)), upper=int(np.count_nonzero(tail == 1))),
        h_counts=dict(inside=int(np.count_nonzero(half == -1)), outside=int(np.count_nonzero(half == 1))),
        nodes=results,
        actual_six_node_mean=dict(full=actual_full_mean, omitted=actual_omitted_mean),
        ideal_six_node_mean_in_a_squared=dict(full=ideal_full_mean, omitted=ideal_omitted_mean),
        clean_preferred=_preferred(results[0]['full_truth_mse'], results[0]['omitted_truth_mse']),
        six_node_preferred=_preferred(actual_full_mean, actual_omitted_mean),
        ideal_clean_preferred=_preferred(0., ideal_omitted_mean),
        ideal_six_node_preferred=_preferred(ideal_full_mean, ideal_omitted_mean),
        ideal_crossover_tail_fraction=R_TAIL_MASS / 4.,
    )


def r_reference_curve(calibration):
    """Audit both unfitted R reference forecasts against original probabilities."""
    if not isinstance(calibration, Calibration) or calibration.setting != 'R':
        raise ValueError('R reference curve requires R calibration')
    return _r_curve(calibration, _row_ids(REFERENCE_SIZE))
