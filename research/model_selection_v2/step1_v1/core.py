"""Step 1 estimators; all protocol constants are fixed before the run."""
import numpy as np
from scipy.stats import t
from research.model_selection_v2.pilot import CANDIDATES, require
from research.model_selection_v2.safe2026 import rank_candidates
from research.model_selection_v2.judge_core.bootstrap import percentile_summary
from research.model_selection_v2.step0_v1.core import simultaneous

MILD = np.array([0., .01, .02, .03, .04, .05])
FULL = np.linspace(0, 1, 23)*.5
ARMS = ('safe', 'selector2', 'plain_loss', 'order_mild', 'accuracy_full', 'selector2_before_check')
SEXES = (('all', None), ('female', 2), ('male', 1))


def skill(y, probability):
    y = np.asarray(y, dtype=float)
    variance = y.mean()*(1-y.mean())
    require(variance > 0, 'Undefined default-rate variance')
    out = 1-np.mean((np.asarray(probability)-y)**2, axis=-1)/variance
    require(np.isfinite(out).all(), 'Undefined accuracy-retained value')
    return out


def score(a, e, robustness):
    value = (a+e+robustness)/3
    require(np.isfinite(value), 'Undefined score')
    return float(value)


def ranking(scores):
    require(set(scores) == set(CANDIDATES) and all(np.isfinite(x) for x in scores.values()),
            'Every candidate needs a finite score')
    return [r['candidate_id'] for r in rank_candidates(scores, CANDIDATES)['rows']]


def choice(ranked, safe, blocked):
    return next(c for c in ranked if c == safe or not blocked[c])


def group_means(values, counts, positions, mask):
    """Record weighted, retaining whole cross-sex/outcome duplicate groups."""
    values = np.asarray(values)
    sums = np.zeros((counts.shape[1], values.shape[1]))
    np.add.at(sums, positions, values*mask[:, None])
    sizes = np.bincount(positions, weights=mask, minlength=counts.shape[1])
    den = counts@sizes
    require(np.all(den > 0), 'Empty bootstrap sex subgroup')
    result = (counts@sums)/den[:, None]
    require(np.isfinite(result).all(), 'Undefined bootstrap mean')
    return result


def fairness(point, draws):
    bounds, metadata = simultaneous(point, draws)
    # Step 0 excludes SD<=1e-12 coordinates as "no bootstrap variation".
    # Such a coordinate supplies no resolved interval and cannot block a switch.
    intervals = [[float(v) if np.isfinite(v) else None for v in bounds[:, j]] for j in range(2)]
    blocked = any(lo is not None and lo > 0 for lo, hi in intervals)
    return {'point_female_male': list(map(float, point)), 'joint_intervals_female_male': intervals,
            'blocked': blocked, 'construction': metadata}


def percentile(values):
    return percentile_summary(values, percentile_bounds=(2.5, 97.5))


def paired_difference(values, other, safe):
    return 0. if other == safe else float(values[other]-values[safe])


def corrected(values, rho, adjusted=False):
    values = np.asarray(values, dtype=float)
    require(len(values) >= 2 and np.isfinite(values).all(), 'Invalid repeat contrast')
    mean, variance = float(values.mean()), float(values.var(ddof=1))
    if variance == 0:
        return {'estimate': mean, 'lower': None, 'upper': None, 'sample_sd': 0.,
                'reason': 'zero between-repeat sample variance; interval not reported', 'rho': float(rho)}
    critical = float(t.ppf(.9875 if adjusted else .975, len(values)-1))
    se = float(np.sqrt((1/len(values)+rho)*variance))
    return {'estimate': mean, 'lower': mean-critical*se, 'upper': mean+critical*se,
            'sample_sd': float(np.sqrt(variance)), 'corrected_se': se, 'rho': float(rho),
            't_quantile': critical, 'df': len(values)-1, 'adjusted': adjusted}


def pattern(safe, order_mild, accuracy_full, accuracy_mild):
    return tuple(int(c != safe) for c in (accuracy_full, order_mild, accuracy_mild))
