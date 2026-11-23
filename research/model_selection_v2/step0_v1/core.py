"""Outcome decomposition and the prespecified conditional interval estimators."""
import numpy as np

MODELS = ('logistic', 'decision_tree', 'random_forest', 'gradient_boosting')
METRICS = ('B0', 'Bs', 'M', 'C', 'deterioration', 'proxy')
SEXES = ('all', 'female', 'male')


def terms(p, q, y):
    p, q, y = (np.asarray(x, dtype=np.float64) for x in (p, q, y))
    m = (q-p)**2
    c = 2*(p-y)*(q-p)
    bs = (q-y)**2
    b0 = np.broadcast_to((p-y)**2, q.shape)
    np.testing.assert_allclose(bs-b0, m+c, rtol=0, atol=1e-12)
    return np.stack((b0, bs, m, c, bs-b0, b0+m), axis=-1)


def ratio(c, m):
    return float(c/m) if m != 0 else None


def interval(draws):
    return np.quantile(draws, [.025, .975], axis=0, method='linear')


def simultaneous(point, draws):
    point, draws = np.asarray(point), np.asarray(draws)
    sd = np.std(draws, axis=0, ddof=1)
    usable = sd > 1e-12
    bounds = np.full((2, len(point)), np.nan)
    critical = None
    if usable.any():
        t = np.max(np.abs((draws[:, usable]-point[usable])/sd[usable]), axis=1)
        critical = float(np.quantile(t, .95, method='higher'))
        bounds[:, usable] = point[usable]+np.array([-1, 1])[:, None]*critical*sd[usable]
    return bounds, {'critical': critical, 'excluded_coordinates': np.flatnonzero(~usable).tolist(),
                    'exclusion_label': 'no bootstrap variation', 'ddof': 1,
                    'quantile_method': 'higher', 'zero_variance_cutoff': 1e-12}


def status(bounds):
    if not np.isfinite(bounds).all() or np.any((bounds[0] <= 0) & (bounds[1] >= 0)):
        return 'unresolved'
    return 'same direction' if np.sign(bounds[0, 0]) == np.sign(bounds[0, 1]) else 'reversed'


def weighted(values, weights, mask):
    """Last axes are record x metric; every mean weights records, not groups."""
    w = np.asarray(weights)*mask
    if w.sum() == 0:
        raise ValueError('Empty bootstrap subgroup')
    return np.einsum('...ik,i->...k', values, w)/w.sum()


def bootstrap_means(values, counts, positions, mask):
    """Aggregate record sums to whole groups; matrix multiplication over draws."""
    # values: disturbance x record x metric
    ng = counts.shape[1]
    sums = np.zeros((ng, values.shape[0], values.shape[2]))
    np.add.at(sums, positions, values.transpose(1, 0, 2)*mask[:, None, None])
    sizes = np.bincount(positions, weights=mask, minlength=ng)
    denominator = counts@sizes
    if np.any(denominator == 0):
        raise ValueError('Empty bootstrap subgroup')
    result = (counts@sums.reshape(ng, -1)).reshape(len(counts), values.shape[0], -1)
    return result/denominator[:, None, None]


def paired_mc(a, b):
    values = np.asarray(a)-np.asarray(b)
    sd = np.std(values, axis=0, ddof=1)
    return sd, sd/np.sqrt(len(values))


def mc_summary_se(realizations):
    """Nine scenarios ordered family then mode; modes share repetition indices."""
    family = np.asarray(realizations).reshape(3, 3, 30, *np.asarray(realizations).shape[2:]).mean(axis=1)
    family_se = np.std(family, axis=1, ddof=1)/np.sqrt(30)
    return np.sqrt(np.sum(family_se**2, axis=0))/3
