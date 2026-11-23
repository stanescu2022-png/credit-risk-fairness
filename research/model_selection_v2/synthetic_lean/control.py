"""No-fit false-veto control for the frozen two-sex joint gate."""

from numbers import Integral

import numpy as np

from research.model_selection_v2.step1_v1.core import fairness

from .generators import child_seed
from .summaries import wilson


CONTROL_REPEATS = 1_000
BOOTSTRAP_DRAWS = 2_000
TRUE_DEFAULT_PROBABILITY = .22
INCUMBENT = .17
CHALLENGER = .27
SEX_SIZES = (2_880, 1_920)  # Female and male, respectively.
OUTCOME_LOSSES = (np.array([INCUMBENT, CHALLENGER])[None, :] - np.arange(2)[:, None])**2


def _sample_bootstrap_cells(observed_cells, rng, draws):
    """Counts by sex and outcome in shared, unstratified singleton-bootstrap draws.

    Every uniform borrower draw falls in a cell with probability cell_size/n.
    The cell totals of n independent such draws are exactly multinomial. The
    bootstrap statistic depends on the draw only through these four totals.
    """
    n = int(observed_cells.sum())
    counts = rng.multinomial(n, observed_cells.ravel()/n, size=draws)
    return counts.reshape(draws, 2, 2)


def _gate_from_cells(observed_cells, bootstrap_cells):
    """Use the same group Brier contrast and joint bound as the fitted gate."""
    def contrast(cells):
        sizes = cells.sum(axis=-1)
        if np.any(sizes == 0):
            raise ValueError('Empty bootstrap sex subgroup')
        loss = (cells @ OUTCOME_LOSSES)/sizes[..., None]
        return loss[..., 1]-loss[..., 0]

    return fairness(contrast(observed_cells), contrast(bootstrap_cells))


def control_trials(repeats, *, purpose='control'):
    """Run independent, explicitly indexed control trials or a discarded timing batch."""
    repeats = tuple(repeats)
    if (any(isinstance(r, bool) or not isinstance(r, Integral) or r < 0 for r in repeats)
            or len(set(repeats)) != len(repeats)):
        raise ValueError('Control repeat numbers must be unique nonnegative integers')
    records = []
    n_female, n_male = SEX_SIZES
    for repeat in repeats:
        repeat = int(repeat)
        outcomes = np.random.default_rng(child_seed(purpose, 'control', repeat, 'outcome'))
        labels = outcomes.binomial(1, TRUE_DEFAULT_PROBABILITY, size=n_female+n_male)
        events = np.array([labels[:n_female].sum(), labels[n_female:].sum()])
        observed_cells = np.column_stack((np.asarray(SEX_SIZES)-events, events))
        bootstrap = np.random.default_rng(child_seed(purpose, 'control', repeat, 'bootstrap'))
        bootstrap_cells = _sample_bootstrap_cells(observed_cells, bootstrap, BOOTSTRAP_DRAWS)
        check = _gate_from_cells(observed_cells, bootstrap_cells)
        records.append(dict(repeat=repeat, purpose=purpose, blocked=check['blocked'],
                            observed_cells_female_male=observed_cells.tolist(), gate=check))
    return records


def summarize_control(trials):
    """Describe the rate for this fixed pair, over every independent control trial."""
    trials = list(trials)
    if (len({trial['repeat'] for trial in trials}) != len(trials)
            or len({trial['purpose'] for trial in trials}) > 1):
        raise ValueError('Control trial identities must be unique within one seed purpose')
    return dict(
        trials=len(trials),
        borrowers_per_trial=sum(SEX_SIZES),
        female_borrowers_per_trial=SEX_SIZES[0],
        male_borrowers_per_trial=SEX_SIZES[1],
        bootstrap_draws=BOOTSTRAP_DRAWS,
        true_default_probability=TRUE_DEFAULT_PROBABILITY,
        forecasts=dict(incumbent=INCUMBENT, challenger=CHALLENGER),
        truth_squared_error_per_forecast=.0025,
        false_vetoes=wilson(sum(trial['blocked'] for trial in trials), len(trials)),
    )
