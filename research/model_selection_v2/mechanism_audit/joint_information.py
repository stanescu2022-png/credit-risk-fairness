"""Finite response-table counterexample, not a SAFE/empirical experiment."""
import numpy as np


def joint_fixture():
    """Same separate responses, distinct joint responses with outcome losses.

    A response is indexed by (record, stress flag, mask flag). Thus the table
    defines two coherent finite functions without assuming a fitted model.
    Nothing asserts these responses arise for the accepted Taiwan candidates.
    """
    outcome = np.array([0, 1, 0, 1])
    separate = {"clean": np.array([.25, .75, .25, .75]),
                "stress_only": np.array([.375, .625, .375, .625]),
                "mask_only": np.array([.5, .5, .5, .5])}
    return {"outcome": outcome,
            "candidate_A": {**{k: v.copy() for k, v in separate.items()},
                            "joint": np.array([.25, .75, .25, .75])},
            "candidate_B": {**{k: v.copy() for k, v in separate.items()},
                            "joint": np.array([.75, .25, .75, .25])}}
