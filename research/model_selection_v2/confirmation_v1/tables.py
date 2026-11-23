"""Frozen prediction contract extended only to retain unknown recorded sex."""
import numpy as np
from research.model_selection_v2.judge_core.tables import (
    PredictionTable, nonempty_label, identifiers, require, immutable_array)


class InclusivePredictionTable(PredictionTable):
    """Same validation as the frozen contract; sex 0 is additionally allowed.

    This subtype lets the unchanged loss functions include unknown-sex records
    in overall results, without assigning them to either audited subgroup.
    """
    def __post_init__(self):
        nonempty_label(self.role_label, 'role_label')
        ids = identifiers(self.ids, 'ID', unique=True)
        groups = identifiers(self.original_group, 'original_group')
        y,sex,p = (np.asarray(v) for v in (self.outcome,self.sex,self.probability))
        require(len(ids)>0, 'Prediction tables must be nonempty.')
        require(all(v.ndim==1 and len(v)==len(ids) for v in (y,sex,p)) and len(groups)==len(ids),
                'Prediction columns must be aligned one-dimensional vectors.')
        require(np.isin(y,[0,1]).all(), 'Outcome must contain only binary 0/1 values.')
        require(np.isin(sex,[0,1,2]).all(), 'SEX must contain project codes 0/1/2.')
        try:
            p = np.asarray(p,dtype=np.float64)
        except (TypeError,ValueError) as error:
            raise ValueError('Probability must be numeric float64 values.') from error
        require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'Probabilities must be finite and in [0,1].')
        object.__setattr__(self,'ids',ids)
        object.__setattr__(self,'original_group',groups)
        object.__setattr__(self,'outcome',immutable_array(y,np.int8))
        object.__setattr__(self,'sex',immutable_array(sex,np.int8))
        object.__setattr__(self,'probability',immutable_array(p,np.float64))
