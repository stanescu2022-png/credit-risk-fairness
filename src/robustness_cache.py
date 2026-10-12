"""Exact sparse storage of paired frozen predictions in canonical holdout order."""
import numpy as np

FORMAT = 'canonical sparse probability cache v1'


def restore_probabilities(cache, key):
    """Fill saved row positions over the shared baseline, without arithmetic loss."""
    if str(cache['cache_format']) != FORMAT:
        raise ValueError('Unexpected robustness prediction-cache format')
    result=np.array(cache['baseline'],copy=True)
    stored=np.asarray(cache[key+'_stored'],dtype=bool)
    values=np.asarray(cache[key])
    if values.shape!=(result.shape[0],int(stored.sum())):
        raise ValueError('Sparse cached probabilities lost record alignment')
    result[:,stored]=values
    return result


def compact_cache(cache, registry):
    result={k:cache[k] for k in ['ID','SEX','actual','duplicate_group_id','baseline']}
    result['cache_format']=np.array(FORMAT)
    for row in registry:
        key=row['cache_key']
        changed=np.asarray(cache[key+'_changed'],dtype=bool)
        # Also retain any floating-point changes on unchanged inputs exactly.
        stored=changed | np.any(cache[key]!=cache['baseline'],axis=0)
        result[key]=cache[key][:,stored]
        result[key+'_stored']=stored
        result[key+'_changed']=changed
        np.testing.assert_array_equal(restore_probabilities(result,key),cache[key])
    return result
