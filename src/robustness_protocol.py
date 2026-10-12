"""Prespecified input corruption: training-only semantic rules and frozen choices."""
import hashlib
import json
import numpy as np
import pandas as pd
from src.baseline import ROOT, CATEGORICAL

MASTER_SEED = 20261007
REPETITIONS = 30
SAFE_LEVELS = [.01, .025, .05, .10]
SEMANTIC_LEVELS = {'primary': .10, 'stronger': .20}
STATUS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
MONETARY = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1,7)] + [f'PAY_AMT{i}' for i in range(1,7)]
GROUPS = {'credit_limit': ['LIMIT_BAL'], 'bill_amounts': [f'BILL_AMT{i}' for i in range(1,7)],
          'payment_amounts': [f'PAY_AMT{i}' for i in range(1,7)], 'repayment_status': STATUS,
          'age': ['AGE'], 'education': ['EDUCATION'], 'marriage': ['MARRIAGE']}
PRIMARY_SAFE = 'safe__group_all_features__p_0.05'
PRIMARY_SEMANTIC = 'semantic__all_features__primary'


def child_seed(scope, repetition, component):
    """SHA-256 stream identity independent of Python hash randomization and level."""
    text = f'{MASTER_SEED}|{scope}|{repetition}|{component}'
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'little')


def semantic_rules(train):
    rules = {}
    for f in train.columns:
        if f in MONETARY:
            integral = bool(np.equal(train[f], np.rint(train[f])).all())
            if not integral:
                raise ValueError('Observed monetary precision differs from inspected integer NT-dollar inputs')
            rules[f] = {'kind': 'money', 'iqr': float(train[f].quantile(.75) - train[f].quantile(.25)),
                        'lower': float(train[f].min()) if f == 'LIMIT_BAL' else (0. if f.startswith('PAY_AMT') else None),
                        'bound_source': 'training-support minimum, not a universal credit-limit rule' if f == 'LIMIT_BAL' else ('nonnegative payments' if f.startswith('PAY_AMT') else 'negative bills allowed'),
                        'upper': None, 'precision': 'integer NT dollars; numpy rint, ties to even'}
        elif f == 'AGE':
            rules[f] = {'kind': 'age', 'lower': int(train[f].min()), 'upper': int(train[f].max()),
                        'bound_source': 'training-support interval, not universal age bounds',
                        'magnitude': {'primary': 1, 'stronger': 2}}
        elif f in STATUS:
            support = sorted(int(x) for x in train[f].unique())
            maps = {}
            for level, step in [('primary',1), ('stronger',2)]:
                maps[level] = {str(s): {str(direction): s + direction*step if s >= 1 and s + direction*step >= 1 and s + direction*step in support else s
                                       for direction in (-1,1)} for s in support}
            rules[f] = {'kind': 'status', 'support': support, 'maps': maps,
                        'special_states_unchanged': [-2,-1,0],
                        'endpoint_rule': 'unsupported exact destination produces no-op; no jumps across missing codes',
                        'semantics': 'ordinary delays 1-8 measured in months; observed training support excludes 9+'}
        else:
            counts = train[f].value_counts().sort_index()
            rules[f] = {'kind': 'category', 'categories': [int(x) for x in counts.index],
                        'counts': [int(x) for x in counts],
                        'frequencies': (counts/counts.sum()).tolist(),
                        'rule': 'training empirical category frequencies conditional on differing from original'}
    return rules


def build_protocol(train, protected, safe_identity):
    order = list(train.columns)
    groups = {**GROUPS, 'all_features': order}
    scenarios = []
    for p in SAFE_LEVELS:
        for f in order:
            scenarios.append({'scenario': f'safe__feature_{f}__p_{p:g}', 'layer': 'safe', 'kind': 'individual',
                              'scope': f, 'features': [f], 'p': p, 'repetitions': 1})
        for scope, features in groups.items():
            scenarios.append({'scenario': f'safe__group_{scope}__p_{p:g}', 'layer': 'safe', 'kind': 'group',
                              'scope': scope, 'features': features, 'p': p, 'repetitions': 1})
    scenarios.append({'scenario': 'safe__group_all_features__p_0.05__permuted', 'layer': 'safe', 'kind': 'group',
                      'scope': 'all_features', 'features': order, 'p': .05, 'repetitions': 1, 'permutation_seed': 20261008})
    seeds = []
    for scope, features in [(f,[f]) for f in order] + [('all_features',order)]:
        for level,q in SEMANTIC_LEVELS.items():
            scenarios.append({'scenario': f'semantic__{scope}__{level}', 'layer': 'semantic', 'kind': 'joint' if scope=='all_features' else 'individual',
                              'scope': scope, 'features': features, 'level': level, 'q': q, 'repetitions': REPETITIONS})
        for repetition in range(1,REPETITIONS+1):
            seeds.append({'scope': scope, 'repetition': repetition,
                          'mask': child_seed(scope,repetition,'mask'),
                          'features': {f: {'sign': child_seed(scope,repetition,'sign:'+f),
                                           'category': child_seed(scope,repetition,'category:'+f)} for f in features}})
    return {'protocol_version': 1, 'protocol_date': '2026-10-08',
            'predictor_order': order, 'groups': groups, 'scenarios': scenarios,
            'primary_scenarios': [PRIMARY_SAFE, PRIMARY_SEMANTIC], 'semantic_rules': semantic_rules(train),
            'training_rows': len(train), 'scales_source': 'locked training split only',
            'master_seed': MASTER_SEED, 'child_seeds': seeds,
            'seed_formula': 'uint64 little-endian first 8 SHA-256 bytes of UTF8 master|scope|1-based repetition|component',
            'coupling': 'level-independent permutation; first floor(.10*N) nested inside first floor(.20*N); shared full-length feature sign and category uniforms across levels; streams independent across features and scopes',
            'signs': 'independent Bernoulli(.5) per record and feature, before level selection',
            'semantic_selection': 'complete holdout uniformly without replacement; eligibility never changes selection',
            'classification': {'threshold': .5, 'rule': 'probability_default > 0.50; equality is class 0',
                               'source': 'performance_evaluation.run_manifest.thresholds.class_rule; verified model.predict equivalence'},
            'metric_definitions': {'movement': 'perturbed minus original probabilities; absolute summaries across all records; changed-record supplement',
                'rank': 'Spearman average ranks; constant inputs undefined',
                'RGR': 'authentic SAFE rga(original probabilities, perturbed probabilities), point estimates only; model-specific magnitude normalization',
                'AP': 'sklearn.metrics.average_precision_score, not trapezoidal PR-AUC',
                'outcome_metrics': 'src.performance_evaluation.performance_metrics; recorded outcomes fixed',
                'calibration': '10 equal-width bins, count-weighted ECE using established reliability_bins',
                'zero_denominators': 'precision and F1 follow repository zero convention; recall/specificity undefined when their class absent',
                'quantiles': 'numpy linear quantiles', 'delta': 'perturbed minus baseline', 'paired': 'GBM minus LR',
                'direction': 'higher RGR/Spearman/AUC/AP/accuracy/precision/recall/specificity/F1 better; higher absolute movement/flip/error scores greater instability or error; signed movement has no good direction'},
            'aggregation': 'compute nonlinear metrics and baseline deltas per repetition, then paired GBM-LR, then mean; sample SD ddof=1 separately; deterministic SD undefined, not fabricated',
            'bootstrap': {'repetitions': 2000, 'seed': 42, 'confidence': .95, 'method': 'whole-original-predictor-group percentile, numpy linear quantiles',
                          'scenarios': [PRIMARY_SAFE,PRIMARY_SEMANTIC], 'metrics': ['mean_absolute_movement','flip_rate','delta_ROC-AUC','delta_Average Precision (AP)','delta_Brier Score'],
                          'sex': 'subset each globally sampled complete group resample; composition varies; never force exact strata',
                          'conditioning': 'frozen models and saved tail-swap realization or 30 saved semantic realizations; no re-perturbing in bootstrap',
                          'RGR_intervals': False},
            'protected_artifact_hashes': protected, 'safe_identity': safe_identity,
            'limits': 'synthetic research choices, not measured error frequencies; layers not severity-equivalent'}


def semantic_transform(X, scenario, rules):
    out = X.copy(deep=True)
    n = len(X)
    scope, repetition, level = scenario['scope'], scenario['repetition'], scenario['level']
    permutation = np.random.default_rng(child_seed(scope,repetition,'mask')).permutation(n)
    selected = np.zeros(n,dtype=bool)
    selected[permutation[:int(np.floor(scenario['q']*n))]] = True
    info = {}
    for feature in scenario['features']:
        rule, original = rules[feature], X[feature].to_numpy().copy()
        eligible = np.ones(n,dtype=bool)
        clipped = np.zeros(n,dtype=bool)
        outside = np.zeros(n,dtype=bool)
        destination = original.copy()
        signs = np.where(np.random.default_rng(child_seed(scope,repetition,'sign:'+feature)).random(n) < .5, -1, 1)
        if rule['kind'] in ('money','age'):
            lower, upper = rule['lower'], rule.get('upper')
            if lower is not None:
                outside |= original < lower
            if upper is not None:
                outside |= original > upper
            eligible &= ~outside
            active = selected & eligible
            magnitude = scenario['q']*rule['iqr'] if rule['kind']=='money' else rule['magnitude'][level]
            proposed = original[active].astype(float) + signs[active]*magnitude
            bounded = proposed.copy()
            if lower is not None:
                bounded = np.maximum(bounded,lower)
            if upper is not None:
                bounded = np.minimum(bounded,upper)
            clipped[active] = bounded != proposed
            destination[active] = np.rint(bounded).astype(original.dtype)
        elif rule['kind'] == 'status':
            eligible = original >= 1
            for i in np.flatnonzero(selected):
                entry = rule['maps'][level].get(str(int(original[i])))
                destination[i] = original[i] if entry is None else entry[str(int(signs[i]))]
        else:
            categories, frequencies = np.array(rule['categories']), np.array(rule['frequencies'])
            uniforms = np.random.default_rng(child_seed(scope,repetition,'category:'+feature)).random(n)
            for source in np.unique(original):
                alternative = categories != source
                positions = np.flatnonzero(selected & (original == source))
                if not alternative.any():
                    eligible[original==source] = False
                    continue
                prob = frequencies[alternative] / frequencies[alternative].sum()
                destination[positions] = categories[alternative][np.minimum(np.searchsorted(np.cumsum(prob),uniforms[positions],side='right'),alternative.sum()-1)]
        out[feature] = destination
        np.testing.assert_array_equal(destination[~selected], original[~selected])
        np.testing.assert_array_equal(destination[outside], original[outside])
        if rule['kind'] in ('category','status'):
            if not set(np.unique(destination)).issubset(set(np.unique(original)) | set(rule.get('categories',rule.get('support',[])))):
                raise AssertionError('Invalid semantic destination')
        info[feature] = {'selected': selected.copy(), 'eligible': eligible, 'clipped': clipped,
                         'outside_support': outside, 'zero_scale': rule['kind']=='money' and rule['iqr']==0,
                         'special_state': (original <= 0) if rule['kind']=='status' else np.zeros(n,dtype=bool)}
    pd.testing.assert_frame_equal(out.loc[~selected],X.loc[~selected],check_exact=True)
    return out, info


RATIONALE = '''# Prespecified robustness protocol — 8 October 2026

Saved before robustness results. The article-level motivation is the project's comparison of predictive performance and trustworthiness dimensions; this experiment does not validate a complete framework or establish novelty. The authentic installed SAFE operation exchanges opposite evaluation-input tails. The supplementary semantic rules are research choices about input errors, not SAFE requirements or measured operational frequencies. These layers are not severity-equivalent.

Canonical holdout input order, both frozen pipelines, all 22 predictors, ID/SEX/outcome associations, and strict p(default=1)>0.50 decisions are locked. No fitting, resplitting, feature selection, recalibration, or mitigation is allowed. Original data, models, split, previous predictions, reports, code and dependency locks are hashed and protected. Baseline reproduction tolerance is 1e-15 absolute for probabilities and 1e-12 for reproduced metric values; classes, IDs and metadata must agree exactly.

SAFE uses p=.01,.025,.05,.10 for every feature and eight specified groups. Its rounding selects min(ceil(p*N), N-ceil((1-p)*N)) pairs. Stable Python value sorting selects the extreme ends; ties can change who is exposed under row permutation. Genuine perturb is used without clipping or repairs. For joint scenarios report union exposure. The canonical all-feature p=.05 result is primary; one shared permutation (seed 20261008) is secondary and restored by ID. Numeric nominal education/marriage ordering is arbitrary; preserved marginals can coexist with broken dependencies and implausible complete records. Evaluation-defined tails are an explicit exception to training-only parameter estimation.

Semantic scopes are each feature and all features jointly, q=.10 and .20, 30 repetitions. Select floor(q*N) uniformly from the complete holdout; eligible records never redefine the sample. The joint scope uses a common mask across features. SHA-256 child seeds from master 20261007 nest primary in stronger masks and reuse signs/uniforms across levels. Signs are independent across records/features. Stronger stress increases both selection and magnitude. Signed fixed monetary offsets are q times each training IQR, never Gaussian or uniform-interval noise. Zero IQR gives zero-scale no-ops. Training amounts are integer NT dollars; numpy rint (ties to even) preserves that observed precision, rather than claiming a universal source precision rule. Payments stay nonnegative. Credit limit has a training-support lower bound (training minimum), with no maximum cap. Bills may be negative. Age uses training-support min/max, ±1 or ±2. Original values outside a bound remain unchanged; bounds and rounding affect selected eligible cells only.

Repayment special states -2,-1,0 stay unchanged. Ordinary delays move by exactly one or two months only if that exact destination is present in feature-specific training support and positive; unsupported endpoints are no-ops rather than jumps. In particular PAY_5/PAY_6 lack code 1. Eligibility denotes ordinary positive source states; directional endpoint rejection can still be a no-op. Category replacement samples training frequencies conditional on a different category. All maps, support, scale, bounds and child seeds are recorded in protocol.json before results.

Movement summaries use all records, plus explicitly labelled changed-input-record supplements. ROC-AUC, AP, Brier, log loss, fixed-rule classification, confusion counts and ten-bin ECE use established definitions. Precision/F1 retain the established zero-denominator convention. Constant-score Spearman and zero-normalization RGR are explicitly undefined. SAFE RGR uses original probability magnitudes and model-specific normalization; higher means more original-order preservation, not guaranteed probability quality, fairness or safety. No RGR intervals are planned. Semantic tests have conventional metrics, not falsely labelled authentic tail RGR.

Compute nonlinear metrics, perturbed-minus-baseline deltas and GBM-minus-LR differences separately per repetition, then average. Between-repetition sample SD describes Monte Carlo spread, not sampling uncertainty. Deterministic SAFE has no Monte Carlo SD. Outcomes remain fixed: changes are recorded-outcome sensitivity, not causal estimates of behavior changes.

The primary joint SAFE and primary joint semantic scenarios receive 2,000 whole-original-predictor-group bootstrap draws (seed 42), shared across models and scenarios. Preserve cross-SEX and cross-outcome groups; subset each resample for descriptive SEX summaries, so composition varies. Exact predictor matches need not be the same person. Percentile 95% intervals cover mean absolute movement, flip rate, changes in AUC/AP/Brier and paired model differences. Condition on saved predictions from the observed SAFE realization or all 30 semantic realizations; recompute each repetition's metrics under each sample, then average. Do not predict, average probabilities, concatenate clients, or generate new corruption inside bootstrap. Weighted sorted-score computations are validated against sklearn and expanded whole-group samples before use. Invalid results retain reasons and valid replicate counts. These intervals do not repeat the entire data-dependent SAFE procedure on new samples.

SEX=1 male / SEX=2 female remains unchanged metadata. Group results subset globally corrupted inputs and distinguish input exposure, model response, baseline distributions and proximity to .50; descriptive differences do not establish discrimination or replace prior fairness results. No significance grid or preferred winner is selected.

Synthetic tests cannot establish real-world error rates, causal default-risk changes, adversarial security, future-population reliability, a universally best model, or framework novelty. Holdout historical inspection remains disclosed. Later stability, PSI, monitoring, mitigation and governance stages are outside scope.
'''
