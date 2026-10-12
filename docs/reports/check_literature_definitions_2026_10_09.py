"""Small definition checks for literature review; no project data/models loaded."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

def transport_checks(a,b):
 a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
 cost=0.5*(a[:,None]-b[None,:])**2
 rows,cols=linear_sum_assignment(cost)
 return {'half_squared_cost_matrix':cost.tolist(),'uniform_product_plan_cost':float(cost.mean()),'optimal_half_squared_transport_cost':float(cost[rows,cols].mean()),'standard_squared_wasserstein':float(2*cost[rows,cols].mean()),'paired_mean_squared_change':float(np.mean((a-b)**2))}
identity=transport_checks([0,1],[0,1])
permutation=transport_checks([0.1,0.9],[0.9,0.1])
toy=transport_checks([0,1],[0,2])
assert identity['uniform_product_plan_cost']==0.25 and identity['optimal_half_squared_transport_cost']==0
assert permutation['optimal_half_squared_transport_cost']==0 and np.isclose(permutation['paired_mean_squared_change'],0.64)
assert toy['uniform_product_plan_cost']==0.75 and toy['optimal_half_squared_transport_cost']==0.25
# Eq. (13)-(17) arithmetic aggregation collapses algebraically to the average of vector means.
a=np.array([0.7,0.3]); e=np.array([1,0.8]); r=np.array([1,0.2])
tensor=(a[:,None,None]+e[None,:,None]+r[None,None,:])/3
assert np.isclose(tensor.mean(),(a.mean()+e.mean()+r.mean())/3)
result={'date':'2026-10-09','scope':'definition checks using constructed numbers only; no credit experiments or replication',
 'wasserstein_identity':identity,'prediction_permutation':permutation,'paper_toy_values':toy,
 'paired_brier_before':float(np.mean((np.array([0,1])-np.array([0.1,0.9]))**2)),
 'paired_brier_after':float(np.mean((np.array([0,1])-np.array([0.9,0.1]))**2)),
 'integrated_arithmetic_tensor_mean':float(tensor.mean()),
 'integrated_arithmetic_mean_of_curve_means':float((a.mean()+e.mean()+r.mean())/3),
 'interpretation':'The printed uniform-coupling formula is not optimal transport in general. Correct marginal transport is permutation invariant; paired loss is not. Independent arithmetic cross-product aggregation cannot encode measured feature-removal/noise interactions.'}
p=Path('docs/reports/literature_definition_checks_2026-10-09.json')
p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))