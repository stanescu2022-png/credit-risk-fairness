from pathlib import Path
import json
rows=[]
for label,loss in [('A',[.11,.24]),('B',[.115,.13]),('C',[.16,.16])]:
 risk=max(loss);rows.append({'candidate':label,'losses':loss,'risk':risk,'score':100*(1-risk/.16)})
gain=rows[0]['risk']-rows[1]['risk']
assert abs(rows[0]['score']+50)<1e-12
assert abs(rows[1]['score']-18.75)<1e-12
assert abs(rows[2]['score'])<1e-12
assert abs(gain-.11)<1e-12
assert abs((max([.12,.24,.18])-max([.125,.13,.15]))-.09)<1e-12
assert abs(max(a-b for a,b in zip([.12,.24,.18],[.125,.13,.15]))-.11)<1e-12
bounds=[.230-.140,.250-.120]
assert abs(bounds[0]-.09)<1e-12 and abs(bounds[1]-.13)<1e-12
result={'scope':'constructed planning illustrations only; no credit data/models loaded','rows':rows,'gain':gain,'relative_gain_percent':100*gain/rows[0]['risk'],'gain_bounds':bounds,'checks_passed':True}
Path('docs/plans/scoring_example_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))