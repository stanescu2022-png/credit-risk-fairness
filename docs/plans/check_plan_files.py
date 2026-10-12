from pathlib import Path
import re,hashlib,json
root=Path.cwd()
files=[root/'docs/plans'/n for n in ['safe_comparison_plan_2026-10-09.md','safe_comparison_technical_protocol_2026-10-09.md','safe_scoring_worked_example_2026-10-09.md']]
for p in files:
 text=p.read_text(encoding='utf-8')
 for link in re.findall(r'\]\(([^)]+)\)',text):
  if not link.startswith(('https://','http://','#')): assert (p.parent/link).resolve().is_file(),(p,link)
 print(p.name,len(text.split()),'words; local links resolve')
count=0;bad=[]
for p in root.glob('outputs/tables/*/run_manifest.json'):
 m=json.loads(p.read_text(encoding='utf-8'))
 for cat in ['protected_artifact_hashes','locked_input_hashes','source_hashes','output_hashes']:
  for name,expected in m.get(cat,{}).items():
   f=root/name;count+=1
   if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:bad.append(name)
assert count==410 and not bad
print('All 410 evaluation artifact hashes still match.')