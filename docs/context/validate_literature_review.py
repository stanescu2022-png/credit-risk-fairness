from pathlib import Path
import hashlib,json
root=Path.cwd()
src=root/'docs/context/sources/literature_2026-10-09'
receipt=json.loads((src/'reading_provenance.json').read_text(encoding='utf-8'))
for s in receipt['sources']:
 p=root/s['archived_path']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==s['sha256']
 assert p.read_bytes()==Path(s['supplied_path']).read_bytes()
 text=(root/s['extracted_text']).read_text(encoding='utf-8')
 assert text.count('===== PDF PAGE ')==s['pages']
assert sum(s['pages'] for s in receipt['sources'])==108
mismatches=[];count=0
for mp in root.glob('outputs/tables/*/run_manifest.json'):
 m=json.loads(mp.read_text(encoding='utf-8'))
 for cat in ('protected_artifact_hashes','locked_input_hashes','source_hashes','output_hashes'):
  for name,expected in m.get(cat,{}).items():
   p=root/name;count+=1
   if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:mismatches.append(name)
assert count==410 and not mismatches
report=root/'docs/reports/safe_full_text_review_2026-10-09.md'
s=report.read_text(encoding='utf-8').replace('a independently specified','an independently specified')
report.write_text(s,encoding='utf-8')
context=(root/'PROJECT_CONTEXT.md').read_text(encoding='utf-8')
assert '## Full-text review of five supplied SAFE papers, 9 October 2026' in context
rendered=[str(p.relative_to(root)) for p in sorted((root/'tmp/pdfs/literature_2026-10-09').glob('*.png'))]
assert len(rendered)==14
validation={'date':'2026-10-09','source_copies_byte_identical':5,'extracted_pages':108,'visually_inspected_pages':rendered,'evaluation_hash_checks':count,'mismatches':mismatches,'new_credit_experiments':False,'definition_check_receipt':'docs/reports/literature_definition_checks_2026-10-09.json'}
(src/'review_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
print('5 source copies identical; 108 extracted pages; 14 rendered pages visually checked; 410 artifact checks passed; dated context update present.')