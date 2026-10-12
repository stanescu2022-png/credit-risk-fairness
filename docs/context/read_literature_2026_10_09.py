from pathlib import Path
import hashlib, json, shutil
import pdfplumber
ROOT = Path.cwd()
DEST = ROOT / 'docs/context/sources/literature_2026-10-09'
FILES = {
 'correlation_metrics': 'risks-13-00178.pdf',
 'rank_graduation_box': '1-s2.0-S0957417424021067-main.pdf',
 'trustworthiness_review': 'electronics-14-02717-v2.pdf',
 'wasserstein': 'Evaluating SAFE AI principles using Wasserstein distance  a comparative study of Machine Learning models.pdf',
 'integrated_safe': '1-s2.0-S266682702500204X-main.pdf',
}
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
DEST.mkdir(parents=True, exist_ok=True)
records=[]
for key,name in FILES.items():
 source=Path('C:/Users/Andrei/Downloads')/name
 target=DEST/name
 if target.exists() and digest(target)!=digest(source): raise RuntimeError('Refusing replacement: '+str(target))
 if not target.exists(): shutil.copy2(source,target)
 with pdfplumber.open(target) as pdf:
  pages=[{'pdf_page':i,'text':p.extract_text(layout=False) or ''} for i,p in enumerate(pdf.pages,1)]
 text_path=DEST/(key+'.extracted.txt')
 text_path.write_text('\n\n'.join('===== PDF PAGE '+str(p['pdf_page'])+' =====\n'+p['text'] for p in pages),encoding='utf-8')
 records.append({'key':key,'supplied_path':str(source),'archived_path':str(target.relative_to(ROOT)),'sha256':digest(target),'bytes':target.stat().st_size,'pages':len(pages),'extracted_text':str(text_path.relative_to(ROOT)),'text_characters':sum(len(p['text']) for p in pages)})
 print(key,len(pages),'pages',sum(len(p['text']) for p in pages),'characters')
checks=[]
for mp in sorted(ROOT.glob('outputs/tables/*/run_manifest.json')):
 manifest=json.loads(mp.read_text(encoding='utf-8'))
 for cat in ('protected_artifact_hashes','locked_input_hashes','source_hashes','output_hashes'):
  for name,expected in manifest.get(cat,{}).items():
   p=ROOT/name
   checks.append({'manifest':str(mp.relative_to(ROOT)),'category':cat,'path':name,'matches':p.is_file() and digest(p)==expected})
receipt={'date':'2026-10-09','purpose':'full-text literature review; no research rerun','sources':records,'evaluation_manifest_checks':len(checks),'evaluation_hash_mismatches':[c for c in checks if not c['matches']]}
(DEST/'reading_provenance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Manifest checks:',len(checks),'mismatches:',len(receipt['evaluation_hash_mismatches']))