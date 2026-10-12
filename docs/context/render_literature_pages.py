from pathlib import Path
import subprocess, shutil
root=Path.cwd()
src=root/'docs/context/sources/literature_2026-10-09'
out=root/'tmp/pdfs/literature_2026-10-09'
out.mkdir(parents=True,exist_ok=True)
renderer=shutil.which('pdftoppm')
if not renderer: raise RuntimeError('PDF renderer not found')
files={
 'correlation_metrics':('risks-13-00178.pdf',[4,5,9]),
 'rank_graduation_box':('1-s2.0-S0957417424021067-main.pdf',[5,6,10]),
 'integrated_safe':('1-s2.0-S266682702500204X-main.pdf',[5,7,9,12]),
 'wasserstein':('Evaluating SAFE AI principles using Wasserstein distance  a comparative study of Machine Learning models.pdf',[8,17,19]),
 'trustworthiness_review':('electronics-14-02717-v2.pdf',[25]),
}
for key,(name,pages) in files.items():
 for p in pages:
  dest=out/(key+'_page_'+str(p))
  subprocess.run([renderer,'-f',str(p),'-l',str(p),'-singlefile','-scale-to','1800','-png',str(src/name),str(dest)],check=True,capture_output=True)
  print(str(dest)+'.png')