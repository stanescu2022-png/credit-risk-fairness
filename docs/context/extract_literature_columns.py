from pathlib import Path
import pdfplumber
root=Path('docs/context/sources/literature_2026-10-09')
for key,name in [('integrated_safe','1-s2.0-S266682702500204X-main.pdf'),('rank_graduation_box','1-s2.0-S0957417424021067-main.pdf')]:
 with pdfplumber.open(root/name) as pdf:
  text=[]
  for i,p in enumerate(pdf.pages,1):
   if i==1: t=p.extract_text()
   else:
    mid=p.width/2
    t=(p.crop((0,0,mid,p.height)).extract_text() or '')+'\n[RIGHT COLUMN]\n'+(p.crop((mid,0,p.width,p.height)).extract_text() or '')
   text.append('===== PDF PAGE '+str(i)+' =====\n'+t)
 (root/(key+'.columns.txt')).write_text('\n\n'.join(text),encoding='utf-8')