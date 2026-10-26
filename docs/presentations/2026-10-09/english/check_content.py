import zipfile,xml.etree.ElementTree as ET,re,json
from pathlib import Path
root=Path.cwd(); source=root/'output/presentations/credit_risk_prezentare_profesor_09_octombrie_2026.pptx'; final=root/'output/presentations/credit_risk_professor_presentation_09_october_2026_EN.pptx'
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
with zipfile.ZipFile(source) as old,zipfile.ZipFile(final) as new:
    slides=sorted([n for n in new.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)])
    notes=[n for n in new.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)]
    assert len(slides)==len(notes)==10
    for n in slides:
        t=' '.join(ET.fromstring(new.read(n)).itertext())
        assert not re.search('[ăâîșțĂÂÎȘȚ]',t),n
    for n in notes:
        t=' '.join(ET.fromstring(new.read(n)).itertext())
        assert 'Project sources:' in t,n
        assert 'Surse în proiect:' not in t,n
    oldcharts=sorted(n for n in old.namelist() if re.fullmatch(r'ppt/slides/charts/[^/]+\.xml',n)); newcharts=sorted(n for n in new.namelist() if re.fullmatch(r'ppt/slides/charts/[^/]+\.xml',n))
    assert len(oldcharts)==len(newcharts)==2
    for oldname,newname in zip(oldcharts,newcharts):
        def vals(z,n):
            x=ET.fromstring(z.read(n)); return [[v.text for v in ser.findall('.//c:val//c:v',ns)] for ser in x.findall('.//c:ser',ns)]
        assert vals(old,oldname)==vals(new,newname),(oldname,newname)
print('10 English slides and notes; 2 native charts retain exact numeric data.')