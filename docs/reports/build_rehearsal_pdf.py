from pathlib import Path
base=Path(__file__).with_name('build_additions_pdf.py').read_text(encoding='utf-8').split('# Page 1')[0]
base=base.replace("credit_risk_additions_beyond_safe.pdf","fisa_repetitie_discutie_1600.pdf").replace('Implemented additions | Credit risk fairness project | 8 October 2026','Fișă de repetiție | Discuția de la 16:00 | 8 octombrie 2026').replace('BEYOND THE PACKAGE METRICS','CREDIT RISK / PREGĂTIRE').replace('IMPLEMENTED SCOPE','EXPLORER OFFLINE').replace('CREDIT RISK RESEARCH / ','TRASEU PRIN EXPLORER / ')
exec(compile(base,str(__file__),'exec'))

def step(title,click,say,remember):
    add(title,'sub');add('<b>În explorer:</b> '+click,'small');add('<b>Poți spune:</b> „'+say+'”');add('<b>De reținut:</b> '+remember,'small')

# Page 1
add('PREGĂTIRE CU COLEGUL / 16:00 / 8 OCTOMBRIE 2026','eyebrow')
add('Fișă de repetiție<br/>pentru discuția de azi','cover')
add('Obiectiv: să vă puneți de acord asupra mesajului, exemplelor din HTML și întrebărilor pentru profesor. Folosește formulările ca sprijin, fără să le înveți cuvânt cu cuvânt.')
note('<b>Ideea centrală:</b> boosting ordonează mai bine cazurile după riscul de default, dar comparația mai largă depinde de ce măsurăm. Performanța, echitatea, explicațiile și robustețea nu dau automat același câștigător.')
add('Deschiderea în aproximativ 30 de secunde','sub')
add('„Comparăm regresia logistică și gradient boosting pe un set public de date despre default. Am evaluat performanța, echitatea, explicabilitatea și robustețea. Vrem să înțelegem ce câștigăm printr-un model mai complex și ce compromisuri apar. Aceasta este baza empirică pe care vrem să o discutăm cu profesorul înainte de extinderea cadrului.”')
step('1. Performance / aproximativ 2 minute','Metric: ROC-AUC; Evaluation sample: Locked holdout. Apoi schimbă Metric la Default recall.', 'Boosting are ROC-AUC 0,778, față de 0,757 pentru logistică. Asta arată o ordonare mai bună a riscului. Totuși, la pragul 0,50, ambele detectează doar aproximativ 35% dintre default-uri, deși acuratețea este în jur de 82%.', 'ROC-AUC măsoară ordonarea; recall arată câte default-uri detectăm. Acuratețea singură nu descrie bine clasa minoritară. Nu am ales un prag nou.')
step('2. Fairness / aproximativ 2 minute','Signed group comparison: Recall gap. Apoi selectează SAFE RGA gap și compară intervalele.', 'Diferența de recall femei minus bărbați este de aproximativ 2,32 puncte procentuale la logistică și 3,35 la boosting. În schimb, boosting are un decalaj absolut SAFE mai mic. Ordinea modelelor depinde de metrica de echitate.', 'Intervalele afișate pentru aceste diferențe includ zero. Nu putem spune că un model este în general mai echitabil sau că lipsesc diferențe relevante. Un flag pozitiv înseamnă risc de default, nu aprobare de credit.')
add('<b>Clarifică împreună:</b> Care sunt cele două idei pe care vrem să le rețină profesorul din performanță și fairness?','small')

# Page 2
new('02','Explicații și robustețe','Păstrează accentul pe sensul rezultatelor. Deschide detaliile metodologice doar dacă aveți nevoie să clarificați o întrebare.')
step('3. Explainability / aproximativ 2 minute','Explanation: Global - top 10 features. Comută Model între Logistic regression și Gradient boosting.', 'Istoricul recent de rambursare, PAY_0, și limita de credit, LIMIT_BAL, conduc explicațiile ambelor modele. Primele trei variabile reprezintă circa 54% din importanța SHAP la logistică și 41% la boosting. Boosting distribuie importanța pe mai multe variabile.', 'SHAP descrie contribuțiile la calculul modelului. În acest proiect, scara este log-odds; valorile barelor nu sunt puncte procentuale de probabilitate. Explicația nu este un efect cauzal.')
add('<b>Exemplu individual:</b> Explanation: Individual - saved cases; Saved example: ID 2. Logistică: risc 26,64%; boosting: 38,69%. Ambele rămân sub 0,50, deși cazul a avut default. Spune: „Putem vedea ce a contribuit la un default ratat, fără să pretindem că un exemplu descrie toate cazurile.”','small')
step('4. Robustness / aproximativ 3 minute','Authentic SAFE - tail swaps; Joint - all 22 features; p = 0.05; Overall. Uită-te la Mean absolute probability movement, apoi la Authentic SAFE RGR.', 'SAFE schimbă valorile din cozi opuse ale distribuțiilor. p=0,05 se referă la fracția din cozi, nu la o creștere cu 5% a fiecărei valori. În scenariul comun, se schimbă efectiv inputuri pentru aproximativ 65% dintre înregistrări. RGR este 0,817 la logistică și 0,827 la boosting.', 'Un RGR mai mare înseamnă o păstrare mai bună a ordonării originale în acel test. Diferența nu stabilește o superioritate generală; normalizarea depinde de model.')
add('<b>Schimbă doar Corruption layer</b> la Semantic - synthetic input errors. Păstrează Joint, q = 0.10, Overall și selectează Mean absolute probability movement.','small')
add('<b>Poți spune:</b> „La aceste erori sintetice mai mici, inputurile se schimbă pentru circa 10% dintre cazuri. Mișcarea medie a probabilității este de 0,51 puncte procentuale la logistică și 0,83 la boosting. Logistică este mai puțin sensibilă în acest scenariu. Diferența pereche este circa 0,32 puncte procentuale, cu interval de 95% condiționat de aproximativ 0,30-0,34.”')
note('<b>Nu compara cele două straturi ca stresuri egale.</b> Au mecanisme și expuneri diferite. Intervalele de eșantionare și variația între cele 30 de repetări semantice sunt lucruri diferite. Rezultatele nu estimează frecvența reală a erorilor și nici schimbarea cauzală a riscului clientului.')
add('<b>Clarifică împreună:</b> Ce exemplu de robustețe putem explica în două fraze, fără să supraîncărcăm demonstrația?','small')

# Page 3
new('03','Întrebări și acorduri pentru mâine','După turul de aproximativ 10 minute, mutați discuția de la cifre la mesaj, roluri și feedbackul pe care îl cereți profesorului.')
step('5. Study & sources / aproximativ 1 minut','Deschide Study & sources. Arată Frozen comparison și Findings across the four dimensions.', 'Am păstrat toate cele 30.000 de înregistrări și am reparat suprapunerea exactă a predictorilor între antrenare și test. Folosim aceleași modele salvate, aceiași 22 de predictori și pragul de referință 0,50. Comparația este reproductibilă, dar rămâne un studiu pe un singur set de date.', 'SEX și ID nu intră în predicție. Holdout-ul fusese inspectat anterior. PSI, monitorizarea și extinderea propusă a cadrului nu sunt încă implementate.')
add('Răspunsuri scurte dacă apar întrebări','sub')
table(['Întrebare','Răspuns pe care îl poți adapta'],[
['Care model este mai bun?','Boosting are avantaje de ranking. Pe celelalte dimensiuni, răspunsul depinde de metrică și scenariu. Ne trebuie criterii de decizie pentru o alegere practică.'],
['De ce păstrăm 0,50?','Este pragul de referință al comparației. Am explorat candidați pe predicții OOF din antrenare, fără să alegem sau să validăm un prag nou pe holdout.'],
['Este modelul fair dacă nu folosește SEX?','Excluderea evită utilizarea directă, dar nu garantează echitatea. Alte variabile pot fi asociate cu grupurile, iar ratele de eroare pot diferi.'],
['HTML-ul rulează analiza?','Nu. Afișează un snapshot extras automat din tabelele salvate. Selectorii explorează rezultate existente; nu antrenează modele și nu lansează teste.']], [155,WIDTH-155])
add('Patru lucruri de stabilit cu colegul','sub')
add('<b>1. Scopul întâlnirii:</b> Ce vrem concret de la profesor? O evaluare a metodologiei actuale și îndrumare pentru pașii următori?')
add('<b>2. Contribuția:</b> Ce prezentăm ca realizat și ce rămâne propunere? Cum le legăm fără să pretindem că am validat deja întregul cadru?')
add('<b>3. Rolurile:</b> Cine introduce motivația și propunerea? Cine explică validarea, rezultatele și demonstrația? Stabiliți și cine preia fiecare tip de întrebare.')
add('<b>4. Întrebările pentru profesor:</b> Este adecvată validarea actuală? Cum ar trebui justificat criteriul de decizie și pragul? Care extensii merită prioritizate?')
note('<b>Încheiere posibilă:</b> „Avem o comparație implementată și verificată. Vrem feedback asupra acestei baze, apoi să alegem împreună ce merită extins. Nu vrem să tragem o concluzie mai puternică decât permit datele.”')
add('Pentru întâlnirea de mâine, de 30 de minute: țintiți 10-12 minute de prezentare și păstrați restul pentru discuție. Luați din această fișă doar demonstrațiile care susțin mesajul comun. Sursa cifrelor: explorerul și tabelele proiectului, snapshot din 8 octombrie 2026.','small')

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=53,bottomMargin=58,title='Fișă de repetiție pentru discuția de la 16:00',author='Credit risk fairness research project',subject='Pregătire în română pentru prezentarea rezultatelor din explorer',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)