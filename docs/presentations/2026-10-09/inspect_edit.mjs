import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='C:/Users/Andrei/Desktop/credit-risk-fairness';
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/output/presentations/credit_risk_prezentare_profesor_09_octombrie_2026.pptx'));
console.log(await p.inspect({kind:'slide',maxChars:18000}));
console.log(JSON.stringify(p.slides.items.map((s,i)=>({slide:i+1,id:s.id,shapes:s.shapes.items.map(sh=>({id:sh.id,text:String(sh.text)})),tables:s.tables.items.map(t=>({rows:t.rows,columns:t.columns})),charts:s.charts.items.map(c=>({categories:c.categories,series:c.series.items.map(x=>({name:x.name,values:x.values}))}))})),null,2));