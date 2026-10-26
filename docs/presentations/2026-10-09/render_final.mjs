import fs from 'node:fs/promises';import path from 'node:path';import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='C:/Users/Andrei/Desktop/credit-risk-fairness';const file=root+'/output/presentations/credit_risk_prezentare_profesor_09_octombrie_2026.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(file));
const out=root+'/tmp/presentation_credit_risk/final-renders';await fs.mkdir(out,{recursive:true});
for(const [i,s] of p.slides.items.entries()){const blob=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));}
console.log('Rendered finalized PPTX: '+p.slides.items.length+' slides');