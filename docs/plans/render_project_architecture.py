from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'diagrams'
OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(14, 12), dpi=170)
fig.patch.set_facecolor('#ffffff')
ax.set(xlim=(0,14.5), ylim=(0,12))
ax.axis('off')
ink='#172b40'
def box(x,y,w,h,title,body='',color='#edf3f8',edge='#8ca2b4',dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=0.12',facecolor=color,edgecolor=edge,linewidth=1.4,linestyle='--' if dashed else '-'))
    ax.text(x+w/2,y+h-(.27 if body else h/2),title,ha='center',va='center',fontsize=12.2,fontweight='bold',color=ink)
    if body: ax.text(x+w/2,y+h/2-.18,body,ha='center',va='center',fontsize=10.8,linespacing=1.5,color=ink)
def arrow(a,b,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,linewidth=1.4,color='#63768a',linestyle='--' if dashed else '-',shrinkA=4,shrinkB=5))
ax.text(.45,11.65,'How the project works today',fontsize=23,fontweight='bold',color=ink)
ax.text(.45,11.28,'Implemented workflow and the comparison still to be built  |  10 October 2026',fontsize=11,color='#5d6c7c')
box(2.0,9.65,10,1.1,'Shared data preparation, exploratory analysis and outer split','22 predictors  |  24,004 training records  |  5,996 holdout records')
ax.text(.65,9.10,'EXISTING PROJECT',ha='left',fontsize=12,fontweight='bold',color='#27735a')
ax.text(7.55,9.10,'ARTICLE IMPLEMENTATION',ha='left',fontsize=12,fontweight='bold',color='#386da0')
arrow((5.5,9.65),(5.5,8.30))
arrow((11.85,9.65),(13.35,8.75))
box(.55,6.7,5.4,1.6,'Original fitted models','Regularized logistic regression\nFull GBM: 22 predictors',color='#e9f4ee',edge='#6ba58a')
box(7.45,7.85,6.5,.85,'Training-only selection','19,203 fitting / 4,801 validation',color='#eaf1fa',edge='#7da3c5')
arrow((10.7,7.85),(10.7,7.50))
box(7.45,6.35,6.5,1.10,'Final models fitted on all training records','Full GBM: 22 inputs  |  Selected GBM: 3 inputs',color='#eaf1fa',edge='#7da3c5')
box(7.45,5.55,6.5,.48,'Article LR unavailable under current design',color='#fff1df',edge='#c69955')
arrow((3.25,6.7),(3.25,4.9))
arrow((10.7,5.55),(10.7,4.9))
box(.55,3.6,5.4,1.2,'Existing assessment','Native SAFE + performance, fairness,\nSHAP, robustness and uncertainty',color='#e9f4ee',edge='#6ba58a')
box(7.45,3.6,6.5,1.2,'Article-based assessment','DeLong, SHAP, Gini / KS and available\nexplanation tests; supporting uncertainty',color='#eaf1fa',edge='#7da3c5')
arrow((3.25,3.6),(3.25,3.00))
arrow((10.7,3.6),(10.7,3.00))
box(.55,2.3,5.4,.60,'Completed profiles and reports',color='#e9f4ee',edge='#6ba58a')
box(7.45,2.3,6.5,.60,'Completed report; missing LR results explicit',color='#eaf1fa',edge='#7da3c5')
arrow((3.25,2.3),(5.0,1.75),True)
arrow((10.7,2.3),(9.0,1.75),True)
box(2,0.65,10,1,'Proposed common comparison — not implemented','A shared decision rule and a measurable outcome criterion',color='#f4f5f7',edge='#8c96a1',dashed=True)
ax.text(.55,.12,'The two full GBMs give identical holdout probabilities. The selected GBM uses PAY_0, LIMIT_BAL and PAY_AMT1.',fontsize=10,color='#4d5f72')
fig.savefig(OUT/'project_architecture_current.png',bbox_inches='tight',facecolor='white')
fig.savefig(OUT/'project_architecture_current.svg',bbox_inches='tight',facecolor='white')
plt.close(fig)
print(OUT/'project_architecture_current.png')

