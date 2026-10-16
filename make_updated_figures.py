"""Regenerate current campaign and boundary vector figures from saved records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent;D=ROOT/'full_search';OUT=ROOT/'updated_figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,title,body,color='#eaf2fa'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc=color,ec='#24608b',lw=1.1));ax.text(x+w/2,y+h-.045,title,ha='center',va='top',weight='bold',fontsize=11);ax.text(x+w/2,y+h/2-.025,body,ha='center',va='center',fontsize=10,linespacing=1.55)
def arrow(ax,a,b,color='#24608b'):
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5,'color':color})
fig,ax=plt.subplots(figsize=(10.4,5.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.5,.98,'Prospectively fixed complete-search campaign',ha='center',va='top',size=14,weight='bold')
box(ax,.02,.59,.44,.29,'1. Search both cases','F3: six variables free | F2: four free\n5 optimizer seeds per case\n8 training draws; 182 × 100 evaluations/run')
box(ax,.54,.59,.44,.29,'2. Screen each final population','All 182 vectors × 3 separate banks\n400 demand scenarios per bank\nCheapest design with CP upper bound ≤ 2.5%')
box(ax,.54,.16,.44,.29,'3. Freeze every decision','30 attempted selections recorded\nNo passing candidate → no selection\nNo fallback or capacity changes after audit')
box(ax,.02,.16,.44,.29,'4. Independent demand audit','2000 previously unused demand draws\nPrimary CP upper bound ≤ 5%\nSecondary simultaneous bounds: α = 0.05/30',color='#e8f4ec')
arrow(ax,(.46,.73),(.54,.73));arrow(ax,(.76,.59),(.76,.45));arrow(ax,(.54,.30),(.46,.30))
ax.text(.5,.045,'Exceedance: three-year unserved hydrogen mass > 0.1% of demand.\nAll bounds concern this demand generator and the fixed 2020–2022 renewable chronology.',ha='center',va='bottom',fontsize=9)
save(fig,'F03_optimization_protocol')
fig,ax=plt.subplots(figsize=(10.4,5.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.5,.98,'Modeled renewable hydrogen supply boundary',ha='center',va='top',fontsize=14,weight='bold')
box(ax,.02,.58,.20,.25,'Renewables','PV + wind\nHourly generation')
box(ax,.39,.58,.25,.25,'Electrolysis','AWE priority, then PEM\nFixed hourly dispatch')
box(ax,.79,.58,.19,.25,'Demand','Hydrogen mass\nsupplied / unserved',color='#e8f4ec')
box(ax,.03,.16,.19,.23,'Battery','10–90% SOC\nElectricity storage')
box(ax,.40,.16,.22,.23,'Curtailment','Unused electricity')
box(ax,.78,.16,.20,.23,'H₂ storage','5% inventory floor\nHydrogen carryover',color='#e8f4ec')
arrow(ax,(.22,.71),(.39,.71));arrow(ax,(.64,.71),(.79,.71),'#23854e');arrow(ax,(.13,.58),(.13,.39));arrow(ax,(.22,.31),(.39,.60));arrow(ax,(.30,.70),(.48,.39));arrow(ax,(.64,.62),(.78,.31),'#23854e');arrow(ax,(.88,.39),(.88,.58),'#23854e')
ax.text(.5,.055,'Blue arrows: electricity. Green arrows: hydrogen.\nCompression, pre-cooling, dispensing, queues and vehicle-event completion are outside the model.',ha='center',fontsize=9)
save(fig,'system_architecture')
p=json.loads((D/'protocol.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10.4,4.1),layout='constrained');colors=plt.cm.tab10.colors
for j,case in enumerate(['F3','F2']):
 ax=axes[j]
 for i,seed in enumerate(p['optimizer_seeds']):
  z=np.load(D/f'{case}_{seed}'/'final.npz');f=z['F'];ax.scatter(f[:,0],100*f[:,1],s=10,alpha=.55,c=[colors[i]],label=str(seed))
 ax.set(title=f'{case}: complete final populations',xlabel='Training mean cost (USD/kg)',ylabel='Training mean unserved mass (%)');ax.grid(alpha=.2);ax.legend(title='Optimizer seed',fontsize=8)
save(fig,'full_search_populations')
fig,axes=plt.subplots(1,2,figsize=(10.4,3.7),layout='constrained')
for j,case in enumerate(['F3','F2']):
 for i,seed in enumerate(p['optimizer_seeds']):
  t=json.loads((D/f'{case}_{seed}'/'trace.json').read_text());axes[j].plot([r['generation'] for r in t],[r['min_cost'] for r in t],c=colors[i],label=str(seed))
 axes[j].set(title=case,xlabel='Generation',ylabel='Minimum population mean cost (USD/kg)');axes[j].grid(alpha=.2);axes[j].legend(title='Optimizer seed',fontsize=8)
save(fig,'full_search_progress')
print(OUT)
