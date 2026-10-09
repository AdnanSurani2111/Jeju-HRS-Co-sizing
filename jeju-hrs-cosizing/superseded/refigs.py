import json,numpy as np,pandas as pd,matplotlib,os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import config as C, optimizer_v2 as O
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,
 'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'figure.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.03,'axes.grid':True,'grid.alpha':0.25,
 'grid.linewidth':0.5,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.3})
BLUE,GREEN,RED,ORANGE,PURPLE,GREY='#2E5E8A','#2E8B6B','#B03A3A','#D98032','#6B4E8A','#7A7A7A'
def save(f,n):
    for e in ('png','pdf'): f.savefig(os.path.join(C.FIG_DIR,f'{n}.{e}'))
    plt.close(f); print('  ',n)
F=np.load('checkpoints/F_pooled.npy'); X=np.load('checkpoints/X_pooled.npy')
R=json.load(open('checkpoints/reconciled.json')); rec=R['recommended']; m=R['metrics']; ho=R['holdout']
ti,W=O.entropy_topsis(F); ok=np.where(F[:,1]<=0.01)[0]; ri=ok[np.argmin(F[ok,0])]
print('entropy weights',np.round(W,4))

fig,ax=plt.subplots(1,3,figsize=(7.1,2.2))
sc=ax[0].scatter(F[:,0],F[:,2]*100,c=F[:,1]*100,cmap='viridis',s=11,lw=0)
ax[0].scatter(F[ri,0],F[ri,2]*100,marker='*',s=170,c='gold',edgecolor='k',lw=.6,zorder=6)
ax[0].scatter(F[ti,0],F[ti,2]*100,marker='D',s=32,c=RED,edgecolor='k',lw=.5,zorder=5)
cb=plt.colorbar(sc,ax=ax[0],pad=.02); cb.set_label('LPSP (%)',fontsize=7); cb.ax.tick_params(labelsize=6)
ax[0].set_xlabel('LCOH (USD/kg)'); ax[0].set_ylabel('Curtailment (%)'); ax[0].set_title('(a) Cost versus curtailment')
ax[1].scatter(F[:,0],F[:,1]*100,s=11,c=BLUE,lw=0)
ax[1].scatter(F[ri,0],F[ri,1]*100,marker='*',s=170,c='gold',edgecolor='k',lw=.6,zorder=6,label='Reliability screened')
ax[1].scatter(F[ti,0],F[ti,1]*100,marker='D',s=32,c=RED,edgecolor='k',lw=.5,zorder=5,label='TOPSIS compromise')
ax[1].axhline(1.0,color=GREY,ls=':',lw=.9); ax[1].set_xlabel('LCOH (USD/kg)'); ax[1].set_ylabel('LPSP (%)')
ax[1].legend(frameon=False); ax[1].set_title('(b) Cost versus service loss')
ax[2].scatter(F[:,2]*100,F[:,1]*100,s=11,c=GREEN,lw=0)
ax[2].scatter(F[ri,2]*100,F[ri,1]*100,marker='*',s=170,c='gold',edgecolor='k',lw=.6,zorder=6)
ax[2].set_xlabel('Curtailment (%)'); ax[2].set_ylabel('LPSP (%)'); ax[2].set_title('(c) Curtailment versus service loss')
save(fig,'F03_pareto')

UN=['kW','kW','kW','','kWh','kg']
fig,ax=plt.subplots(2,3,figsize=(7.1,3.6))
for a,j,nm,u in zip(ax.ravel(),range(6),C.VAR_NAMES,UN):
    a.scatter(F[:,0],X[:,j],s=8,c=GREEN,lw=0,alpha=.75)
    a.scatter(F[ri,0],X[ri,j],marker='*',s=150,c='gold',edgecolor='k',lw=.6,zorder=6)
    a.set_ylabel(f'{nm}'+(f' ({u})' if u else ''))
for a in ax[1]: a.set_xlabel('LCOH (USD/kg)')
fig.tight_layout(h_pad=.9,w_pad=1.2); save(fig,'F04_capacity_map')

fig,ax=plt.subplots(1,3,figsize=(6.0,2.05))
for a,(t,k,f,fm) in zip(ax,[('LCOH (USD/kg)','lcoh',1,'{:.2f}'),('Service loss (%)','lpsp',100,'{:.2f}'),
                            ('Curtailment (%)','curtailment',100,'{:.2f}')]):
    v=[m[k]*f,ho[k]*f]; a.bar([0,1],v,color=[BLUE,ORANGE],width=.55)
    for xx,yy in zip([0,1],v): a.text(xx,yy+max(v)*.02,fm.format(yy),ha='center',fontsize=6.5)
    a.set_xticks([0,1]); a.set_xticklabels(['Design\n2020 to 2022','Hold out\n2023'],fontsize=6.5)
    a.set_ylabel(t); a.set_ylim(0,max(v)*1.25)
    if k=='lpsp': a.axhline(1.0,color=RED,ls=':',lw=.9); a.text(0.5,1.06,'design target',fontsize=5.5,color=RED,ha='center')
fig.tight_layout(w_pad=1.3); save(fig,'F07_holdout')

dec=pd.DataFrame([('Annualised CAPEX',m['ann_capex']),('O&M',m['ann_om']),
                  ('Replacement',m['ann_repl']),('Cyclic ageing',m['ann_cyc'])],columns=['Component','USD_per_kg'])
dec['Share_pct']=100*dec.USD_per_kg/dec.USD_per_kg.sum(); dec.round(4).to_csv('tables/T05_lcoh_decomposition.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(5.6,2.2)); cc=[BLUE,GREEN,ORANGE,RED]
ax[0].bar(range(4),dec.USD_per_kg,color=cc,width=.6)
for i,v in enumerate(dec.USD_per_kg): ax[0].text(i,v+.12,f'{v:.2f}',ha='center',fontsize=6.5)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(['CAPEX','O&M','Replace','Cyclic'],fontsize=7)
ax[0].set_ylabel('USD/kg'); ax[0].set_ylim(0,dec.USD_per_kg.max()*1.2); ax[0].set_title('(a) Cost components')
ax[1].pie(dec.USD_per_kg,labels=['CAPEX','O&M','Replace','Cyclic'],colors=cc,autopct='%1.1f%%',
          textprops={'fontsize':6.5},startangle=90,wedgeprops={'linewidth':.5,'edgecolor':'white'})
ax[1].set_title('(b) Cost shares'); save(fig,'F08_lcoh_decomposition')
print('shares',dec.round(3).to_dict('records'))
