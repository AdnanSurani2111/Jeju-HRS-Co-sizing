import json,glob,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import config as C, optimizer_v2 as O
plt.rcParams.update({'font.size':9,'figure.dpi':200,'savefig.bbox':'tight','axes.grid':True,'grid.alpha':.3})
def save(f,n):
    for e in ('png','pdf'): f.savefig(f'figures/{n}.{e}')
    plt.close(f); print(' figure:',n)

# ---- FIX: star must mark the RECOMMENDED (reliability-constrained) design ----
F=np.load('checkpoints/F_gen.npy'); X=np.load('checkpoints/X_gen.npy')
ti,_=O.entropy_topsis(F)
ok=np.where(F[:,1]<=0.01)[0]; ri=ok[np.argmin(F[ok,0])]
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
sc=ax[0].scatter(F[:,0],F[:,2]*100,c=F[:,1]*100,cmap='viridis',s=14)
ax[0].scatter(F[ri,0],F[ri,2]*100,marker='*',s=280,c='gold',edgecolor='k',zorder=6,label='Recommended (LPSP$\\leq$1%)')
ax[0].scatter(F[ti,0],F[ti,2]*100,marker='D',s=60,c='crimson',edgecolor='k',zorder=5,label='Entropy-TOPSIS')
plt.colorbar(sc,ax=ax[0],label='LPSP (%)')
ax[0].set_xlabel('LCOH (USD/kg)'); ax[0].set_ylabel('Curtailment (%)'); ax[0].legend(fontsize=6)
ax[1].scatter(F[:,0],F[:,1]*100,s=14,c='steelblue')
ax[1].scatter(F[ri,0],F[ri,1]*100,marker='*',s=280,c='gold',edgecolor='k',zorder=6)
ax[1].scatter(F[ti,0],F[ti,1]*100,marker='D',s=60,c='crimson',edgecolor='k',zorder=5)
ax[1].axhline(1.0,ls='--',c='grey',lw=.8); ax[1].set_xlabel('LCOH (USD/kg)'); ax[1].set_ylabel('LPSP (%)')
save(fig,'F03_pareto')
fig,ax=plt.subplots(2,3,figsize=(9.4,4.6))
for a,j,nm in zip(ax.ravel(),range(6),C.VAR_NAMES):
    a.scatter(F[:,0],X[:,j],s=10,c='teal')
    a.scatter(F[ri,0],X[ri,j],marker='*',s=220,c='gold',edgecolor='k',zorder=6)
    a.scatter(F[ti,0],X[ti,j],marker='D',s=50,c='crimson',edgecolor='k',zorder=5)
    a.set_xlabel('LCOH (USD/kg)'); a.set_ylabel(nm)
fig.suptitle('Design variables across the Pareto set (star = recommended, diamond = TOPSIS)',y=1.01,fontsize=9)
save(fig,'F04_capacity_map')

# ---- F11 epsilon sweep ----
rows=[]
for tech in ('awe','pem','hybrid'):
    for sd in (17,23):
        try: Ff=np.load(f'checkpoints/eps_F_{tech}_{sd}.npy')
        except: 
            try: Ff=np.load(f'checkpoints/eps_F_{tech}.npy')
            except: continue
        for e in (0.01,0.005,0.001,0.0):
            okk=np.where(Ff[:,1]<=e+1e-12)[0]
            rows.append(dict(tech=tech,seed=sd,eps=e,
                             LCOH=float(Ff[okk,0].min()) if len(okk) else np.nan))
E=pd.DataFrame(rows).dropna()
P=E.groupby(['tech','eps']).LCOH.agg(['mean','min','max','count']).reset_index()
P.round(3).to_csv('tables/T09_eps_sweep.csv',index=False)
fig,ax=plt.subplots(figsize=(5.4,3.0))
lbl={'awe':'AWE-only','pem':'PEM-only','hybrid':'Hybrid AWE-PEM'}
for tech,c,mk in (('awe','tab:blue','o'),('pem','tab:green','s'),('hybrid','tab:red','^')):
    d=P[P.tech==tech].sort_values('eps',ascending=False)
    xx=100*d.eps.values
    ax.errorbar(xx,d['mean'],yerr=[d['mean']-d['min'],d['max']-d['mean']],
                marker=mk,color=c,capsize=3,label=lbl[tech],lw=1.4)
ax.invert_xaxis(); ax.set_xlabel('Reliability constraint, LPSP $\\leq \\epsilon$ (%)')
ax.set_ylabel('Minimum LCOH (USD/kg)'); ax.legend(fontsize=7)
ax.set_title('Matched-reliability comparison (2 seeds, identical budget)',fontsize=8)
save(fig,'F11_eps_sweep')
print(P.round(3).to_string(index=False))

# ---- F12 ablation ----
A=[]
for c_ in ('F0','F1','F2'):
    d=json.load(open(f'checkpoints/abl_{c_}.json')); A.append(dict(Case=c_,LCOH=d['F'][0],LPSP=d['F'][1],CURT=d['F'][2]))
Fh=np.load('checkpoints/eps_F_hybrid.npy')
okh=np.where(Fh[:,1]<=0.01)[0]; ih=okh[np.argmin(Fh[okh,0])]
A.append(dict(Case='F3',LCOH=float(Fh[ih,0]),LPSP=float(Fh[ih,1]),CURT=float(Fh[ih,2])))
Ad=pd.DataFrame(A); base=Ad.loc[Ad.Case=='F0','LCOH'].iloc[0]
Ad['Delta']=Ad.LCOH-base
Ad.round(4).to_csv('tables/T10_ablation.csv',index=False)
d1=Ad.loc[Ad.Case=='F1','Delta'].iloc[0]; d2=Ad.loc[Ad.Case=='F2','Delta'].iloc[0]; d3=Ad.loc[Ad.Case=='F3','Delta'].iloc[0]
inter=d3-d1-d2
fig,ax=plt.subplots(1,2,figsize=(8.0,2.9))
names=['F0\nnothing free','F1\nrenewable\nonly','F2\nconversion\n+storage only','F3\njoint']
ax[0].bar(names,Ad.LCOH,color=['grey','tab:blue','tab:green','tab:red'])
for i,v in enumerate(Ad.LCOH): ax[0].text(i,v+.2,f'{v:.2f}',ha='center',fontsize=7)
ax[0].set_ylabel('LCOH (USD/kg)'); ax[0].tick_params(labelsize=6.5)
ax[1].bar(['Renewable\nsizing','Conversion\n+storage sizing','Joint\ninteraction'],
          [-d1,-d2,-inter],color=['tab:blue','tab:green','tab:purple'])
for i,v in enumerate([-d1,-d2,-inter]): ax[1].text(i,v+.1,f'{v:.2f}',ha='center',fontsize=7)
ax[1].set_ylabel('LCOH reduction (USD/kg)'); ax[1].tick_params(labelsize=6.5)
ax[1].set_title('Decomposition of the co-sizing gain',fontsize=8)
save(fig,'F12_ablation')
print(Ad.round(3).to_string(index=False)); print(f'interaction term {inter:.3f}')
