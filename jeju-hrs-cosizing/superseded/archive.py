"""External archive built to a strict protocol:
every design from every run is RE-EVALUATED on one common set of demand
realizations, objectives recomputed from raw dispatch, nondominance recomputed
globally, and selection applied afterwards with an explicit reliability margin."""
import glob, os, json
import numpy as np, pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import config as C, data_v2 as D
from objectives_v2 import evaluate_population

re_=D.load_re(); ps,pw=re_['train']; ts,tw=re_['test']
# common evaluation set: 8 realizations, none of them the two used in optimization
REV=[D.demand_series(len(ps),seed=200+s)[0] for s in range(8)]

srcs=[]
for f in sorted(glob.glob('checkpoints/cont_X_*.npy')): srcs.append(('continuous',f))
for f in sorted(glob.glob('checkpoints/eps_X_hybrid*.npy')): srcs.append(('multi-start',f))
srcs.append(('restarted',  'checkpoints/X_gen.npy'))

X=[];tag=[]
for t,f in srcs:
    x=np.load(f); X.append(x); tag += [t]*len(x)
X=np.vstack(X); tag=np.array(tag)
print(f'candidate designs pooled: {len(X)} from {len(srcs)} runs')

# re-evaluate ALL designs on the common realization set
L=np.zeros((len(X),len(REV))); P=np.zeros_like(L); Cu=np.zeros_like(L)
for k,dh in enumerate(REV):
    r=evaluate_population(X,ps,pw,dh)
    L[:,k],P[:,k],Cu[:,k]=r['lcoh'],r['lpsp'],r['curtailment']
Fm=np.column_stack([L.mean(1),P.mean(1),Cu.mean(1)])
q95=np.quantile(P,0.95,axis=1)

keep=NonDominatedSorting().do(Fm,only_non_dominated_front=True)
print(f'nondominated after common re-evaluation: {len(keep)}')

rows=[]
for eps,name in [(0.01,'eps<=1%'),(0.005,'eps<=0.5%'),(0.001,'eps<=0.1%')]:
    ok=[i for i in keep if q95[i]<=eps]
    if not ok: rows.append(dict(Screen=name,n=0)); continue
    i=ok[int(np.argmin(Fm[ok,0]))]
    dht,_=D.demand_series(len(ts),seed=99)
    h=evaluate_population(X[i][None,:],ts,tw,dht)
    rows.append(dict(Screen=name,n=len(ok),source=tag[i],
                     LCOH=Fm[i,0],LPSP_mean=Fm[i,1],LPSP_q95=q95[i],CURT=Fm[i,2],
                     holdout_LCOH=float(h['lcoh'][0]),holdout_LPSP=float(h['lpsp'][0]),
                     P_sol=X[i,0],P_win=X[i,1],P_elz=X[i,2],xPEM=X[i,3],BESS=X[i,4],HSS=X[i,5],idx=i))
R=pd.DataFrame(rows)
print(R.round(4).to_string(index=False))
R.round(5).to_csv('tables/T14_margin_screen.csv',index=False)

# full 20-realization vector for the eps<=0.1% (margin) design
best=int(R.loc[R.Screen=='eps<=0.1%','idx'].iloc[0]) if (R.Screen=='eps<=0.1%').any() and R.loc[R.Screen=='eps<=0.1%','n'].iloc[0]>0 else int(R.loc[R.Screen=='eps<=1%','idx'].iloc[0])
V=[]
for s in range(20):
    dq,_=D.demand_series(len(ps),seed=300+s)
    m=evaluate_population(X[best][None,:],ps,pw,dq)
    V.append(dict(realization=s,LCOH=float(m['lcoh'][0]),LPSP=float(m['lpsp'][0]),CURT=float(m['curtailment'][0])))
Vd=pd.DataFrame(V); Vd.round(6).to_csv('tables/T15_realization_vector.csv',index=False)
print(f"\n20-realization vector for the selected design (index {best}, source {tag[best]}):")
print(f"  LCOH {Vd.LCOH.mean():.4f} +- {Vd.LCOH.std(ddof=1):.4f}  [{Vd.LCOH.min():.3f}, {Vd.LCOH.max():.3f}]")
print(f"  LPSP {Vd.LPSP.mean():.5f} +- {Vd.LPSP.std(ddof=1):.5f}  max {Vd.LPSP.max():.5f}  q95 {Vd.LPSP.quantile(.95):.5f}")
print(f"  CURT {Vd.CURT.mean():.4f} +- {Vd.CURT.std(ddof=1):.4f}")
json.dump(dict(design=dict(zip(C.VAR_NAMES,X[best].tolist())),source=str(tag[best]),
               screens=R.drop(columns=['idx']).to_dict('records'),
               realization_summary=dict(lcoh_mean=float(Vd.LCOH.mean()),lcoh_sd=float(Vd.LCOH.std(ddof=1)),
                 lpsp_mean=float(Vd.LPSP.mean()),lpsp_max=float(Vd.LPSP.max()),
                 curt_mean=float(Vd.CURT.mean()))),open('checkpoints/archive.json','w'),indent=2)
