"""FINAL PIPELINE. Archive from uninterrupted state-preserved runs only; the
chunk-restarted run is excluded. Common re-evaluation, margin screen, then
validation on fresh realizations and the 2023 chronology."""
import glob,os,json,numpy as np,pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import config as C, data_v2 as D
from objectives_v2 import evaluate_population
re_=D.load_re(); ps,pw=re_['train']; ts,tw=re_['test']
SEL=[D.demand_series(len(ps),seed=200+s)[0] for s in range(8)]
def ev(X,R,a=None,b=None):
    a=ps if a is None else a; b=pw if b is None else b
    L=np.zeros((len(X),len(R))); P=np.zeros_like(L); Cu=np.zeros_like(L)
    for k,d in enumerate(R):
        r=evaluate_population(X,a,b,d); L[:,k],P[:,k],Cu[:,k]=r['lcoh'],r['lpsp'],r['curtailment']
    return L,P,Cu

# ---- archive: uninterrupted runs ONLY (X_gen.npy is the restarted run -> excluded)
src=[]
for f in sorted(glob.glob('checkpoints/cont_X_*.npy')): src.append(('continuous',f))
for f in sorted(glob.glob('checkpoints/eps_X_hybrid*.npy')): src.append(('multistart',f))
X=[];tg=[]
for t,f in src:
    x=np.load(f); X.append(x); tg+=[t]*len(x)
# polished candidates, labelled
pv=json.load(open('checkpoints/polish_val.json'))
for k in ('seed42_polished','seed7_polished'):
    X.append(np.array(pv[k])[None,:]); tg.append('polished')
X=np.vstack(X); tg=np.array(tg)
print(f'archive candidates (uninterrupted only): {len(X)} from {len(src)} runs + 2 polished')
L,P,Cu=ev(X,SEL)
Fm=np.column_stack([L.mean(1),P.mean(1),Cu.mean(1)]); q95=np.quantile(P,.95,axis=1)
keep=NonDominatedSorting().do(Fm,only_non_dominated_front=True)
print(f'nondominated after common re-evaluation: {len(keep)}')

rows=[]
for eps in (0.01,0.005,0.001):
    ok=[i for i in keep if q95[i]<=eps]
    if not ok: rows.append(dict(screen=eps,n=0)); continue
    i=ok[int(np.argmin(Fm[ok,0]))]
    rows.append(dict(screen=eps,n=len(ok),src=tg[i],LCOH=Fm[i,0],q95=q95[i],CURT=Fm[i,2],idx=i))
S=pd.DataFrame(rows); print(S.round(4).to_string(index=False))

sel=int(S.loc[S.screen==0.001,'idx'].iloc[0]) if (S.screen==0.001).any() and S.loc[S.screen==0.001,'n'].iloc[0]>0 else int(S.loc[S.screen==0.01,'idx'].iloc[0])
x=X[sel]
print(f"\nFINAL DESIGN (margin screen q95<=0.1%, source {tg[sel]}):")
for k,v in zip(C.VAR_NAMES,x): print(f"   {k:<9}{v:10.1f}")

VAL=[D.demand_series(len(ps),seed=500+s)[0] for s in range(20)]
Lv,Pv,Cv=ev(x[None,:],VAL)
print(f"\n20 fresh demand realizations (none used in optimization, refinement or screening):")
print(f"   LCOH mean {Lv.mean():.4f}  sd {Lv.std(ddof=1):.4f}  p05 {np.quantile(Lv,.05):.3f}  p50 {np.median(Lv):.3f}  p95 {np.quantile(Lv,.95):.3f}")
print(f"   LPSP mean {Pv.mean():.5f}  q95 {np.quantile(Pv,.95):.5f}  max {Pv.max():.5f}")
print(f"   CURT mean {Cv.mean():.4f}  sd {Cv.std(ddof=1):.4f}")
dht=[D.demand_series(len(ts),seed=99+s)[0] for s in range(4)]
Lh,Ph,Ch=ev(x[None,:],dht,ts,tw)
print(f"\nHELD OUT 2023 chronology (4 demand draws):")
print(f"   LCOH {Lh.mean():.4f}  LPSP mean {Ph.mean():.5f} max {Ph.max():.5f}  CURT {Ch.mean():.4f}")
print(f"   -> {'PASSES' if np.quantile(Ph,.95)<=0.01 else 'FAILS'} the 1% service target out of sample")
det=evaluate_population(x[None,:],ps,pw,SEL[0])
d={k:float(np.asarray(v).ravel()[0]) for k,v in det.items() if v is not None and np.asarray(v).size==1}
json.dump(dict(design=dict(zip(C.VAR_NAMES,x.tolist())),source=str(tg[sel]),
  screens=S.drop(columns=['idx'],errors='ignore').to_dict('records'),
  fresh=dict(lcoh_mean=float(Lv.mean()),lcoh_sd=float(Lv.std(ddof=1)),
             lpsp_mean=float(Pv.mean()),lpsp_q95=float(np.quantile(Pv,.95)),lpsp_max=float(Pv.max()),
             curt_mean=float(Cv.mean())),
  holdout=dict(lcoh=float(Lh.mean()),lpsp_mean=float(Ph.mean()),lpsp_max=float(Ph.max()),curt=float(Ch.mean())),
  detail=d),open('checkpoints/FINAL.json','w'),indent=2)
print(f"\ndetail: H2 prod {d['h2_produced']:.0f} served {d['h2_served']:.0f} discarded {d['h2_discarded']:.4f}")
print(f"        AWE life {d['awe_life']:.1f} PEM life {d['pem_life']:.1f} CAPEX {d['ann_capex']:.3f} repl {d['ann_repl']:.3f}")
