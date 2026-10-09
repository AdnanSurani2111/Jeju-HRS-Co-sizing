"""Pool all hybrid fronts into one nondominated archive (multi-start), then select.
This replaces the single 'production run' designation, which is indefensible because
a chunk-restarted 182x100 run was beaten by unrestarted 92x30 runs."""
import numpy as np, pandas as pd, json, glob, os
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import config as C, data_v2 as D
from objectives_v2 import evaluate_population

re_=D.load_re(); ps,pw=re_['train']; ts,tw=re_['test']
dh,_=D.demand_series(len(ps),seed=0)

runs=[('production (chunk restarted)',182,100,42,'checkpoints/F_gen.npy','checkpoints/X_gen.npy')]
for f in sorted(glob.glob('checkpoints/eps_F_hybrid*.npy')):
    sd=os.path.basename(f)[:-4].split('_')[-1]
    sd=sd if sd.isdigit() else '17'
    runs.append((f'multi-start seed {sd}',92,30,int(sd),f,f.replace('eps_F','eps_X')))

rows=[];Fa=[];Xa=[]
for name,pop,gen,sd,ff,xf in runs:
    F=np.load(ff); X=np.load(xf); Fa.append(F); Xa.append(X)
    ok=np.where(F[:,1]<=0.01)[0]; i=ok[np.argmin(F[ok,0])]
    rows.append(dict(Run=name,Population=pop,Generations=gen,Seed=sd,Realizations=2,
                     Front=len(F),LCOH=F[i,0],LPSP=F[i,1],CURT=F[i,2],x_PEM=X[i,3]))
Fa=np.vstack(Fa); Xa=np.vstack(Xa)
keep=NonDominatedSorting().do(Fa,only_non_dominated_front=True)
Fp,Xp=Fa[keep],Xa[keep]
ok=np.where(Fp[:,1]<=0.01)[0]; j=ok[np.argmin(Fp[ok,0])]
best=Xp[j]
rows.append(dict(Run='POOLED archive (recommended)',Population='-',Generations='-',Seed='all',
                 Realizations=2,Front=len(Fp),LCOH=Fp[j,0],LPSP=Fp[j,1],CURT=Fp[j,2],x_PEM=best[3]))
P=pd.DataFrame(rows).round(4)
P.to_csv('tables/T13_provenance.csv',index=False)
print(P.to_string(index=False))

def one(x,a,b,dd):
    r=evaluate_population(np.asarray(x,float)[None,:],a,b,dd)
    return {k:float(np.asarray(v).ravel()[0]) for k,v in r.items() if np.asarray(v).size==1}

sel=one(best,ps,pw,dh)
# robustness of the recommended design across many demand realizations
lc=[];lp=[]
for s in range(20):
    dq,_=D.demand_series(len(ps),seed=100+s)
    m=one(best,ps,pw,dq); lc.append(m['lcoh']); lp.append(m['lpsp'])
lc=np.array(lc); lp=np.array(lp)
dht,_=D.demand_series(len(ts),seed=99); tst=one(best,ts,tw,dht)
print(f"\nRECOMMENDED (pooled): {dict(zip(C.VAR_NAMES,np.round(best,1)))}")
print(f"  LCOH {sel['lcoh']:.4f}  LPSP {sel['lpsp']:.6f}  CURT {sel['curtailment']:.4f}  SI {sel['si']:.4f}")
print(f"  H2 prod {sel['h2_produced']:.0f} served {sel['h2_served']:.0f} discarded {sel['h2_discarded']:.4f} resid {sel['mass_residual']:.2e}")
print(f"  AWE life {sel['awe_life']:.1f} PEM life {sel['pem_life']:.1f} AWEh {sel['awe_hours']:.0f} PEMh {sel['pem_hours']:.0f} BESScyc {sel['bess_cycles']:.0f}")
print(f"  CAPEX {sel['ann_capex']:.4f} O&M {sel['ann_om']:.4f} repl {sel['ann_repl']:.4f} cyc {sel['ann_cyc']:.4f}")
print(f"\n20 demand realizations: LCOH {lc.mean():.4f} +- {lc.std(ddof=1):.4f}  [{lc.min():.3f}, {lc.max():.3f}]")
print(f"                        LPSP {lp.mean():.6f} +- {lp.std(ddof=1):.6f}  max {lp.max():.6f}")
print(f"HOLDOUT 2023: LCOH {tst['lcoh']:.4f} LPSP {tst['lpsp']:.6f} CURT {tst['curtailment']:.4f} SI {tst['si']:.4f}")
json.dump(dict(recommended=dict(zip(C.VAR_NAMES,best.tolist())),metrics=sel,holdout=tst,
               demand_robust=dict(lcoh_mean=float(lc.mean()),lcoh_sd=float(lc.std(ddof=1)),
                                  lcoh_min=float(lc.min()),lcoh_max=float(lc.max()),
                                  lpsp_mean=float(lp.mean()),lpsp_max=float(lp.max()),n=20),
               pooled_front=len(Fp)),open('checkpoints/reconciled.json','w'),indent=2)
np.save('checkpoints/F_pooled.npy',Fp); np.save('checkpoints/X_pooled.npy',Xp)
