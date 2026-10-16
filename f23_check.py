"""Local convergence check for F2 and F3: Latin hypercube around the incumbent plus
coordinate-wise perturbations, all re-evaluated on the same eight common demand
realizations with the same q95 reliability screen."""
import json, numpy as np, pandas as pd
from scipy.stats import qmc
import config as C, data_v2 as D
from objectives_v2 import evaluate_population
re_=D.load_re(); ps,pw=re_['train']
REV=[D.demand_series(len(ps),seed=200+s)[0] for s in range(8)]
LO=np.array(C.VAR_LOWER,float); HI=np.array(C.VAR_UPPER,float)

def common(X):
    L=np.zeros((len(X),len(REV))); P=np.zeros_like(L)
    for k,d in enumerate(REV):
        r=evaluate_population(X,ps,pw,d); L[:,k],P[:,k]=r['lcoh'],r['lpsp']
    return L.mean(1), np.quantile(P,.95,axis=1)

def check(x0, free, tag, n_lhs=180, seed=5):
    x0=np.asarray(x0,float); cand=[x0.copy()]
    # coordinate-wise perturbations on the free variables
    for j in free:
        for f in (0.90,0.95,1.05,1.10):
            y=x0.copy(); y[j]=np.clip(y[j]*f,LO[j],HI[j]); cand.append(y)
    # Latin hypercube in a +-30% box around the incumbent, free variables only
    s=qmc.LatinHypercube(d=len(free),seed=seed).random(n_lhs)
    for row in s:
        y=x0.copy()
        for k,j in enumerate(free):
            lo=max(LO[j],x0[j]*0.7); hi=min(HI[j],x0[j]*1.3)
            y[j]=lo+row[k]*(hi-lo)
        cand.append(y)
    X=np.array(cand); L,Q=common(X)
    base_L,base_Q=L[0],Q[0]
    feas=Q<=0.01
    print(f'{tag}: incumbent LCOH {base_L:.4f} (q95 {base_Q:.5f}), {len(X)} candidates, feasible {feas.sum()}')
    if not feas.any():
        print('   no feasible candidate'); return base_L
    i=np.where(feas)[0][np.argmin(L[feas])]
    imp=base_L-L[i]
    print(f'   best feasible LCOH {L[i]:.4f} (q95 {Q[i]:.5f})  improvement {imp:+.4f} USD/kg')
    if i==0: print('   incumbent is best -> locally converged')
    else:
        print('   improved point:', np.round(X[i],1))
    return L[i]

# F2 incumbent: seed 42 gen 100
x2=np.array(json.load(open('checkpoints/ablc_F2.json'))['x'])
r2=check(x2,[2,3,4,5],'F2 (conversion+storage free)')
# F3 incumbent: seed 42 continuous gen 100, reliability screened
F=np.load('checkpoints/cont_F_42.npy'); X3=np.load('checkpoints/cont_X_42.npy')
ok=np.where(F[:,1]<=0.01)[0]; x3=X3[ok[np.argmin(F[ok,0])]]
r3=check(x3,[0,1,2,3,4,5],'F3 (all six free)')
f1=json.load(open('checkpoints/f1_grid.json'))
print(f"\nPOST-CHECK DECOMPOSITION (F0 {f1['F0_LCOH']:.4f})")
print(f"  F1 grid   {f1['LCOH']:.4f}  benefit {f1['F0_LCOH']-f1['LCOH']:.4f}")
print(f"  F2 polish {r2:.4f}  benefit {f1['F0_LCOH']-r2:.4f}")
print(f"  F3 polish {r3:.4f}  benefit {f1['F0_LCOH']-r3:.4f}")
I=(f1['F0_LCOH']-r3)-((f1['F0_LCOH']-f1['LCOH'])+(f1['F0_LCOH']-r2))
print(f"  interaction I = {I:.4f} USD/kg")
json.dump(dict(F0=f1['F0_LCOH'],F1=f1['LCOH'],F2=r2,F3=r3,interaction=I),
          open('checkpoints/ablation_final.json','w'),indent=2)
