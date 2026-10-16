"""Three checks: (a) does the F3 polish survive on FRESH demand realizations, or was
it overfitting to the eight used for selection? (b) second F3 seed with the same
polish protocol. (c) epsilon-near-optimal design envelope."""
import json, numpy as np, pandas as pd
from scipy.stats import qmc
import config as C, data_v2 as D
from objectives_v2 import evaluate_population
re_=D.load_re(); ps,pw=re_['train']
SEL=[D.demand_series(len(ps),seed=200+s)[0] for s in range(8)]    # selection set
VAL=[D.demand_series(len(ps),seed=400+s)[0] for s in range(8)]    # fresh validation set
LO=np.array(C.VAR_LOWER,float); HI=np.array(C.VAR_UPPER,float)

def ev(X,R):
    L=np.zeros((len(X),len(R))); P=np.zeros_like(L)
    for k,d in enumerate(R):
        r=evaluate_population(X,ps,pw,d); L[:,k],P[:,k]=r['lcoh'],r['lpsp']
    return L.mean(1), np.quantile(P,.95,axis=1)

def polish(x0,seed=5,n=180):
    cand=[x0.copy()]
    for j in range(6):
        for f in (0.90,0.95,1.05,1.10):
            y=x0.copy(); y[j]=np.clip(y[j]*f,LO[j],HI[j]); cand.append(y)
    s=qmc.LatinHypercube(d=6,seed=seed).random(n)
    for row in s:
        y=np.clip(x0*(0.7+0.6*row),LO,HI); cand.append(y)
    X=np.array(cand); L,Q=ev(X,SEL); f=Q<=0.01
    i=np.where(f)[0][np.argmin(L[f])]
    return X[i], X, L, Q, f

# ---- (a) overfitting check, seed 42
F=np.load('checkpoints/cont_F_42.npy'); X42=np.load('checkpoints/cont_X_42.npy')
ok=np.where(F[:,1]<=0.01)[0]; inc42=X42[ok[np.argmin(F[ok,0])]]
p42,Xa,La,Qa,fa=polish(inc42)
pair=np.array([inc42,p42])
Ls,Qs=ev(pair,SEL); Lv,Qv=ev(pair,VAL)
print('(a) F3 polish overfitting check, seed 42')
print(f"    incumbent  selection {Ls[0]:.4f} (q95 {Qs[0]:.4f}) | fresh {Lv[0]:.4f} (q95 {Qv[0]:.4f})")
print(f"    polished   selection {Ls[1]:.4f} (q95 {Qs[1]:.4f}) | fresh {Lv[1]:.4f} (q95 {Qv[1]:.4f})")
print(f"    gain on selection set {Ls[0]-Ls[1]:+.4f} | gain on fresh set {Lv[0]-Lv[1]:+.4f}")
print(f"    -> {'holds' if (Lv[0]-Lv[1])>0.05 else 'does NOT hold: polish was selection-set specific'}")

# ---- (b) second seed
x7=np.array(json.load(open('checkpoints/ablc_F3_7.json'))['x'])
p7,_,_,_,_=polish(x7,seed=11)
pr=np.array([x7,p7]); Ls7,Qs7=ev(pr,SEL); Lv7,Qv7=ev(pr,VAL)
print('\n(b) second F3 seed (seed 7, gen 40) under the same polish protocol')
print(f"    incumbent  selection {Ls7[0]:.4f} (q95 {Qs7[0]:.4f}) | fresh {Lv7[0]:.4f} (q95 {Qv7[0]:.4f})")
print(f"    polished   selection {Ls7[1]:.4f} (q95 {Qs7[1]:.4f}) | fresh {Lv7[1]:.4f} (q95 {Qv7[1]:.4f})")

# ---- (c) near-optimal envelope on the selection set
best=La[fa].min(); tol=best+0.10
sub=Xa[fa][La[fa]<=tol]
print(f'\n(c) epsilon-near-optimal set, LCOH within 0.10 USD/kg of {best:.4f}: {len(sub)} designs')
E=pd.DataFrame(sub,columns=C.VAR_NAMES)
print(E.describe().loc[['min','max']].round(1).to_string())
E.round(2).to_csv('tables/T16_near_optimal_set.csv',index=False)
json.dump(dict(seed42_polished=p42.tolist(),seed42_sel=float(Ls[1]),seed42_fresh=float(Lv[1]),
               seed42_inc_fresh=float(Lv[0]),seed7_polished=p7.tolist(),
               seed7_sel=float(Ls7[1]),seed7_fresh=float(Lv7[1]),n_near=int(len(sub))),
          open('checkpoints/polish_val.json','w'),indent=2)
