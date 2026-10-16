"""F1 converged by deterministic 2-D search. Only P_sol and P_wind are free, so a
grid plus local refinement is the appropriate method and removes the stochastic
under-convergence that contaminated the NSGA-III F1 runs."""
import numpy as np, pandas as pd, json
import config as C, data_v2 as D
from objectives_v2 import evaluate_population
re_=D.load_re(); ps,pw=re_['train']
REV=[D.demand_series(len(ps),seed=200+s)[0] for s in range(8)]
V1=np.array([5500.,1400.,1738.6,0.2314,375.8,108.7])   # conversion/storage fixed at published design

def eval_grid(S,W):
    X=np.column_stack([S,W,np.full(len(S),V1[2]),np.full(len(S),V1[3]),
                       np.full(len(S),V1[4]),np.full(len(S),V1[5])])
    L=np.zeros((len(S),len(REV))); P=np.zeros_like(L); Cu=np.zeros_like(L)
    for k,d in enumerate(REV):
        r=evaluate_population(X,ps,pw,d)
        L[:,k],P[:,k],Cu[:,k]=r['lcoh'],r['lpsp'],r['curtailment']
    return L.mean(1),np.quantile(P,.95,axis=1),Cu.mean(1)

def search(slo,shi,wlo,whi,n,tag):
    sv=np.linspace(slo,shi,n); wv=np.linspace(wlo,whi,n)
    S,W=[a.ravel() for a in np.meshgrid(sv,wv)]
    lc,q95,cu=eval_grid(S,W)
    feas=q95<=0.01
    if not feas.any():
        print(f'{tag}: no feasible point (q95<=1%)'); return None
    i=np.where(feas)[0][np.argmin(lc[feas])]
    print(f'{tag}: n={n}x{n}, feasible {feas.sum()}/{len(S)} | best LCOH {lc[i]:.4f} '
          f'at P_sol {S[i]:.0f} P_win {W[i]:.0f} (q95 {q95[i]:.4f}, CURT {cu[i]:.3f})')
    return S[i],W[i],lc[i],q95[i],cu[i]

print('coarse sweep over the full F1 bound range')
c=search(C.VAR_LOWER[0],C.VAR_UPPER[0],C.VAR_LOWER[1],C.VAR_UPPER[1],18,'coarse')
s0,w0=c[0],c[1]
print('\nrefinement around the coarse optimum')
r1=search(max(C.VAR_LOWER[0],s0*0.6),s0*1.4,max(C.VAR_LOWER[1],w0*0.6),w0*1.4,18,'refine 1')
s1,w1=r1[0],r1[1]
r2=search(max(C.VAR_LOWER[0],s1*0.85),s1*1.15,max(C.VAR_LOWER[1],w1*0.85),w1*1.15,18,'refine 2')
S,W,L,Q,CU=r2
print(f'\nF1 CONVERGED: P_sol {S:.1f}  P_win {W:.1f}  LCOH {L:.4f}  q95 LPSP {Q:.5f}  CURT {CU:.4f}')
# F0 on the same protocol
l0,q0,c0=eval_grid(np.array([V1[0]]),np.array([V1[1]]))
print(f'F0 reference : LCOH {l0[0]:.4f}  q95 {q0[0]:.5f}  CURT {c0[0]:.4f}')
print(f'\nrenewable-only benefit (F0 - F1) = {l0[0]-L:.4f} USD/kg')
print(f'  previous NSGA-III F1 estimates were 1.75 (seed 42) and 0.92 (seed 7)')
json.dump(dict(P_sol=float(S),P_wind=float(W),LCOH=float(L),q95=float(Q),CURT=float(CU),
               F0_LCOH=float(l0[0]),benefit=float(l0[0]-L)),open('checkpoints/f1_grid.json','w'),indent=2)
