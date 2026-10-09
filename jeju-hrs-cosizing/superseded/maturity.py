"""Demand-maturity sensitivity: re-optimise on the mature-station basis under the
identical budget and selection rule used for the full-record case."""
import json, numpy as np
import config as C, data_v2 as D
from objectives_v2 import objectives
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
re=D.load_re(); ps,pw=re['train']
DL=[D.demand_series(len(ps),seed=s,basis='mature')[0] for s in range(2)]
class P(Problem):
    def __init__(s): super().__init__(n_var=6,n_obj=3,xl=np.array(C.VAR_LOWER),xu=np.array(C.VAR_UPPER))
    def _evaluate(s,X,out,*a,**k): out['F']=objectives(X,ps,pw,DL)
r=minimize(P(),NSGA3(pop_size=92,ref_dirs=get_reference_directions('das-dennis',3,n_partitions=10)),
           ('n_gen',30),seed=17,verbose=False)
ok=np.where(r.F[:,1]<=0.01)[0]; i=ok[np.argmin(r.F[ok,0])]
json.dump(dict(x=r.X[i].tolist(),F=r.F[i].tolist()),open('checkpoints/maturity.json','w'))
print(f'mature basis: LCOH {r.F[i,0]:6.2f} LPSP {r.F[i,1]:.4f} CURT {r.F[i,2]:.3f} | '
      f'Psol {r.X[i,0]:6.0f} Pwin {r.X[i,1]:5.0f} Pelz {r.X[i,2]:6.0f} xPEM {r.X[i,3]:.3f} '
      f'BESS {r.X[i,4]:5.0f} HSS {r.X[i,5]:6.0f}')
