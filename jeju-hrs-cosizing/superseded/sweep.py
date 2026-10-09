import sys, json, os, numpy as np
import config as C, data_v2 as D, optimizer_v2 as O
from objectives_v2 import objectives
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
re=D.load_re(); ps,pw=re['train']
for sc in [float(a) for a in sys.argv[1:]]:
    DL=[D.demand_series(len(ps),seed=0,scale=sc)[0]]
    class P(Problem):
        def __init__(s): super().__init__(n_var=6,n_obj=3,xl=np.array(C.VAR_LOWER),xu=np.array(C.VAR_UPPER))
        def _evaluate(s,X,out,*a,**k): out['F']=objectives(X,ps,pw,DL)
    r=minimize(P(),NSGA3(pop_size=40,ref_dirs=get_reference_directions('das-dennis',3,n_partitions=6)),
               ('n_gen',18),seed=11,verbose=False)
    ok=np.where(r.F[:,1]<=0.01)[0]
    i=ok[np.argmin(r.F[ok,0])] if len(ok) else int(np.argmin(r.F[:,0]))
    d=dict(scale=sc, demand=181.3*sc, x=r.X[i].tolist(), F=r.F[i].tolist(), constrained=bool(len(ok)))
    json.dump(d,open(f'checkpoints/sweep_{sc}.json','w'))
    print(f"{181.3*sc:6.1f} kg/d | LCOH {r.F[i,0]:6.2f} LPSP {r.F[i,1]:.4f} CURT {r.F[i,2]:.3f} | "
          f"Psol {r.X[i,0]:6.0f} Pwin {r.X[i,1]:5.0f} Pelz {r.X[i,2]:5.0f} xPEM {r.X[i,3]:.3f} HSS {r.X[i,5]:6.0f}", flush=True)
