import sys, json, os, numpy as np, pandas as pd
import config as C, data_v2 as D, optimizer_v2 as O
from objectives_v2 import objectives, evaluate_population
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
which=sys.argv[1]
re=D.load_re(); ps,pw=re['train']
DL=[D.demand_series(len(ps),seed=0)[0]]          # single realisation for benchmarks
lo=np.array(C.VAR_LOWER,float); hi=np.array(C.VAR_UPPER,float)
if which=='awe':   hi=np.array([hi[0],hi[1],hi[2],1e-6,hi[4],hi[5]])
elif which=='pem': lo=np.array([lo[0],lo[1],lo[2],1-1e-6,lo[4],lo[5]])
elif which=='fixed': lo=np.array([5500.,1400.,lo[2],lo[3],lo[4],lo[5]]); hi=np.array([5500.,1400.,hi[2],hi[3],hi[4],hi[5]])
class P(Problem):
    def __init__(s): super().__init__(n_var=6,n_obj=3,xl=lo,xu=hi)
    def _evaluate(s,X,out,*a,**k): out['F']=objectives(X,ps,pw,DL)
r=minimize(P(),NSGA3(pop_size=40,ref_dirs=get_reference_directions('das-dennis',3,n_partitions=6)),
           ('n_gen',18),seed=7,verbose=False)
i,_=O.entropy_topsis(r.F)
ok=np.where(r.F[:,1]<=0.01)[0]
j=ok[np.argmin(r.F[ok,0])] if len(ok) else i
json.dump(dict(x=r.X[j].tolist(),F=r.F[j].tolist()),open(f'checkpoints/bench_{which}.json','w'))
print(which,'->',np.round(r.X[j],1),np.round(r.F[j],4))
