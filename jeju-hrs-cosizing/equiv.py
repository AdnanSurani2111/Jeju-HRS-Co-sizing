"""Pause/resume equivalence test. Runs seed 99 for 12 generations continuously,
then again as 6+6 with serialization between, and compares objective vectors."""
import os, numpy as np, dill as pickle
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
import config as C, data_v2 as D
from objectives_v2 import objectives
re_=D.load_re(); ps,pw=re_['train']
DL=[D.demand_series(len(ps),seed=s)[0] for s in range(C.N_DEMAND_REALISATIONS)]
class P(Problem):
    def __init__(s): super().__init__(n_var=6,n_obj=3,xl=np.array(C.VAR_LOWER),xu=np.array(C.VAR_UPPER))
    def _evaluate(s,X,out,*a,**k): out['F']=objectives(X,ps,pw,DL)
def mk(seed):
    a=NSGA3(pop_size=60,ref_dirs=get_reference_directions('das-dennis',3,n_partitions=8))
    a.setup(P(),termination=('n_gen',12),seed=seed,save_history=False,verbose=False); return a
# continuous
a=mk(99)
for _ in range(12): a.next()
Fc=a.result().F; Xc=a.result().X
# paused and resumed
b=mk(99)
for _ in range(6): b.next()
blob=pickle.dumps(b); del b
b=pickle.loads(blob)
for _ in range(6): b.next()
Fr=b.result().F; Xr=b.result().X
print(f'continuous front {len(Fc)}, resumed front {len(Fr)}')
if Fc.shape==Fr.shape:
    dF=np.abs(np.sort(Fc,axis=0)-np.sort(Fr,axis=0)).max()
    dX=np.abs(np.sort(Xc,axis=0)-np.sort(Xr,axis=0)).max()
    print(f'max abs difference: objectives {dF:.3e}   design variables {dX:.3e}')
    print('EQUIVALENT' if dF<1e-9 else 'NOT bit-identical; difference reported above')
else:
    print('front sizes differ -> NOT equivalent')
