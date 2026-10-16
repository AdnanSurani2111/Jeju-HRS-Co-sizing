"""F0-F3 ablation under the uninterrupted optimizer with full state preservation."""
import os,sys,json,numpy as np,dill as pickle
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
import config as C, data_v2 as D
from objectives_v2 import objectives, evaluate_population
case=sys.argv[1]; NG=int(sys.argv[2]); TGT=int(sys.argv[3]) if len(sys.argv)>3 else 100
SEED=int(sys.argv[4]) if len(sys.argv)>4 else 42
re_=D.load_re(); ps,pw=re_['train']
DL=[D.demand_series(len(ps),seed=s)[0] for s in range(C.N_DEMAND_REALISATIONS)]
V1=np.array([5500.,1400.,1738.6,0.2314,375.8,108.7])
lo=np.array(C.VAR_LOWER,float); hi=np.array(C.VAR_UPPER,float)
if case=='F1':
    lo=V1.copy(); hi=V1.copy(); lo[0],hi[0]=C.VAR_LOWER[0],C.VAR_UPPER[0]; lo[1],hi[1]=C.VAR_LOWER[1],C.VAR_UPPER[1]
elif case=='F2':
    lo[0]=hi[0]=5500.; lo[1]=hi[1]=1400.
# F3 keeps full bounds
ST=f'checkpoints/abl_state_{case}_{SEED}.pkl'
class P(Problem):
    def __init__(s): super().__init__(n_var=6,n_obj=3,xl=lo,xu=hi)
    def _evaluate(s,X,out,*a,**k): out['F']=objectives(X,ps,pw,DL)
if os.path.exists(ST):
    algo=pickle.load(open(ST,'rb')); print(f'resumed {case} at gen {algo.n_gen-1}',flush=True)
else:
    algo=NSGA3(pop_size=C.NSGA3_POP,ref_dirs=get_reference_directions('das-dennis',3,n_partitions=C.DAS_DENNIS_P))
    algo.setup(P(),termination=('n_gen',TGT),seed=SEED,save_history=False,verbose=False)
    print(f'init {case}',flush=True)
d=0
while algo.has_next() and d<NG:
    algo.next(); d+=1
pickle.dump(algo,open(ST,'wb'))
r=algo.result(); F,X=r.F,r.X
np.save(f'checkpoints/ablc_F_{case}_{SEED}.npy',F); np.save(f'checkpoints/ablc_X_{case}_{SEED}.npy',X)
ok=np.where(F[:,1]<=0.01)[0]
i=ok[np.argmin(F[ok,0])] if len(ok) else int(np.argmin(F[:,0]))
json.dump(dict(case=case,gen=int(algo.n_gen-1),x=X[i].tolist(),F=F[i].tolist(),feasible=bool(len(ok))),
          open(f'checkpoints/ablc_{case}_{SEED}.json','w'))
print(f'{case} seed {SEED} gen {algo.n_gen-1}: LCOH {F[i,0]:.4f} LPSP {F[i,1]:.5f} CURT {F[i,2]:.4f} | '
      f'Psol {X[i,0]:.0f} Pwin {X[i,1]:.0f} Pelz {X[i,2]:.0f} xPEM {X[i,3]:.3f}',flush=True)
