"""F0-F3 co-sizing ablation. All optimised cases use identical budget
(pop 92, gen 30, seed 17, 2 demand realisations) and identical selection
(cheapest design with LPSP <= 1%), so differences are attributable to which
variables are free, not to optimiser effort."""
import sys, json, numpy as np
import config as C, data_v2 as D
from objectives_v2 import objectives, evaluate_population
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

re = D.load_re(); ps, pw = re['train']
DL = [D.demand_series(len(ps), seed=s)[0] for s in range(2)]
V1 = np.array([5500., 1400., 1738.6, 0.2314, 375.8, 108.7])   # published v1 design

def opt(lo, hi, tag, gen=30):
    class P(Problem):
        def __init__(s): super().__init__(n_var=6, n_obj=3, xl=lo, xu=hi)
        def _evaluate(s, X, out, *a, **k): out['F'] = objectives(X, ps, pw, DL)
    r = minimize(P(), NSGA3(pop_size=92, ref_dirs=get_reference_directions('das-dennis',3,n_partitions=10)),
                 ('n_gen', gen), seed=17, verbose=False)
    ok = np.where(r.F[:,1] <= 0.01)[0]
    i = ok[np.argmin(r.F[ok,0])] if len(ok) else int(np.argmin(r.F[:,0]))
    return r.X[i], r.F[i]

case = sys.argv[1]
lo = np.array(C.VAR_LOWER, float); hi = np.array(C.VAR_UPPER, float)
if case == 'F0':          # everything fixed at the published v1 design
    m = evaluate_population(V1[None,:], ps, pw, DL[0])
    x = V1; F = np.array([m['lcoh'][0], m['lpsp'][0], m['curtailment'][0]])
elif case == 'F1':        # renewable free, conversion+storage fixed at v1
    lo = V1.copy(); hi = V1.copy()
    lo[0], hi[0] = C.VAR_LOWER[0], C.VAR_UPPER[0]
    lo[1], hi[1] = C.VAR_LOWER[1], C.VAR_UPPER[1]
    x, F = opt(lo, hi, case)
elif case == 'F2':        # renewable fixed at v1 plant, everything else free
    lo[0] = hi[0] = 5500.; lo[1] = hi[1] = 1400.
    x, F = opt(lo, hi, case)
else:                     # F3 fully joint
    x, F = opt(lo, hi, case)
json.dump(dict(case=case, x=x.tolist(), F=F.tolist()), open(f'checkpoints/abl_{case}.json','w'))
print(f'{case}: LCOH {F[0]:6.2f}  LPSP {F[1]:.4f}  CURT {F[2]:.3f} | '
      f'Psol {x[0]:6.0f} Pwin {x[1]:5.0f} Pelz {x[2]:6.1f} xPEM {x[3]:.3f} BESS {x[4]:6.1f} HSS {x[5]:6.1f}')
