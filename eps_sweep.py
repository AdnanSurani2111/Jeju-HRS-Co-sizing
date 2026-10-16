"""Matched-reliability comparison: cheapest design per technology at each LPSP target.
Saves the FULL front so any epsilon can be queried without re-optimising."""
import sys, json, numpy as np
import config as C, data_v2 as D
from objectives_v2 import objectives
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

which = sys.argv[1]; gen = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 17
re = D.load_re(); ps, pw = re['train']
DL = [D.demand_series(len(ps), seed=s)[0] for s in range(2)]
lo = np.array(C.VAR_LOWER, float); hi = np.array(C.VAR_UPPER, float)
if which == 'awe':   hi[3] = 1e-6
elif which == 'pem': lo[3] = 1 - 1e-6

class P(Problem):
    def __init__(s): super().__init__(n_var=6, n_obj=3, xl=lo, xu=hi)
    def _evaluate(s, X, out, *a, **k): out['F'] = objectives(X, ps, pw, DL)

r = minimize(P(), NSGA3(pop_size=92, ref_dirs=get_reference_directions('das-dennis', 3, n_partitions=10)),
             ('n_gen', gen), seed=SEED, verbose=False)
np.save(f'checkpoints/eps_F_{which}_{SEED}.npy', r.F)
np.save(f'checkpoints/eps_X_{which}_{SEED}.npy', r.X)
print(f'{which}: front {len(r.F)}, LPSP min {r.F[:,1].min():.5f}')
for e in (0.01, 0.005, 0.001, 0.0):
    ok = np.where(r.F[:, 1] <= e + 1e-12)[0]
    if len(ok):
        i = ok[np.argmin(r.F[ok, 0])]
        print(f'   eps<={e:<6}: n={len(ok):3d} LCOH {r.F[i,0]:6.2f} CURT {r.F[i,2]:.3f} '
              f'Psol {r.X[i,0]:6.0f} Pwin {r.X[i,1]:5.0f} Pelz {r.X[i,2]:5.0f} HSS {r.X[i,5]:6.0f}')
    else:
        print(f'   eps<={e:<6}: INFEASIBLE on this front')
