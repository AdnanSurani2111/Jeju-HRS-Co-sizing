"""run_continuous.py - a genuine uninterrupted NSGA-III run.

The earlier run_chunk.py rebuilt a fresh NSGA3 object each chunk from the
population alone, discarding survival, niching, and ideal/nadir state. That is
not a continuation. Here the ENTIRE algorithm object is pickled between chunks,
so resuming is bit-identical to never having stopped.

Usage: python3 run_continuous.py <seed> <generations_this_call>
"""
import os
import sys
import dill as pickle
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
import config as C
import data_v2 as D
from objectives_v2 import objectives

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
NGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 25
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 100
STATE = f'checkpoints/cont_state_{SEED}.pkl'

re_ = D.load_re()
ps, pw = re_['train']
# Fixed demand realizations, identical for every candidate in every generation.
DL = [D.demand_series(len(ps), seed=s)[0] for s in range(C.N_DEMAND_REALISATIONS)]


class HRSProblem(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=3, n_ieq_constr=0,
                         xl=np.array(C.VAR_LOWER), xu=np.array(C.VAR_UPPER))

    def _evaluate(self, X, out, *a, **k):
        out['F'] = objectives(X, ps, pw, DL)


if os.path.exists(STATE):
    with open(STATE, 'rb') as f:
        algo, hist = pickle.load(f)
    print(f'resumed at generation {algo.n_gen - 1} with full optimizer state', flush=True)
else:
    ref = get_reference_directions('das-dennis', 3, n_partitions=C.DAS_DENNIS_P)
    algo = NSGA3(pop_size=C.NSGA3_POP, ref_dirs=ref)
    algo.setup(HRSProblem(), termination=('n_gen', TARGET), seed=SEED,
               save_history=False, verbose=False)
    hist = []
    print('initialized fresh run', flush=True)

ind = HV(ref_point=np.array(C.HV_REF))
done = 0
while algo.has_next() and done < NGEN:
    algo.next()
    done += 1
    F = algo.opt.get('F')
    hist.append(dict(gen=int(algo.n_gen - 1), hv=float(ind(F)), n_front=int(len(F))))

with open(STATE, 'wb') as f:
    pickle.dump((algo, hist), f)

res = algo.result()
np.save(f'checkpoints/cont_F_{SEED}.npy', res.F)
np.save(f'checkpoints/cont_X_{SEED}.npy', res.X)
import json
json.dump(hist, open(f'checkpoints/cont_hv_{SEED}.json', 'w'), indent=2)

F, X = res.F, res.X
ok = np.where(F[:, 1] <= 0.01)[0]
g = int(algo.n_gen - 1)
if len(ok):
    i = ok[np.argmin(F[ok, 0])]
    print(f'seed {SEED} gen {g}: front {len(F)}, HV {hist[-1]["hv"]:.6f}, '
          f'best feasible LCOH {F[i,0]:.4f} LPSP {F[i,1]:.5f} CURT {F[i,2]:.4f} '
          f'xPEM {X[i,3]:.3f}', flush=True)
else:
    print(f'seed {SEED} gen {g}: front {len(F)}, HV {hist[-1]["hv"]:.6f}, '
          f'no solution meets LPSP<=1%', flush=True)
