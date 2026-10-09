"""Resumable NSGA-III. Each invocation advances CHUNK generations and
checkpoints the full population, so the run fits inside a command time limit."""
import os
import sys
import json
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import config as C
import data_v2 as D
from objectives_v2 import objectives

CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 20
STATE = 'checkpoints/state.npz'

re = D.load_re()
ps, pw = re['train']
DL = [D.demand_series(len(ps), seed=s)[0] for s in range(C.N_DEMAND_REALISATIONS)]


class P(Problem):
    def __init__(self):
        super().__init__(n_var=6, n_obj=3, n_ieq_constr=0,
                         xl=np.array(C.VAR_LOWER), xu=np.array(C.VAR_UPPER))

    def _evaluate(self, X, out, *a, **k):
        out['F'] = objectives(X, ps, pw, DL)


ref = get_reference_directions('das-dennis', 3, n_partitions=C.DAS_DENNIS_P)
if os.path.exists(STATE):
    s = np.load(STATE)
    X0, done, arch = s['X'], int(s['gen']), s['arch']
    sampling = X0
    print(f'resuming from generation {done}', flush=True)
else:
    sampling, done, arch = FloatRandomSampling(), 0, None

algo = NSGA3(pop_size=C.NSGA3_POP, ref_dirs=ref, sampling=sampling)
res = minimize(P(), algo, ('n_gen', CHUNK), seed=C.NSGA3_SEED + done, verbose=False,
               save_history=False)

Xpop = res.pop.get('X')
Fopt = res.F
arch = Fopt if arch is None else np.vstack([arch, Fopt])
arch = arch[NonDominatedSorting().do(arch, only_non_dominated_front=True)]
hv = float(HV(ref_point=np.array(C.HV_REF))(arch))

hist = []
if os.path.exists('checkpoints/hv_hist.json'):
    hist = json.load(open('checkpoints/hv_hist.json'))
hist.append(dict(gen=done + CHUNK, hv=hv, n_front=int(len(arch))))
json.dump(hist, open('checkpoints/hv_hist.json', 'w'), indent=2)

np.savez(STATE, X=Xpop, gen=done + CHUNK, arch=arch)
np.save('checkpoints/F.npy', arch)
# keep the design vectors matching the archive front from the latest generation
np.save('checkpoints/F_gen.npy', Fopt)
np.save('checkpoints/X_gen.npy', res.X)
print(f'generation {done + CHUNK}: archive front {len(arch)}, HV {hv:.6f}', flush=True)
