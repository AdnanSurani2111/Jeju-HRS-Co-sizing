"""optimizer_v2.py - NSGA-III over the six variable design space."""
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
import config as C
from objectives_v2 import objectives, evaluate_population


class HRSProblem(Problem):
    def __init__(self, pu_solar, pu_wind, demand_list, **kw):
        super().__init__(n_var=6, n_obj=C.N_OBJ, n_ieq_constr=0,
                         xl=np.array(C.VAR_LOWER), xu=np.array(C.VAR_UPPER))
        self.ps, self.pw, self.dl, self.kw = pu_solar, pu_wind, demand_list, kw

    def _evaluate(self, X, out, *a, **k):
        out['F'] = objectives(X, self.ps, self.pw, self.dl, **self.kw)


def run(pu_solar, pu_wind, demand_list, pop=None, gen=None, seed=None, verbose=True):
    pop = C.NSGA3_POP if pop is None else pop
    gen = C.NSGA3_GEN if gen is None else gen
    seed = C.NSGA3_SEED if seed is None else seed
    ref = get_reference_directions('das-dennis', C.N_OBJ, n_partitions=C.DAS_DENNIS_P)
    prob = HRSProblem(pu_solar, pu_wind, demand_list)
    algo = NSGA3(pop_size=pop, ref_dirs=ref)
    res = minimize(prob, algo, ('n_gen', gen), seed=seed, verbose=verbose,
                   save_history=True)
    return res


def hv_trajectory(res):
    """Hypervolume over the CUMULATIVE non-dominated archive -> monotone by construction."""
    ind = HV(ref_point=np.array(C.HV_REF))
    arch, traj = None, []
    for h in res.history:
        F = h.opt.get('F')
        arch = F if arch is None else np.vstack([arch, F])
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        keep = NonDominatedSorting().do(arch, only_non_dominated_front=True)
        arch = arch[keep]
        traj.append(ind(arch))
    return np.array(traj)


def entropy_topsis(F):
    """Entropy objective weights + TOPSIS closeness. Returns (index, weights)."""
    F = np.asarray(F, dtype=float)
    rng = F.max(0) - F.min(0)
    rng[rng < 1e-12] = 1.0
    Z = (F - F.min(0)) / rng                     # 0 = best (all objectives minimised)
    P = Z + 1e-12
    P = P / P.sum(0, keepdims=True)
    n = F.shape[0]
    E = -(P * np.log(P)).sum(0) / np.log(n)
    d = 1.0 - E
    w = d / d.sum() if d.sum() > 1e-12 else np.full(F.shape[1], 1.0 / F.shape[1])
    V = Z * w
    dp = np.linalg.norm(V - V.min(0), axis=1)
    dm = np.linalg.norm(V - V.max(0), axis=1)
    C_ = dm / (dp + dm + 1e-12)
    return int(np.argmax(C_)), w
