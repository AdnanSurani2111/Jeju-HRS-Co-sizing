import json
import numpy as np
import config as C
import data_v2 as D
import optimizer_v2 as O
from objectives_v2 import evaluate_population

re = D.load_re()
ps, pw = re['train']
DL = [D.demand_series(len(ps), seed=s)[0] for s in range(C.N_DEMAND_REALISATIONS)]
print("hours:", len(ps), "demand realisations:", len(DL), flush=True)

res = O.run(ps, pw, DL, verbose=False)
F, X = res.F, res.X
idx, w = O.entropy_topsis(F)
hv = O.hv_trajectory(res)
np.save('checkpoints/F.npy', F)
np.save('checkpoints/X.npy', X)
np.save('checkpoints/hv.npy', hv)

best = X[idx]
det = evaluate_population(best[None, :], ps, pw, DL[0])
out = dict(x=dict(zip(C.VAR_NAMES, best.tolist())),
           F=dict(lcoh=float(F[idx, 0]), lpsp=float(F[idx, 1]), curt=float(F[idx, 2])),
           entropy_weights=dict(zip(['LCOH', 'LPSP', 'CURT'], w.tolist())),
           detail={k: float(np.asarray(v).ravel()[0]) for k, v in det.items()
                   if np.asarray(v).size == 1},
           hv_final=float(hv[-1]), hv_gen=hv.tolist(), n_pareto=int(len(F)))
json.dump(out, open('checkpoints/topsis.json', 'w'), indent=2)
print(json.dumps({k: out[k] for k in ('x', 'F', 'entropy_weights', 'n_pareto')}, indent=2))
