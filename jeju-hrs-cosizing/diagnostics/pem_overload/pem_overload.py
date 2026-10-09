"""PEM overload sensitivity, fixed-design re-evaluation.

The frozen selected design is re-evaluated on the 20 primary validation draws
(seeds 500-519) with PEM_MAX_LOAD_FRAC set to 1.00, 1.10 and 1.20. The design
is NOT re-optimized, so this answers whether the reported performance depends
on the overload allowance. It does not answer whether the selected capacities
would change; that requires re-optimization.

Basis: PRIMARY (20 fresh demand realizations, seeds 500-519).
"""
import json
import os
import numpy as np
import pandas as pd
import config as C
import data_v2 as D
import importlib
import objectives_v2

re_ = D.load_re()
ps, pw = re_['train']
VAL = [D.demand_series(len(ps), seed=500 + s)[0] for s in range(20)]

# FINAL.json is looked up beside this script first, then in the repository root.
cand = [os.path.join(C.BASE_DIR, 'checkpoints', 'FINAL.json'),
        os.path.abspath(os.path.join(C.BASE_DIR, '..', '..', 'checkpoints', 'FINAL.json'))]
fp = next(p for p in cand if os.path.exists(p))
x = np.array(list(json.load(open(fp))['design'].values()))
P_PEM = x[2] * x[3]
print(f'selected design: {np.round(x,1)}')
print(f'installed PEM rating {P_PEM:.1f} kW\n')

rows = []
for cap in (1.00, 1.10, 1.20):
    C.PEM_MAX_LOAD_FRAC = cap
    importlib.reload(objectives_v2)          # pick up the changed module constant
    objectives_v2.C.PEM_MAX_LOAD_FRAC = cap
    import dispatch_v2
    dispatch_v2.C.PEM_MAX_LOAD_FRAC = cap
    L, P, Cu, S = [], [], [], []
    for d in VAL:
        r = objectives_v2.evaluate_population(x[None, :], ps, pw, d)
        L.append(float(r['lcoh'][0])); P.append(float(r['lpsp'][0]))
        Cu.append(float(r['curtailment'][0])); S.append(float(r['h2_served'][0]))
    L, P, Cu, S = map(np.array, (L, P, Cu, S))
    rows.append(dict(pem_cap=cap, lcoh_mean=L.mean(), lcoh_sd=L.std(ddof=1),
                     lpsp_mean=100 * P.mean(), lpsp_q95=100 * np.quantile(P, .95),
                     lpsp_max=100 * P.max(), curt_mean=100 * Cu.mean(),
                     h2_served=S.mean()))
    print(f'  cap {cap:.2f}  LCOH {L.mean():.4f} +- {L.std(ddof=1):.4f}   '
          f'LPSP mean {100*P.mean():.4f}%  q95 {100*np.quantile(P,.95):.4f}%  '
          f'CURT {100*Cu.mean():.2f}%  served {S.mean():.0f} kg/yr')

T = pd.DataFrame(rows)
base = T[T.pem_cap == 1.20].iloc[0]
T['d_lcoh'] = T.lcoh_mean - base.lcoh_mean
T['d_lpsp_q95'] = T.lpsp_q95 - base.lpsp_q95
T.round(5).to_csv('pem_overload_sensitivity.csv', index=False)
print('\nchange relative to the reported 1.20 cap:')
for _, r in T.iterrows():
    print(f"  cap {r.pem_cap:.2f}: dLCOH {r.d_lcoh:+.4f} USD/kg   "
          f"d q95 service loss {r.d_lpsp_q95:+.4f} pp")
json.dump(dict(basis='PRIMARY', realizations=20, seeds='500-519',
               design_frozen=True, reoptimized=False,
               installed_pem_kw=float(P_PEM),
               results=T.to_dict('records')),
          open('pem_overload_sensitivity.json', 'w'), indent=2)
print('\nwritten: pem_overload_sensitivity.csv / .json')
