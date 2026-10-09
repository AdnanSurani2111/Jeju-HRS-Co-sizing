"""finalize.py - all designs re-evaluated on ONE common demand realisation so the
comparison table is budget-neutral, plus held-out 2023, LCOH decomposition and
the remaining figures. Pure evaluation; no optimisation."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as C
import data_v2 as D
import optimizer_v2 as O
from objectives_v2 import evaluate_population

plt.rcParams.update({'font.size': 9, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.grid': True, 'grid.alpha': 0.3})


def save(fig, n):
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(C.FIG_DIR, f'{n}.{e}'))
    plt.close(fig)
    print('  figure:', n)


def one(x, ps, pw, dh):
    r = evaluate_population(np.asarray(x, float)[None, :], ps, pw, dh)
    return {k: float(np.asarray(v).ravel()[0]) for k, v in r.items()
            if np.asarray(v).size == 1}


re = D.load_re()
ps, pw = re['train']
ts, tw = re['test']
dh, _ = D.demand_series(len(ps), seed=0)          # common basis
F = np.load('checkpoints/F_gen.npy')
X = np.load('checkpoints/X_gen.npy')
hist = json.load(open('checkpoints/hv_hist.json'))
idx, w = O.entropy_topsis(F)

ok = np.where(F[:, 1] <= 0.01)[0]
rec = X[ok[np.argmin(F[ok, 0])]]                   # recommended: cheapest with LPSP<=1%
topsis = X[idx]
cheap = X[int(np.argmin(F[:, 0]))]

designs = [('Proposed hybrid (reliability-constrained)', rec),
           ('Entropy-TOPSIS compromise', topsis),
           ('Cost-optimal corner', cheap)]
for k, lbl in (('awe', 'AWE-only'), ('pem', 'PEM-only'),
               ('fixed', 'Fixed 5.5 MW + 1.4 MW plant (v1 premise)')):
    p = f'checkpoints/bench_{k}.json'
    if os.path.exists(p):
        designs.append((lbl, np.array(json.load(open(p))['x'])))

rows = []
for lbl, x in designs:
    m = one(x, ps, pw, dh)
    rows.append(dict(Configuration=lbl, P_solar=x[0], P_wind=x[1],
                     P_AWE=x[2] * (1 - x[3]), P_PEM=x[2] * x[3], x_PEM=x[3],
                     E_BESS=x[4], M_HSS=x[5], LCOH=m['lcoh'], LPSP=m['lpsp'],
                     Curtailment=m['curtailment'], SI=m['si'],
                     BESS_cycles=m['bess_cycles'], H2_served=m['h2_served'],
                     H2_discarded=m['h2_discarded']))
B = pd.DataFrame(rows).round(4)
B.to_csv(os.path.join(C.TABLE_DIR, 'T03_benchmarks.csv'), index=False)
print(B[['Configuration', 'P_solar', 'P_wind', 'x_PEM', 'M_HSS', 'LCOH', 'LPSP',
         'Curtailment']].to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(9.8, 3.0))
lb = [r['Configuration'].replace(' (', '\n(').replace(' 5.5', '\n5.5') for r in rows]
for a, k, t in zip(ax, ('LCOH', 'LPSP', 'Curtailment'),
                   ('LCOH (USD/kg)', 'LPSP', 'Curtailment fraction')):
    v = [r[k] for r in rows]
    a.bar(range(len(v)), v, color='steelblue')
    a.set_xticks(range(len(v)))
    a.set_xticklabels(lb, fontsize=5, rotation=25, ha='right')
    a.set_ylabel(t)
save(fig, 'F06_benchmarks')

# ---- held-out 2023 ---------------------------------------------------------
dht, _ = D.demand_series(len(ts), seed=99)
sel = one(rec, ps, pw, dh)
tst = one(rec, ts, tw, dht)
T = pd.DataFrame([dict(Metric='LCOH (USD/kg)', Design_2020_2022=sel['lcoh'], Test_2023=tst['lcoh']),
                  dict(Metric='LPSP', Design_2020_2022=sel['lpsp'], Test_2023=tst['lpsp']),
                  dict(Metric='Curtailment', Design_2020_2022=sel['curtailment'], Test_2023=tst['curtailment']),
                  dict(Metric='HSS stability index', Design_2020_2022=sel['si'], Test_2023=tst['si'])]).round(6)
T.to_csv(os.path.join(C.TABLE_DIR, 'T04_holdout_2023.csv'), index=False)
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.4))
for a, k, t in zip(ax, ('lcoh', 'lpsp', 'curtailment'), ('LCOH (USD/kg)', 'LPSP', 'Curtailment')):
    a.bar([0, 1], [sel[k], tst[k]], color=['steelblue', 'indianred'])
    a.set_xticks([0, 1]); a.set_xticklabels(['2020-22\ndesign', '2023\nhold-out'], fontsize=7)
    a.set_ylabel(t)
save(fig, 'F07_holdout')

# ---- LCOH decomposition ----------------------------------------------------
dec = pd.DataFrame([dict(Component='Annualised CAPEX', USD_per_kg=sel['ann_capex']),
                    dict(Component='O&M', USD_per_kg=sel['ann_om']),
                    dict(Component='Replacement', USD_per_kg=sel['ann_repl']),
                    dict(Component='Cyclic ageing', USD_per_kg=sel['ann_cyc'])])
dec['Share_pct'] = 100 * dec.USD_per_kg / dec.USD_per_kg.sum()
dec.round(4).to_csv(os.path.join(C.TABLE_DIR, 'T05_lcoh_decomposition.csv'), index=False)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax[0].bar(dec.Component, dec.USD_per_kg, color=['steelblue', 'seagreen', 'darkorange', 'indianred'])
ax[0].set_ylabel('USD/kg'); ax[0].tick_params(axis='x', rotation=18, labelsize=7)
ax[1].pie(dec.USD_per_kg, labels=dec.Component, autopct='%1.1f%%', textprops={'fontsize': 6.5})
save(fig, 'F08_lcoh_decomposition')

# ---- economic sensitivity --------------------------------------------------
rw = []
for dr in (0.04, 0.06, 0.08):
    rw.append(dict(Case=f'Discount {dr:.0%}',
                   LCOH=float(evaluate_population(rec[None, :], ps, pw, dh, discount_rate=dr)['lcoh'][0])))
for g in (0.7, 1.0, 1.3):
    rw.append(dict(Case=f'Degradation x{g}',
                   LCOH=float(evaluate_population(rec[None, :], ps, pw, dh, deg_factor=g)['lcoh'][0])))
E = pd.DataFrame(rw).round(4)
E.to_csv(os.path.join(C.TABLE_DIR, 'T07_sensitivity.csv'), index=False)
fig, ax = plt.subplots(figsize=(5.0, 2.5))
ax.bar(E.Case, E.LCOH, color='slateblue'); ax.set_ylabel('LCOH (USD/kg)')
ax.tick_params(axis='x', rotation=22, labelsize=7)
save(fig, 'F10_sensitivity')

# ---- selected design table -------------------------------------------------
sd = pd.DataFrame([('Solar capacity', f"{rec[0]:.1f} kW"), ('Wind capacity', f"{rec[1]:.1f} kW"),
                   ('AWE capacity', f"{rec[2]*(1-rec[3]):.1f} kW"),
                   ('PEM capacity', f"{rec[2]*rec[3]:.1f} kW"),
                   ('PEM fraction', f"{rec[3]:.4f}"), ('BESS capacity', f"{rec[4]:.1f} kWh"),
                   ('HSS capacity', f"{rec[5]:.1f} kg"), ('LCOH', f"{sel['lcoh']:.4f} USD/kg"),
                   ('LPSP', f"{sel['lpsp']:.6f}"), ('Curtailment', f"{100*sel['curtailment']:.2f} %"),
                   ('HSS stability index', f"{sel['si']:.4f}"),
                   ('BESS cycles', f"{sel['bess_cycles']:.0f} /yr"),
                   ('H2 produced', f"{sel['h2_produced']:.0f} kg/yr"),
                   ('H2 served', f"{sel['h2_served']:.0f} kg/yr"),
                   ('H2 discarded', f"{sel['h2_discarded']:.4f} kg/yr"),
                   ('Mass-balance residual', f"{sel['mass_residual']:.2e} kg/yr"),
                   ('AWE lifetime', f"{sel['awe_life']:.1f} yr"),
                   ('PEM lifetime', f"{sel['pem_life']:.1f} yr"),
                   ('AWE operating hours', f"{sel['awe_hours']:.0f} h/yr"),
                   ('PEM operating hours', f"{sel['pem_hours']:.0f} h/yr")],
                  columns=['Quantity', 'Value'])
sd.to_csv(os.path.join(C.TABLE_DIR, 'T08_selected_design.csv'), index=False)

json.dump(dict(recommended=dict(zip(C.VAR_NAMES, rec.tolist())),
               topsis=dict(zip(C.VAR_NAMES, topsis.tolist())),
               metrics=sel, holdout=tst,
               entropy_weights=dict(zip(['LCOH', 'LPSP', 'CURT'], w.tolist())),
               hv_history=hist, n_pareto=int(len(F)),
               benchmarks=B.to_dict('records')),
          open('checkpoints/results_numbers.json', 'w'), indent=2)
print('\nfinalize done')
