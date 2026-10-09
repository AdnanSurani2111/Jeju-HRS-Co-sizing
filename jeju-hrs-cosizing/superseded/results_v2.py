"""results_v2.py - benchmarks, held-out year, sensitivities, tables and figures.

Every benchmark is evaluated with the same evaluator as the proposed method.
No numerical value in any table is taken from another publication.
"""
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
FIG = C.FIG_DIR
TAB = C.TABLE_DIR


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'))
    plt.close(fig)
    print('  figure:', name)


def one(x, ps, pw, dh):
    r = evaluate_population(np.asarray(x, float)[None, :], ps, pw, dh)
    return {k: float(np.asarray(v).ravel()[0]) for k, v in r.items()
            if np.asarray(v).size == 1}


def main(stage='a'):
    re = D.load_re()
    ps, pw = re['train']
    ts, tw = re['test']
    DL = [D.demand_series(len(ps), seed=s)[0] for s in range(C.N_DEMAND_REALISATIONS)]
    dh = DL[0]
    F = np.load('checkpoints/F_gen.npy')
    X = np.load('checkpoints/X_gen.npy')
    hv = np.array([h['hv'] for h in json.load(open('checkpoints/hv_hist.json'))])
    hvg = np.array([h['gen'] for h in json.load(open('checkpoints/hv_hist.json'))])
    idx, w = O.entropy_topsis(F)
    best = X[idx]
    sel = one(best, ps, pw, dh)
    print('selected design:', dict(zip(C.VAR_NAMES, np.round(best, 2))))

    # ---------------- F01 renewable resource -------------------------------
    stats = D.re_statistics()
    stats.to_csv(os.path.join(TAB, 'T01_re_statistics.csv'), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    b = np.arange(4)
    ax[0].bar(b - 0.2, stats.solar_cf, 0.4, label='Solar')
    ax[0].bar(b + 0.2, stats.wind_cf, 0.4, label='Wind')
    ax[0].set_xticks(b); ax[0].set_xticklabels(stats.Year)
    ax[0].set_ylabel('Capacity factor (%)'); ax[0].legend()
    ax[0].set_title('(a) Capacity factors, year-specific fleet basis')
    h = np.arange(24)
    s24 = ps.reshape(-1, 24).mean(0) * C.REF_SOLAR_KW
    w24 = pw.reshape(-1, 24).mean(0) * C.REF_WIND_KW
    ax[1].plot(h, s24, label='Solar'); ax[1].plot(h, w24, label='Wind')
    ax[1].plot(h, s24 + w24, label='Combined')
    ax[1].set_xlabel('Hour'); ax[1].set_ylabel('Mean power (kW)')
    ax[1].set_title('(b) Mean hourly profile, 2020-2022'); ax[1].legend()
    save(fig, 'F01_resource')

    # ---------------- F02 measured demand ----------------------------------
    d = D.load_demand_records()
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 2.6))
    ax[0].plot(d.date, d.kg, lw=0.8)
    ax[0].axhline(d.kg.mean(), color='r', ls='--', label=f'mean {d.kg.mean():.0f} kg/d')
    ax[0].set_ylabel('H$_2$ dispensed (kg/d)'); ax[0].legend(fontsize=7)
    ax[0].tick_params(axis='x', rotation=30); ax[0].set_title('(a) Measured daily record')
    ax[1].hist(d.kg, bins=30, color='steelblue')
    ax[1].set_xlabel('kg/day'); ax[1].set_ylabel('Days'); ax[1].set_title('(b) Distribution')
    m = d.groupby(d.date.dt.to_period('M')).kg.mean()
    ax[2].plot(range(len(m)), m.values, 'o-')
    ax[2].set_xticks(range(len(m))); ax[2].set_xticklabels([str(x)[2:] for x in m.index],
                                                          rotation=45, fontsize=6)
    ax[2].set_ylabel('Monthly mean (kg/d)'); ax[2].set_title('(c) Throughput growth')
    save(fig, 'F02_demand')
    pd.DataFrame([dict(records=len(d), start=str(d.date.min().date()),
                       end=str(d.date.max().date()), mean=d.kg.mean(), std=d.kg.std(),
                       cv_pct=100 * d.kg.std() / d.kg.mean(), p05=d.kg.quantile(.05),
                       p95=d.kg.quantile(.95), min=d.kg.min(), max=d.kg.max(),
                       lag1_autocorr=np.corrcoef(d.kg[:-1], d.kg[1:])[0, 1])]).to_csv(
        os.path.join(TAB, 'T02_demand_statistics.csv'), index=False)

    # ---------------- F03 Pareto -------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    sc = ax[0].scatter(F[:, 0], F[:, 2] * 100, c=F[:, 1] * 100, cmap='viridis', s=14)
    ax[0].scatter(F[idx, 0], F[idx, 2] * 100, marker='*', s=260, c='gold',
                  edgecolor='k', zorder=5, label='TOPSIS')
    plt.colorbar(sc, ax=ax[0], label='LPSP (%)')
    ax[0].set_xlabel('LCOH (USD/kg)'); ax[0].set_ylabel('Curtailment (%)')
    ax[0].legend(fontsize=7); ax[0].set_title('(a) Cost-curtailment projection')
    ax[1].scatter(F[:, 0], F[:, 1] * 100, s=14, c='steelblue')
    ax[1].scatter(F[idx, 0], F[idx, 1] * 100, marker='*', s=260, c='gold', edgecolor='k', zorder=5)
    ax[1].set_xlabel('LCOH (USD/kg)'); ax[1].set_ylabel('LPSP (%)')
    ax[1].set_title('(b) Cost-reliability projection')
    save(fig, 'F03_pareto')

    # ---------------- F04 capacity decisions --------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(9.4, 4.6))
    for a, j, nm in zip(ax.ravel(), range(6), C.VAR_NAMES):
        a.scatter(F[:, 0], X[:, j], s=10, c='teal')
        a.scatter(F[idx, 0], X[idx, j], marker='*', s=200, c='gold', edgecolor='k', zorder=5)
        a.set_xlabel('LCOH (USD/kg)'); a.set_ylabel(nm)
    fig.suptitle('Design variables across the Pareto set', y=1.01)
    save(fig, 'F04_capacity_map')

    # ---------------- F05 hypervolume (cumulative archive) ------------------
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.plot(hvg, hv, 'o-', lw=1.4)
    ax.set_xlabel('Generation'); ax.set_ylabel('Hypervolume (cumulative archive)')
    ax.set_title(f'Monotone by construction; final {hv[-1]:.4f}')
    save(fig, 'F05_hypervolume')

    if stage == 'a':
        json.dump(dict(selected=dict(zip(C.VAR_NAMES, best.tolist())),
                       entropy_weights=dict(zip(['LCOH','LPSP','CURT'], w.tolist())),
                       metrics={k: sel[k] for k in sel}), open('checkpoints/stage_a.json','w'), indent=2)
        print('stage a done'); return

    # ---------------- benchmarks -------------------------------------------
    def opt_fixed(mask_fn, label, pop=40, gen=18, seed=7):
        """Re-optimise with a restricted design space, same evaluator."""
        lo, hi = np.array(C.VAR_LOWER, float), np.array(C.VAR_UPPER, float)
        lo, hi = mask_fn(lo, hi)
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.optimize import minimize
        from objectives_v2 import objectives as OBJ

        class P(Problem):
            def __init__(s): super().__init__(n_var=6, n_obj=3, xl=lo, xu=hi)
            def _evaluate(s, XX, out, *a, **k): out['F'] = OBJ(XX, ps, pw, DL)
        r = minimize(P(), NSGA3(pop_size=pop,
                                ref_dirs=get_reference_directions('das-dennis', 3, n_partitions=8)),
                     ('n_gen', gen), seed=seed, verbose=False)
        i, _ = O.entropy_topsis(r.F)
        return r.X[i], label

    bench = [('NSGA-III hybrid (proposed)', best)]
    for fn, lbl in [((lambda l, h: (l.copy(), np.array([h[0], h[1], h[2], 1e-6, h[4], h[5]]))), 'AWE-only'),
                    ((lambda l, h: (np.array([l[0], l[1], l[2], 1 - 1e-6, l[4], l[5]]), h.copy())), 'PEM-only'),
                    ((lambda l, h: (np.array([5500., 1400., l[2], l[3], l[4], l[5]]),
                                    np.array([5500., 1400., h[2], h[3], h[4], h[5]]))),
                     'Fixed 5.5/1.4 MW plant')]:
        xx, _ = opt_fixed(fn, lbl)
        bench.append((lbl, xx))

    # cost-optimal corner of the same Pareto front (what a cost-only objective selects)
    bench.append(('Cost-only corner', X[int(np.argmin(F[:, 0]))]))
    # reliability-constrained selection: cheapest design meeting LPSP <= 1%
    ok = np.where(F[:, 1] <= 0.01)[0]
    if len(ok):
        bench.append(('Reliability-constrained (LPSP<=1%)', X[ok[np.argmin(F[ok, 0])]]))

    B = []
    for lbl, xx in bench:
        m = one(xx, ps, pw, dh)
        B.append(dict(Configuration=lbl, P_solar_kW=xx[0], P_wind_kW=xx[1],
                      P_AWE_kW=xx[2] * (1 - xx[3]), P_PEM_kW=xx[2] * xx[3], x_PEM=xx[3],
                      E_BESS_kWh=xx[4], M_HSS_kg=xx[5], LCOH=m['lcoh'],
                      LPSP=m['lpsp'], Curtailment=m['curtailment'], SI=m['si'],
                      BESS_cycles=m['bess_cycles']))
    Bd = pd.DataFrame(B).round(4)
    Bd.to_csv(os.path.join(TAB, 'T03_benchmarks.csv'), index=False)
    print(Bd.to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(9.4, 2.8))
    lbl = [b['Configuration'].replace(' ', '\n') for b in B]
    for a, key, t in zip(ax, ('LCOH', 'LPSP', 'Curtailment'),
                         ('LCOH (USD/kg)', 'LPSP', 'Curtailment')):
        v = [b[key] for b in B]
        a.bar(range(len(v)), v, color='steelblue')
        a.set_xticks(range(len(v))); a.set_xticklabels(lbl, fontsize=5.5)
        a.set_ylabel(t)
    save(fig, 'F06_benchmarks')

    if stage == 'b':
        print('stage b done'); return

    # ---------------- held-out 2023 -----------------------------------------
    dh_t, _ = D.demand_series(len(ts), seed=99)
    tst = one(best, ts, tw, dh_t)
    T = pd.DataFrame([dict(Metric='LCOH (USD/kg)', Design=sel['lcoh'], Test=tst['lcoh']),
                      dict(Metric='LPSP', Design=sel['lpsp'], Test=tst['lpsp']),
                      dict(Metric='Curtailment', Design=sel['curtailment'], Test=tst['curtailment']),
                      dict(Metric='HSS stability index', Design=sel['si'], Test=tst['si'])]).round(6)
    T.to_csv(os.path.join(TAB, 'T04_holdout_2023.csv'), index=False)
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.4))
    for a, k, t in zip(ax, ('lcoh', 'lpsp', 'curtailment'), ('LCOH (USD/kg)', 'LPSP', 'Curtailment')):
        a.bar([0, 1], [sel[k], tst[k]], color=['steelblue', 'indianred'])
        a.set_xticks([0, 1]); a.set_xticklabels(['2020-22\ndesign', '2023\nhold-out'], fontsize=7)
        a.set_ylabel(t)
    save(fig, 'F07_holdout')

    # ---------------- LCOH decomposition ------------------------------------
    dec = pd.DataFrame([dict(Component='Annualised CAPEX', USD_per_kg=sel['ann_capex']),
                        dict(Component='O&M', USD_per_kg=sel['ann_om']),
                        dict(Component='Replacement', USD_per_kg=sel['ann_repl']),
                        dict(Component='Cyclic ageing', USD_per_kg=sel['ann_cyc'])])
    dec['Share_pct'] = 100 * dec.USD_per_kg / dec.USD_per_kg.sum()
    dec.round(4).to_csv(os.path.join(TAB, 'T05_lcoh_decomposition.csv'), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax[0].bar(dec.Component, dec.USD_per_kg, color=['steelblue', 'seagreen', 'darkorange', 'indianred'])
    ax[0].set_ylabel('USD/kg'); ax[0].tick_params(axis='x', rotation=20, labelsize=7)
    ax[1].pie(dec.USD_per_kg, labels=dec.Component, autopct='%1.1f%%', textprops={'fontsize': 6.5})
    save(fig, 'F08_lcoh_decomposition')

    if stage == 'c':
        print('stage c done'); return

    # ---------------- demand sensitivity ------------------------------------
    sw = []
    for sc in (0.5, 0.75, 1.0, 1.5, 2.0):
        dsc = [D.demand_series(len(ps), seed=s, scale=sc)[0] for s in range(2)]
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.optimize import minimize
        from objectives_v2 import objectives as OBJ

        class P(Problem):
            def __init__(s): super().__init__(n_var=6, n_obj=3,
                                              xl=np.array(C.VAR_LOWER), xu=np.array(C.VAR_UPPER))
            def _evaluate(s, XX, out, *a, **k): out['F'] = OBJ(XX, ps, pw, dsc)
        r = minimize(P(), NSGA3(pop_size=60,
                                ref_dirs=get_reference_directions('das-dennis', 3, n_partitions=8)),
                     ('n_gen', 40), seed=11, verbose=False)
        i, _ = O.entropy_topsis(r.F)
        xx = r.X[i]
        sw.append(dict(demand_kg_day=181.3 * sc, LCOH=r.F[i, 0], LPSP=r.F[i, 1],
                       CURT=r.F[i, 2], x_PEM=xx[3], P_elz=xx[2],
                       P_solar=xx[0], P_wind=xx[1], M_HSS=xx[5]))
        print('  demand sweep', round(181.3 * sc, 1), 'x_PEM', round(xx[3], 3))
    S = pd.DataFrame(sw).round(4)
    S.to_csv(os.path.join(TAB, 'T06_demand_sensitivity.csv'), index=False)
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 2.6))
    ax[0].plot(S.demand_kg_day, S.LCOH, 'o-'); ax[0].set_ylabel('LCOH (USD/kg)')
    ax[1].plot(S.demand_kg_day, S.x_PEM, 'o-', color='seagreen'); ax[1].set_ylabel('$x_{PEM}$')
    ax[2].plot(S.demand_kg_day, S.P_solar, 'o-', label='Solar')
    ax[2].plot(S.demand_kg_day, S.P_wind, 's-', label='Wind')
    ax[2].plot(S.demand_kg_day, S.P_elz, '^-', label='Electrolyser')
    ax[2].set_ylabel('Capacity (kW)'); ax[2].legend(fontsize=7)
    for a in ax: a.set_xlabel('H$_2$ demand (kg/day)')
    save(fig, 'F09_demand_sensitivity')

    # ---------------- econ sensitivity --------------------------------------
    rowsx = []
    for dr in (0.04, 0.06, 0.08):
        r = evaluate_population(best[None, :], ps, pw, dh, discount_rate=dr)
        rowsx.append(dict(Case=f'Discount {dr:.0%}', LCOH=float(r['lcoh'][0])))
    for df_ in (0.7, 1.0, 1.3):
        r = evaluate_population(best[None, :], ps, pw, dh, deg_factor=df_)
        rowsx.append(dict(Case=f'Degradation x{df_}', LCOH=float(r['lcoh'][0])))
    E = pd.DataFrame(rowsx).round(4)
    E.to_csv(os.path.join(TAB, 'T07_sensitivity.csv'), index=False)
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.bar(E.Case, E.LCOH, color='slateblue'); ax.set_ylabel('LCOH (USD/kg)')
    ax.tick_params(axis='x', rotation=25, labelsize=7)
    save(fig, 'F10_sensitivity')

    # ---------------- selected design table ---------------------------------
    sd = pd.DataFrame([
        dict(Quantity='Solar capacity', Value=f"{best[0]:.1f} kW"),
        dict(Quantity='Wind capacity', Value=f"{best[1]:.1f} kW"),
        dict(Quantity='AWE capacity', Value=f"{best[2]*(1-best[3]):.1f} kW"),
        dict(Quantity='PEM capacity', Value=f"{best[2]*best[3]:.1f} kW"),
        dict(Quantity='PEM fraction', Value=f"{best[3]:.4f}"),
        dict(Quantity='BESS capacity', Value=f"{best[4]:.1f} kWh"),
        dict(Quantity='HSS capacity', Value=f"{best[5]:.1f} kg"),
        dict(Quantity='LCOH', Value=f"{sel['lcoh']:.4f} USD/kg"),
        dict(Quantity='LPSP', Value=f"{sel['lpsp']:.6f}"),
        dict(Quantity='Curtailment', Value=f"{100*sel['curtailment']:.2f} %"),
        dict(Quantity='HSS stability index', Value=f"{sel['si']:.4f}"),
        dict(Quantity='BESS cycles', Value=f"{sel['bess_cycles']:.0f} /yr"),
        dict(Quantity='H2 produced', Value=f"{sel['h2_produced']:.0f} kg/yr"),
        dict(Quantity='H2 served', Value=f"{sel['h2_served']:.0f} kg/yr"),
        dict(Quantity='H2 discarded', Value=f"{sel['h2_discarded']:.3f} kg/yr"),
        dict(Quantity='AWE life', Value=f"{sel['awe_life']:.1f} yr"),
        dict(Quantity='PEM life', Value=f"{sel['pem_life']:.1f} yr"),
    ])
    sd.to_csv(os.path.join(TAB, 'T08_selected_design.csv'), index=False)

    json.dump(dict(selected=dict(zip(C.VAR_NAMES, best.tolist())),
                   metrics={k: sel[k] for k in
                            ('lcoh', 'lpsp', 'curtailment', 'si', 'bess_cycles',
                             'h2_produced', 'h2_served', 'h2_discarded',
                             'ann_capex', 'ann_om', 'ann_repl', 'ann_cyc',
                             'awe_life', 'pem_life', 'awe_hours', 'pem_hours',
                             'mass_residual', 'hss_imbalance')},
                   entropy_weights=dict(zip(['LCOH', 'LPSP', 'CURT'], w.tolist())),
                   holdout={k: tst[k] for k in ('lcoh', 'lpsp', 'curtailment', 'si')},
                   hv_final=float(hv[-1]), n_pareto=int(len(F))),
              open('checkpoints/results_numbers.json', 'w'), indent=2)
    print('\nDONE. tables and figures written.')


import sys
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'a')
