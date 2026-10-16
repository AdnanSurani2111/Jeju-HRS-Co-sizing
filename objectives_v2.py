"""
objectives_v2.py - LCOH / LPSP / curtailment for the corrected model.

LCOH numerator now includes the renewable plant the optimiser actually chooses,
so oversizing generation is paid for rather than assumed free.
"""
import numpy as np
import config as C
from dispatch_v2 import simulate


def crf(r=None, n=None):
    r = C.DISCOUNT_RATE if r is None else float(r)
    n = float(C.PROJECT_LIFE_YR) if n is None else float(n)
    if r < 1e-9:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1.0)


def pv_replacement(capex, life, r=None, horizon=None):
    r = C.DISCOUNT_RATE if r is None else float(r)
    horizon = float(C.PROJECT_LIFE_YR) if horizon is None else float(horizon)
    pv = np.zeros_like(np.asarray(capex, dtype=float))
    life = np.maximum(np.asarray(life, dtype=float), 0.5)
    kmax = int(np.ceil(horizon / max(float(np.min(life)), 0.5)))
    for k in range(1, kmax + 2):
        t = k * life
        pv = pv + np.where(t < horizon, capex / (1.0 + r) ** t, 0.0)
    return pv * crf(r, horizon)


def evaluate_population(X, pu_solar, pu_wind, demand_h,
                        deg_factor=1.0, cyclic_factor=1.0, discount_rate=None):
    """X : (N,6) -> dict of (N,) results."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    P_sol, P_win, P_elz, x_pem, E_b, M_h = (X[:, i] for i in range(6))
    r = C.DISCOUNT_RATE if discount_rate is None else float(discount_rate)

    sim = simulate(P_sol, P_win, P_elz, x_pem, E_b, M_h,
                   pu_solar, pu_wind, demand_h, deg_factor=deg_factor)

    P_awe = P_elz * (1.0 - x_pem)
    P_pem = P_elz * x_pem
    cx_sol = C.SOLAR_CAPEX_PER_KW * P_sol
    cx_win = C.WIND_CAPEX_PER_KW * P_win
    cx_awe = C.AWE_CAPEX_PER_KW * P_awe
    cx_pem = C.PEM_CAPEX_PER_KW * P_pem
    cx_bes = C.BESS_CAPEX_PER_KWH * E_b
    cx_hss = C.HSS_CAPEX_PER_KG * M_h
    cx_re = cx_sol + cx_win
    cx_tot = cx_re + cx_awe + cx_pem + cx_bes + cx_hss

    ann_capex = cx_tot * crf(r)
    ann_om = (cx_awe * C.AWE_OM_FRAC + cx_pem * C.PEM_OM_FRAC +
              cx_bes * C.BESS_OM_FRAC + cx_hss * C.HSS_OM_FRAC +
              cx_re * C.RE_OM_FRAC)

    awe_life = np.clip(C.AWE_REPLACE_THRESHOLD_UV / np.maximum(sim['awe_deg'], 1e-6),
                       0.5, C.PROJECT_LIFE_YR + 0.1)
    pem_life = np.clip(C.PEM_REPLACE_THRESHOLD_UV / np.maximum(sim['pem_deg'], 1e-6),
                       0.5, C.PROJECT_LIFE_YR + 0.1)
    ann_repl = (pv_replacement(cx_bes, np.full_like(cx_bes, float(C.BESS_LIFE_YR)), r) +
                pv_replacement(cx_awe, awe_life, r) +
                pv_replacement(cx_pem, pem_life, r))
    ann_cyc = cyclic_factor * (sim['awe_cycles'] * C.AWE_CYCLE_COST +
                               sim['pem_cycles'] * C.PEM_CYCLE_COST)

    h2 = np.maximum(sim['h2_served'], 1.0)
    lcoh = (ann_capex + ann_om + ann_repl + ann_cyc) / h2

    out = dict(sim)
    out.update(lcoh=lcoh, ann_capex=ann_capex / h2, ann_om=ann_om / h2,
               ann_repl=ann_repl / h2, ann_cyc=ann_cyc / h2,
               capex_total=cx_tot, capex_re=cx_re, capex_awe=cx_awe,
               capex_pem=cx_pem, capex_bess=cx_bes, capex_hss=cx_hss,
               awe_life=awe_life, pem_life=pem_life,
               P_awe=P_awe, P_pem=P_pem)
    return out


def objectives(X, pu_solar, pu_wind, demand_list, **kw):
    """Average the three objectives over the supplied demand realisations."""
    F = None
    for dh in demand_list:
        r = evaluate_population(X, pu_solar, pu_wind, dh, **kw)
        f = np.column_stack([r['lcoh'], r['lpsp'], r['curtailment']])
        F = f if F is None else F + f
    return F / len(demand_list)
