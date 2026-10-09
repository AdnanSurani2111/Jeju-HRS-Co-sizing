"""Warm-up sensitivity for the locked selected design on validation seeds 500--519."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data_v2 as D
from objectives_v2 import evaluate_population


ROOT = Path(__file__).resolve().parent
DESIGN = np.array([[1846.7825617959973, 668.0003220745918,
                    1028.7109795926112, 0.4005987732169565,
                    55.24554888853066, 1855.8464836442977]])


def scalar(result, key):
    return float(np.asarray(result[key]).reshape(-1)[0])


def main():
    re = D.load_re()
    solar, wind = re['train']
    target_passes = {0: '0', 1: '1 (production)', 3: '3', 5: '5'}
    detail = []
    for seed in range(500, 520):
        demand, _ = D.demand_series(len(solar), seed=seed)
        soc = np.array([0.50])
        hss = np.array([0.50 * DESIGN[0, 5]])
        iterative_recorded = False
        for pass_number in range(51):
            result = evaluate_population(
                DESIGN, solar, wind, demand, warmup_passes=0,
                initial_soc=soc, initial_hss=hss)
            record = {
                    'seed': seed,
                    'passes_used': pass_number,
                    'lcoh_usd_per_kg': scalar(result, 'lcoh'),
                    'lpsp_fraction': scalar(result, 'lpsp'),
                    'curtailment_fraction': scalar(result, 'curtailment'),
                    'bess_start_soc': scalar(result, 'soc_start'),
                    'bess_end_soc': scalar(result, 'soc_end'),
                    'bess_terminal_mismatch': scalar(result, 'soc_imbalance'),
                    'hss_start_kg': scalar(result, 'hss_start'),
                    'hss_end_kg': scalar(result, 'hss_end'),
                    'hss_terminal_mismatch_fraction': scalar(result, 'hss_imbalance'),
                }
            if pass_number in target_passes:
                detail.append({'case': target_passes[pass_number], **record})
            closure = max(record['bess_terminal_mismatch'],
                          record['hss_terminal_mismatch_fraction'])
            if closure <= 1e-10 and not iterative_recorded:
                detail.append({'case': 'iterative', **record})
                iterative_recorded = True
            if pass_number >= 5 and iterative_recorded:
                break
            soc = np.array([record['bess_end_soc']])
            hss = np.array([record['hss_end_kg']])
        if not iterative_recorded:
            detail.append({'case': 'iterative (not converged)', **record})
    df = pd.DataFrame(detail)
    df.to_csv(ROOT / 'warmup_sensitivity_draws.csv', index=False)

    rows = []
    cases = ['0', '1 (production)', '3', '5', 'iterative']
    for label in cases:
        g = df[df['case'] == label]
        if g.empty:
            continue
        rows.append({
            'case': label,
            'passes_used_min': int(g.passes_used.min()),
            'passes_used_max': int(g.passes_used.max()),
            'mean_lcoh_usd_per_kg': g.lcoh_usd_per_kg.mean(),
            'mean_lpsp_percent': 100 * g.lpsp_fraction.mean(),
            'q95_lpsp_percent': 100 * g.lpsp_fraction.quantile(0.95),
            'mean_curtailment_percent': 100 * g.curtailment_fraction.mean(),
            'max_bess_terminal_mismatch': g.bess_terminal_mismatch.max(),
            'max_hss_terminal_mismatch_percent':
                100 * g.hss_terminal_mismatch_fraction.max(),
            'mean_bess_start_soc': g.bess_start_soc.mean(),
            'mean_bess_end_soc': g.bess_end_soc.mean(),
            'mean_hss_start_kg': g.hss_start_kg.mean(),
            'mean_hss_end_kg': g.hss_end_kg.mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / 'warmup_sensitivity_summary.csv', index=False)
    print(summary.to_string(index=False))

    meta = {
        'design': DESIGN[0].tolist(),
        'validation_seeds': [500, 519],
        'chronology_hours': len(solar),
        'production_warmup_passes': 1,
        'iterative_tolerance': 1e-10,
        'iterative_max_passes': 50,
        'note': ('Diagnostic only; it does not replace explicit terminal-state '
                 'constraints in optimization.'),
    }
    (ROOT / 'warmup_sensitivity_meta.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
