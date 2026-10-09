"""figures_final.py - rebuild every figure with consistent publication styling."""
import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config as C
import data_v2 as D
import optimizer_v2 as O

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.3,
})
BLUE, GREEN, RED, ORANGE, PURPLE, GREY = '#2E5E8A', '#2E8B6B', '#B03A3A', '#D98032', '#6B4E8A', '#7A7A7A'


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(C.FIG_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print('  ', name)


def annot(ax, xs, ys, fmt='{:.2f}', dy=0.01, size=6):
    span = max(ys) - min(0, min(ys))
    for x, y in zip(xs, ys):
        ax.text(x, y + span * dy, fmt.format(y), ha='center', va='bottom', fontsize=size)


re = D.load_re()
ps, pw = re['train']
ts, tw = re['test']
F = np.load('checkpoints/F_gen.npy')
X = np.load('checkpoints/X_gen.npy')
ti, W = O.entropy_topsis(F)
ok = np.where(F[:, 1] <= 0.01)[0]
ri = ok[np.argmin(F[ok, 0])]
RES = json.load(open('checkpoints/results_numbers.json'))

# ------------------------------------------------ F01 resource
stats = D.re_statistics()
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.15))
b = np.arange(4)
ax[0].bar(b - 0.2, stats.solar_cf, 0.4, label='Solar', color=ORANGE)
ax[0].bar(b + 0.2, stats.wind_cf, 0.4, label='Wind', color=BLUE)
ax[0].set_xticks(b); ax[0].set_xticklabels(stats.Year.astype(str))
ax[0].set_ylabel('Capacity factor (%)'); ax[0].legend(frameon=False, loc='upper left')
ax[0].set_title('(a) Estimated fleet normalized CF'); ax[0].set_ylim(0, 34)
h = np.arange(24)
s24 = ps.reshape(-1, 24).mean(0) * C.REF_SOLAR_KW
w24 = pw.reshape(-1, 24).mean(0) * C.REF_WIND_KW
ax[1].plot(h, s24, color=ORANGE, label='Solar')
ax[1].plot(h, w24, color=BLUE, label='Wind')
ax[1].plot(h, s24 + w24, color=GREY, ls='--', label='Combined')
ax[1].set_xlabel('Hour of day'); ax[1].set_ylabel('Mean power (kW)')
ax[1].set_xticks([0, 6, 12, 18, 23]); ax[1].legend(frameon=False)
ax[1].set_title('(b) Mean diurnal profile, 2020 to 2022')
tot = (ps * C.REF_SOLAR_KW + pw * C.REF_WIND_KW)
dur = np.sort(tot)[::-1]
ax[2].plot(100 * np.arange(len(dur)) / len(dur), dur, color=GREEN)
ax[2].set_xlabel('Fraction of hours exceeded (%)'); ax[2].set_ylabel('Combined power (kW)')
ax[2].set_title('(c) Load duration curve of resource')
save(fig, 'F01_resource')

# ------------------------------------------------ F02 demand
d = D.load_demand_records()
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.15))
ax[0].plot(d.date, d.kg, lw=0.7, color=BLUE)
ax[0].axhline(d.kg.mean(), color=RED, ls='--', lw=1.0, label=f'mean {d.kg.mean():.0f} kg/d')
ax[0].set_ylabel('H$_2$ dispensed (kg/d)'); ax[0].legend(frameon=False)
ax[0].tick_params(axis='x', rotation=32); ax[0].set_title('(a) Measured daily record')
for _t in ax[0].get_xticklabels():
    _t.set_ha('right')
    _t.set_fontsize(6)
ax[1].hist(d.kg, bins=28, color=BLUE, edgecolor='white', linewidth=0.4)
ax[1].axvline(d.kg.mean(), color=RED, ls='--', lw=1.0)
ax[1].set_xlabel('Daily throughput (kg/d)'); ax[1].set_ylabel('Number of days')
ax[1].set_title(f'(b) Distribution, CV = {100*d.kg.std()/d.kg.mean():.1f}%')
m = d.groupby(d.date.dt.to_period('M')).kg.mean()
ax[2].plot(range(len(m)), m.values, 'o-', color=GREEN, ms=3.5)
ax[2].set_xticks(range(len(m)))
ax[2].set_xticklabels([str(x)[2:] for x in m.index], rotation=45, fontsize=6, ha='right')
ax[2].set_ylabel('Monthly mean (kg/d)'); ax[2].set_title('(c) Throughput growth during ramp up')
save(fig, 'F02_demand')

# ------------------------------------------------ F03 Pareto
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.2))
sc = ax[0].scatter(F[:, 0], F[:, 2] * 100, c=F[:, 1] * 100, cmap='viridis', s=12, lw=0)
ax[0].scatter(F[ri, 0], F[ri, 2] * 100, marker='*', s=170, c='gold', edgecolor='k', lw=0.6, zorder=6)
ax[0].scatter(F[ti, 0], F[ti, 2] * 100, marker='D', s=34, c=RED, edgecolor='k', lw=0.5, zorder=5)
cb = plt.colorbar(sc, ax=ax[0], pad=0.02); cb.set_label('LPSP (%)', fontsize=7); cb.ax.tick_params(labelsize=6)
ax[0].set_xlabel('LCOH (USD/kg)'); ax[0].set_ylabel('Curtailment (%)')
ax[0].set_title('(a) Cost versus curtailment')
ax[1].scatter(F[:, 0], F[:, 1] * 100, s=12, c=BLUE, lw=0)
ax[1].scatter(F[ri, 0], F[ri, 1] * 100, marker='*', s=170, c='gold', edgecolor='k', lw=0.6, zorder=6,
              label='Reliability constrained')
ax[1].scatter(F[ti, 0], F[ti, 1] * 100, marker='D', s=34, c=RED, edgecolor='k', lw=0.5, zorder=5,
              label='TOPSIS compromise')
ax[1].axhline(1.0, color=GREY, ls=':', lw=0.9)
ax[1].text(F[:, 0].max(), 1.4, r'$\epsilon = 1\%$', ha='right', fontsize=6, color=GREY)
ax[1].set_xlabel('LCOH (USD/kg)'); ax[1].set_ylabel('LPSP (%)')
ax[1].legend(frameon=False, loc='upper right'); ax[1].set_title('(b) Cost versus supply loss')
ax[2].scatter(F[:, 2] * 100, F[:, 1] * 100, s=12, c=GREEN, lw=0)
ax[2].scatter(F[ri, 2] * 100, F[ri, 1] * 100, marker='*', s=170, c='gold', edgecolor='k', lw=0.6, zorder=6)
ax[2].scatter(F[ti, 2] * 100, F[ti, 1] * 100, marker='D', s=34, c=RED, edgecolor='k', lw=0.5, zorder=5)
ax[2].set_xlabel('Curtailment (%)'); ax[2].set_ylabel('LPSP (%)')
ax[2].set_title('(c) Curtailment versus supply loss')
save(fig, 'F03_pareto')

# ------------------------------------------------ F04 capacity map
UNITS = ['kW', 'kW', 'kW', '', 'kWh', 'kg']
fig, ax = plt.subplots(2, 3, figsize=(7.1, 3.6))
for a, j, nm, u in zip(ax.ravel(), range(6), C.VAR_NAMES, UNITS):
    a.scatter(F[:, 0], X[:, j], s=9, c=GREEN, lw=0, alpha=0.75)
    a.scatter(F[ri, 0], X[ri, j], marker='*', s=150, c='gold', edgecolor='k', lw=0.6, zorder=6)
    a.scatter(F[ti, 0], X[ti, j], marker='D', s=30, c=RED, edgecolor='k', lw=0.5, zorder=5)
    a.set_ylabel(f'{nm}' + (f' ({u})' if u else ''))
for a in ax[1]:
    a.set_xlabel('LCOH (USD/kg)')
fig.tight_layout(h_pad=0.9, w_pad=1.2)
save(fig, 'F04_capacity_map')

# ------------------------------------------------ F05 hypervolume
hist = json.load(open('checkpoints/hv_hist.json'))
g = [h['gen'] for h in hist]; hv = [h['hv'] for h in hist]; nf = [h['n_front'] for h in hist]
fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.05))
ax[0].plot(g, hv, 'o-', color=BLUE, ms=4)
ax[0].set_xlabel('Generation'); ax[0].set_ylabel('Hypervolume')
ax[0].set_title('(a) Cumulative archive hypervolume')
for gi, hvi in zip(g, hv):
    ax[0].annotate(f'{hvi:.3f}', (gi, hvi), textcoords='offset points', xytext=(0, 5),
                   ha='center', fontsize=5.5)
ax[1].plot(g, nf, 's-', color=GREEN, ms=4)
ax[1].set_xlabel('Generation'); ax[1].set_ylabel('Archive size')
ax[1].set_title('(b) Nondominated archive growth')
save(fig, 'F05_hypervolume')

# ------------------------------------------------ F06 benchmarks
B = pd.read_csv(os.path.join(C.TABLE_DIR, 'T03_benchmarks.csv'))
SHORT = {'Proposed hybrid (reliability-constrained)': 'Proposed hybrid',
         'Entropy-TOPSIS compromise': 'TOPSIS compromise',
         'Cost-optimal corner': 'Cost optimal corner',
         'AWE-only': 'AWE only', 'PEM-only': 'PEM only',
         'Fixed 5.5 MW + 1.4 MW plant (v1 premise)': 'Fixed plant 5.5/1.4 MW'}
lb = [SHORT.get(c, c) for c in B.Configuration]
xx = np.arange(len(lb))
cols = [RED if 'Proposed' in c else BLUE for c in B.Configuration]
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.9))
for a, k, t, f in zip(ax, ('LCOH', 'LPSP', 'Curtailment'),
                      ('LCOH (USD/kg)', 'LPSP (%)', 'Curtailment (%)'),
                      (1, 100, 100)):
    v = (B[k] * f).values
    a.bar(xx, v, color=cols, width=0.65)
    annot(a, xx, v, '{:.2f}' if k == 'LCOH' else '{:.1f}')
    a.set_xticks(xx); a.set_xticklabels(lb, fontsize=6.2, rotation=38, ha='right')
    a.set_ylabel(t); a.set_ylim(0, max(v) * 1.22 if max(v) > 0 else 1)
fig.tight_layout(w_pad=1.4)
save(fig, 'F06_benchmarks')

# ------------------------------------------------ F07 holdout
T = pd.read_csv(os.path.join(C.TABLE_DIR, 'T04_holdout_2023.csv'))
fig, ax = plt.subplots(1, 3, figsize=(6.0, 2.05))
specs = [('LCOH (USD/kg)', 0, 1.0, '{:.2f}'), ('LPSP (%)', 1, 100, '{:.3f}'),
         ('Curtailment (%)', 2, 100, '{:.2f}')]
for a, (t, i, f, fm) in zip(ax, specs):
    v = [T.Design_2020_2022[i] * f, T.Test_2023[i] * f]
    a.bar([0, 1], v, color=[BLUE, ORANGE], width=0.55)
    annot(a, [0, 1], v, fm)
    a.set_xticks([0, 1]); a.set_xticklabels(['Design\n2020 to 2022', 'Hold out\n2023'], fontsize=6.5)
    a.set_ylabel(t); a.set_ylim(0, max(v) * 1.25 if max(v) > 0 else 1)
fig.tight_layout(w_pad=1.3)
save(fig, 'F07_holdout')

# ------------------------------------------------ F08 LCOH decomposition
dec = pd.read_csv(os.path.join(C.TABLE_DIR, 'T05_lcoh_decomposition.csv'))
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.2))
cc = [BLUE, GREEN, ORANGE, RED]
ax[0].bar(range(len(dec)), dec.USD_per_kg, color=cc, width=0.6)
annot(ax[0], range(len(dec)), dec.USD_per_kg.values, '{:.2f}')
ax[0].set_xticks(range(len(dec)))
ax[0].set_xticklabels(['CAPEX', 'O&M', 'Replace', 'Cyclic'], fontsize=7)
ax[0].set_ylabel('USD/kg'); ax[0].set_ylim(0, dec.USD_per_kg.max() * 1.2)
ax[0].set_title('(a) Cost components')
ax[1].pie(dec.USD_per_kg, labels=['CAPEX', 'O&M', 'Replace', 'Cyclic'], colors=cc,
          autopct='%1.1f%%', textprops={'fontsize': 6.5}, startangle=90,
          wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax[1].set_title('(b) Cost shares')
save(fig, 'F08_lcoh_decomposition')

# ------------------------------------------------ F09 demand sweep
S = pd.read_csv(os.path.join(C.TABLE_DIR, 'T06_demand_sensitivity.csv'))
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.15))
ax[0].plot(S.demand_kg_day, S.LCOH, 'o-', color=RED, ms=4)
ax[0].set_ylabel('LCOH (USD/kg)'); ax[0].set_ylim(0, 15)
ax[0].set_title('(a) LCOH is near scale invariant')
ax[1].plot(S.demand_kg_day, S.x_PEM, 'o-', color=GREEN, ms=4)
ax[1].set_ylabel(r'$x_{\mathrm{PEM}}$'); ax[1].set_ylim(0, 0.55)
ax[1].set_title('(b) PEM share falls toward baseload')
ax[2].plot(S.demand_kg_day, S.P_solar, 'o-', color=ORANGE, ms=4, label='Solar')
ax[2].plot(S.demand_kg_day, S.P_wind, 's-', color=BLUE, ms=4, label='Wind')
ax[2].plot(S.demand_kg_day, S.P_elz, '^-', color=PURPLE, ms=4, label='Electrolyzer')
ax[2].set_ylabel('Capacity (kW)'); ax[2].legend(frameon=False)
ax[2].set_title('(c) Capacities scale with demand')
for a in ax:
    a.set_xlabel('H$_2$ demand (kg/day)')
fig.tight_layout(w_pad=1.3)
save(fig, 'F09_demand_sensitivity')

# ------------------------------------------------ F10 econ sensitivity
E = pd.read_csv(os.path.join(C.TABLE_DIR, 'T07_sensitivity.csv'))
fig, ax = plt.subplots(figsize=(3.5, 2.1))
v = E.LCOH.values
ax.bar(range(len(E)), v, color=[BLUE] * 3 + [GREEN] * 3, width=0.6)
annot(ax, range(len(E)), v, '{:.2f}')
ax.set_xticks(range(len(E)))
ax.set_xticklabels([c.replace('Discount ', 'r = ').replace('Degradation ', 'deg. ') for c in E.Case],
                   rotation=32, fontsize=6.2, ha='right')
ax.set_ylabel('LCOH (USD/kg)'); ax.set_ylim(0, v.max() * 1.18)
save(fig, 'F10_sensitivity')

print('core figures done')
