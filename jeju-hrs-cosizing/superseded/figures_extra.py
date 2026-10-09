"""figures_extra.py - graphical abstract, data pipeline, algorithm flow, F11 to F14."""
import json, os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import config as C
import data_v2 as D

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


def box(ax, x, y, w, h, text, fc, ec, fs=6.4, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012',
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight=weight)


def arrow(ax, p1, p2, color='#444444', style='-|>', rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=8,
                                 color=color, lw=lw,
                                 connectionstyle=f'arc3,rad={rad}'))


re = D.load_re()
ps, pw = re['train']
RES = json.load(open('checkpoints/results_numbers.json'))
rec = RES['recommended']

# ============================================ GRAPHICAL ABSTRACT
fig = plt.figure(figsize=(7.3, 3.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.42)

axf = fig.add_subplot(gs[0]); axf.axis('off')
axf.set_xlim(0, 1); axf.set_ylim(0, 1)
axf.text(0.5, 0.965, 'Joint six variable design space', ha='center', fontsize=7.6, weight='bold')
box(axf, 0.04, 0.75, 0.42, 0.115, 'Solar capacity', '#E8F0F7', BLUE)
box(axf, 0.54, 0.75, 0.42, 0.115, 'Wind capacity', '#E8F0F7', BLUE)
box(axf, 0.04, 0.575, 0.42, 0.115, 'Electrolyzer size', '#E9F4EF', GREEN)
box(axf, 0.54, 0.575, 0.42, 0.115, r'AWE/PEM split', '#E9F4EF', GREEN)
box(axf, 0.04, 0.40, 0.42, 0.115, 'Battery', '#FBEEE4', ORANGE)
box(axf, 0.54, 0.40, 0.42, 0.115, 'H$_2$ storage', '#FBEEE4', ORANGE)
box(axf, 0.10, 0.20, 0.80, 0.125, '26,304 h chronological dispatch\nmeasured station demand', '#F2F2F2', GREY, fs=6.2)
box(axf, 0.10, 0.035, 0.80, 0.115, 'NSGA-III: LCOH, LPSP, curtailment', '#F7E9E9', RED, fs=6.4)
arrow(axf, (0.5, 0.40), (0.5, 0.332))
arrow(axf, (0.5, 0.20), (0.5, 0.155))

axa = fig.add_subplot(gs[1])
cases = ['F0', 'F1', 'F2', 'F3']
vals = [21.24, 19.64, 17.23, 10.80]
bars = axa.bar(range(4), vals, color=[GREY, BLUE, GREEN, RED], width=0.62)
for i, v in enumerate(vals):
    axa.text(i, v + 0.4, f'{v:.2f}', ha='center', fontsize=6.6, weight='bold')
axa.set_xticks(range(4)); axa.set_xticklabels(cases, fontsize=7.2)
axa.set_xlabel('F0 fixed  F1 renewable  F2 conversion  F3 joint', fontsize=5.8)
axa.set_ylabel('LCOH (USD/kg)'); axa.set_ylim(0, 25); axa.set_xlim(-0.75, 3.75)
axa.set_title('Co-sizing decomposition', fontsize=7.8, weight='bold')
axa.annotate('', xy=(3.0, 11.6), xytext=(0.0, 22.0),
             arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.1,
                             connectionstyle='arc3,rad=0.28'))
axa.text(1.5, 6.4, '$-$10.45 USD/kg', fontsize=7.0, color=RED, weight='bold', ha='center')
axa.text(1.5, 4.3, 'interaction $-$4.82', fontsize=6.3, color=RED, ha='center')

axr = fig.add_subplot(gs[2]); axr.axis('off')
axr.set_xlim(0, 1); axr.set_ylim(0, 1)
axr.text(0.5, 0.965, 'Recommended design', ha='center', fontsize=7.8, weight='bold')
rows = [('Solar', f"{rec['P_solar']:.0f} kW"), ('Wind', f"{rec['P_wind']:.0f} kW"),
        ('AWE', f"{rec['P_elz']*(1-rec['x_PEM']):.0f} kW"),
        ('PEM', f"{rec['P_elz']*rec['x_PEM']:.0f} kW"),
        ('Battery', f"{rec['E_BESS']:.0f} kWh"), ('H$_2$ store', f"{rec['M_HSS']:.0f} kg")]
for i, (k, v) in enumerate(rows):
    y = 0.855 - i * 0.098
    axr.add_patch(Rectangle((0.03, y - 0.038), 0.94, 0.078,
                            fc='#F6F8FA' if i % 2 == 0 else 'white', ec='none'))
    axr.text(0.07, y, k, fontsize=6.8, va='center')
    axr.text(0.93, y, v, fontsize=6.8, va='center', ha='right', weight='bold')
box(axr, 0.03, 0.115, 0.94, 0.135,
    'LCOH 11.29 USD/kg   |   LPSP 0\nCurtailment 34.95%', '#F7E9E9', RED, fs=6.8, weight='bold')
axr.text(0.5, 0.045, 'Held out 2023: 11.65 USD/kg, LPSP 0.08%',
         ha='center', fontsize=6.0, style='italic', color='#444444')
save(fig, 'GA_graphical_abstract')

# ============================================ DATA PIPELINE
raw_s = pd.read_csv(os.path.join(C.DATA_DIR, 'raw_KPX_solar_2019-12_2023-12.csv'),
                    encoding='cp949') if os.path.exists(
    os.path.join(C.DATA_DIR, 'raw_KPX_solar_2019-12_2023-12.csv')) else None
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.15))
if raw_s is not None:
    raw_s.columns = ['date'] + [f'H{i:02d}' for i in range(24)]
    raw_s['date'] = pd.to_datetime(raw_s['date']); raw_s['Year'] = raw_s.date.dt.year
    H = [f'H{i:02d}' for i in range(24)]
    yrs = [2020, 2021, 2022, 2023]
    p999 = [np.percentile(raw_s[raw_s.Year == y][H].values, 99.9) for y in yrs]
    jun = [raw_s[(raw_s.Year == y) & (raw_s.date.dt.month == 6)][['H11', 'H12', 'H13']].values.mean()
           for y in yrs]
    ax[0].bar(np.arange(4) - 0.2, p999, 0.4, color=BLUE, label='99.9th pct hourly')
    ax[0].bar(np.arange(4) + 0.2, jun, 0.4, color=ORANGE, label='June midday mean')
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([str(y) for y in yrs])
    ax[0].set_ylabel('Fleet output (MWh/h)'); ax[0].legend(frameon=False, fontsize=6)
    ax[0].set_title('(a) Raw fleet output grows\nwith installed capacity')
stats = D.re_statistics()
ax[1].plot(stats.Year, stats.solar_cf, 'o-', color=ORANGE, ms=4, label='Solar, year specific')
ax[1].plot(stats.Year, stats.wind_cf, 's-', color=BLUE, ms=4, label='Wind, year specific')
ax[1].plot(stats.Year, [8.28, 8.74, 10.34, 12.36], 'o--', color=ORANGE, ms=3, alpha=0.5,
           label='Solar, fixed denominator')
ax[1].set_xticks(stats.Year); ax[1].set_ylabel('Capacity factor (%)')
ax[1].legend(frameon=False, fontsize=5.6); ax[1].set_ylim(0, 34)
ax[1].set_title('(b) Normalization removes\nfleet growth artifact')
tot = ps * C.REF_SOLAR_KW + pw * C.REF_WIND_KW
ax[2].plot(np.arange(720) / 24, tot[:720], lw=0.6, color=GREEN)
ax[2].set_xlabel('Day of January 2020'); ax[2].set_ylabel('Plant output (kW)')
ax[2].set_title('(c) Per unit profile scaled to\nthe optimized plant')
fig.tight_layout(w_pad=1.3)
save(fig, 'F00_data_pipeline')

# ============================================ METHOD FLOWCHART
fig, ax = plt.subplots(figsize=(7.1, 3.5))
ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
box(ax, 0.02, 0.80, 0.20, 0.15, 'Raw KPX fleet series\n2020 to 2023\nhourly solar and wind', '#E8F0F7', BLUE)
box(ax, 0.02, 0.58, 0.20, 0.15, 'Hamdeok dispensing\n255 measured days', '#E8F0F7', BLUE)
box(ax, 0.27, 0.80, 0.20, 0.15, 'Algorithm 1\nyear specific\nnormalization', '#E9F4EF', GREEN)
box(ax, 0.27, 0.58, 0.20, 0.15, 'Algorithm 2\nblock bootstrap\ndemand chronology', '#E9F4EF', GREEN)
box(ax, 0.52, 0.68, 0.21, 0.27,
    'Algorithm 3\nchronological dispatch\n\nproduction cap\nEq. (12)\nmass balance\nEq. (16)', '#FBEEE4', ORANGE, fs=6.0)
box(ax, 0.78, 0.68, 0.20, 0.27,
    'Objectives\n\nLCOH  Eq. (17)\nLPSP  Eq. (18)\nCURT  Eq. (19)', '#F7E9E9', RED, fs=6.2)
box(ax, 0.52, 0.36, 0.21, 0.20, 'Algorithm 4\nNSGA-III\npop 182, gen 100', '#F0EAF6', PURPLE)
box(ax, 0.78, 0.36, 0.20, 0.20,
    'Algorithm 5\nreliability screen\nLPSP $\\leq$ 1%\nthen least cost', '#F0EAF6', PURPLE, fs=6.2)
box(ax, 0.30, 0.06, 0.42, 0.17,
    'Validation: held out 2023, ablation F0 to F3,\nmatched reliability sweep, tornado, seeds', '#F2F2F2', GREY, fs=6.4)
arrow(ax, (0.22, 0.875), (0.27, 0.875))
arrow(ax, (0.22, 0.655), (0.27, 0.655))
arrow(ax, (0.47, 0.875), (0.52, 0.84))
arrow(ax, (0.47, 0.655), (0.52, 0.72))
arrow(ax, (0.73, 0.815), (0.78, 0.815))
arrow(ax, (0.885, 0.68), (0.885, 0.565))
arrow(ax, (0.78, 0.46), (0.73, 0.46))
arrow(ax, (0.625, 0.36), (0.55, 0.23))
arrow(ax, (0.625, 0.56), (0.625, 0.68), rad=-0.55, color=PURPLE)
ax.text(0.545, 0.625, 'iterate', fontsize=5.8, color=PURPLE, rotation=90, va='center')
save(fig, 'F0A_method_flow')

# ============================================ F11 eps sweep
Dd = pd.read_csv(os.path.join(C.TABLE_DIR, 'T09_eps_sweep_raw.csv'))
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.35))
lbl = {'awe': 'AWE only', 'pem': 'PEM only', 'hybrid': 'Hybrid AWE-PEM'}
for tech, c, mk in (('awe', BLUE, 'o'), ('pem', GREEN, 's'), ('hybrid', RED, '^')):
    dd = Dd[Dd.tech == tech].dropna(subset=['LCOH'])
    es = sorted(dd.eps.unique(), reverse=True)
    m = [dd[dd.eps == e].LCOH.mean() for e in es]
    lo = [dd[dd.eps == e].LCOH.min() for e in es]
    hi = [dd[dd.eps == e].LCOH.max() for e in es]
    xx = [100 * e for e in es]
    ax[0].errorbar(xx, m, yerr=[np.array(m) - lo, np.array(hi) - np.array(m)],
                   marker=mk, color=c, capsize=2.5, ms=4, label=lbl[tech])
    for _, r in dd.iterrows():
        ax[0].plot(100 * r.eps, r.LCOH, '.', color=c, ms=3, alpha=0.45)
    ax[1].plot(xx, lo, marker=mk, color=c, ms=4, label=lbl[tech])
for a in ax:
    a.invert_xaxis(); a.set_xlabel(r'Reliability target, LPSP $\leq \epsilon$ (%)')
ax[0].set_ylabel('Minimum LCOH (USD/kg)'); ax[0].legend(frameon=False)
ax[0].set_title('(a) Mean and range over five seeds')
ax[1].set_ylabel('Best over seeds LCOH (USD/kg)')
ax[1].set_title('(b) Tightest bound on each optimum')
fig.tight_layout(w_pad=1.4)
save(fig, 'F11_eps_sweep')

# ============================================ F12 ablation
A = pd.read_csv(os.path.join(C.TABLE_DIR, 'T10_ablation.csv'))
d1, d2, d3 = A.Delta[1], A.Delta[2], A.Delta[3]
inter = d3 - d1 - d2
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
names = ['F0\nnothing free', 'F1\nrenewable\nonly', 'F2\nconversion\n+ storage', 'F3\njoint']
cols = [GREY, BLUE, GREEN, RED]
ax[0].bar(range(4), A.LCOH, color=cols, width=0.62)
for i, v in enumerate(A.LCOH):
    ax[0].text(i, v + 0.35, f'{v:.2f}', ha='center', fontsize=6.6)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names, fontsize=6.3)
ax[0].set_ylabel('LCOH (USD/kg)'); ax[0].set_ylim(0, 25)
ax[0].set_title('(a) Ablation of the design space')
comp = [-d1, -d2, -inter]
cl = ['Renewable\nsizing', 'Conversion\n+ storage', 'Joint\ninteraction']
ax[1].bar(range(3), comp, color=[BLUE, GREEN, PURPLE], width=0.55)
for i, v in enumerate(comp):
    ax[1].text(i, v + 0.12, f'{v:.2f}', ha='center', fontsize=6.8, weight='bold')
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(cl, fontsize=6.3)
ax[1].set_ylabel('LCOH reduction (USD/kg)'); ax[1].set_ylim(0, max(comp) * 1.22)
ax[1].set_title('(b) Decomposition of the total gain')
fig.tight_layout(w_pad=1.4)
save(fig, 'F12_ablation')

# ============================================ F13 tornado
T = pd.read_csv(os.path.join(C.TABLE_DIR, 'T11_tornado.csv'))
Tt = T[T.pct > 0].copy()
base = 11.2851
Tt['swing'] = (Tt[['hyb_low', 'hyb_high']].max(1) - Tt[['hyb_low', 'hyb_high']].min(1))
Tt = Tt.sort_values('swing')
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for i, (_, r) in enumerate(Tt.iterrows()):
    ax.barh(i, r.hyb_high - base, left=base, color=RED, height=0.6, label='+30%' if i == 0 else '')
    ax.barh(i, r.hyb_low - base, left=base, color=BLUE, height=0.6, label='-30%' if i == 0 else '')
ax.axvline(base, color='k', lw=1.0)
ax.text(base, len(Tt) - 0.3, f'base {base:.2f}', fontsize=6, ha='center')
ax.set_yticks(range(len(Tt)))
ax.set_yticklabels([f"{r.Parameter} ±{r.pct}%" for _, r in Tt.iterrows()], fontsize=6.4)
ax.set_xlabel('LCOH (USD/kg)'); ax.legend(frameon=False, loc='lower right', fontsize=6.2)
save(fig, 'F13_tornado')

# ============================================ F14 mechanism
M = pd.read_csv(os.path.join(C.TABLE_DIR, 'T12_xpem_mechanism.csv'))
i = int(np.argmin(M.LCOH.values))
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.2))
ax[0].plot(M.x_PEM, M.LCOH, 'o-', color=RED, ms=3.5)
ax[0].axvline(M.x_PEM[i], color=GREY, ls='--', lw=0.9)
ax[0].annotate(f'min {M.LCOH[i]:.2f}\nat {M.x_PEM[i]:.2f}', (M.x_PEM[i], M.LCOH[i]),
               textcoords='offset points', xytext=(6, 14), fontsize=6.2)
ax[0].set_xlabel(r'$x_{\mathrm{PEM}}$'); ax[0].set_ylabel('LCOH (USD/kg)')
ax[0].set_title('(a) Composition cost curve')
ax[1].plot(M.x_PEM, M.awe_cyc, 'o-', color=BLUE, ms=3.5, label='AWE starts')
ax[1].set_xlabel(r'$x_{\mathrm{PEM}}$'); ax[1].set_ylabel('AWE start cycles (1/yr)', color=BLUE)
a2 = ax[1].twinx(); a2.plot(M.x_PEM, M.repl, 's-', color=ORANGE, ms=3.5)
a2.set_ylabel('Replacement (USD/kg)', color=ORANGE); a2.grid(False)
ax[1].set_title('(b) PEM suppresses cycling\nand replacement cost')
ax[2].plot(M.x_PEM, M.capex, 'o-', color=BLUE, ms=3.5, label='CAPEX')
ax[2].plot(M.x_PEM, M.repl, 's-', color=ORANGE, ms=3.5, label='Replacement')
ax[2].plot(M.x_PEM, M.capex + M.repl, '^-', color=RED, ms=3.5, label='Sum')
ax[2].set_xlabel(r'$x_{\mathrm{PEM}}$'); ax[2].set_ylabel('USD/kg')
ax[2].legend(frameon=False, fontsize=6.2)
ax[2].set_title('(c) Capital rises, replacement falls')
fig.tight_layout(w_pad=1.5)
save(fig, 'F14_mechanism')
print('extra figures done')
