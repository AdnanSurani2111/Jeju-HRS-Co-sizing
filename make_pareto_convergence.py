"""Pareto front (3D + three projections) and convergence figures.
SEARCH basis: seed-42 uninterrupted run, objectives evaluated on the two
optimization demand realizations (seeds 0-1). These are NOT the 20-draw
validation values and must never be compared with 11.7184 USD/kg.
"""
import json, hashlib, os, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 6.8, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
})
# Okabe-Ito
BLUE, ORANGE, GREEN, RED, GREY, GOLD = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#666666', '#F0C808'
OUT = 'figures'

F = np.load('checkpoints/cont_F_42.npy')
X = np.load('checkpoints/cont_X_42.npy')
sel = np.array(list(json.load(open('checkpoints/FINAL.json'))['design'].values()))
i_sel = int(np.argmin(np.linalg.norm(X - sel, axis=1)))
lc, lp, cu = F[:, 0], 100 * F[:, 1], 100 * F[:, 2]

# ---------------------------------------------------------------- Pareto
fig = plt.figure(figsize=(7.1, 5.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1.0], hspace=0.30, wspace=0.40)

ax3 = fig.add_subplot(gs[0, :], projection='3d')
ax3.set_box_aspect((1.9, 1.25, 0.95), zoom=1.28)
p = ax3.scatter(lc, cu, lp, c=lp, cmap='viridis', s=26, depthshade=False,
                edgecolor='k', linewidth=0.25)
ax3.scatter([lc[i_sel]], [cu[i_sel]], [lp[i_sel]], marker='*', s=340, c=GOLD,
            edgecolor='k', linewidth=0.6, depthshade=False, label='Selected design')
ax3.set_xlabel('LCOH (USD/kg)', labelpad=2)
ax3.set_ylabel('Curtailment (%)', labelpad=2)
ax3.set_zlabel('Service loss (%)', labelpad=-2)
ax3.set_zlim(0, max(lp) * 1.05)          # anchor at exactly zero: no negative padding
ax3.view_init(elev=20, azim=-58)
ax3.zaxis.set_rotate_label(False)
ax3.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.02, 0.99))
ax3.set_title('(a) Nondominated set in the three objective dimensions', pad=2)

def proj(ax, xv, yv, xl, yl, ttl, col):
    ax.scatter(xv, yv, s=16, c=col, edgecolor='k', linewidth=0.2, alpha=0.85)
    ax.scatter(xv[i_sel], yv[i_sel], marker='*', s=190, c=GOLD, edgecolor='k',
               linewidth=0.5, zorder=5)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)

axb = fig.add_subplot(gs[1, 0])
proj(axb, lc, lp, 'LCOH (USD/kg)', 'Service loss (%)', '(b) Cost vs service loss', BLUE)
axb.axhline(0.1, color=RED, ls='--', lw=0.9)
axb.set_ylim(-3.2, max(lp) * 1.06)


axc = fig.add_subplot(gs[1, 1])
proj(axc, lc, cu, 'LCOH (USD/kg)', 'Curtailment (%)', '(c) Cost vs curtailment', GREEN)

axd = fig.add_subplot(gs[1, 2])
proj(axd, cu, lp, 'Curtailment (%)', 'Service loss (%)', '(d) Curtailment vs service loss', ORANGE)
axd.set_ylim(-2.5, max(lp) * 1.06)

for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT}/F16_pareto_front.{ext}')
plt.close(fig)
print('  F16_pareto_front')

# ------------------------------------------------------------ convergence
hv = json.load(open('checkpoints/cont_hv_42.json'))
g = np.array([h['gen'] for h in hv])
v = np.array([h['hv'] for h in hv])
nf = np.array([h['n_front'] for h in hv])
best = np.maximum.accumulate(v)

fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.25))
ax[0].plot(g, v, lw=0.9, color=BLUE, label='Per-generation front')
ax[0].plot(g, best, lw=1.4, color=RED, label='Best so far')
ax[0].set_xlabel('Generation'); ax[0].set_ylabel('Hypervolume')
ax[0].legend(frameon=False, loc='lower right')
ax[0].set_title('(a) Hypervolume')

ax[1].plot(g, nf, lw=1.1, color=GREEN)
ax[1].set_xlabel('Generation'); ax[1].set_ylabel('Nondominated solutions')
ax[1].set_title('(b) Front size')

w = 10
tail = v[-w:]
ax[2].plot(g[-40:], v[-40:], 'o-', ms=2.6, lw=0.9, color=BLUE)
ax[2].axhline(tail.mean(), color=GREY, ls='--', lw=0.9)
ax[2].set_xlabel('Generation'); ax[2].set_ylabel('Hypervolume')
ax[2].set_title(f'(c) Final 40 generations\nlast-10 spread {tail.max()-tail.min():.5f}')
fig.tight_layout(w_pad=1.3)
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT}/F17_convergence.{ext}')
plt.close(fig)
print('  F17_convergence')
print(f'  front {len(F)} pts | selected at index {i_sel} | last-10 HV spread {tail.max()-tail.min():.6f}')

# ------------------------------------------------------------ sidecars
def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

now = datetime.datetime.now().isoformat(timespec='seconds')
meta = {
 'F16_pareto_front': dict(
    source_files=['checkpoints/cont_F_42.npy', 'checkpoints/cont_X_42.npy',
                  'checkpoints/FINAL.json'],
    source_keys_or_columns=['objective vectors (LCOH, LPSP, CURT)', 'design vectors',
                            'design'],
    basis='SEARCH',
    number_of_realizations=2,
    note=('Objectives evaluated on the two optimization demand realizations '
          '(seeds 0-1). Not comparable with the 20-draw validation LCOH of '
          '11.7184 USD/kg.')),
 'F17_convergence': dict(
    source_files=['checkpoints/cont_hv_42.json'],
    source_keys_or_columns=['gen', 'hv', 'n_front'],
    basis='SEARCH',
    number_of_realizations=2,
    note=('Per-generation front hypervolume is not monotone because the '
          'indicator is computed on the current front, not a cumulative '
          'archive. The best-so-far curve is monotone by construction.')),
}
for fid, m in meta.items():
    m.update(figure_id=fid, design_identifier='FINAL.json selected design',
             generated_at=now, script='make_pareto_convergence.py',
             script_hash=sha('make_pareto_convergence.py'),
             png_file=f'{fid}.png', png_sha256=sha(f'{OUT}/{fid}.png'),
             pdf_file=f'{fid}.pdf', pdf_sha256=sha(f'{OUT}/{fid}.pdf'),
             main_text=True)
    json.dump(m, open(f'{OUT}/{fid}.meta.json', 'w'), indent=2)
print('  sidecars written')
