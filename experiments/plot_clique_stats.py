"""
Visualize maximal clique size statistics for DIP / BioGRID / Mann.

Layout
------
Top-left  : Parallel coordinates (normalized 0-1) 鈥?shows cross-metric relationships
Top-right : Bar chart 鈥?number of maximal cliques per dataset
Bottom row: Size-frequency histograms (one per dataset) with mean / median markers
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
RESULT_DIR = os.path.join(_ROOT, 'results', 'clique_size_stats')

with open(os.path.join(RESULT_DIR, 'clique_stats.json')) as fh:
    _raw = json.load(fh)

DATASETS = ['DIP', 'BioGRID', 'Mann']
COLORS   = ['#4878D0', '#EE854A', '#6ACC65']   # blue / orange / green
MARKERS  = ['o', 's', '^']

data = {
    name: {
        'num_cliques': _raw[name]['num_cliques'],
        'max_size':    _raw[name]['max_size'],
        'mean_size':   _raw[name]['mean_size'],
        'median_size': _raw[name]['median_size'],
        'size_dist':   {int(k): v for k, v in _raw[name]['size_distribution'].items()},
    }
    for name in DATASETS
}

# 鈹€鈹€ helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

METRIC_KEYS    = ['num_cliques', 'max_size', 'mean_size', 'median_size']
METRIC_LABELS  = ['# Cliques', 'Max size', 'Mean size', 'Median size']


def _normalize(vals):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


raw_per_metric = {m: [data[n][m] for n in DATASETS] for m in METRIC_KEYS}
norm_per_metric = {m: _normalize(raw_per_metric[m]) for m in METRIC_KEYS}


# 鈹€鈹€ figure layout 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

fig = plt.figure(figsize=(15, 8.5))
gs  = gridspec.GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[1.4, 1.0],
    hspace=0.52,
    wspace=0.36,
)

ax_para  = fig.add_subplot(gs[0, :2])   # parallel coordinates (top-left, wide)
ax_count = fig.add_subplot(gs[0, 2])    # num_cliques bar     (top-right)
ax_hist  = [fig.add_subplot(gs[1, c]) for c in range(3)]   # histograms (bottom)


# 鈹€鈹€ Panel A: parallel coordinates 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

N_AXES = len(METRIC_KEYS)
x_pos  = np.arange(N_AXES)

# draw vertical axis rules
for j in range(N_AXES):
    ax_para.axvline(j, color='#c8c8c8', linewidth=1.4, zorder=1)

# draw one poly-line per dataset
for i, (name, color, marker) in enumerate(zip(DATASETS, COLORS, MARKERS)):
    y_vals = [norm_per_metric[m][i] for m in METRIC_KEYS]
    ax_para.plot(x_pos, y_vals,
                 color=color, linewidth=2.6, marker=marker,
                 markersize=9, zorder=3, alpha=0.92, label=name,
                 markeredgecolor='white', markeredgewidth=0.8)

# annotate actual values beside each axis tick
#   label_offsets: (dx, dy, ha, va) per (axis_idx, dataset_idx)
#   Computed to avoid overlap given the actual normalised positions.
label_cfg = {
    # axis 0  num_cliques: DIP鈮?  Mann鈮?.07  BioGRID=1
    (0, 0): (-0.07, -0.13, 'right', 'top'),     # DIP   (bottom)
    (0, 1): (-0.07,  0.06, 'right', 'bottom'),  # BioGRID (top)
    (0, 2): ( 0.07,  0.06, 'left',  'bottom'),  # Mann  (just above DIP)

    # axis 1  max_size:  DIP=0  BioGRID鈮?.57  Mann=1
    (1, 0): ( 0.07, -0.13, 'left', 'top'),
    (1, 1): ( 0.07,  0.06, 'left', 'bottom'),
    (1, 2): ( 0.07,  0.06, 'left', 'bottom'),

    # axis 2  mean_size: DIP鈮?  BioGRID鈮?.36  Mann=1
    (2, 0): ( 0.07, -0.13, 'left', 'top'),
    (2, 1): ( 0.07,  0.06, 'left', 'bottom'),
    (2, 2): ( 0.07,  0.06, 'left', 'bottom'),

    # axis 3  median_size: DIP=0  BioGRID鈮?.16  Mann=1
    (3, 0): ( 0.07, -0.13, 'left', 'top'),
    (3, 1): ( 0.07,  0.06, 'left', 'bottom'),
    (3, 2): ( 0.07,  0.06, 'left', 'bottom'),
}

for i, (name, color) in enumerate(zip(DATASETS, COLORS)):
    for j, m in enumerate(METRIC_KEYS):
        y = norm_per_metric[m][i]
        rv = raw_per_metric[m][i]
        label = f'{rv:,}' if m == 'num_cliques' else f'{rv:.1f}' if isinstance(rv, float) else str(rv)
        dx, dy, ha, va = label_cfg[(j, i)]
        ax_para.text(j + dx, y + dy, label,
                     color=color, fontsize=8.2, ha=ha, va=va, fontweight='bold')

ax_para.set_xticks(x_pos)
ax_para.set_xticklabels(METRIC_LABELS, fontsize=11.5)
ax_para.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_para.set_yticklabels(['Min', '25%', '50%', '75%', 'Max'], fontsize=9)
ax_para.set_xlim(-0.35, N_AXES - 0.65)
ax_para.set_ylim(-0.28, 1.28)
ax_para.set_ylabel('Normalised value', fontsize=10)
ax_para.set_title('(A)  Parallel coordinates 鈥?cross-metric comparison (normalised)',
                  fontsize=11, fontweight='bold', loc='left')
ax_para.legend(fontsize=10, loc='upper right',
               framealpha=0.85, edgecolor='#cccccc')
ax_para.grid(axis='y', linestyle='--', alpha=0.35)
ax_para.spines[['top', 'right']].set_visible(False)


# 鈹€鈹€ Panel B: num_cliques bar 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

counts = [data[n]['num_cliques'] for n in DATASETS]
bars   = ax_count.bar(DATASETS, counts, color=COLORS,
                      alpha=0.85, edgecolor='white', linewidth=0.6, width=0.5)
for bar, val in zip(bars, counts):
    ax_count.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 800,
                  f'{val:,}',
                  ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                  color='#333333')

ax_count.set_ylabel('Number of maximal cliques', fontsize=10)
ax_count.set_title('(B)  # Maximal cliques', fontsize=11, fontweight='bold', loc='left')
ax_count.set_ylim(0, max(counts) * 1.18)
ax_count.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax_count.spines[['top', 'right']].set_visible(False)
ax_count.tick_params(axis='x', labelsize=10.5)


# 鈹€鈹€ Panel C: size-frequency histograms 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

for i, (name, color, ax) in enumerate(zip(DATASETS, COLORS, ax_hist)):
    dist   = data[name]['size_dist']
    sizes  = sorted(dist.keys())
    counts_h = [dist[s] for s in sizes]
    mean_v = data[name]['mean_size']
    med_v  = data[name]['median_size']

    ax.bar(sizes, counts_h, color=color, alpha=0.75,
           edgecolor='white', linewidth=0.4, width=0.9)
    ax.axvline(mean_v,   color='#222222', linewidth=1.6,
               linestyle='--', label=f'Mean {mean_v:.1f}',   zorder=4)
    ax.axvline(med_v,    color='#666666', linewidth=1.6,
               linestyle=':',  label=f'Median {med_v:.0f}',  zorder=4)

    ax.set_title(f'(C{i+1})  {name} 鈥?size distribution',
                 fontsize=10.5, fontweight='bold', loc='left')
    ax.set_xlabel('Clique size', fontsize=9.5)
    ax.set_ylabel('Count', fontsize=9.5)
    ax.legend(fontsize=8.5, framealpha=0.8, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))


# 鈹€鈹€ save 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

fig.suptitle(
    'Maximal clique statistics across PPI datasets  (DIP / BioGRID / Mann)',
    fontsize=13.5, fontweight='bold', y=1.01,
)

out_path = os.path.join(RESULT_DIR, 'clique_stats_plot.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')



