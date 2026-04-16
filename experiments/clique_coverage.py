"""
Coverage analysis between gold standard protein complexes and maximal cliques.

For each gold complex R and every maximal clique C, three pairwise metrics are
computed; the best-matching clique value is recorded for each complex:

  OS(C, R)      = |C 鈭?R|虏 / (|C| 脳 |R|)         -- overlap score
  Contain(C, R) = |C 鈭?R| / |R|                    -- containment (recall)
  Cover(C, R)   = |C 鈭?R| / min(|C|, |R|)          -- min-set coverage

Datasets: DIP, BioGRID, Mann (Saccharomyces cerevisiae).
Results saved to results/clique_coverage/.
"""

import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import networkx as nx

# 鈹€鈹€ paths 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.join(_HERE, '..')
_SC       = os.path.join(_ROOT, 'data', 'Saccharomyces_cerevisiae')
_SC_PPI   = os.path.join(_SC, 'PPI')
RESULT_DIR = os.path.join(_ROOT, 'results', 'clique_coverage')

GOLD_PATH = os.path.join(_SC, 'protein_complex', 'AdaPPI_golden_standard.txt')

_DATASETS = [
    ('DIP',     os.path.join(_SC_PPI, 'AdaPPI_Dataset', 'DIP',        'dip.txt'),            'tsv'),
    ('BioGRID', os.path.join(_SC_PPI, 'AdaPPI_Dataset', 'BIOGRID',    'biogrid.txt'),         'tsv'),
    ('Mann',    os.path.join(_SC_PPI, 'Mann_PPI.csv'),                                        'mann'),
]

COLORS  = {'DIP': '#4878D0', 'BioGRID': '#EE854A', 'Mann': '#6ACC65'}
DS_NAMES = ['DIP', 'BioGRID', 'Mann']


# 鈹€鈹€ data loading 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def load_gold(path):
    """Return list of frozensets; each frozenset = one gold complex."""
    complexes = []
    with open(path) as fh:
        for line in fh:
            members = line.strip().split()
            if len(members) >= 2:
                complexes.append(frozenset(members))
    return complexes


def load_ppi_edges(path, tag):
    if tag == 'tsv':
        df = pd.read_csv(path, sep='\t', header=None,
                         names=['p1', 'p2'], dtype=str)
        df = df[df['p1'] != df['p2']]
        return df[['p1', 'p2']].values.tolist()
    if tag == 'mann':
        df = pd.read_csv(path, sep=';', dtype=str)
        df = (
            df.assign(target=df['target'].str.split(';'))
            .explode('target')
            .reset_index(drop=True)[['source', 'target']]
            .query('source != target')
        )
        df.columns = ['p1', 'p2']
        return df[['p1', 'p2']].values.tolist()
    raise ValueError(tag)


def find_cliques(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    return list(nx.find_cliques(G)), G.nodes()


# 鈹€鈹€ coverage computation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def compute_coverage(gold_complexes, cliques):
    """
    For each gold complex R, find the clique C* maximising each metric.

    Metrics
    -------
    OS(C, R)      = |C 鈭?R|虏 / (|C| 脳 |R|)
    Contain(C, R) = |C 鈭?R| / |R|
    Cover(C, R)   = |C 鈭?R| / min(|C|, |R|)

    Returns a list of dicts (one per gold complex) with keys:
      complex_size, n_ppi_proteins, best_os, best_contain, best_cover
    """
    # Build inverted index: protein -> set of clique indices (plain dict)
    clique_sets = [set(c) for c in cliques]
    inv = {}
    for i, cs in enumerate(clique_sets):
        for p in cs:
            if p not in inv:
                inv[p] = set()
            inv[p].add(i)

    ppi_protein_set = set(inv.keys())   # fixed set; never modified after this

    records = []
    for R in gold_complexes:
        r_size = len(R)
        # Candidate cliques: those sharing 鈮? protein with R
        candidate_idx = set()
        for p in R:
            if p in inv:                # plain dict: no phantom entries
                candidate_idx |= inv[p]

        n_ppi = len(R & ppi_protein_set)   # proteins of R that appear in PPI

        best_os, best_contain, best_cover = 0.0, 0.0, 0.0
        for i in candidate_idx:
            C = clique_sets[i]
            inter = len(C & R)
            if inter == 0:
                continue
            c_size = len(C)
            os      = inter ** 2 / (c_size * r_size)
            contain = inter / r_size
            cover   = inter / min(c_size, r_size)
            if os      > best_os:      best_os      = os
            if contain > best_contain: best_contain = contain
            if cover   > best_cover:   best_cover   = cover

        records.append({
            'complex_size':    r_size,
            'n_ppi_proteins':  n_ppi,
            'best_os':         best_os,
            'best_contain':    best_contain,
            'best_cover':      best_cover,
        })
    return records


# 鈹€鈹€ statistics helper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _dist_stats(arr):
    a = np.array(arr)
    return {
        'mean':  round(float(a.mean()), 4),
        'std':   round(float(a.std()),  4),
        'p25':   round(float(np.percentile(a, 25)), 4),
        'p50':   round(float(np.percentile(a, 50)), 4),
        'p75':   round(float(np.percentile(a, 75)), 4),
        'p90':   round(float(np.percentile(a, 90)), 4),
        'p95':   round(float(np.percentile(a, 95)), 4),
        'gt_02': int((a > 0.2).sum()),
        'gt_05': int((a > 0.5).sum()),
        'gt_07': int((a > 0.7).sum()),
        'eq_0':  int((a == 0.0).sum()),
    }


def summarise(records, n_total):
    arr_os      = [r['best_os']      for r in records]
    arr_contain = [r['best_contain'] for r in records]
    arr_cover   = [r['best_cover']   for r in records]

    return {
        'n_complexes':          n_total,
        'n_with_ppi_overlap':   sum(1 for r in records if r['n_ppi_proteins'] > 0),
        'overlap_score':        _dist_stats(arr_os),
        'containment':          _dist_stats(arr_contain),
        'min_coverage':         _dist_stats(arr_cover),
    }


# 鈹€鈹€ plotting 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def plot_coverage(all_records, out_path):
    """
    3-panel figure: one CDF curve per dataset, one panel per metric.
    X-axis: score in [0, 1]; Y-axis: fraction of complexes with score >= x.
    """
    metrics = [
        ('best_os',      'Overlap score  OS(C,R) = |C鈭㏑|虏 / (|C|路|R|)'),
        ('best_contain', 'Containment  = |C鈭㏑| / |R|'),
        ('best_cover',   'Min-coverage  = |C鈭㏑| / min(|C|,|R|)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    x_grid = np.linspace(0, 1, 500)

    for ax, (key, title) in zip(axes, metrics):
        for name in DS_NAMES:
            vals = np.array([r[key] for r in all_records[name]])
            # survival function: fraction of complexes with score >= x
            y = np.array([(vals >= t).mean() for t in x_grid])
            ax.plot(x_grid, y, color=COLORS[name], linewidth=2.2,
                    label=name, alpha=0.9)

        ax.axvline(0.2, color='#aaaaaa', linewidth=1, linestyle='--', alpha=0.7)
        ax.axvline(0.5, color='#888888', linewidth=1, linestyle='--', alpha=0.7)
        ax.set_xlabel('Score threshold', fontsize=11)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=10, framealpha=0.85)

    axes[0].set_ylabel('Fraction of gold complexes with\nbest score 鈮?threshold', fontsize=10)

    fig.suptitle(
        'Coverage of gold standard complexes by maximal cliques\n'
        '(survival function: fraction of complexes achieving score 鈮?threshold)',
        fontsize=12, fontweight='bold', y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# 鈹€鈹€ main 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def run():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print('Loading gold standard complexes...')
    gold = load_gold(GOLD_PATH)
    print(f'  {len(gold)} complexes loaded')

    all_records = {}
    all_summary = {}

    for name, path, tag in _DATASETS:
        print(f'\n[{name}] Loading PPI and finding cliques...')
        t0 = time.time()
        edges   = load_ppi_edges(path, tag)
        cliques, ppi_nodes = find_cliques(edges)
        print(f'  {len(cliques)} cliques found in {time.time()-t0:.1f}s')

        print(f'[{name}] Computing coverage for {len(gold)} complexes...')
        t1 = time.time()
        records = compute_coverage(gold, cliques)
        print(f'  Done in {time.time()-t1:.1f}s')

        all_records[name] = records
        all_summary[name] = summarise(records, len(gold))

        # Per-dataset CSV
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(RESULT_DIR, f'coverage_{name}.csv'), index=False)

    # Print summary table
    print('\n' + '='*70)
    for name, s in all_summary.items():
        print(f'\n[{name}]  complexes={s["n_complexes"]}  '
              f'with_ppi_proteins={s["n_with_ppi_overlap"]}')
        for metric in ('overlap_score', 'containment', 'min_coverage'):
            d = s[metric]
            print(f'  {metric:15s}  mean={d["mean"]:.3f}  '
                  f'p50={d["p50"]:.3f}  p75={d["p75"]:.3f}  '
                  f'p90={d["p90"]:.3f}  '
                  f'>0.5:{d["gt_05"]}  >0.7:{d["gt_07"]}  =0:{d["eq_0"]}')

    # Save JSON summary
    json_path = os.path.join(RESULT_DIR, 'coverage_stats.json')
    with open(json_path, 'w') as fh:
        json.dump(all_summary, fh, indent=2)
    print(f'\nSaved: {json_path}')

    # Plot
    plot_coverage(all_records,
                  os.path.join(RESULT_DIR, 'coverage_plot.png'))


if __name__ == '__main__':
    run()



