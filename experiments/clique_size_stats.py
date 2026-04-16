"""
Maximal clique size distribution statistics across PPI datasets.

Each dataset's maximal cliques (used as hyperedges in the hypergraph)
are analyzed for size distribution. Results are saved to
results/clique_size_stats/.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_SC_PPI = os.path.join(_ROOT, 'data', 'Saccharomyces_cerevisiae', 'PPI')
RESULT_DIR = os.path.join(_ROOT, 'results', 'clique_size_stats')

# Dataset definitions: name -> (path, loader_tag)
_DATASETS = [
    ('DIP',        os.path.join(_SC_PPI, 'AdaPPI_Dataset', 'DIP',        'dip.txt'),           'tsv'),
    ('BioGRID',    os.path.join(_SC_PPI, 'AdaPPI_Dataset', 'BIOGRID',    'biogrid.txt'),        'tsv'),
    ('Collins',    os.path.join(_SC_PPI, 'AdaPPI_Dataset', 'COLLINS',    'collins.txt'),        'tsv'),
    ('Krogan-core',os.path.join(_SC_PPI, 'AdaPPI_Dataset', 'Krogan-core','krogan2006core.txt'), 'tsv'),
    ('Krogan14k',  os.path.join(_SC_PPI, 'AdaPPI_Dataset', 'Krogan14k',  'krogan14k.txt'),      'tsv'),
    ('Mann',       os.path.join(_SC_PPI, 'Mann_PPI.csv'),                                       'mann'),
]


def _load_tsv(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['p1', 'p2'], dtype=str)
    df = df[df['p1'] != df['p2']]
    return df[['p1', 'p2']].values.tolist()


def _load_mann(path):
    df = pd.read_csv(path, sep=';', dtype=str)
    df = (
        df.assign(target=df['target'].str.split(';'))
        .explode('target')
        .reset_index(drop=True)[['source', 'target']]
        .query('source != target')
    )
    df.columns = ['p1', 'p2']
    return df[['p1', 'p2']].values.tolist()


def _load_edges(path, tag):
    if tag == 'tsv':
        return _load_tsv(path)
    if tag == 'mann':
        return _load_mann(path)
    raise ValueError(f'Unknown loader tag: {tag}')


def _graph_stats(G, cliques):
    sizes = np.array([len(c) for c in cliques], dtype=int)
    dist = Counter(sizes.tolist())
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'num_cliques': int(len(sizes)),
        'min_size': int(sizes.min()),
        'max_size': int(sizes.max()),
        'mean_size': round(float(sizes.mean()), 4),
        'median_size': float(np.median(sizes)),
        'std_size': round(float(sizes.std()), 4),
        'p25': float(np.percentile(sizes, 25)),
        'p75': float(np.percentile(sizes, 75)),
        'p90': float(np.percentile(sizes, 90)),
        'p95': float(np.percentile(sizes, 95)),
        'p99': float(np.percentile(sizes, 99)),
        'size_distribution': {str(k): int(v) for k, v in sorted(dist.items())},
    }


def run():
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_stats = {}

    for name, path, tag in _DATASETS:
        print(f'[{name}] Loading edges...', flush=True)
        t0 = time.time()

        edges = _load_edges(path, tag)
        G = nx.Graph()
        G.add_edges_from(edges)

        print(f'[{name}] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. '
              f'Finding cliques...', flush=True)

        cliques = list(nx.find_cliques(G))
        elapsed = round(time.time() - t0, 2)

        stats = _graph_stats(G, cliques)
        stats['elapsed_sec'] = elapsed
        all_stats[name] = stats

        print(f'[{name}] cliques={stats["num_cliques"]}, '
              f'size=[{stats["min_size"]}, {stats["max_size"]}], '
              f'mean={stats["mean_size"]:.2f}, '
              f'median={stats["median_size"]:.1f}, '
              f'time={elapsed}s', flush=True)

    # --- Save results ---
    json_path = os.path.join(RESULT_DIR, 'clique_stats.json')
    with open(json_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f'\nSaved: {json_path}')

    summary_rows = []
    for name, s in all_stats.items():
        summary_rows.append({
            'dataset':     name,
            'num_nodes':   s['num_nodes'],
            'num_edges':   s['num_edges'],
            'num_cliques': s['num_cliques'],
            'min_size':    s['min_size'],
            'max_size':    s['max_size'],
            'mean_size':   s['mean_size'],
            'median_size': s['median_size'],
            'std_size':    s['std_size'],
            'p25':         s['p25'],
            'p75':         s['p75'],
            'p90':         s['p90'],
            'p95':         s['p95'],
            'p99':         s['p99'],
            'elapsed_sec': s['elapsed_sec'],
        })
    csv_path = os.path.join(RESULT_DIR, 'clique_stats.csv')
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')

    for name, s in all_stats.items():
        df_dist = pd.DataFrame(
            [{'clique_size': int(k), 'count': v}
             for k, v in s['size_distribution'].items()]
        )
        dist_path = os.path.join(RESULT_DIR, f'size_dist_{name}.csv')
        df_dist.to_csv(dist_path, index=False)
        print(f'Saved: {dist_path}')


if __name__ == '__main__':
    run()



