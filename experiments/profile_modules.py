"""
Per-module runtime and memory profiling for KSHL-PC across DIP, BioGRID, and Mann-PPI datasets.

Modules profiled:
  A. Hypergraph Construction (maximal clique enumeration)
  B. SHE (Spectral Hypergraph Embedding)
  C. KEFT (Knowledge-Enhanced Feature Transformation)
  D. PCpredict Classifier (1-fold, 100-epoch training)
"""

import sys
import os
import time
import tracemalloc
import gc

import numpy as np
import pandas as pd
import torch
import networkx as nx
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 鈹€鈹€ path setup 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

from she_core import she_embed, SHEConfig
from she_embed import build_H0_from_cliques, compute_she_embedding
from knowledge_features import build_knowledge_features
from fusion import fuse_embeddings
from utils import (
    Nested_list_dup, count_unique_elements,
    convert_ppi, load_txt_list, try_gpu, negative_on_distribution,
)
from models import PCpredict
from Train_PC import train_DNN

# 鈹€鈹€ constants 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
DATA_ROOT   = os.path.join(PROJECT_ROOT, 'data', 'Saccharomyces_cerevisiae')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'figs')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQUENCE_PATH = os.path.join(
    DATA_ROOT, 'protein_feature', 'uniprot-sequences-2023.05.10-01.31.31.11.tsv'
)
GO_SLIM_PATH  = os.path.join(DATA_ROOT, 'GO', 'go_slim_mapping.tab.txt')
SERIES_PATH   = os.path.join(DATA_ROOT, 'expression', 'series_matrix.txt')
PC_PATH       = os.path.join(DATA_ROOT, 'protein_complex', 'AdaPPI_golden_standard.txt')

SHE_R     = 64
SHE_K     = 64
SHE_T     = 10
SHE_ALPHA = 0.1
SEED       = 42

KEFT_EPOCHS  = 100
KEFT_LR      = 1e-3
KEFT_OUT_DIM = 128

CLF_MEASURE_EPOCHS = 10   # epochs actually timed; extrapolated to 500 in output
CLF_LR             = 0.001
CLF_DROP           = 0.4
CLF_FULL_EPOCHS    = 500  # target for extrapolation

MODULE_LABELS = ['Hypergraph\nConstruction', 'SHE', 'KEFT', 'PCpredict\nClassifier']
MODULE_KEYS   = ['hg', 'she', 'keft', 'clf']
DATASETS      = ['DIP', 'BioGRID', 'Mann']


# 鈹€鈹€ memory helper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def profile(func, *args, **kwargs):
    """Return (result, wall_seconds, peak_ram_mb)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mem_before = _rss_mb()
    tracemalloc.start()

    t0     = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_after = _rss_mb()
    peak_ram  = max(mem_after - mem_before, peak_traced / 1024 ** 2)

    return result, elapsed, peak_ram


# 鈹€鈹€ fast preprocessing (not profiled) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _build_gene_index(sequence_df: pd.DataFrame) -> dict:
    """Build gene_symbol -> Entry mapping in O(n) instead of O(n*m)."""
    gene_to_entry = {}
    for _, row in sequence_df.iterrows():
        entry = row['Entry']
        field = str(row.get('Gene Names', ''))
        if field == 'nan' or not field:
            continue
        for gene in field.replace(';', ' ').replace(',', ' ').split():
            gene_to_entry[gene.strip().upper()] = entry
    return gene_to_entry


def load_ppi_raw(dataset: str) -> pd.DataFrame:
    if dataset.upper() == 'MANN':
        path = os.path.join(DATA_ROOT, 'PPI', 'Mann_PPI.csv')
        ppi  = pd.read_csv(path, sep=';')
        ppi  = (
            ppi.assign(target=ppi['target'].str.split(';'))
            .explode('target')
            .reset_index(drop=True)[['source', 'target']]
            .query('source != target')
        )
        rev  = ppi[['target', 'source']].copy()
        rev.columns = ['protein1', 'protein2']
        ppi.columns = ['protein1', 'protein2']
        return pd.concat([ppi, rev], axis=0).reset_index(drop=True)
    else:
        fname = dataset.lower() + '.txt'
        path  = os.path.join(
            DATA_ROOT, 'PPI', 'AdaPPI_Dataset', dataset.upper(), fname
        )
        ppi = pd.read_csv(path, sep='\t', header=None,
                          names=['protein1', 'protein2'])
        rev = ppi[['protein2', 'protein1']].copy()
        rev.columns = ['protein1', 'protein2']
        return pd.concat([ppi, rev], axis=0).reset_index(drop=True)


def fast_preprocessing(ppi_raw: pd.DataFrame,
                       sequence_df: pd.DataFrame,
                       gene_index: dict):
    """Vectorised version of preprocessing_PPI using pre-built gene index."""
    proteins = list(set(ppi_raw['protein1'].unique()) |
                    set(ppi_raw['protein2'].unique()))

    entries  = {p: gene_index.get(p.upper(), 'NA') for p in proteins}
    valid_entries = set(sequence_df['Entry'].values)

    prot_df = pd.DataFrame(
        [(sym, ent) for sym, ent in entries.items() if ent != 'NA' and ent in valid_entries],
        columns=['Gene_symbol', 'Entry']
    )

    valid_syms = set(prot_df['Gene_symbol'])
    ppi_filt   = ppi_raw[
        ppi_raw['protein1'].isin(valid_syms) &
        ppi_raw['protein2'].isin(valid_syms)
    ].copy()

    syms_after = set(ppi_filt['protein1'].unique()) | set(ppi_filt['protein2'].unique())
    prot_df    = prot_df[prot_df['Gene_symbol'].isin(syms_after)].copy()
    prot_df    = prot_df.sort_values('Gene_symbol').reset_index(drop=True)

    id_map            = {sym: i for i, sym in enumerate(prot_df['Gene_symbol'])}
    prot_df['ID']     = prot_df['Gene_symbol'].map(id_map)
    ppi_filt['protein1'] = ppi_filt['protein1'].map(id_map)
    ppi_filt['protein2'] = ppi_filt['protein2'].map(id_map)
    ppi_filt = ppi_filt.dropna().reset_index(drop=True)

    return ppi_filt, prot_df


def prepare_dataset(dataset: str, sequence: pd.DataFrame, gene_index: dict):
    """Preprocess PPI 鈫?(ppi_list, prot_dict, G)."""
    ppi_raw          = load_ppi_raw(dataset)
    ppi, prot_dict   = fast_preprocessing(ppi_raw, sequence, gene_index)

    ppi_list = ppi.values.tolist()
    ppi_list = [[int(a), int(b)] for a, b in ppi_list]
    ppi_list = Nested_list_dup(ppi_list)

    G = nx.Graph()
    G.add_edges_from(ppi_list)

    return ppi_list, prot_dict, G


# 鈹€鈹€ module A: hypergraph construction 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def module_hypergraph(G, ppi_list):
    cliques      = list(nx.find_cliques(G))
    unique_nodes = count_unique_elements(cliques)
    return {
        'num_vertices':     len(unique_nodes),
        'PPI_edge_list':    ppi_list,
        'PPI_cliques_list': cliques,
    }


# 鈹€鈹€ module B: SHE 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def module_she(edge_list_data):
    Z_v, _ = compute_she_embedding(
        edge_list_data, None,
        r=SHE_R, k=SHE_K, T=SHE_T, alpha=SHE_ALPHA, seed=SEED,
    )
    return Z_v


# 鈹€鈹€ module C: KEFT 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def module_keft(she_emb, prot_dict):
    Z_know, _ = build_knowledge_features(
        protein_dict=prot_dict,
        go_slim_path=GO_SLIM_PATH,
        series_path=SERIES_PATH,
        T=12, k=1, normalize=True,
        use_cache=False,
    )
    Z_fused = fuse_embeddings(
        she_emb=she_emb,
        know_features=Z_know,
        method='ae',
        out_dim=KEFT_OUT_DIM,
        epochs=KEFT_EPOCHS,
        lr=KEFT_LR,
        verbose=False,
    )
    return Z_fused


# 鈹€鈹€ module D: PCpredict classifier (1 fold) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def module_classifier(embedding, prot_dict, edge_list_data):
    ppi_dict = convert_ppi(edge_list_data['PPI_edge_list'])
    PC_raw   = load_txt_list(
        os.path.dirname(PC_PATH), '/' + os.path.basename(PC_PATH),
        display_flag=False,
    )
    p_map = dict(zip(prot_dict['Gene_symbol'], prot_dict['ID'].tolist()))

    PCs = []
    for pc in PC_raw:
        if len(pc) > 2 and set(pc).issubset(set(p_map.keys())):
            PCs.append(sorted([p_map[s] for s in pc]))

    if len(PCs) < 5:
        raise RuntimeError(f'Too few valid complexes: {len(PCs)}')

    np.random.seed(SEED)
    idx   = np.random.permutation(len(PCs))
    split = max(int(len(PCs) * 0.8), 1)
    train_pcs = [PCs[i] for i in idx[:split]]
    test_pcs  = [PCs[i] for i in idx[split:]]

    train_neg = negative_on_distribution(train_pcs, list(ppi_dict.keys()), 5)
    test_neg  = negative_on_distribution(test_pcs,  list(ppi_dict.keys()), 5)

    train_all = train_pcs + train_neg
    train_lbl = torch.cat([
        torch.ones(len(train_pcs),  1, dtype=torch.float),
        torch.zeros(len(train_neg), 1, dtype=torch.float),
    ])
    _, model = train_DNN(
        embedding, train_lbl, train_all, test_pcs + test_neg,
        CLF_MEASURE_EPOCHS, CLF_LR, CLF_DROP,
    )
    return model


# 鈹€鈹€ main profiling loop 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def run_profiling():
    print('Loading sequence data...', flush=True)
    sequence   = pd.read_csv(SEQUENCE_PATH, sep='\t')
    gene_index = _build_gene_index(sequence)
    print(f'Gene index built: {len(gene_index)} entries', flush=True)

    results = {ds: {} for ds in DATASETS}

    for dataset in DATASETS:
        print(f'\n{"=" * 60}', flush=True)
        print(f'Dataset: {dataset}', flush=True)
        print('=' * 60, flush=True)

        # preprocessing (not profiled)
        print('  Preprocessing...', flush=True)
        ppi_list, prot_dict, G = prepare_dataset(dataset, sequence, gene_index)
        print(f'  Proteins={G.number_of_nodes()}, Edges={G.number_of_edges()}', flush=True)

        # 鈹€鈹€ A: Hypergraph Construction 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        print('  [A] Hypergraph Construction...', flush=True)
        edge_list_data, t_hg, m_hg = profile(module_hypergraph, G, ppi_list)
        n_cliques = len(edge_list_data['PPI_cliques_list'])
        print(f'      {n_cliques} cliques | {t_hg:.2f}s | {m_hg:.1f} MB', flush=True)
        results[dataset]['hg'] = {'time': t_hg, 'mem': m_hg}

        # 鈹€鈹€ B: SHE 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        print('  [B] SHE...', flush=True)
        she_emb, t_she, m_she = profile(module_she, edge_list_data)
        print(f'      shape={she_emb.shape} | {t_she:.2f}s | {m_she:.1f} MB', flush=True)
        results[dataset]['she'] = {'time': t_she, 'mem': m_she}

        # 鈹€鈹€ C: KEFT 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        print('  [C] KEFT...', flush=True)
        she_gpu  = she_emb.to(try_gpu())
        fused, t_keft, m_keft = profile(module_keft, she_gpu, prot_dict)
        print(f'      shape={fused.shape} | {t_keft:.2f}s | {m_keft:.1f} MB', flush=True)
        results[dataset]['keft'] = {'time': t_keft, 'mem': m_keft}

        # 鈹€鈹€ D: PCpredict Classifier 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        print(f'  [D] PCpredict Classifier (1 fold, {CLF_MEASURE_EPOCHS} ep measured -> extrapolated to {CLF_FULL_EPOCHS} ep)...', flush=True)
        _, t_clf, m_clf = profile(
            module_classifier, fused, prot_dict, edge_list_data
        )
        scale   = CLF_FULL_EPOCHS / CLF_MEASURE_EPOCHS
        t_clf_e = t_clf * scale
        print(f'      measured {CLF_MEASURE_EPOCHS} ep: {t_clf:.2f}s => extrapolated {CLF_FULL_EPOCHS} ep: {t_clf_e:.1f}s | {m_clf:.1f} MB', flush=True)
        results[dataset]['clf'] = {'time': t_clf_e, 'mem': m_clf}

    return results


# 鈹€鈹€ visualization 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']


def plot_results(results: dict):
    n_modules = len(MODULE_KEYS)

    times = np.zeros((n_modules, len(DATASETS)))
    mems  = np.zeros((n_modules, len(DATASETS)))
    for j, ds in enumerate(DATASETS):
        for i, mk in enumerate(MODULE_KEYS):
            times[i, j] = results[ds][mk]['time']
            mems[i, j]  = results[ds][mk]['mem']

    x       = np.arange(len(DATASETS))
    width   = 0.18
    offsets = np.linspace(-(n_modules - 1) / 2,
                           (n_modules - 1) / 2,
                           n_modules) * width

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f'KSHL-PC Per-Module Runtime and Memory Consumption\n'
        f'(PCpredict Classifier: 1-fold, {CLF_FULL_EPOCHS}-epoch extrapolation)',
        fontsize=13, fontweight='bold',
    )

    for ax, data, ylabel, title in [
        (axes[0], times, 'Time (s)',       'Runtime (seconds)'),
        (axes[1], mems,  'Memory (MB)',    'Peak Memory Consumption (MB)'),
    ]:
        vmax = data.max()
        for i, (mk, label, color) in enumerate(zip(MODULE_KEYS, MODULE_LABELS, COLORS)):
            bars = ax.bar(x + offsets[i], data[i], width,
                          label=label, color=color, edgecolor='white', linewidth=0.6)
            for bar, val in zip(bars, data[i]):
                if val >= 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vmax * 0.01, 0.3),
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=7.5, color='#333333',
                    )
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, fontsize=11)
        ax.set_ylim(0, vmax * 1.22)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(
            handles=[Patch(color=c, label=lbl.replace('\n', ' '))
                     for c, lbl in zip(COLORS, MODULE_LABELS)],
            fontsize=8.5, loc='upper left',
        )

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        out = os.path.join(RESULTS_DIR, f'module_profile.{ext}')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved: {out}', flush=True)
    plt.close()


def print_summary(results: dict):
    col_w = 18
    header = f"{'Module':<28}" + ''.join(f'{ds:>{col_w}}' for ds in DATASETS)
    sep    = '=' * (28 + col_w * len(DATASETS))

    for metric, unit in [('time', 's'), ('mem', 'MB')]:
        title = 'Runtime (s)' if metric == 'time' else 'Peak Memory (MB)'
        print(f'\n{sep}')
        print(title)
        print(header)
        print('-' * (28 + col_w * len(DATASETS)))
        for mk, label in zip(MODULE_KEYS, MODULE_LABELS):
            row = f"{label.replace(chr(10), ' '):<28}"
            for ds in DATASETS:
                row += f"{results[ds][mk][metric]:>{col_w}.2f}"
            print(row)
    print(sep, flush=True)


if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    results = run_profiling()
    print_summary(results)
    plot_results(results)



