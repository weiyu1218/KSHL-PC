"""
Volume Scale Sensitivity Experiment
====================================
Compares two volume (vol) definitions for the structural basis F in Eq.(7):

  vol_traditional : vol = H.sum() = sum_v d(v)
                    Classical spectral hypergraph volume (Zhou et al., NeurIPS 2006).

  vol_normalized  : vol = (D_e^{-1/2} H D_v^{-1/2}).sum()
                    Sum of the doubly-normalised incidence matrix stated in the paper.

Efficient design: per dataset, data loading and clique enumeration happen ONCE.
Only the SHE embedding (fast, seconds) differs between the two vol_types.

Usage (from project root):
    python experiments/vol_scale_comparison.py --datasets DIP BioGRID Mann \
        --n_repeats 30 --seed 42 --out_root results/vol_scale
"""

import os
import sys
import json
import csv
import argparse
import time
import numpy as np
import pandas as pd
import networkx as nx
import torch
from types import SimpleNamespace
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils import (preprocessing_PPI, Nested_list_dup, count_unique_elements,
                   convert_ppi, load_txt_list, try_gpu)
from she_embed import compute_she_embedding
from knowledge_features import build_knowledge_features
from fusion import fuse_embeddings
from Train_PC import HGC_DNN


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare traditional vs. normalised volume for SHE structural basis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', nargs='+', default=['DIP', 'BioGRID', 'Mann'],
                        choices=['DIP', 'BioGRID', 'Mann'])
    parser.add_argument('--species', type=str, default='Saccharomyces_cerevisiae')
    parser.add_argument('--data_path', type=str, default='data')

    parser.add_argument('--she_r', type=int, default=64)
    parser.add_argument('--she_k', type=int, default=64)
    parser.add_argument('--she_T', type=int, default=10)
    parser.add_argument('--she_alpha', type=float, default=0.1)

    parser.add_argument('--n_repeats', type=int, default=30)
    parser.add_argument('--neg_ratio', type=int, default=5)
    parser.add_argument('--pooling_type', type=str, default='mean',
                        choices=['mean', 'max', 'attention'])
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--fusion_method', type=str, default='ae', choices=['ae', 'concat'])
    parser.add_argument('--fusion_dim', type=int, default=128)
    parser.add_argument('--fusion_epochs', type=int, default=100)
    parser.add_argument('--fusion_lr', type=float, default=1e-3)
    parser.add_argument('--keft_T', type=int, default=12)
    parser.add_argument('--keft_k', type=int, default=1)

    parser.add_argument('--out_root', type=str, default='results/vol_scale')
    parser.add_argument('--resume', action='store_true',
                        help='Skip configurations whose result JSON already exists')
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Data loading (expensive; done ONCE per dataset)                              #
# --------------------------------------------------------------------------- #

def load_dataset(dataset, data_path, species):
    """Load PPI data and enumerate maximal cliques. Returns shared data structures."""
    print(f"\n[data] Loading {dataset} ...")
    t0 = time.perf_counter()

    ppi_dataset = dataset.upper()
    species_dir = os.path.join(data_path, species)
    seq_path = os.path.join(species_dir, 'protein_feature',
                            'uniprot-sequences-2023.05.10-01.31.31.11.tsv')
    Sequence = pd.read_csv(seq_path, sep='\t')

    if ppi_dataset == 'MANN':
        ppi_file = os.path.join(species_dir, 'PPI', 'Mann_PPI.csv')
        PPI = pd.read_csv(ppi_file, sep=';')
        PPI = (PPI.assign(target=PPI['target'].str.split(';'))
               .explode('target').reset_index(drop=True)
               [['source', 'target']].query('source != target'))
        PPI_rev = PPI[['target', 'source']].copy()
        PPI_rev.columns = ['protein1', 'protein2']
        PPI.columns = ['protein1', 'protein2']
        PPI = pd.concat([PPI, PPI_rev], axis=0).reset_index(drop=True)
    else:
        ppi_file = os.path.join(species_dir, 'PPI', 'AdaPPI_Dataset',
                                ppi_dataset, f'{ppi_dataset.lower()}.txt')
        PPI = pd.read_csv(ppi_file, sep='\t', header=None, names=['protein1', 'protein2'])
        PPI_rev = PPI[['protein2', 'protein1']].copy()
        PPI_rev.columns = ['protein1', 'protein2']
        PPI = pd.concat([PPI, PPI_rev], axis=0).reset_index(drop=True)

    PPI, Protein_dict = preprocessing_PPI(PPI, Sequence)
    PPI_list = Nested_list_dup(PPI.values.tolist())

    print(f"[data] Building PPI graph and enumerating cliques ...")
    G = nx.Graph()
    G.add_edges_from(PPI_list)
    cliques = list(nx.find_cliques(G))
    unique_elements = count_unique_elements(cliques)

    edge_list_data = {
        "num_vertices": len(unique_elements),
        "PPI_edge_list": PPI_list,
        "PPI_cliques_list": cliques,
    }

    PPI_dict = convert_ppi(PPI_list)

    pc_path = os.path.join(species_dir, 'protein_complex')
    PC = load_txt_list(pc_path, '/AdaPPI_golden_standard.txt')
    protein_dict = dict(zip(Protein_dict['Gene_symbol'], list(Protein_dict['ID'])))

    elapsed = time.perf_counter() - t0
    print(f"[data] Loaded {dataset}: {len(unique_elements)} proteins, "
          f"{len(cliques)} cliques, elapsed {elapsed:.1f}s")

    return edge_list_data, Protein_dict, PPI_dict, PC, protein_dict


def load_knowledge_features(Protein_dict, data_path, species, keft_T, keft_k):
    """Build (or load cached) knowledge features."""
    species_dir = os.path.join(data_path, species)
    go_slim_path = os.path.join(species_dir, 'GO', 'go_slim_mapping.tab.txt')
    series_path = os.path.join(species_dir, 'expression', 'series_matrix.txt')

    if not os.path.exists(go_slim_path) or not os.path.exists(series_path):
        print("[know] Knowledge files not found; skipping fusion.")
        return None, None

    Z_know, metadata = build_knowledge_features(
        protein_dict=Protein_dict,
        go_slim_path=go_slim_path,
        series_path=series_path,
        T=keft_T,
        k=keft_k,
        normalize=True,
        use_cache=True,
    )
    return Z_know, metadata


# --------------------------------------------------------------------------- #
# Single-vol-type evaluation                                                   #
# --------------------------------------------------------------------------- #

def run_vol_type(vol_type, edge_list_data, Z_know, Protein_dict,
                 PPI_dict, PC, protein_dict, args, out_dir, resume):
    """Compute SHE 鈫?fuse 鈫?evaluate for one vol_type. Returns metrics dict."""
    result_json = os.path.join(out_dir, f'results_SHE_{vol_type}.json')
    if resume and os.path.exists(result_json):
        print(f"[skip] {vol_type}: result exists at {result_json}")
        with open(result_json) as f:
            return json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    # --- SHE embedding (seconds) ---
    print(f"\n[she] Computing embedding (vol_type={vol_type}) ...")
    t0 = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    Z_v, emb_time = compute_she_embedding(
        edge_list_data, None,
        r=args.she_r, k=args.she_k,
        T=args.she_T, alpha=args.she_alpha,
        seed=args.seed, vol_type=vol_type,
    )
    print(f"[she] Embedding done in {emb_time:.2f}s, shape={Z_v.shape}")

    # --- Knowledge fusion ---
    if Z_know is not None:
        print(f"[fuse] Fusing with knowledge features ...")
        Embedding = fuse_embeddings(
            she_emb=Z_v.to(device=try_gpu()),
            know_features=Z_know,
            method=args.fusion_method,
            out_dim=args.fusion_dim,
            epochs=args.fusion_epochs,
            lr=args.fusion_lr,
            verbose=True,
        )
    else:
        Embedding = Z_v.to(device=try_gpu())

    # --- DNN evaluation ---
    print(f"[eval] Running HGC_DNN (n_repeats={args.n_repeats}) ...")
    results = HGC_DNN(
        PC, protein_dict, PPI_dict, Embedding,
        n_repeats=args.n_repeats,
        export_preds=False,
        export_val=False,
        dataset=args.dataset,
        variant=f'vol_{vol_type}',
        neg_ratio=args.neg_ratio,
        pooling_type=args.pooling_type,
    )

    # Attach metadata
    results['vol_type'] = vol_type
    results['emb_time'] = emb_time
    results['total_time'] = time.perf_counter() - t0

    with open(result_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[save] Results written to {result_json}")

    return results


# --------------------------------------------------------------------------- #
# Summary helpers                                                              #
# --------------------------------------------------------------------------- #

METRIC_KEYS = ['f1', 'auprc', 'auroc', 'precision', 'recall',
               'complex_f1', 'complex_acc', 'complex_sn', 'complex_ppv']


def extract_metrics(results):
    bm = results.get('binary_metrics', {})
    cm = results.get('complex_metrics', {})
    return {
        'f1':          bm.get('f1'),
        'auprc':       bm.get('auprc'),
        'auroc':       bm.get('auroc'),
        'precision':   bm.get('precision'),
        'recall':      bm.get('recall'),
        'complex_f1':  cm.get('f1'),
        'complex_acc': cm.get('acc'),
        'complex_sn':  cm.get('sn'),
        'complex_ppv': cm.get('ppv'),
    }


def write_raw_csv(records, out_root):
    path = os.path.join(out_root, 'vol_scale_raw.csv')
    fieldnames = ['dataset', 'vol_type'] + METRIC_KEYS + ['emb_time', 'timestamp']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    print(f"\nRaw results: {path}")


def write_diff_csv(diff_records, out_root):
    path = os.path.join(out_root, 'vol_scale_diff.csv')
    diff_keys = [f'delta_{k}' for k in METRIC_KEYS]
    fieldnames = ['dataset', 'timestamp'] + diff_keys
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for rec in diff_records:
            w.writerow(rec)
    print(f"Differences: {path}")


def print_summary(records, diff_records):
    print(f"\n{'='*72}")
    print("  Volume Scale Sensitivity: Performance Comparison")
    print(f"{'='*72}")
    print(f"{'Dataset':<12} {'vol_type':<14} {'F1':>8} {'AUPRC':>8} {'Cplx-F1':>10}")
    print('-' * 54)
    for r in records:
        f1  = f"{r['f1']:.4f}"         if r.get('f1')         is not None else 'N/A'
        au  = f"{r['auprc']:.4f}"      if r.get('auprc')      is not None else 'N/A'
        cf1 = f"{r['complex_f1']:.4f}" if r.get('complex_f1') is not None else 'N/A'
        print(f"{r['dataset']:<12} {r['vol_type']:<14} {f1:>8} {au:>8} {cf1:>10}")

    print(f"\n{'='*72}")
    print("  Absolute Differences |traditional - normalized|")
    print(f"{'='*72}")
    print(f"{'Dataset':<12} {'|dF1|':>10} {'|dAUPRC|':>10} {'|dCplx-F1|':>12}")
    print('-' * 46)
    for r in diff_records:
        df1 = f"{r['delta_f1']:.4f}"         if r.get('delta_f1')         is not None else 'N/A'
        dau = f"{r['delta_auprc']:.4f}"      if r.get('delta_auprc')      is not None else 'N/A'
        dcf = f"{r['delta_complex_f1']:.4f}" if r.get('delta_complex_f1') is not None else 'N/A'
        print(f"{r['dataset']:<12} {df1:>10} {dau:>10} {dcf:>12}")
    print(f"{'='*72}\n")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    vol_types = ['traditional', 'normalized']
    all_records = []
    all_diff = []

    for dataset in args.datasets:
        args.dataset = dataset  # used inside run_vol_type for export labelling

        dataset_out = os.path.join(args.out_root, dataset)

        # If resume and both vol_type results already exist, skip data loading entirely
        if args.resume:
            cached = {}
            for vt in vol_types:
                rj = os.path.join(dataset_out, vt, f'results_SHE_{vt}.json')
                if os.path.exists(rj):
                    with open(rj) as f:
                        cached[vt] = json.load(f)
            if len(cached) == len(vol_types):
                print(f"\n[skip] {dataset}: all vol_type results exist, skipping data loading.")
                results_by_vol = {vt: extract_metrics(cached[vt]) for vt in vol_types}
                for vt in vol_types:
                    m = results_by_vol[vt]
                    all_records.append({
                        'dataset': dataset, 'vol_type': vt,
                        'emb_time': cached[vt].get('emb_time'),
                        'timestamp': str(datetime.now()), **m,
                    })
                trad = results_by_vol.get('traditional', {})
                norm = results_by_vol.get('normalized', {})
                diff_row = {'dataset': dataset, 'timestamp': str(datetime.now())}
                for k in METRIC_KEYS:
                    t_val, n_val = trad.get(k), norm.get(k)
                    diff_row[f'delta_{k}'] = abs(t_val - n_val) if (t_val is not None and n_val is not None) else None
                all_diff.append(diff_row)
                continue

        # --- Expensive step: done ONCE per dataset ---
        edge_list_data, Protein_dict, PPI_dict, PC, protein_dict = load_dataset(
            dataset, args.data_path, args.species
        )

        # --- Knowledge features: done ONCE per dataset (cached) ---
        Z_know, _ = load_knowledge_features(
            Protein_dict, args.data_path, args.species, args.keft_T, args.keft_k
        )

        os.makedirs(dataset_out, exist_ok=True)

        results_by_vol = {}
        for vol_type in vol_types:
            out_dir = os.path.join(dataset_out, vol_type)
            res = run_vol_type(
                vol_type, edge_list_data, Z_know,
                Protein_dict, PPI_dict, PC, protein_dict,
                args, out_dir, args.resume,
            )
            m = extract_metrics(res)
            results_by_vol[vol_type] = m
            all_records.append({
                'dataset': dataset,
                'vol_type': vol_type,
                'emb_time': res.get('emb_time'),
                'timestamp': str(datetime.now()),
                **m,
            })

        # Compute differences
        trad = results_by_vol.get('traditional', {})
        norm = results_by_vol.get('normalized', {})
        diff_row = {'dataset': dataset, 'timestamp': str(datetime.now())}
        for k in METRIC_KEYS:
            t_val, n_val = trad.get(k), norm.get(k)
            diff_row[f'delta_{k}'] = abs(t_val - n_val) if (t_val is not None and n_val is not None) else None
        all_diff.append(diff_row)

    # Write outputs
    write_raw_csv(all_records, args.out_root)
    write_diff_csv(all_diff, args.out_root)
    print_summary(all_records, all_diff)

    combined = {
        'config': {
            'datasets': args.datasets,
            'vol_types': vol_types,
            'n_repeats': args.n_repeats,
            'seed': args.seed,
            'she_r': args.she_r,
            'she_k': args.she_k,
            'she_T': args.she_T,
            'she_alpha': args.she_alpha,
            'timestamp': str(datetime.now()),
        },
        'results': all_records,
        'differences': all_diff,
    }
    combined_json = os.path.join(args.out_root, 'vol_scale_combined.json')
    with open(combined_json, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"Combined JSON: {combined_json}")


if __name__ == '__main__':
    main()



