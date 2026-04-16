"""
Negative sample ratio sensitivity experiment.

Runs the full KSHL-PC classification pipeline using pre-computed fused embeddings
for neg:pos ratios in {1:1, 3:1, 5:1}. For each ratio:
  - Training/validation/test negative samples are generated independently per fold
  - Decision threshold theta is calibrated on the validation set per fold (same
    protocol as the main experiments), decoupling threshold selection from the
    choice of neg_ratio
  - Reports AUPRC (ranking-based) and complex-level metrics (Precision, Recall,
    F1, Acc, Sn, PPV) as mean +/- s.e.m. across all repeats

Usage (run from repo root):
  python experiments/neg_ratio_sensitivity.py [--dataset Mann] [--n_repeats 10] \
      [--out_dir results/neg_ratio_sensitivity] [--seed 42] [--data_path data] \
      [--species Saccharomyces_cerevisiae] [--embedding fused]
"""

import os
import sys
import json
import argparse
import csv
from datetime import datetime

import numpy as np
import torch
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup: ensure src/ is importable
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(HERE, '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import convert_ppi, load_txt_list, try_gpu
from Train_PC import HGC_DNN


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_protein_dict(species_dir):
    """Load Gene_symbol -> integer ID mapping from Protein_list.csv."""
    path = os.path.join(species_dir, 'Gene_Entry_ID_list', 'Protein_list.csv')
    df = pd.read_csv(path, sep='\t', header=None, names=['Gene_symbol', 'Entry', 'ID'])
    return dict(zip(df['Gene_symbol'], df['ID'].astype(int)))


def load_ppi_dict(species_dir):
    """Load PPI edge list and return adjacency dict {protein_id: [neighbor_ids]}."""
    path = os.path.join(species_dir, 'PPI', 'ID_Change_PPI.txt')
    ppi_df = pd.read_csv(path, sep='\t', header=None, names=['p1', 'p2'])
    ppi_list = ppi_df.values.tolist()
    return convert_ppi(ppi_list)


def load_protein_complexes(species_dir):
    """Load positive protein complexes from AdaPPI_golden_standard.txt."""
    path = os.path.join(species_dir, 'protein_complex')
    return load_txt_list(path, '/AdaPPI_golden_standard.txt', display_flag=False)


def load_embedding(species_dir, embedding_type):
    """Load pre-computed protein embeddings."""
    feature_dir = os.path.join(species_dir, 'protein_feature')
    if embedding_type == 'fused':
        pt_path = os.path.join(feature_dir, 'protein_feature_fused.pt')
    elif embedding_type == 'she':
        pt_path = os.path.join(feature_dir, 'protein_feature_SHE.pt')
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}. Choose 'fused' or 'she'.")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Embedding file not found: {pt_path}")

    emb = torch.load(pt_path, map_location='cpu')
    return emb.to(device=try_gpu())


# ---------------------------------------------------------------------------
# Result I/O helpers
# ---------------------------------------------------------------------------

def parse_metrics(result_dict):
    m = result_dict.get('binary_metrics', {})
    c = result_dict.get('complex_metrics', {})
    return {
        'auprc':       m.get('auprc'),
        'auprc_se':    m.get('auprc_se'),
        'auroc':       m.get('auroc'),
        'auroc_se':    m.get('auroc_se'),
        'f1':          m.get('f1'),
        'f1_se':       m.get('f1_se'),
        'complex_precision':  c.get('precision'),
        'complex_recall':     c.get('recall'),
        'complex_f1':         c.get('f1'),
        'complex_acc':        c.get('acc'),
        'complex_sn':         c.get('sn'),
        'complex_ppv':        c.get('ppv'),
        'complex_precision_se': c.get('precision_se'),
        'complex_recall_se':    c.get('recall_se'),
        'complex_f1_se':        c.get('f1_se'),
        'complex_acc_se':       c.get('acc_se'),
        'complex_sn_se':        c.get('sn_se'),
        'complex_ppv_se':       c.get('ppv_se'),
    }


def write_summary(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'summary_neg_ratio.csv')
    fieldnames = [
        'neg_ratio', 'n_repeats',
        'auprc', 'auprc_se',
        'auroc', 'auroc_se',
        'f1', 'f1_se',
        'complex_precision', 'complex_precision_se',
        'complex_recall', 'complex_recall_se',
        'complex_f1', 'complex_f1_se',
        'complex_acc', 'complex_acc_se',
        'complex_sn', 'complex_sn_se',
        'complex_ppv', 'complex_ppv_se',
        'timestamp',
    ]
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    return summary_path


def print_table(records):
    header = f"{'Ratio':>7}  {'AUPRC':>10}  {'AUROC':>10}  {'cplx-F1':>10}  {'cplx-Acc':>10}  {'cplx-Sn':>10}  {'cplx-PPV':>10}"
    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in records:
        ratio_str = f"neg:{r['neg_ratio']}:1"

        def fmt(v, se=None):
            if v is None:
                return f"{'N/A':>10}"
            if se is not None:
                return f"{v:.4f}+/-{se:.4f}"
            return f"{v:.4f}"

        print(f"{ratio_str:>7}  "
              f"{fmt(r.get('auprc'), r.get('auprc_se')):>18}  "
              f"{fmt(r.get('auroc'), r.get('auroc_se')):>18}  "
              f"{fmt(r.get('complex_f1'), r.get('complex_f1_se')):>18}  "
              f"{fmt(r.get('complex_acc'), r.get('complex_acc_se')):>18}  "
              f"{fmt(r.get('complex_sn'), r.get('complex_sn_se')):>18}  "
              f"{fmt(r.get('complex_ppv'), r.get('complex_ppv_se')):>18}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Negative sample ratio sensitivity experiment for KSHL-PC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_path', type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--species', type=str, default='Saccharomyces_cerevisiae',
                        help='Species directory name')
    parser.add_argument('--dataset', type=str, default='Mann',
                        help='PPI dataset label (for output naming only)')
    parser.add_argument('--embedding', type=str, default='fused',
                        choices=['fused', 'she'],
                        help="Which pre-computed embedding to use: 'fused' (KSHL-PC full) or 'she'")
    parser.add_argument('--neg_ratios', type=int, nargs='+', default=[1, 3, 5],
                        help='List of neg:pos ratios to evaluate')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='Number of repeats with different negative sampling per ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed')
    parser.add_argument('--out_dir', type=str, default='results/neg_ratio_sensitivity',
                        help='Output directory for results')
    parser.add_argument('--resume', action='store_true',
                        help='Skip ratios with existing result JSON')
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to repo root (parent of experiments/)
    repo_root = os.path.abspath(os.path.join(HERE, '..'))
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(repo_root, args.data_path)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(repo_root, args.out_dir)
    species_dir = os.path.join(data_path, args.species)

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading data from: {species_dir}")
    protein_dict = load_protein_dict(species_dir)
    ppi_dict = load_ppi_dict(species_dir)
    pc_list = load_protein_complexes(species_dir)
    embedding = load_embedding(species_dir, args.embedding)

    print(f"Proteins: {len(protein_dict)}, PPI nodes: {len(ppi_dict)}, "
          f"Complexes: {len(pc_list)}, Embedding shape: {tuple(embedding.shape)}")
    print(f"Embedding device: {embedding.device}")
    print(f"Neg ratios to evaluate: {args.neg_ratios}")
    print(f"Repeats per ratio: {args.n_repeats}")
    print()

    os.makedirs(out_dir, exist_ok=True)

    records = []

    for ratio in args.neg_ratios:
        ratio_label = f"neg_ratio_{ratio}"
        result_json = os.path.join(out_dir, f'results_{args.dataset}_{ratio_label}.json')

        if args.resume and os.path.exists(result_json):
            print(f"[skip] neg_ratio={ratio}: results exist at {result_json}")
            with open(result_json) as f:
                saved = json.load(f)
            metrics = parse_metrics(saved)
            records.append({'neg_ratio': ratio, 'n_repeats': args.n_repeats, **metrics,
                            'timestamp': saved.get('timestamp', '')})
            continue

        print(f"\n{'='*60}")
        print(f"Running neg:pos = {ratio}:1")
        print(f"{'='*60}")

        export_dir = os.path.join(out_dir, 'preds')
        results = HGC_DNN(
            pc_list, protein_dict, ppi_dict, embedding,
            n_repeats=args.n_repeats,
            export_preds=False,
            export_dir=export_dir,
            export_val=False,
            dataset=args.dataset,
            variant=ratio_label,
            neg_ratio=ratio,
        )

        results['neg_ratio'] = ratio
        results['n_repeats'] = args.n_repeats
        results['embedding'] = args.embedding
        results['dataset'] = args.dataset
        results['timestamp'] = str(datetime.now())

        with open(result_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {result_json}")

        metrics = parse_metrics(results)
        records.append({'neg_ratio': ratio, 'n_repeats': args.n_repeats, **metrics,
                        'timestamp': results['timestamp']})

    summary_path = write_summary(records, out_dir)
    print(f"\nSummary CSV saved to: {summary_path}")
    print_table(records)


if __name__ == '__main__':
    main()



