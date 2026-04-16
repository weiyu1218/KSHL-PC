import os
import sys
import json
import argparse
import csv
import numpy as np
import torch
import pandas as pd
from datetime import datetime

here = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(here, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils import try_gpu, convert_ppi, load_txt_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pooling aggregator ablation for PCpredict (mean vs max vs attention)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='Mann',
                        choices=['DIP', 'BioGRID', 'Mann'],
                        help='PPI dataset to use')
    parser.add_argument('--species', type=str, default='Saccharomyces_cerevisiae',
                        help='Species directory name')
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to data root')
    parser.add_argument('--embedding', type=str, default='fused',
                        choices=['fused', 'she'],
                        help='Which pre-computed embedding to use: fused (SHE+knowledge) or she (SHE only)')
    parser.add_argument('--n_repeats', type=int, default=30,
                        help='Num repeats for negative sampling')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--out_root', type=str, default='results/pooling_ablation',
                        help='Root output directory')
    parser.add_argument('--resume', action='store_true',
                        help='Skip variants with existing results')
    parser.add_argument('--pooling_types', type=str, nargs='+',
                        default=['mean', 'max', 'attention'],
                        choices=['mean', 'max', 'attention'],
                        help='Pooling variants to run')
    return parser.parse_args()


def load_embedding(args):
    feature_path = os.path.join(args.data_path, args.species, 'protein_feature')
    if args.embedding == 'fused':
        emb_path = os.path.join(feature_path, 'protein_feature_fused.pt')
        if not os.path.exists(emb_path):
            print(f"Fused embedding not found at {emb_path}, falling back to SHE.")
            emb_path = os.path.join(feature_path, 'protein_feature_SHE.pt')
    else:
        emb_path = os.path.join(feature_path, 'protein_feature_SHE.pt')

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding not found: {emb_path}")

    Embedding = torch.load(emb_path, map_location=try_gpu())
    print(f"Loaded embedding from {emb_path}, shape: {Embedding.shape}")
    return Embedding


def load_ppi(args):
    ppi_file = os.path.join(args.data_path, args.species, 'PPI', 'ID_Change_PPI.txt')
    if not os.path.exists(ppi_file):
        raise FileNotFoundError(f"PPI file not found: {ppi_file}")
    ppi_df = pd.read_csv(ppi_file, sep='\t', header=None, names=['protein1', 'protein2'])
    ppi_list = ppi_df.values.tolist()
    PPI_dict = convert_ppi(ppi_list)
    print(f"Loaded PPI: {len(ppi_list)} edges, {len(PPI_dict)} proteins")
    return PPI_dict


def load_protein_dict(args):
    protein_list_file = os.path.join(args.data_path, args.species, 'Gene_Entry_ID_list', 'Protein_list.csv')
    if not os.path.exists(protein_list_file):
        raise FileNotFoundError(f"Protein list not found: {protein_list_file}")
    protein_df = pd.read_csv(protein_list_file, sep='\t', header=None)
    protein_df.columns = ['Gene_symbol', 'Entry', 'ID'] if protein_df.shape[1] == 3 else ['Gene_symbol', 'ID']
    protein_dict = dict(zip(protein_df['Gene_symbol'], protein_df['ID']))
    print(f"Loaded protein dict: {len(protein_dict)} proteins")
    return protein_dict


def load_complexes(args):
    pc_path = os.path.join(args.data_path, args.species, 'protein_complex')
    PC = load_txt_list(pc_path, '/AdaPPI_golden_standard.txt')
    print(f"Loaded {len(PC)} protein complexes")
    return PC


def run_one_pooling(pooling_type, PC, protein_dict, PPI_dict, Embedding, args, out_dir):
    from Train_PC import HGC_DNN

    result_json = os.path.join(out_dir, f'results_{args.dataset}_{pooling_type}.json')
    if args.resume and os.path.exists(result_json):
        print(f"[skip] {pooling_type}: results exist at {result_json}")
        return True, result_json

    os.makedirs(out_dir, exist_ok=True)
    export_dir = os.path.join(out_dir, 'preds')

    print(f"\n{'='*60}")
    print(f"Pooling variant: {pooling_type}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    results = HGC_DNN(
        PC, protein_dict, PPI_dict, Embedding,
        n_repeats=args.n_repeats,
        export_preds=True,
        export_dir=export_dir,
        export_val=True,
        dataset=args.dataset,
        variant=f'pooling_{pooling_type}',
        pooling_type=pooling_type,
    )

    results['config'] = {
        'dataset': args.dataset,
        'pooling_type': pooling_type,
        'embedding': args.embedding,
        'n_repeats': args.n_repeats,
        'seed': args.seed,
        'timestamp': str(datetime.now()),
    }

    with open(result_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {result_json}")
    return True, result_json


def parse_metrics(path):
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
        m = obj.get('binary_metrics', {})
        c = obj.get('complex_metrics', {})
        return {
            'auprc': m.get('auprc'),
            'auprc_se': m.get('auprc_se'),
            'auroc': m.get('auroc'),
            'auroc_se': m.get('auroc_se'),
            'f1': m.get('f1'),
            'f1_se': m.get('f1_se'),
            'precision': m.get('precision'),
            'recall': m.get('recall'),
            'complex_f1': c.get('f1'),
            'complex_f1_se': c.get('f1_se'),
            'complex_acc': c.get('acc'),
            'complex_sn': c.get('sn'),
            'complex_ppv': c.get('ppv'),
        }
    except Exception:
        return {}


def write_summary(out_root, records):
    summary_csv = os.path.join(out_root, 'summary_pooling_ablation.csv')
    fieldnames = [
        'pooling_type', 'result_path', 'success',
        'auprc', 'auprc_se', 'auroc', 'auroc_se',
        'f1', 'f1_se', 'precision', 'recall',
        'complex_f1', 'complex_f1_se', 'complex_acc', 'complex_sn', 'complex_ppv',
        'timestamp',
    ]
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k, '') for k in fieldnames})
    print(f"\nSummary saved to: {summary_csv}")
    return summary_csv


def print_comparison_table(records):
    print(f"\n{'='*80}")
    print("Pooling Aggregator Comparison")
    print(f"{'='*80}")
    header = f"{'Pooling':<12} {'AUPRC':>10} {'AUROC':>10} {'F1':>10} {'Complex-F1':>12} {'Complex-Acc':>13}"
    print(header)
    print('-' * 80)
    for rec in records:
        if not rec.get('success'):
            continue
        auprc = f"{rec['auprc']:.4f}" if rec.get('auprc') is not None else 'N/A'
        auroc = f"{rec['auroc']:.4f}" if rec.get('auroc') is not None else 'N/A'
        f1 = f"{rec['f1']:.4f}" if rec.get('f1') is not None else 'N/A'
        cf1 = f"{rec['complex_f1']:.4f}" if rec.get('complex_f1') is not None else 'N/A'
        cacc = f"{rec['complex_acc']:.4f}" if rec.get('complex_acc') is not None else 'N/A'
        print(f"{rec['pooling_type']:<12} {auprc:>10} {auroc:>10} {f1:>10} {cf1:>12} {cacc:>13}")
    print(f"{'='*80}")


def main():
    base = parse_args()

    np.random.seed(base.seed)
    torch.manual_seed(base.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base.seed)

    Embedding = load_embedding(base)
    PPI_dict = load_ppi(base)
    protein_dict = load_protein_dict(base)
    PC = load_complexes(base)

    os.makedirs(base.out_root, exist_ok=True)
    records = []

    for pooling_type in base.pooling_types:
        variant_dir = os.path.join(base.out_root, pooling_type)
        ok, path = run_one_pooling(pooling_type, PC, protein_dict, PPI_dict, Embedding, base, variant_dir)
        metrics = parse_metrics(path) if ok else {}
        records.append({
            'pooling_type': pooling_type,
            'result_path': path,
            'success': ok,
            **metrics,
            'timestamp': str(datetime.now()),
        })

    summary_csv = write_summary(base.out_root, records)
    print_comparison_table(records)
    print(f"\nAll done. Summary: {summary_csv}")


if __name__ == '__main__':
    main()



