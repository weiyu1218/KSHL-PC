import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict


def build_go_labels(go_slim_path, protein_list_path, aspect='BP', min_freq=30):
    if not os.path.exists(go_slim_path):
        raise FileNotFoundError(f"GO slim file not found: {go_slim_path}")
    if not os.path.exists(protein_list_path):
        raise FileNotFoundError(f"Protein list file not found: {protein_list_path}")

    protein_dict = pd.read_csv(protein_list_path, sep='\t', header=None,
                               names=['Gene_symbol', 'UniProt', 'ID'])
    symbol_to_id = dict(zip(protein_dict['Gene_symbol'], protein_dict['ID']))
    n_proteins = len(protein_dict)

    if aspect == 'BP':
        go_tag = 'P'
        root_term_id = 'GO:0008150'
        root_term_name = 'biological_process'
    elif aspect == 'MF':
        go_tag = 'F'
        root_term_id = 'GO:0003674'
        root_term_name = 'molecular_function'
    else:
        raise ValueError(f"Invalid aspect: {aspect}. Must be 'BP' or 'MF'")

    protein_go = defaultdict(set)
    go_term_counts = defaultdict(int)
    go_id_to_name = {}

    with open(go_slim_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue

            gene_symbol = parts[0].upper()
            aspect_tag = parts[3]
            go_name = parts[4]
            go_id = parts[5]

            if aspect_tag != go_tag:
                continue

            if go_id == root_term_id or go_name == root_term_name:
                continue

            if gene_symbol not in symbol_to_id:
                continue

            protein_id = symbol_to_id[gene_symbol]

            if go_id and go_name:
                protein_go[protein_id].add(go_id)
                go_term_counts[go_id] += 1
                go_id_to_name[go_id] = go_name

    filtered_terms = {go_id for go_id, count in go_term_counts.items() if count >= min_freq}

    if len(filtered_terms) == 0:
        raise ValueError(f"No GO terms found with frequency >= {min_freq} for aspect {aspect}")

    term_list = sorted(list(filtered_terms))
    term_to_idx = {term: idx for idx, term in enumerate(term_list)}
    term_names = [go_id_to_name[term] for term in term_list]

    label_matrix = np.zeros((n_proteins, len(term_list)), dtype=np.float32)

    for protein_id, terms in protein_go.items():
        for term in terms:
            if term in term_to_idx:
                label_matrix[protein_id, term_to_idx[term]] = 1.0

    proteins_with_labels = np.sum(label_matrix.sum(axis=1) > 0)
    coverage = proteins_with_labels / n_proteins

    print(f"GO-{aspect} label matrix: {label_matrix.shape}")
    print(f"  Total terms before filtering: {len(go_term_counts)}")
    print(f"  Terms after freq>={min_freq} filtering: {len(term_list)}")
    print(f"  Proteins with labels: {proteins_with_labels}/{n_proteins} ({coverage:.2%})")
    print(f"  Average labels per protein: {label_matrix.sum() / proteins_with_labels:.2f}")
    print(f"  Sparsity: {1 - label_matrix.sum() / label_matrix.size:.4f}")

    return label_matrix, term_list, term_names


def main():
    parser = argparse.ArgumentParser(description='Build GO-BP/MF multi-label classification labels')
    parser.add_argument('--aspect', required=True, choices=['BP', 'MF'],
                        help='GO aspect: BP (Biological Process) or MF (Molecular Function)')
    parser.add_argument('--go_slim_path', required=True,
                        help='Path to go_slim_mapping.tab.txt')
    parser.add_argument('--protein_list', default=None,
                        help='Path to Protein_list.csv (default: auto-detect from go_slim_path)')
    parser.add_argument('--min_freq', type=int, default=30,
                        help='Minimum term frequency to keep (default: 30)')
    parser.add_argument('--out_npz', required=True,
                        help='Output path for labels .npz file')
    args = parser.parse_args()

    if args.protein_list is None:
        data_root = os.path.dirname(os.path.dirname(args.go_slim_path))
        args.protein_list = os.path.join(data_root, 'Gene_Entry_ID_list', 'Protein_list.csv')
        print(f"Auto-detected protein_list: {args.protein_list}")

    label_matrix, term_ids, term_names = build_go_labels(
        args.go_slim_path,
        args.protein_list,
        aspect=args.aspect,
        min_freq=args.min_freq
    )

    label_sparse = sp.csr_matrix(label_matrix)

    os.makedirs(os.path.dirname(args.out_npz) or '.', exist_ok=True)

    np.savez_compressed(
        args.out_npz,
        labels=label_sparse.data,
        indices=label_sparse.indices,
        indptr=label_sparse.indptr,
        shape=label_sparse.shape,
        term_ids=np.array(term_ids),
        term_names=np.array(term_names)
    )

    print(f"\nLabels saved to: {args.out_npz}")
    print(f"  Shape: {label_matrix.shape}")
    print(f"  Format: scipy sparse CSR matrix")


if __name__ == '__main__':
    main()



