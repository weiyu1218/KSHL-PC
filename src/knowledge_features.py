import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple

def build_go_cc_matrix(go_slim_path: str, protein_dict: pd.DataFrame) -> Tuple[np.ndarray, List[str], float]:
    symbol_to_id = dict(zip(protein_dict['Gene_symbol'], protein_dict['ID']))
    n_proteins = len(protein_dict)

    if not os.path.exists(go_slim_path):
        raise FileNotFoundError(f"GO slim file not found: {go_slim_path}")

    go_cc_terms = set()
    protein_go_cc = {}

    with open(go_slim_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue

            gene_symbol = parts[0].upper()
            go_tag = parts[3]
            go_term = parts[4]

            if go_tag == 'C' and gene_symbol in symbol_to_id:
                protein_id = symbol_to_id[gene_symbol]
                if protein_id not in protein_go_cc:
                    protein_go_cc[protein_id] = set()
                if go_term:
                    protein_go_cc[protein_id].add(go_term)
                    go_cc_terms.add(go_term)

    go_cc_list = sorted(list(go_cc_terms))
    go_cc_to_idx = {term: idx for idx, term in enumerate(go_cc_list)}

    cc_matrix = np.zeros((n_proteins, len(go_cc_list)), dtype=np.float32)

    for protein_id, terms in protein_go_cc.items():
        for term in terms:
            if term in go_cc_to_idx:
                cc_matrix[protein_id, go_cc_to_idx[term]] = 1.0

    coverage = np.sum(cc_matrix.sum(axis=1) > 0) / n_proteins

    print(f"GO-CC matrix: {cc_matrix.shape}, coverage: {coverage:.2%}, {len(go_cc_list)} terms")

    return cc_matrix, go_cc_list, coverage


def build_temporal_activity(series_path: str, protein_dict: pd.DataFrame, T: int = 12, k: int = 1) -> Tuple[np.ndarray, float]:
    symbol_to_id = dict(zip(protein_dict['Gene_symbol'], protein_dict['ID']))
    n_proteins = len(protein_dict)

    if not os.path.exists(series_path):
        raise FileNotFoundError(f"Series matrix file not found: {series_path}")

    tt_matrix = np.zeros((n_proteins, T), dtype=np.float32)
    expression_data = {}

    with open(series_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < T * 3 + 2:
                continue

            gene_symbol = parts[1].upper()
            if gene_symbol not in symbol_to_id:
                continue

            protein_id = symbol_to_id[gene_symbol]

            expr_values = []
            for t in range(T):
                try:
                    val1 = float(parts[2 + t])
                    val2 = float(parts[2 + T + t])
                    val3 = float(parts[2 + 2 * T + t])
                    expr_values.append((val1 + val2 + val3) / 3.0)
                except (ValueError, IndexError):
                    break

            if len(expr_values) == T:
                expression_data[protein_id] = np.array(expr_values)

    for protein_id, expr in expression_data.items():
        mean_val = np.mean(expr)
        sd_val = np.std(expr, ddof=1) if len(expr) > 1 else 0.0

        thresh_kSD = mean_val + k * sd_val * (sd_val / (1 + sd_val)) if sd_val > 0 else mean_val

        for t in range(T):
            if expr[t] >= thresh_kSD:
                tt_matrix[protein_id, t] = 1.0

    coverage = np.sum(tt_matrix.sum(axis=1) > 0) / n_proteins

    print(f"Temporal activity matrix: {tt_matrix.shape}, coverage: {coverage:.2%}")

    return tt_matrix, coverage


def build_knowledge_features(
    protein_dict: pd.DataFrame,
    go_slim_path: str,
    series_path: str,
    T: int = 12,
    k: int = 1,
    normalize: bool = True,
    cache_dir: str = None,
    use_cache: bool = True
) -> Tuple[np.ndarray, Dict[str, any]]:

    if cache_dir is None:
        cache_dir = os.path.dirname(go_slim_path)

    cache_path = os.path.join(cache_dir, f"knowledge_features_T{T}_k{k}_norm{normalize}.npz")

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached knowledge features from: {cache_path}")
        try:
            cached = np.load(cache_path, allow_pickle=True)
            Z_know = cached['features']
            metadata = cached['metadata'].item()

            if Z_know.shape[0] == len(protein_dict):
                print(f"Cache loaded: {Z_know.shape}, CC: {metadata['cc_dim']}, TT: {metadata['tt_dim']}")
                print(f"  CC coverage: {metadata['cc_coverage']:.2%}, TT coverage: {metadata['tt_coverage']:.2%}")
                return Z_know, metadata
            else:
                print(f"Cache size mismatch ({Z_know.shape[0]} vs {len(protein_dict)}), rebuilding...")
        except Exception as e:
            print(f"Cache loading failed: {e}, rebuilding...")

    print("Building knowledge features (CC/TT)...")

    cc_matrix, cc_terms, cc_coverage = build_go_cc_matrix(go_slim_path, protein_dict)

    tt_matrix, tt_coverage = build_temporal_activity(series_path, protein_dict, T=T, k=k)

    Z_know = np.concatenate([cc_matrix, tt_matrix], axis=1)

    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Z_know = scaler.fit_transform(Z_know)

    metadata = {
        'cc_dim': cc_matrix.shape[1],
        'tt_dim': tt_matrix.shape[1],
        'total_dim': Z_know.shape[1],
        'cc_coverage': cc_coverage,
        'tt_coverage': tt_coverage,
        'cc_terms': cc_terms,
        'T': T,
        'k': k,
        'normalized': normalize
    }

    print(f"Knowledge features: {Z_know.shape}, CC: {cc_matrix.shape[1]}, TT: {tt_matrix.shape[1]}")

    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_path, features=Z_know, metadata=metadata)
            print(f"Knowledge features cached to: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    return Z_know, metadata



