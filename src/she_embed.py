import torch
import numpy as np
import scipy.sparse as sp
from she_core import she_embed, SHEConfig


def build_H0_from_cliques(cliques, num_vertices):
    row_indices = []
    col_indices = []

    for edge_idx, clique in enumerate(cliques):
        for vertex in clique:
            row_indices.append(edge_idx)
            col_indices.append(vertex)

    data = np.ones(len(row_indices))
    H0 = sp.csr_matrix((data, (row_indices, col_indices)), shape=(len(cliques), num_vertices))

    return H0


def compute_she_embedding(edge_list_data, X, r, k, T, alpha, seed=None, vol_type='traditional'):
    cliques = edge_list_data["PPI_cliques_list"]
    num_vertices = edge_list_data["num_vertices"]

    H0 = build_H0_from_cliques(cliques, num_vertices)

    X_np = X.cpu().numpy() if torch.is_tensor(X) else X

    config = SHEConfig(
        embedding_dim=k,
        svd_rank=r,
        svd_tol=0,
        window=T,
        alpha=alpha,
        seed=seed,
        vol_type=vol_type
    )

    Z_v, elapsed_time = she_embed(H0, X_np, config)

    Z_v_tensor = torch.FloatTensor(Z_v)

    return Z_v_tensor, elapsed_time



