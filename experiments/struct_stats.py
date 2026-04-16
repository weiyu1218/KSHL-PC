import numpy as np
import torch
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _dict_to_array(d: dict, n: int) -> np.ndarray:
    arr = np.zeros(n, dtype=np.float32)
    for k, v in d.items():
        if 0 <= int(k) < n:
            arr[int(k)] = float(v)
    return arr


def compute_structstats_embedding(
    G: nx.Graph,
    cliques: list,
    num_vertices: int,
    target_dim: int = 64,
    seed=None,
) -> torch.FloatTensor:
    """
    Build a non-spectral structural feature embedding for each vertex using
    fast graph statistics (no SVD/eigendecomposition).

    Features (all per-node, standardized across nodes):
      - degree
      - local clustering coefficient
      - k-core number
      - PageRank
      - triangle count
      - clique participation count
      - average clique size participated

    The resulting feature matrix is aligned to `target_dim` by PCA (if d0 >= k)
    or zero-padding (if d0 < k). No trainable parameters are used.
    """

    n = int(num_vertices)

    # Basic graph statistics
    deg_dict = dict(G.degree())
    clustering_dict = nx.clustering(G)
    # core number may fail for empty graphs; handle gracefully
    try:
        core_dict = nx.core_number(G) if G.number_of_edges() > 0 else {u: 0 for u in G.nodes()}
    except Exception:
        core_dict = {u: 0 for u in G.nodes()}

    # PageRank can be costly on huge graphs but fine for our scales
    try:
        pr_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-06)
    except Exception:
        # fallback: degree-based proxy if PageRank fails
        total_deg = max(1.0, float(sum(deg_dict.values())))
        pr_dict = {u: deg_dict.get(u, 0.0) / total_deg for u in G.nodes()}

    # Triangle counts
    try:
        tri_dict = nx.triangles(G)
    except Exception:
        tri_dict = {u: 0 for u in G.nodes()}

    # Clique participation statistics
    clique_count = np.zeros(n, dtype=np.float32)
    clique_size_sum = np.zeros(n, dtype=np.float32)
    for c in cliques:
        sz = float(len(c))
        for v in c:
            if 0 <= int(v) < n:
                clique_count[int(v)] += 1.0
                clique_size_sum[int(v)] += sz

    # Assemble feature matrix (n x d0)
    deg = _dict_to_array(deg_dict, n)
    clustering = _dict_to_array(clustering_dict, n)
    core = _dict_to_array(core_dict, n)
    pr = _dict_to_array(pr_dict, n)
    tri = _dict_to_array(tri_dict, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_clique_size = np.divide(clique_size_sum, clique_count, out=np.zeros_like(clique_count), where=clique_count>0)

    X = np.vstack([
        deg,
        clustering,
        core,
        pr,
        tri,
        clique_count,
        avg_clique_size,
    ]).T  # shape: (n, d0)

    # Standardize per-feature
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    d0 = X_std.shape[1]
    k = int(target_dim) if target_dim is not None else d0

    if k <= 0:
        k = d0

    # Align dimension to target_dim without introducing trainable params
    if k <= d0:
        pca = PCA(n_components=k, svd_solver='auto', random_state=seed)
        X_k = pca.fit_transform(X_std)
    else:
        # Zero-pad to k dimensions to keep fairness on capacity
        pad_width = k - d0
        X_k = np.pad(X_std, ((0, 0), (0, pad_width)), mode='constant')

    emb = torch.FloatTensor(X_k)
    return emb



