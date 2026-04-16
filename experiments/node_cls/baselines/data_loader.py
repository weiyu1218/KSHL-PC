"""
鏁版嵁鍔犺浇妯″潡 - 绾疨yTorch鐗堟湰
"""
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp


def load_ppi_graph_pytorch(ppi_path, protein_list_path, feature_path=None):
    """
    鍔犺浇PPI鍥炬暟鎹苟鏋勫缓閫傚悎绾疨yTorch鐨勬牸寮?

    Returns:
        x: 鑺傜偣鐗瑰緛 [num_nodes, feature_dim]
        adj: 閭绘帴鐭╅樀 (scipy sparse)
        edge_index: 杈圭储寮?[2, num_edges]
        num_nodes: 鑺傜偣鏁伴噺
    """
    protein_df = pd.read_csv(protein_list_path, sep='\t', header=None,
                              names=['Gene_symbol', 'UniProt', 'ID'])
    num_nodes = len(protein_df)
    symbol_to_id = dict(zip(protein_df['Gene_symbol'], protein_df['ID']))

    ppi_df = pd.read_csv(ppi_path, sep=';')

    # 鏋勫缓閭绘帴鐭╅樀鍜岃竟绱㈠紩
    edges = []
    for _, row in ppi_df.iterrows():
        source_gene = row['Source Gene names (SGD/UniProt-primary or ordered locus)']
        target_gene = row['Target Gene names  (SGD/UniProt-primary or ordered locus)']

        if source_gene in symbol_to_id and target_gene in symbol_to_id:
            source_id = symbol_to_id[source_gene]
            target_id = symbol_to_id[target_gene]
            edges.append((source_id, target_id))
            edges.append((target_id, source_id))  # 鏃犲悜鍥?

    # 鏋勫缓閭绘帴鐭╅樀
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    data = np.ones(len(edges))
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # 娣诲姞鑷幆
    adj = adj + sp.eye(num_nodes)

    # 杈圭储寮?
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 鍔犺浇鑺傜偣鐗瑰緛
    if feature_path:
        x = torch.load(feature_path, map_location='cpu')
        if x.device.type == 'cuda':
            x = x.cpu()
        if x.shape[0] != num_nodes:
            raise ValueError(f"Feature size mismatch: {x.shape[0]} vs {num_nodes}")
    else:
        x = torch.eye(num_nodes, dtype=torch.float)

    return x, adj, edge_index, num_nodes


def normalize_adj_torch(adj):
    """
    瀵圭О褰掍竴鍖栭偦鎺ョ煩闃? D^{-1/2}AD^{-1/2}
    杩斿洖torch sparse tensor
    """
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    # 杞崲涓簍orch sparse tensor
    indices = torch.from_numpy(
        np.vstack((adj_normalized.row, adj_normalized.col)).astype(np.int64))
    values = torch.from_numpy(adj_normalized.data.astype(np.float32))
    shape = torch.Size(adj_normalized.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_default_data_paths(data_root='data/Saccharomyces_cerevisiae'):
    """
    鑾峰彇榛樿鏁版嵁璺緞
    """
    import os

    paths = {
        'ppi': os.path.join(data_root, 'PPI', 'Mann_PPI.csv'),
        'protein_list': os.path.join(data_root, 'Gene_Entry_ID_list', 'Protein_list.csv'),
        'feature_fused': os.path.join(data_root, 'protein_feature', 'protein_feature_fused.pt'),
        'feature_ct': os.path.join(data_root, 'protein_feature', 'protein_feature_CT.pt'),
    }

    return paths



