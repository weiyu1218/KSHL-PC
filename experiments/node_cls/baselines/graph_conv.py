"""
绾疨yTorch瀹炵幇鐨勫浘鍗风Н灞?
涓嶄緷璧杢orch_geometric
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


def normalize_adj(adj):
    """
    瀵圭О褰掍竴鍖栭偦鎺ョ煩闃? D^{-1/2}AD^{-1/2}
    """
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class GCNConv(nn.Module):
    """鍥惧嵎绉眰"""
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Args:
            x: [num_nodes, in_features]
            adj: [num_nodes, num_nodes] 褰掍竴鍖栧悗鐨勯偦鎺ョ煩闃?
        """
        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        return x


class GATConv(nn.Module):
    """Helper."""
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edge_index):
        """
        Args:
            x: [num_nodes, in_features]
            edge_index: [2, num_edges] 杈圭储寮?
        """
        h = torch.mm(x, self.W)  # [N, out_features]
        N = h.size()[0]

        # 鍑嗗娉ㄦ剰鍔涙満鍒剁殑杈撳叆
        a_input = self._prepare_attentional_mechanism_input(h, edge_index)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 璁＄畻娉ㄦ剰鍔涚郴鏁?
        attention = self._normalize_attention(e, edge_index, N)

        # 浣跨敤娉ㄦ剰鍔涚郴鏁拌仛鍚堥偦灞呯壒寰?
        h_prime = self._aggregate_neighbors(h, attention, edge_index, N)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, h, edge_index):
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]  # [num_edges, out_features]
        h_dst = h[dst]  # [num_edges, out_features]
        a_input = torch.cat([h_src, h_dst], dim=1)  # [num_edges, 2*out_features]
        return a_input.unsqueeze(2)  # [num_edges, 2*out_features, 1]

    def _normalize_attention(self, e, edge_index, num_nodes):
        # 浣跨敤softmax褰掍竴鍖栨敞鎰忓姏绯绘暟
        src, dst = edge_index[0], edge_index[1]

        # 瀵规瘡涓妭鐐圭殑鍏ヨ竟杩涜softmax
        attention = torch.zeros(e.size(0), device=e.device)
        for i in range(num_nodes):
            neighbors = (dst == i).nonzero(as_tuple=True)[0]
            if len(neighbors) > 0:
                attention[neighbors] = F.softmax(e[neighbors], dim=0)

        attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

    def _aggregate_neighbors(self, h, attention, edge_index, num_nodes):
        src, dst = edge_index[0], edge_index[1]

        # 鑱氬悎閭诲眳鐗瑰緛
        h_prime = torch.zeros(num_nodes, self.out_features, device=h.device)
        for edge_idx in range(edge_index.size(1)):
            s, d = src[edge_idx].item(), dst[edge_idx].item()
            h_prime[d] += attention[edge_idx] * h[s]

        return h_prime


class GINConv(nn.Module):
    """鍥惧悓鏋勭綉缁滃眰"""
    def __init__(self, in_features, out_features):
        super(GINConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj):
        """
        Args:
            x: [num_nodes, in_features]
            adj: [num_nodes, num_nodes] 閭绘帴鐭╅樀
        """
        # 鑱氬悎閭诲眳: (1 + eps) * x + sum(neighbors)
        neighbor_sum = torch.sparse.mm(adj, x)
        out = self.mlp((1 + self.eps) * x + neighbor_sum)
        return out


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Helper."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



