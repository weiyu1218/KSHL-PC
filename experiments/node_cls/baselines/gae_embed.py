"""
GAE (Graph Autoencoder) 宓屽叆鐢熸垚 - 绾疨yTorch鐗堟湰
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys


sys.path.append(os.path.dirname(__file__))
from graph_conv import GCNConv


class GAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super(GAEEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, out_dim))

        self.dropout = dropout

    def forward(self, x, adj):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj)

        return x


def train_gae(x, adj, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5,
              lr=0.01, epochs=200, device='cpu', verbose=False):
    """
    璁粌GAE妯″瀷鐢熸垚鑺傜偣宓屽叆
    """
    model = GAEEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout).to(device)
    x = x.to(device)
    adj = adj.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    adj_dense = adj.to_dense()
    pos_edges = (adj_dense > 0).nonzero(as_tuple=False)

    def recon_loss(z, pos_edges, num_nodes):
        pos_idx = torch.randperm(pos_edges.size(0))[:1000]
        pos_edge_sample = pos_edges[pos_idx]

        pos_loss = -torch.log(
            torch.sigmoid((z[pos_edge_sample[:, 0]] * z[pos_edge_sample[:, 1]]).sum(dim=1)) + 1e-15
        ).mean()

        neg_edges = torch.randint(0, num_nodes, (1000, 2), device=z.device)
        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edges[:, 0]] * z[neg_edges[:, 1]]).sum(dim=1)) + 1e-15
        ).mean()

        return pos_loss + neg_loss

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(x, adj)
        loss = recon_loss(z, pos_edges, x.size(0))
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        embeddings = model(x, adj)

    return embeddings.cpu()


def main():
    parser = argparse.ArgumentParser(description='Generate GAE embeddings')
    parser.add_argument('--data_root', default='data/Saccharomyces_cerevisiae')
    parser.add_argument('--feature_path', default=None)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    from data_loader import load_ppi_graph_pytorch, normalize_adj_torch, get_default_data_paths

    paths = get_default_data_paths(args.data_root)
    feature_path = args.feature_path or paths['feature_fused']

    print(f"Loading PPI graph from {paths['ppi']}")
    print(f"Using features from {feature_path}")

    x, adj, edge_index, num_nodes = load_ppi_graph_pytorch(
        paths['ppi'],
        paths['protein_list'],
        feature_path
    )

    adj_norm = normalize_adj_torch(adj)

    print(f"Graph: {num_nodes} nodes, {edge_index.size(1)} edges")
    print(f"Feature dimension: {x.size(1)}")
    print(f"Training GAE model on {args.device}...")

    embeddings = train_gae(
        x, adj_norm,
        in_dim=x.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        verbose=args.verbose
    )

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    torch.save(embeddings, args.out_path)

    print(f"\nEmbeddings saved to: {args.out_path}")
    print(f"  Shape: {embeddings.shape}")


if __name__ == '__main__':
    main()



