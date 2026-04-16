import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import try_gpu
import time


class LightweightAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dim=256):
        super(LightweightAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def fuse_embeddings(
    she_emb: torch.Tensor,
    know_features: np.ndarray,
    method: str = 'ae',
    out_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = None,
    verbose: bool = True
) -> torch.Tensor:
    """
    Fuse SHE embeddings with knowledge features.

    Args:
        she_emb: SHE structural embeddings
        know_features: Knowledge features (GO-CC/TT)
        method: 'ae' for autoencoder fusion, 'concat' for simple concatenation
        out_dim: Output dimension for fused embeddings
        epochs: Training epochs for AE
        lr: Learning rate for AE
        device: Compute device
        verbose: Print progress

    Returns:
        Fused embeddings
    """

    if device is None:
        device = try_gpu()

    start_time = time.time()

    if isinstance(she_emb, torch.Tensor):
        she_np = she_emb.cpu().detach().numpy()
    else:
        she_np = she_emb

    scaler_she = StandardScaler()
    she_norm = scaler_she.fit_transform(she_np)

    scaler_know = StandardScaler()
    know_norm = scaler_know.fit_transform(know_features)

    concat_features = np.concatenate([she_norm, know_norm], axis=1)
    input_dim = concat_features.shape[1]

    if method == 'ae':
        X_train = torch.FloatTensor(concat_features).to(device)

        model = LightweightAutoencoder(input_dim, out_dim, hidden_dim=256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            encoded, decoded = model(X_train)
            loss = criterion(decoded, X_train)
            loss.backward()
            optimizer.step()

            if verbose and (epoch == 0 or (epoch + 1) % 20 == 0):
                print(f"  Fusion AE epoch {epoch + 1}/{epochs}, loss: {loss.item():.5f}")

        model.eval()
        with torch.no_grad():
            Z_fused, _ = model(X_train)

    elif method == 'concat':
        Z_fused = torch.FloatTensor(concat_features).to(device)

    else:
        raise ValueError(f"Unknown fusion method: {method}. Choose 'ae' or 'concat'.")

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"Fusion completed in {elapsed_time:.2f}s, fused shape: {Z_fused.shape}")

    return Z_fused



