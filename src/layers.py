import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Concentration(nn.Module):
    def __init__(self, hidden_channels: int, pooling_type: str = 'mean'):
        super(Concentration, self).__init__()
        self.pooling_type = pooling_type
        if pooling_type == 'attention':
            self.attn = nn.Linear(hidden_channels, 1, bias=False)

    def forward(self, X, GP_info):
        n_complexes = len(GP_info)
        max_size = max(len(e) for e in GP_info)

        padded_idx = np.zeros((n_complexes, max_size), dtype=np.int64)
        mask_np = np.zeros((n_complexes, max_size), dtype=np.float32)
        for i, complex_members in enumerate(GP_info):
            size = len(complex_members)
            padded_idx[i, :size] = complex_members
            mask_np[i, :size] = 1.0

        idx_t = torch.from_numpy(padded_idx).to(X.device)
        mask_t = torch.from_numpy(mask_np).to(X.device)
        x_pad = X[idx_t]

        if self.pooling_type == 'mean':
            count = mask_t.sum(1, keepdim=True).clamp(min=1)
            Z = (x_pad * mask_t.unsqueeze(-1)).sum(1) / count
        elif self.pooling_type == 'max':
            x_pad = x_pad.masked_fill(mask_t.unsqueeze(-1) == 0, float('-inf'))
            Z = x_pad.max(1).values
        elif self.pooling_type == 'attention':
            scores = self.attn(x_pad).squeeze(-1)
            scores = scores.masked_fill(mask_t == 0, float('-inf'))
            scores = torch.softmax(scores, dim=1)
            Z = (scores.unsqueeze(-1) * x_pad).sum(1)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
        return Z


class Classifier_DNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(Classifier_DNN, self).__init__()

        self.fc1 = nn.Linear(in_features, int(in_features / 1))
        self.bn1 = nn.BatchNorm1d(int(in_features / 1))

        self.fc2 = nn.Linear(int(in_features / 1), int(in_features / 2))
        self.bn2 = nn.BatchNorm1d(int(in_features / 2))

        self.fc3 = nn.Linear(int(in_features / 2), int(in_features / 4))
        self.bn3 = nn.BatchNorm1d(int(in_features / 4))

        self.fc5 = nn.Linear(int(in_features / 4), out_features)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.sigmoid(self.fc5(x))
        return x

