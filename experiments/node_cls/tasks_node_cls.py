import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, auc, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP visualization will be skipped.")


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_labels(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels_sparse = sp.csr_matrix((data['labels'], data['indices'], data['indptr']),
                                   shape=data['shape'])
    labels = labels_sparse.toarray()
    term_ids = data['term_ids']
    term_names = data['term_names']
    return labels, term_ids, term_names


def load_embedding(emb_path):
    if emb_path.endswith('.pt'):
        emb = torch.load(emb_path, map_location='cpu')
        if isinstance(emb, torch.Tensor):
            emb = emb.numpy()
    elif emb_path.endswith('.npy'):
        emb = np.load(emb_path)
    elif emb_path.endswith('.npz'):
        data = np.load(emb_path)
        emb = data['embedding']
    else:
        raise ValueError(f"Unsupported embedding format: {emb_path}")
    return emb


def compute_fmax_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold


def train_mlp(X_train, y_train, X_val, y_val, hidden_dims, dropout, lr=0.001, epochs=100,
              batch_size=64, patience=10, device='cpu', verbose=False):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = MLPClassifier(input_dim, hidden_dims, output_dim, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_multilabel(y_true, y_pred, y_scores):
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    micro_auprc = average_precision_score(y_true, y_scores, average='micro')

    per_label_auprc = []
    n_labels = y_true.shape[1]
    for i in range(n_labels):
        if y_true[:, i].sum() > 0:
            auprc_i = average_precision_score(y_true[:, i], y_scores[:, i])
            per_label_auprc.append(auprc_i)
        else:
            per_label_auprc.append(0.0)

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_auprc': micro_auprc,
        'per_label_auprc': per_label_auprc
    }


def run_cross_validation(X, y, clf_type='lr', folds=5, repeats=5, seed=42,
                         lr_C_values=[0.1, 1, 10], mlp_hidden_dims=[128, 128],
                         mlp_dropout=0.5, mlp_lr=0.001, mlp_epochs=100,
                         mlp_batch_size=64, device='cpu', verbose=False):

    n_samples, n_features = X.shape
    n_labels = y.shape[1]

    all_results = []

    for repeat_idx in range(repeats):
        repeat_seed = seed + repeat_idx
        np.random.seed(repeat_seed)

        label_counts = y.sum(axis=0)
        stratify_labels = y[:, np.argmax(label_counts)]

        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=repeat_seed)

        for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(X, stratify_labels)):
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]

            val_size = int(0.15 * len(train_val_idx))
            train_idx = np.arange(len(X_train_val) - val_size)
            val_idx = np.arange(len(X_train_val) - val_size, len(X_train_val))

            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            if clf_type == 'lr':
                best_val_auprc = -1
                best_C = None
                best_model = None

                for C in lr_C_values:
                    clf = OneVsRestClassifier(
                        LogisticRegression(C=C, max_iter=500, class_weight='balanced',
                                         random_state=repeat_seed, solver='lbfgs')
                    )
                    clf.fit(X_train, y_train)
                    y_val_scores = clf.predict_proba(X_val)
                    val_auprc = average_precision_score(y_val, y_val_scores, average='micro')

                    if val_auprc > best_val_auprc:
                        best_val_auprc = val_auprc
                        best_C = C
                        best_model = clf

                if verbose:
                    print(f"  Repeat {repeat_idx+1}/{repeats}, Fold {fold_idx+1}/{folds}: "
                          f"Best C={best_C}, val_auprc={best_val_auprc:.4f}")

                y_test_scores = best_model.predict_proba(X_test)

                thresholds = []
                for label_idx in range(n_labels):
                    if y_val[:, label_idx].sum() > 0:
                        thresh = compute_fmax_threshold(y_val[:, label_idx],
                                                        clf.predict_proba(X_val)[:, label_idx])
                    else:
                        thresh = 0.5
                    thresholds.append(thresh)

                y_test_pred = (y_test_scores >= np.array(thresholds)).astype(int)

            elif clf_type == 'mlp':
                if verbose:
                    print(f"  Repeat {repeat_idx+1}/{repeats}, Fold {fold_idx+1}/{folds}: "
                          f"Training MLP...")

                model = train_mlp(X_train, y_train, X_val, y_val,
                                hidden_dims=mlp_hidden_dims, dropout=mlp_dropout,
                                lr=mlp_lr, epochs=mlp_epochs, batch_size=mlp_batch_size,
                                patience=10, device=device, verbose=verbose)

                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test).to(device)
                    y_test_logits = model(X_test_tensor).cpu().numpy()
                    y_test_scores = 1 / (1 + np.exp(-y_test_logits))

                    X_val_tensor = torch.FloatTensor(X_val).to(device)
                    y_val_logits = model(X_val_tensor).cpu().numpy()
                    y_val_scores = 1 / (1 + np.exp(-y_val_logits))

                thresholds = []
                for label_idx in range(n_labels):
                    if y_val[:, label_idx].sum() > 0:
                        thresh = compute_fmax_threshold(y_val[:, label_idx],
                                                        y_val_scores[:, label_idx])
                    else:
                        thresh = 0.5
                    thresholds.append(thresh)

                y_test_pred = (y_test_scores >= np.array(thresholds)).astype(int)

            else:
                raise ValueError(f"Unknown classifier type: {clf_type}")

            metrics = evaluate_multilabel(y_test, y_test_pred, y_test_scores)

            result = {
                'repeat': repeat_idx,
                'fold': fold_idx,
                'micro_f1': metrics['micro_f1'],
                'macro_f1': metrics['macro_f1'],
                'micro_auprc': metrics['micro_auprc'],
                'per_label_auprc': metrics['per_label_auprc']
            }
            all_results.append(result)

            if verbose:
                print(f"    Test: micro_f1={metrics['micro_f1']:.4f}, "
                      f"macro_f1={metrics['macro_f1']:.4f}, "
                      f"micro_auprc={metrics['micro_auprc']:.4f}")

    return all_results


def plot_umap(X, y, term_names, out_path):
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping UMAP plot")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)

    label_sums = y.sum(axis=0)
    top_label_idx = np.argmax(label_sums)
    label_mask = y[:, top_label_idx] > 0

    plt.figure(figsize=(10, 8))
    plt.scatter(X_umap[~label_mask, 0], X_umap[~label_mask, 1],
                c='lightgray', alpha=0.3, s=10, label='Others')
    plt.scatter(X_umap[label_mask, 0], X_umap[label_mask, 1],
                c='steelblue', alpha=0.6, s=20, label=term_names[top_label_idx])
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.title(f'UMAP Projection (top label: {term_names[top_label_idx]})', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"UMAP plot saved to: {out_path}")


def plot_per_label_auprc(per_label_auprc, term_names, out_path, topk=20):
    per_label_auprc_mean = np.mean(per_label_auprc, axis=0)

    top_indices = np.argsort(per_label_auprc_mean)[::-1][:topk]
    top_names = [term_names[i] for i in top_indices]
    top_scores = per_label_auprc_mean[top_indices]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_names)), top_scores, color='steelblue', alpha=0.7)
    plt.yticks(range(len(top_names)), top_names, fontsize=10)
    plt.xlabel('AUPRC', fontsize=12)
    plt.title(f'Top {topk} Labels by AUPRC', fontsize=14)
    plt.gca().invert_yaxis()

    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        plt.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Per-label AUPRC plot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-label node classification for GO-BP/MF')
    parser.add_argument('--aspect', required=True, choices=['BP', 'MF'])
    parser.add_argument('--labels_npz', required=True, help='Path to labels .npz file')
    parser.add_argument('--embedding', required=True, help='Path to embedding file (.pt/.npy/.npz)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clf', choices=['lr', 'mlp'], default='mlp',
                        help='Classifier: lr (Logistic Regression) or mlp (MLP)')
    parser.add_argument('--lr_C', nargs='+', type=float, default=[0.1, 1, 10],
                        help='C values for LR grid search')
    parser.add_argument('--mlp_hidden', nargs='+', type=int, default=[128, 128],
                        help='MLP hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--mlp_lr', type=float, default=0.001)
    parser.add_argument('--mlp_epochs', type=int, default=100)
    parser.add_argument('--mlp_batch_size', type=int, default=64)
    parser.add_argument('--device', default='cpu', help='Device for MLP training')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Multi-label Node Classification: GO-{args.aspect}")
    print(f"{'='*60}\n")

    print(f"Loading labels from: {args.labels_npz}")
    labels, term_ids, term_names = load_labels(args.labels_npz)
    print(f"  Labels shape: {labels.shape}")
    print(f"  Number of terms: {len(term_ids)}")

    print(f"\nLoading embedding from: {args.embedding}")
    embedding = load_embedding(args.embedding)
    print(f"  Embedding shape: {embedding.shape}")

    if embedding.shape[0] != labels.shape[0]:
        raise ValueError(f"Embedding and labels size mismatch: {embedding.shape[0]} vs {labels.shape[0]}")

    mask = labels.sum(axis=1) > 0
    X = embedding[mask]
    y = labels[mask]
    print(f"\nFiltered to proteins with labels: {X.shape[0]}/{embedding.shape[0]}")

    print(f"\nRunning {args.folds}-Fold 脳 {args.repeats} repeats cross-validation...")
    print(f"  Classifier: {args.clf}")
    if args.clf == 'lr':
        print(f"  LR C values: {args.lr_C}")
    else:
        print(f"  MLP hidden: {args.mlp_hidden}")
        print(f"  Dropout: {args.dropout}")
        print(f"  Learning rate: {args.mlp_lr}")
        print(f"  Epochs: {args.mlp_epochs}")
        print(f"  Batch size: {args.mlp_batch_size}")
        print(f"  Device: {args.device}")

    results = run_cross_validation(
        X, y,
        clf_type=args.clf,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        lr_C_values=args.lr_C,
        mlp_hidden_dims=args.mlp_hidden,
        mlp_dropout=args.dropout,
        mlp_lr=args.mlp_lr,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        device=args.device,
        verbose=args.verbose
    )

    micro_f1_scores = [r['micro_f1'] for r in results]
    macro_f1_scores = [r['macro_f1'] for r in results]
    micro_auprc_scores = [r['micro_auprc'] for r in results]
    per_label_auprc_all = [r['per_label_auprc'] for r in results]

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Micro F1:    {np.mean(micro_f1_scores):.4f} 卤 {np.std(micro_f1_scores):.4f}")
    print(f"Macro F1:    {np.mean(macro_f1_scores):.4f} 卤 {np.std(macro_f1_scores):.4f}")
    print(f"Micro AUPRC: {np.mean(micro_auprc_scores):.4f} 卤 {np.std(micro_auprc_scores):.4f}")
    print(f"{'='*60}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    metrics_dict = {
        'aspect': args.aspect,
        'classifier': args.clf,
        'folds': args.folds,
        'repeats': args.repeats,
        'micro_f1_mean': float(np.mean(micro_f1_scores)),
        'micro_f1_std': float(np.std(micro_f1_scores)),
        'macro_f1_mean': float(np.mean(macro_f1_scores)),
        'macro_f1_std': float(np.std(macro_f1_scores)),
        'micro_auprc_mean': float(np.mean(micro_auprc_scores)),
        'micro_auprc_std': float(np.std(micro_auprc_scores)),
        'all_results': results
    }

    metrics_path = os.path.join(args.out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    umap_path = os.path.join(args.out_dir, 'umap.png')
    plot_umap(X, y, term_names, umap_path)

    per_label_path = os.path.join(args.out_dir, 'per_label_auprc_top20.png')
    plot_per_label_auprc(per_label_auprc_all, term_names, per_label_path, topk=20)

    print(f"\nAll outputs saved to: {args.out_dir}")


if __name__ == '__main__':
    main()



