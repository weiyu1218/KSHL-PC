import argparse
import json
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize SHE grid search results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--summary_csv', type=str, default='results/grid_she/summary_she.csv',
                        help="Path to summary_she.csv from grid_she.py")
    parser.add_argument('--best_json', type=str, default='results/grid_she/best_she.json',
                        help="Path to best_she.json from grid_she.py")
    parser.add_argument('--out_fig', type=str, default='results/figs/grid_search_sensitivity.png',
                        help="Output figure path")
    parser.add_argument('--dpi', type=int, default=300,
                        help="Figure resolution")

    return parser.parse_args()


def load_best_config(best_json_path: str) -> Dict:
    with open(best_json_path, 'r') as f:
        best_data = json.load(f)
    return best_data['parameters']


def plot_grid_search_results(summary_csv: str, best_json: str, out_fig: str, dpi: int):
    df = pd.read_csv(summary_csv)
    df = df[df['success'] == True].copy()
    df = df.drop_duplicates(subset=['r', 'k', 'T', 'alpha'], keep='first')

    if df.empty:
        raise ValueError("No successful runs found in summary CSV")

    best_config = load_best_config(best_json)
    best_r = best_config['r']
    best_k = best_config['k']
    best_T = best_config['T']
    best_alpha = best_config['alpha']

    print(f"\nBest configuration: r={best_r}, k={best_k}, T={best_T}, alpha={best_alpha}")
    print(f"Best AUPRC: {df[(df['r']==best_r) & (df['k']==best_k) & (df['T']==best_T) & (np.isclose(df['alpha'], best_alpha))]['auprc'].values[0]:.4f}")

    unique_r = sorted(df['r'].unique())
    unique_k = sorted(df['k'].unique())
    unique_alpha = sorted(df['alpha'].unique())

    print(f"\nGrid search space:")
    print(f"  r: {unique_r}")
    print(f"  k: {unique_k}")
    print(f"  alpha: {unique_alpha}")
    print(f"  Total configs: {len(df)}")

    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor('white')

    for idx, alpha_val in enumerate(unique_alpha):
        ax = plt.subplot(1, 2, idx + 1)

        subset = df[np.isclose(df['alpha'], alpha_val, atol=1e-6)]

        pivot = subset.pivot_table(values='auprc', index='r', columns='k', aggfunc='mean')

        r_labels = [str(int(r)) for r in pivot.index]
        k_labels = [str(int(k)) for k in pivot.columns]

        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0.91, vmax=0.94)

        ax.set_xticks(range(len(k_labels)))
        ax.set_yticks(range(len(r_labels)))
        ax.set_xticklabels(k_labels, fontsize=11)
        ax.set_yticklabels(r_labels, fontsize=11)

        ax.set_xlabel('Embedding Dimension (k)', fontsize=13, fontweight='bold', labelpad=8)
        ax.set_ylabel('SVD Rank (r)', fontsize=13, fontweight='bold', labelpad=8)
        ax.set_title(f'伪 = {alpha_val}', fontsize=15, fontweight='bold', pad=12)

        for i in range(len(r_labels)):
            for j in range(len(k_labels)):
                value = pivot.values[i, j]
                r_val = pivot.index[i]
                k_val = pivot.columns[j]

                is_best = (r_val == best_r) and (k_val == best_k) and np.isclose(alpha_val, best_alpha)

                if is_best:
                    text_color = 'white'
                    fontweight = 'bold'
                    fontsize = 12
                    rect = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9,
                                        fill=False, edgecolor='blue', linewidth=4)
                    ax.add_patch(rect)
                elif value > 0.93:
                    text_color = 'white'
                    fontweight = 'normal'
                    fontsize = 10
                else:
                    text_color = 'black'
                    fontweight = 'normal'
                    fontsize = 10

                if not np.isnan(value):
                    ax.text(j, i, f'{value:.4f}', ha='center', va='center',
                           color=text_color, fontsize=fontsize, fontweight=fontweight)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('AUPRC', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

    plt.suptitle(f'SHE Hyperparameter Grid Search (T={best_T})\nBest: r={best_r}, k={best_k}, 伪={best_alpha}',
                 fontsize=17, fontweight='bold', y=1.02)

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.savefig(out_fig, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"\nGrid search visualization saved to: {out_fig}")
    plt.close()

    print_summary_statistics(df, best_config)


def print_summary_statistics(df: pd.DataFrame, best_config: Dict):
    print(f"\n{'='*80}")
    print("Grid Search Summary Statistics")
    print(f"{'='*80}")

    print(f"\nBest Configuration:")
    print(f"  r={best_config['r']}, k={best_config['k']}, T={best_config['T']}, alpha={best_config['alpha']}")

    best_row = df[
        (df['r'] == best_config['r']) &
        (df['k'] == best_config['k']) &
        (df['T'] == best_config['T']) &
        (np.isclose(df['alpha'], best_config['alpha'], atol=1e-6))
    ]

    if not best_row.empty:
        best_row = best_row.iloc[0]
        print(f"  AUPRC: {best_row['auprc']:.4f}")
        print(f"  AUROC: {best_row['auroc']:.4f}")
        print(f"  Complex F1: {best_row['complex_f1']:.4f}")
        print(f"  Runtime: {best_row['elapsed_time']:.1f}s")

    print(f"\nOverall Statistics:")
    print(f"  Total successful runs: {len(df)}")
    print(f"  AUPRC range: [{df['auprc'].min():.4f}, {df['auprc'].max():.4f}]")
    print(f"  AUPRC mean 卤 std: {df['auprc'].mean():.4f} 卤 {df['auprc'].std():.4f}")
    print(f"  Complex F1 range: [{df['complex_f1'].min():.4f}, {df['complex_f1'].max():.4f}]")
    print(f"  Complex F1 mean 卤 std: {df['complex_f1'].mean():.4f} 卤 {df['complex_f1'].std():.4f}")
    print(f"  Runtime range: [{df['elapsed_time'].min():.1f}s, {df['elapsed_time'].max():.1f}s]")
    print(f"  Runtime mean 卤 std: {df['elapsed_time'].mean():.1f}s 卤 {df['elapsed_time'].std():.1f}s")

    unique_r = sorted(df['r'].unique())
    if len(unique_r) > 1:
        print(f"\nSVD Rank (r) Analysis:")
        for r in unique_r:
            subset = df[df['r'] == r]
            print(f"  r={r}: AUPRC={subset['auprc'].mean():.4f} 卤 {subset['auprc'].std():.4f} (n={len(subset)})")

    unique_k = sorted(df['k'].unique())
    if len(unique_k) > 1:
        print(f"\nEmbedding Dimension (k) Analysis:")
        for k in unique_k:
            subset = df[df['k'] == k]
            print(f"  k={k}: AUPRC={subset['auprc'].mean():.4f} 卤 {subset['auprc'].std():.4f} (n={len(subset)})")

    unique_T = sorted(df['T'].unique())
    if len(unique_T) > 1:
        print(f"\nWindow Size (T) Analysis:")
        for T in unique_T:
            subset = df[df['T'] == T]
            print(f"  T={T}: AUPRC={subset['auprc'].mean():.4f} 卤 {subset['auprc'].std():.4f} (n={len(subset)})")

    unique_alpha = sorted(df['alpha'].unique())
    if len(unique_alpha) > 1:
        print(f"\nAlpha (伪) Analysis:")
        for alpha in unique_alpha:
            subset = df[np.isclose(df['alpha'], alpha, atol=1e-6)]
            print(f"  伪={alpha:.2f}: AUPRC={subset['auprc'].mean():.4f} 卤 {subset['auprc'].std():.4f} (n={len(subset)})")

    print(f"{'='*80}\n")


def main():
    args = parse_args()

    if not os.path.exists(args.summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {args.summary_csv}")
    if not os.path.exists(args.best_json):
        raise FileNotFoundError(f"Best config JSON not found: {args.best_json}")

    print(f"\n{'='*80}")
    print("SHE Grid Search Sensitivity Analysis")
    print(f"{'='*80}")
    print(f"Input CSV: {args.summary_csv}")
    print(f"Best config JSON: {args.best_json}")
    print(f"Output figure: {args.out_fig}")
    print(f"{'='*80}\n")

    plot_grid_search_results(
        summary_csv=args.summary_csv,
        best_json=args.best_json,
        out_fig=args.out_fig,
        dpi=args.dpi
    )

    print("\nVisualization completed successfully!")


if __name__ == "__main__":
    main()



