import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

SERIES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                           'Saccharomyces_cerevisiae', 'expression', 'series_matrix.txt')
PROTEIN_LIST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                                 'Saccharomyces_cerevisiae', 'Gene_Entry_ID_list', 'Protein_list.csv')
T = 12
K0_DEFAULT = 1.0
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity_analysis')


def load_protein_dict(path):
    return pd.read_csv(path, sep='\t', header=None, names=['Gene_symbol', 'Entry', 'ID'])


def load_expression(series_path, protein_dict, T=12):
    symbol_to_id = dict(zip(protein_dict['Gene_symbol'], protein_dict['ID']))
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
    return expression_data


def compute_activity(expression_data, n_proteins, T, use_sigma_sq, k0):
    tt_matrix = np.zeros((n_proteins, T), dtype=np.float32)
    for protein_id, expr in expression_data.items():
        mu = np.mean(expr)
        sigma = np.std(expr, ddof=1) if len(expr) > 1 else 0.0
        if sigma == 0:
            tau = mu
        elif use_sigma_sq:
            tau = mu + k0 * sigma ** 2 / (1 + sigma)
        else:
            tau = mu + k0 * sigma / (1 + sigma)
        for t in range(T):
            if expr[t] >= tau:
                tt_matrix[protein_id, t] = 1.0
    return tt_matrix


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    protein_dict = load_protein_dict(PROTEIN_LIST_PATH)
    n_proteins = len(protein_dict)
    expression_data = load_expression(SERIES_PATH, protein_dict, T)
    print(f"Proteins with expression data: {len(expression_data)} / {n_proteins}")

    proteins_with_expr = list(expression_data.keys())
    sigmas = np.array([np.std(expression_data[p], ddof=1) for p in proteins_with_expr])
    mus = np.array([np.mean(expression_data[p]) for p in proteins_with_expr])

    # Activity matrices at default k0
    tt_sigma = compute_activity(expression_data, n_proteins, T, use_sigma_sq=False, k0=K0_DEFAULT)
    tt_sigma_sq = compute_activity(expression_data, n_proteins, T, use_sigma_sq=True, k0=K0_DEFAULT)
    active_sigma = tt_sigma.sum(axis=1)
    active_sigma_sq = tt_sigma_sq.sum(axis=1)
    coverage_sigma = np.mean(active_sigma > 0)
    coverage_sigma_sq = np.mean(active_sigma_sq > 0)

    # k0 sensitivity
    k0_vals = np.linspace(0.1, 3.0, 30)
    cov_sigma_k0 = []
    cov_sigma_sq_k0 = []
    for k0 in k0_vals:
        tt_s = compute_activity(expression_data, n_proteins, T, use_sigma_sq=False, k0=k0)
        tt_s2 = compute_activity(expression_data, n_proteins, T, use_sigma_sq=True, k0=k0)
        cov_sigma_k0.append(np.mean(tt_s.sum(axis=1) > 0))
        cov_sigma_sq_k0.append(np.mean(tt_s2.sum(axis=1) > 0))
        print(f"  k0={k0:.2f}  sigma-cov={cov_sigma_k0[-1]:.4f}  sigma2-cov={cov_sigma_sq_k0[-1]:.4f}")

    # Per-protein sigma for stratification
    sigma_protein = np.zeros(n_proteins)
    for p_id, expr in expression_data.items():
        sigma_protein[p_id] = np.std(expr, ddof=1) if len(expr) > 1 else 0.0

    # --- Plot ---
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    COLOR_SIGMA = '#2166AC'
    COLOR_SIGMA_SQ = '#D6604D'

    # Panel (a): Mathematical increment functions
    ax1 = fig.add_subplot(gs[0, 0])
    sigma_range = np.linspace(0, 5, 500)
    incr_sigma = sigma_range / (1 + sigma_range)
    incr_sigma_sq = sigma_range ** 2 / (1 + sigma_range)
    ax1.plot(sigma_range, incr_sigma, color=COLOR_SIGMA, linewidth=2,
             label=r'$\sigma/(1+\sigma)$')
    ax1.plot(sigma_range, incr_sigma_sq, color=COLOR_SIGMA_SQ, linewidth=2, linestyle='--',
             label=r'$\sigma^2/(1+\sigma)$')
    ax1.axvline(x=1, color='gray', linestyle=':', linewidth=1.2, alpha=0.8)
    ax1.text(1.05, 0.05, r'$\sigma=1$', fontsize=8, color='gray')
    ax1.set_xlabel(r'$\sigma_i$', fontsize=11)
    ax1.set_ylabel('Threshold increment ($k_0=1$)', fontsize=10)
    ax1.set_title('(a) Increment Functions', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel (b): Sigma distribution in real data
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(sigmas, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label=r'$\sigma=1$ (crossover)')
    pct_low = np.mean(sigmas < 1) * 100
    pct_high = np.mean(sigmas >= 1) * 100
    ax2.set_xlabel(r'$\sigma_i$', fontsize=11)
    ax2.set_ylabel('Number of proteins', fontsize=10)
    ax2.set_title('(b) Protein Expression Variance Distribution', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.text(0.58, 0.82, f'$\\sigma<1$: {pct_low:.1f}%\n$\\sigma\\geq1$: {pct_high:.1f}%',
             transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax2.grid(alpha=0.3)

    # Panel (c): Threshold difference vs sigma
    ax3 = fig.add_subplot(gs[0, 2])
    tau_sigma = mus + K0_DEFAULT * sigmas / (1 + sigmas)
    tau_sigma_sq = mus + K0_DEFAULT * sigmas ** 2 / (1 + sigmas)
    diff = tau_sigma_sq - tau_sigma
    sc = ax3.scatter(sigmas, diff, c=sigmas, cmap='RdYlBu_r', s=6, alpha=0.5, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label(r'$\sigma_i$', fontsize=9)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_xlabel(r'$\sigma_i$', fontsize=11)
    ax3.set_ylabel(r'$\tau_{\sigma^2} - \tau_{\sigma}$', fontsize=11)
    ax3.set_title('(c) Threshold Difference ($\\sigma^2$ form $-$ $\\sigma$ form)', fontsize=11)
    ax3.grid(alpha=0.3)

    # Panel (d): Active time points distribution
    ax4 = fig.add_subplot(gs[1, 0])
    bins = np.arange(-0.5, T + 1.5, 1)
    ax4.hist(active_sigma[active_sigma > 0], bins=bins, alpha=0.7,
             label=fr'$\sigma$ form (cov={coverage_sigma:.2%})',
             color=COLOR_SIGMA, edgecolor='white')
    ax4.hist(active_sigma_sq[active_sigma_sq > 0], bins=bins, alpha=0.7,
             label=fr'$\sigma^2$ form (cov={coverage_sigma_sq:.2%})',
             color=COLOR_SIGMA_SQ, edgecolor='white')
    ax4.set_xlabel('Active time points per protein', fontsize=10)
    ax4.set_ylabel('Number of proteins', fontsize=10)
    ax4.set_title('(d) Active Time Points Distribution', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # Panel (e): Coverage vs k0
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(k0_vals, cov_sigma_k0, color=COLOR_SIGMA, linewidth=2, marker='o', markersize=3,
             label=r'$\sigma$ form')
    ax5.plot(k0_vals, cov_sigma_sq_k0, color=COLOR_SIGMA_SQ, linewidth=2,
             linestyle='--', marker='s', markersize=3, label=r'$\sigma^2$ form')
    ax5.axvline(x=K0_DEFAULT, color='gray', linestyle=':', linewidth=1.5,
                alpha=0.8, label=f'$k_0={K0_DEFAULT}$ (default)')
    ax5.set_xlabel(r'$k_0$', fontsize=11)
    ax5.set_ylabel(r'Coverage (proteins with $\geq$1 active time point)', fontsize=9)
    ax5.set_title('(e) Coverage Sensitivity to $k_0$', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # Panel (f): Mean difference in active time points by sigma group
    ax6 = fig.add_subplot(gs[1, 2])
    groups = [(0, 0.5, r'$\sigma<0.5$'),
              (0.5, 1.0, r'$0.5\leq\sigma<1$'),
              (1.0, 2.0, r'$1\leq\sigma<2$'),
              (2.0, 100, r'$\sigma\geq2$')]
    group_labels = [g[2] for g in groups]
    diff_means = []
    diff_stds = []
    group_counts = []
    for lo, hi, _ in groups:
        mask = (sigma_protein > lo) & (sigma_protein <= hi) & (sigma_protein > 0)
        group_counts.append(int(mask.sum()))
        if mask.sum() > 0:
            d = active_sigma_sq[mask] - active_sigma[mask]
            diff_means.append(d.mean())
            diff_stds.append(d.std())
        else:
            diff_means.append(0.0)
            diff_stds.append(0.0)

    x_pos = np.arange(len(groups))
    bar_colors = ['#4393C3', '#92C5DE', '#F4A582', '#D6604D']
    bars = ax6.bar(x_pos, diff_means, yerr=diff_stds, capsize=5,
                   color=bar_colors, alpha=0.85, edgecolor='white')
    ax6.axhline(y=0, color='black', linewidth=1)
    for i, (rect, cnt) in enumerate(zip(bars, group_counts)):
        ax6.text(rect.get_x() + rect.get_width() / 2, 0.02,
                 f'n={cnt}', ha='center', va='bottom', fontsize=8, color='black')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(group_labels, fontsize=9)
    ax6.set_xlabel(r'$\sigma_i$ group', fontsize=10)
    ax6.set_ylabel(r'Mean $\Delta$ active time points ($\sigma^2$$-$$\sigma$)', fontsize=9)
    ax6.set_title('(f) Impact by Variance Group', fontsize=11)
    ax6.grid(alpha=0.3, axis='y')

    fig.suptitle(
        'Sensitivity Analysis: Threshold Form Comparison\n'
        r'$\sigma$ form: $\tau_i = \mu_i + k_0\sigma_i/(1+\sigma_i)$'
        r'     $\sigma^2$ form: $\tau_i = \mu_i + k_0\sigma_i^2/(1+\sigma_i)$',
        fontsize=12, y=1.02
    )

    pdf_path = os.path.join(OUTPUT_DIR, 'threshold_sensitivity_comparison.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'threshold_sensitivity_comparison.png')
    plt.savefig(pdf_path, bbox_inches='tight', dpi=150)
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    print(f"\nFigures saved:\n  {pdf_path}\n  {png_path}")

    # Summary statistics
    print("\n=== Summary ===")
    print(f"Proteins with expression data: {len(expression_data)}")
    print(f"sigma range: [{sigmas.min():.4f}, {sigmas.max():.4f}], mean={sigmas.mean():.4f}, median={np.median(sigmas):.4f}")
    print(f"sigma < 1: {pct_low:.1f}%    sigma >= 1: {pct_high:.1f}%")
    print()
    print(f"sigma   form coverage (k0={K0_DEFAULT}): {coverage_sigma:.4f} ({coverage_sigma*100:.2f}%)")
    print(f"sigma^2 form coverage (k0={K0_DEFAULT}): {coverage_sigma_sq:.4f} ({coverage_sigma_sq*100:.2f}%)")
    print()
    n_more = np.sum(active_sigma_sq > active_sigma)
    n_less = np.sum(active_sigma_sq < active_sigma)
    n_same = np.sum(active_sigma_sq == active_sigma)
    print(f"Proteins with MORE active time points under sigma^2 form: {n_more} ({n_more/n_proteins*100:.1f}%)")
    print(f"Proteins with FEWER active time points under sigma^2 form: {n_less} ({n_less/n_proteins*100:.1f}%)")
    print(f"Proteins with SAME active time points: {n_same} ({n_same/n_proteins*100:.1f}%)")
    print()
    for i, (lo, hi, label) in enumerate(groups):
        print(f"Group {label}: n={group_counts[i]}, mean diff={diff_means[i]:.4f} +/- {diff_stds[i]:.4f}")


if __name__ == '__main__':
    main()



