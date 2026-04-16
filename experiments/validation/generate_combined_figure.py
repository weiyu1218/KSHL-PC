#!/usr/bin/env python3
"""
鐢熸垚缁煎悎楠岃瘉鍥?- 鍖呭惈GO璇箟鐩镐技鎬с€佸叡瀹氫綅涓€鑷存€с€佽〃杈句竴鑷存€?

甯冨眬锛?
- 鍥続锛欸O璇箟鐩镐技鎬э紙BP, CC, MF妯悜鎺掑垪鍦ㄤ竴寮犲浘涓級
- 鍥綛锛氬畾浣嶄竴鑷存€э紙浠匧ocalization锛?
- 鍥綜锛氳〃杈句竴鑷存€э紙Coexpression Pearson鍜孲pearman锛?
- 缁熶竴鍥句緥鍦ㄥ彸涓婅
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from matplotlib.patches import Rectangle

# 璁剧疆缁樺浘鏍峰紡
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# 鏁版嵁鐩綍
project_root = Path(__file__).resolve().parent.parent.parent
RESULT_DIR = project_root / 'results/validation_experiment'

# 鍔犺浇缁撴灉鏁版嵁
print("鍔犺浇楠岃瘉缁撴灉鏁版嵁...")
with open(RESULT_DIR / 'known_results.json') as f:
    known_results = json.load(f)
with open(RESULT_DIR / 'predicted_results.json') as f:
    predicted_results = json.load(f)
with open(RESULT_DIR / 'random_results.json') as f:
    random_results = json.load(f)

print(f"  Known: {len(known_results)}, Predicted: {len(predicted_results)}, Random: {len(random_results)}")


def extract_metric(results_dict, metric_path):
    """Helper."""
    values = []
    for result in results_dict.values():
        data = result
        for key in metric_path:
            if key in data:
                data = data[key]
            else:
                data = None
                break
        if data is not None and isinstance(data, (int, float)) and not np.isnan(data):
            values.append(data)
    return values


def calculate_significance(group1, group2):
    """Helper."""
    if len(group1) < 2 or len(group2) < 2:
        return None, 'ns'
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    if p_value < 0.001:
        return p_value, '***'
    elif p_value < 0.01:
        return p_value, '**'
    elif p_value < 0.05:
        return p_value, '*'
    else:
        return p_value, 'ns'


def plot_single_comparison(ax, known_vals, pred_vals, rand_vals, title, ylabel,
                           colors, show_ylabel=True, y_range=None):
    """Helper."""
    data_list = [known_vals, pred_vals, rand_vals]
    positions = [1, 2, 3]

    # 缁樺埗绠辩嚎鍥?
    bp = ax.boxplot(data_list,
                    positions=positions,
                    widths=0.5,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=3,
                                   linestyle='none', markeredgecolor='none', alpha=0.4))

    # 璁剧疆绠变綋棰滆壊
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    # 璁剧疆涓綅绾?
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    # 缁樺埗鏁ｇ偣
    for i, (pos, values, color) in enumerate(zip(positions, data_list, colors)):
        n = len(values)
        if n > 0:
            jitter = np.random.normal(0, 0.03, n)
            x = np.ones(n) * pos + jitter
            ax.scatter(x, values, alpha=0.25, s=12, color=color, edgecolors='none')

    # 娣诲姞鏄捐憲鎬ф爣璁帮紙妯法鎵€鏈変笁缁勶級
    p_pr, sig_pr = calculate_significance(pred_vals, rand_vals)
    if sig_pr != 'ns':
        all_max = max(max(known_vals) if known_vals else 0,
                     max(pred_vals) if pred_vals else 0,
                     max(rand_vals) if rand_vals else 0)
        y_pos = all_max * 1.1
        ax.plot([1, 3], [y_pos, y_pos], 'k-', linewidth=1.2)
        ax.text(2, y_pos * 1.01, sig_pr, ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    # 璁剧疆鏍囩
    ax.set_xticks(positions)
    ax.set_xticklabels(['Known', 'Predicted', 'Random'], fontsize=9)

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8, loc='center')

    # 璁剧疆Y杞磋寖鍥?
    if y_range:
        ax.set_ylim(y_range)
    else:
        ax.set_ylim([0, 1.15])

    # 缃戞牸
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)


def plot_go_combined(ax, known_results, pred_results, rand_results, colors):
    """缁樺埗鍚堝苟鐨凣O璇箟鐩镐技鎬у浘锛堜笁涓狦O绫诲埆鍦ㄤ竴寮犲浘涓級"""
    aspects = ['BP', 'CC', 'MF']
    aspect_labels = ['GO:BP', 'GO:CC', 'GO:MF']

    # 鎻愬彇鏁版嵁
    all_data = {}
    for aspect in aspects:
        all_data[aspect] = {
            'known': extract_metric(known_results, ['metrics', 'go_similarity', aspect]),
            'pred': extract_metric(pred_results, ['metrics', 'go_similarity', aspect]),
            'rand': extract_metric(rand_results, ['metrics', 'go_similarity', aspect])
        }

    # 璁剧疆浣嶇疆
    width = 0.25
    x_groups = np.arange(len(aspects))

    all_boxes = []
    all_positions = []

    # 涓烘瘡涓狦O绫诲埆缁樺埗涓夌粍绠辩嚎鍥?
    for i, aspect in enumerate(aspects):
        base_pos = i * 4  # 姣忕粍涔嬮棿闂撮殧4涓崟浣?
        positions = [base_pos + 0, base_pos + 1, base_pos + 2]

        data_list = [
            all_data[aspect]['known'],
            all_data[aspect]['pred'],
            all_data[aspect]['rand']
        ]

        # 缁樺埗绠辩嚎鍥?
        bp = ax.boxplot(data_list,
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showfliers=True,
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=2,
                                       linestyle='none', markeredgecolor='none', alpha=0.3))

        # 璁剧疆绠变綋棰滆壊
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)

        # 璁剧疆涓綅绾?
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        # 缁樺埗鏁ｇ偣
        for pos, values, color in zip(positions, data_list, colors):
            n = len(values)
            if n > 0:
                jitter = np.random.normal(0, 0.02, n)
                x = np.ones(n) * pos + jitter
                ax.scatter(x, values, alpha=0.2, s=8, color=color, edgecolors='none')

        # 娣诲姞鏄捐憲鎬ф爣璁?
        p_pr, sig_pr = calculate_significance(data_list[1], data_list[2])
        if sig_pr != 'ns':
            all_max = max(max(data_list[0]) if data_list[0] else 0,
                         max(data_list[1]) if data_list[1] else 0,
                         max(data_list[2]) if data_list[2] else 0)
            y_pos = all_max * 1.08
            ax.plot([positions[0], positions[2]], [y_pos, y_pos], 'k-', linewidth=1.0)
            ax.text((positions[0] + positions[2]) / 2, y_pos * 1.01, sig_pr,
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 璁剧疆X杞存爣绛?
    x_ticks = [i * 4 + 1 for i in range(len(aspects))]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(aspect_labels, fontsize=10)

    # 璁剧疆鏍囬鍜屾爣绛?
    ax.set_ylabel('GO Semantic Similarity', fontsize=10, fontweight='bold')
    ax.set_ylim([0, 1.15])

    # 缃戞牸
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)


# 鍒涘缓鍥捐〃锛?x2甯冨眬锛屽洓涓浘澶у皬涓€鑷?
# 绗竴琛岋細鍥続锛圙O鍚堝苟鍥撅級銆佸浘B锛堝畾浣嶄竴鑷存€э級
# 绗簩琛岋細鍥綜鐨勪袱涓瓙鍥撅紙Pearson銆丼pearman锛?
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                      left=0.08, right=0.96, top=0.92, bottom=0.08)

# 棰滆壊鏂规锛欿nown=娣辩孩, Predicted=娴呯孩/绮? Random=钃?
colors = ['#C85A5A', '#F5A5A5', '#7BAFD4']

# ========== 绗竴琛岀涓€鍒楋細鍥続 - GO璇箟鐩镐技鎬э紙鍚堝苟鍥撅級==========
print("\n缁樺埗GO璇箟鐩镐技鎬э紙鍚堝苟鍥撅級...")

ax_go = fig.add_subplot(gs[0, 0])
plot_go_combined(ax_go, known_results, predicted_results, random_results, colors)

# ========== 绗竴琛岀浜屽垪锛氬浘B - 瀹氫綅涓€鑷存€?==========
print("\n缁樺埗瀹氫綅涓€鑷存€?..")

ax_loc = fig.add_subplot(gs[0, 1])
known_loc = extract_metric(known_results, ['metrics', 'localization', 'avg_jaccard'])
pred_loc = extract_metric(predicted_results, ['metrics', 'localization', 'avg_jaccard'])
rand_loc = extract_metric(random_results, ['metrics', 'localization', 'avg_jaccard'])

print(f"  Localization: Known={len(known_loc)}, Predicted={len(pred_loc)}, Random={len(rand_loc)}")

plot_single_comparison(ax_loc, known_loc, pred_loc, rand_loc,
                      'Localization', 'Jaccard Score', colors, True)

# ========== 绗簩琛岀涓€鍒楋細鍥綜 - 琛ㄨ揪涓€鑷存€э紙Pearson锛?=========
print("\n缁樺埗鍏辫〃杈句竴鑷存€?..")

# Coexpression - Pearson
ax_pear = fig.add_subplot(gs[1, 0])
known_pear = extract_metric(known_results, ['metrics', 'coexpression', 'pearson'])
pred_pear = extract_metric(predicted_results, ['metrics', 'coexpression', 'pearson'])
rand_pear = extract_metric(random_results, ['metrics', 'coexpression', 'pearson'])

print(f"  Pearson: Known={len(known_pear)}, Predicted={len(pred_pear)}, Random={len(rand_pear)}")

plot_single_comparison(ax_pear, known_pear, pred_pear, rand_pear,
                      'C1: Coexpression (Pearson)', 'Correlation', colors, True, [-0.2, 0.9])

# ========== 绗簩琛岀浜屽垪锛氬浘C - 琛ㄨ揪涓€鑷存€э紙Spearman锛?=========
# Coexpression - Spearman
ax_spear = fig.add_subplot(gs[1, 1])
known_spear = extract_metric(known_results, ['metrics', 'coexpression', 'spearman'])
pred_spear = extract_metric(predicted_results, ['metrics', 'coexpression', 'spearman'])
rand_spear = extract_metric(random_results, ['metrics', 'coexpression', 'spearman'])

print(f"  Spearman: Known={len(known_spear)}, Predicted={len(pred_spear)}, Random={len(rand_spear)}")

plot_single_comparison(ax_spear, known_spear, pred_spear, rand_spear,
                      'C2: Coexpression (Spearman)', 'Correlation', colors, True, [0, 0.9])

# ========== 娣诲姞鍚勯儴鍒嗘爣棰橈紙閬垮厤涓庡浘鏍囬閲嶅彔锛?=========
# 璁＄畻瀛愬浘鐨勫疄闄呬腑蹇冧綅缃?
# 甯冨眬锛歭eft=0.08, right=0.96, wspace=0.3
# 鎬诲搴?= 0.88, 鍗曞垪瀹藉害 = 0.88 / 2.3 鈮?0.383
# 宸﹀垪涓績 鈮?0.27, 鍙冲垪涓績 鈮?0.77, 鏁翠綋涓績 = 0.52

# 鍥続鏍囬锛堝乏涓婏紝灞呬腑锛?
fig.text(0.27, 0.96, '(A)  Protein complex subunits functional similarity',
         fontsize=12, fontweight='bold', va='top', ha='center')

# 鍥綛鏍囬锛堝彸涓婏紝灞呬腑锛?
fig.text(0.77, 0.96, '(B)  Protein complex subunits localization consistency',
         fontsize=12, fontweight='bold', va='top', ha='center')

# 鍥綜鏍囬锛堝乏涓嬶紝璺ㄨ秺涓や釜瀛愬浘锛屾暣浣撳眳涓級
fig.text(0.52, 0.48, '(C)  Protein complex subunits expression consistency',
         fontsize=12, fontweight='bold', va='top', ha='center')

# ========== 娣诲姞缁熶竴鍥句緥 ==========
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#C85A5A', edgecolor='black', alpha=0.7, label='Known'),
    Patch(facecolor='#F5A5A5', edgecolor='black', alpha=0.7, label='Predicted'),
    Patch(facecolor='#7BAFD4', edgecolor='black', alpha=0.7, label='Random')
]

fig.legend(handles=legend_elements, loc='upper right',
          bbox_to_anchor=(0.98, 0.92), fontsize=10, frameon=True,
          edgecolor='black', framealpha=0.9)

# 淇濆瓨
output_path = RESULT_DIR / 'validation_combined_figure.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n鉁?淇濆瓨缁煎悎鍥? {output_path}")
print(f"  鏂囦欢澶у皬: {output_path.stat().st_size / 1024:.1f} KB")

plt.close()

print("\n" + "="*80)
print(" 缁煎悎楠岃瘉鍥剧敓鎴愬畬鎴愶紒")
print("="*80)
print("\n鍖呭惈鍐呭:")
print("  (A) Functional similarity: GO:BP, GO:CC, GO:MF")
print("  (B) 瀹氫綅涓€鑷存€? Localization")
print("  (C) 琛ㄨ揪涓€鑷存€? Coexpression (Pearson), Coexpression (Spearman)")
print(f"\n杈撳嚭鏂囦欢: {output_path}")



