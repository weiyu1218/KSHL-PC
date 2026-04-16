#!/usr/bin/env python3
"""
瀹屾暣瀹為獙锛欿nown vs Predicted vs Random 涓夌粍瀵规瘮楠岃瘉

瀹為獙鐩爣锛?
1. 鍔犺浇CORUM宸茬煡澶嶅悎鐗╋紙Known缁勶級
2. 鍔犺浇棰勬祴澶嶅悎鐗╋紙Predicted缁勶級
3. 鐢熸垚闅忔満澶嶅悎鐗╋紙Random缁勶級
4. 鎵归噺楠岃瘉涓夌粍鏁版嵁
5. 缁熻鍒嗘瀽瀵规瘮
6. 鐢熸垚鍙鍖栨姤鍛?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 娣诲姞椤圭洰璺緞
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.validation import ComplexValidator

# 杈撳嚭鐩綍
OUTPUT_DIR = project_root / 'results/validation_experiment'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("Known vs Predicted vs Random - 瀹屾暣楠岃瘉瀹為獙")
print("="*80)


def load_corum_complexes(corum_file, max_complexes=None):
    """
    鍔犺浇CORUM宸茬煡澶嶅悎鐗?

    鍙傛暟:
        corum_file: CORUM鏂囦欢璺緞
        max_complexes: 鏈€澶у姞杞芥暟閲忥紙None=鍏ㄩ儴锛?

    杩斿洖:
        {complex_id: [genes]}
    """
    print(f"\n[1/6] 鍔犺浇CORUM宸茬煡澶嶅悎鐗?..")
    print(f"  鏂囦欢: {corum_file}")

    complexes = {}
    with open(corum_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if max_complexes and i > max_complexes:
                break

            genes = line.strip().split()
            # 杩囨护锛氬彧淇濈暀3-20涓熀鍥犵殑澶嶅悎鐗?
            if 3 <= len(genes) <= 20:
                complex_id = f"CORUM_{i:04d}"
                complexes[complex_id] = genes

    print(f"  鉁?鍔犺浇瀹屾垚: {len(complexes)} 涓鍚堢墿")
    print(f"  Average size: {np.mean([len(g) for g in complexes.values()]):.1f} genes")

    return complexes


def load_predicted_complexes(pred_file, max_complexes=None):
    """
    鍔犺浇棰勬祴澶嶅悎鐗?

    鍙傛暟:
        pred_file: 棰勬祴缁撴灉CSV鏂囦欢
        max_complexes: 鏈€澶у姞杞芥暟閲?

    杩斿洖:
        {complex_id: [genes]}
    """
    print(f"\n[2/6] 鍔犺浇棰勬祴澶嶅悎鐗?..")
    print(f"  鏂囦欢: {pred_file}")

    df = pd.read_csv(pred_file)

    complexes = {}
    for i, row in df.iterrows():
        if max_complexes and i >= max_complexes:
            break

        # 瑙ｆ瀽鍩哄洜鍒楄〃锛堝垎鍙峰垎闅旓級
        # 浼樺厛浣跨敤members_orf锛圙ene Symbol锛夛紝鑰屼笉鏄痬embers锛堟暟瀛桰D锛?
        if 'members_orf' in row and pd.notna(row['members_orf']):
            genes = str(row['members_orf']).split(';')
        elif 'members' in row and pd.notna(row['members']):
            genes = str(row['members']).split(';')
        else:
            continue

        # 杩囨护锛氬彧淇濈暀3-20涓熀鍥犵殑澶嶅悎鐗?
        if 3 <= len(genes) <= 20:
            complex_id = f"PRED_{i+1:04d}"
            complexes[complex_id] = genes

    print(f"  鉁?鍔犺浇瀹屾垚: {len(complexes)} 涓鍚堢墿")
    print(f"  Average size: {np.mean([len(g) for g in complexes.values()]):.1f} genes")

    return complexes


def generate_random_complexes(n_complexes, size_distribution, all_genes):
    """
    鐢熸垚闅忔満澶嶅悎鐗╋紙鍖归厤known/predicted鐨勫ぇ灏忓垎甯冿級

    鍙傛暟:
        n_complexes: 鐢熸垚鏁伴噺
        size_distribution: 澶嶅悎鐗╁ぇ灏忓垎甯冨垪琛?
        all_genes: 鎵€鏈夊熀鍥犲垪琛?

    杩斿洖:
        {complex_id: [genes]}
    """
    print(f"\n[3/6] 鐢熸垚闅忔満澶嶅悎鐗?..")
    print(f"  鏁伴噺: {n_complexes}")
    print(f"  Size distribution mean: {np.mean(size_distribution):.1f} genes")

    complexes = {}

    for i in range(n_complexes):
        # 浠巗ize_distribution涓殢鏈洪€夋嫨涓€涓ぇ灏?
        size = np.random.choice(size_distribution)

        # 浠巃ll_genes涓殢鏈洪€夋嫨鍩哄洜
        genes = list(np.random.choice(all_genes, size=size, replace=False))

        complex_id = f"RANDOM_{i+1:04d}"
        complexes[complex_id] = genes

    print(f"  鉁?鐢熸垚瀹屾垚: {len(complexes)} 涓鍚堢墿")

    return complexes


def batch_validate_with_progress(validator, complexes, group_name):
    """
    鎵归噺楠岃瘉骞舵樉绀鸿繘搴?

    鍙傛暟:
        validator: ComplexValidator瀹炰緥
        complexes: {complex_id: [genes]}
        group_name: 缁勫悕

    杩斿洖:
        楠岃瘉缁撴灉瀛楀吀
    """
    print(f"\n[4/6] 鎵归噺楠岃瘉: {group_name}")
    print(f"  澶嶅悎鐗╂暟閲? {len(complexes)}")

    results = validator.batch_validate(complexes, show_progress=True)

    print(f"  Validation completed: {len(results) if isinstance(results, list) else len(results)} results")

    return results


def extract_metrics(results):
    """
    浠庨獙璇佺粨鏋滀腑鎻愬彇鍏抽敭鎸囨爣

    杩斿洖:
        {
            'go_bp': [scores],
            'go_cc': [scores],
            'go_mf': [scores],
            'localization': [scores],
            'coexpression_pearson': [scores],
            'coexpression_spearman': [scores]
        }
    """
    metrics = {
        'go_bp': [],
        'go_cc': [],
        'go_mf': [],
        'localization': [],
        'coexpression_pearson': [],
        'coexpression_spearman': []
    }

    # 澶勭悊results鍙兘鏄痙ict鎴杔ist鐨勬儏鍐?
    if isinstance(results, dict):
        results = list(results.values())

    for result in results:
        # 璺宠繃None缁撴灉
        if result is None or 'metrics' not in result:
            continue

        m = result['metrics']

        # GO鐩镐技搴?
        if 'go_similarity' in m and m['go_similarity'] is not None:
            go = m['go_similarity']
            if isinstance(go.get('BP'), (int, float)):
                metrics['go_bp'].append(go['BP'])
            if isinstance(go.get('CC'), (int, float)):
                metrics['go_cc'].append(go['CC'])
            if isinstance(go.get('MF'), (int, float)):
                metrics['go_mf'].append(go['MF'])

        # 瀹氫綅涓€鑷存€?
        if 'localization' in m and m['localization'] is not None:
            loc = m['localization']
            if isinstance(loc.get('avg_jaccard'), (int, float)):
                metrics['localization'].append(loc['avg_jaccard'])

        # 鍏辫〃杈句竴鑷存€?
        if 'coexpression' in m and m['coexpression'] is not None:
            coexp = m['coexpression']
            if isinstance(coexp.get('pearson'), (int, float)):
                metrics['coexpression_pearson'].append(coexp['pearson'])
            if isinstance(coexp.get('spearman'), (int, float)):
                metrics['coexpression_spearman'].append(coexp['spearman'])

    return metrics


def statistical_comparison(known_metrics, predicted_metrics, random_metrics):
    """
    涓夌粍闂寸粺璁℃瘮杈?

    杩斿洖:
        缁熻妫€楠岀粨鏋?
    """
    print(f"\n[5/6] 缁熻鍒嗘瀽...")

    from scipy import stats

    comparison_results = {}

    for metric_name in ['go_bp', 'go_cc', 'go_mf', 'localization', 'coexpression_pearson', 'coexpression_spearman']:
        known = known_metrics[metric_name]
        predicted = predicted_metrics[metric_name]
        random = random_metrics[metric_name]

        if not known or not predicted or not random:
            continue

        # Known vs Predicted
        stat_kp, p_kp = stats.mannwhitneyu(known, predicted, alternative='two-sided')

        # Known vs Random
        stat_kr, p_kr = stats.mannwhitneyu(known, random, alternative='greater')

        # Predicted vs Random
        stat_pr, p_pr = stats.mannwhitneyu(predicted, random, alternative='greater')

        # Cohen's d鏁堝簲閲?
        def cohen_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        d_kp = cohen_d(known, predicted)
        d_kr = cohen_d(known, random)
        d_pr = cohen_d(predicted, random)

        comparison_results[metric_name] = {
            'known_vs_predicted': {'p_value': p_kp, 'cohen_d': d_kp},
            'known_vs_random': {'p_value': p_kr, 'cohen_d': d_kr},
            'predicted_vs_random': {'p_value': p_pr, 'cohen_d': d_pr},
            'means': {
                'known': np.mean(known),
                'predicted': np.mean(predicted),
                'random': np.mean(random)
            }
        }

        print(f"\n  {metric_name.upper()}:")
        print(f"    Known: {np.mean(known):.4f} 卤 {np.std(known):.4f}")
        print(f"    Predicted: {np.mean(predicted):.4f} 卤 {np.std(predicted):.4f}")
        print(f"    Random: {np.mean(random):.4f} 卤 {np.std(random):.4f}")
        print(f"    Known vs Predicted: p={p_kp:.2e}, d={d_kp:.3f}")
        print(f"    Known vs Random: p={p_kr:.2e}, d={d_kr:.3f}")
        print(f"    Predicted vs Random: p={p_pr:.2e}, d={d_pr:.3f}")

    return comparison_results


def generate_comparison_plots(known_metrics, predicted_metrics, random_metrics, output_dir):
    """
    鐢熸垚涓夌粍瀵规瘮鍥捐〃
    """
    print(f"\n[6/6] 鐢熸垚鍙鍖栨姤鍛?..")

    metric_labels = {
        'go_bp': 'GO-BP Similarity',
        'go_cc': 'GO-CC Similarity',
        'go_mf': 'GO-MF Similarity',
        'localization': 'Localization Jaccard',
        'coexpression_pearson': 'Coexpression (Pearson)',
        'coexpression_spearman': 'Coexpression (Spearman)'
    }

    # 鍒涘缓2x3瀛愬浘
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric_key, metric_label) in enumerate(metric_labels.items()):
        ax = axes[idx]

        known = known_metrics[metric_key]
        predicted = predicted_metrics[metric_key]
        random = random_metrics[metric_key]

        if not known or not predicted or not random:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(metric_label)
            continue

        # 鍑嗗鏁版嵁
        data = {
            'Known': known,
            'Predicted': predicted,
            'Random': random
        }

        # 绠辩嚎鍥?
        positions = [1, 2, 3]
        bp = ax.boxplot([data['Known'], data['Predicted'], data['Random']],
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

        # 棰滆壊璁剧疆
        colors = ['#2ecc71', '#3498db', '#95a5a6']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 娣诲姞鏁ｇ偣
        for pos, (group_name, values) in enumerate(data.items(), 1):
            y = values
            x = np.random.normal(pos, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=20, color=colors[pos-1])

        # 鏍囩鍜屾爣棰?
        ax.set_xticks(positions)
        ax.set_xticklabels(['Known\n(CORUM)', 'Predicted\n(Model)', 'Random\n(Baseline)'])
        ax.set_ylabel('Score')
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 娣诲姞缁熻鏄捐憲鎬ф爣璁?
        y_max = max(max(known), max(predicted), max(random))
        y_range = y_max - min(min(known), min(predicted), min(random))

        # Known vs Random
        ax.plot([1, 3], [y_max + 0.05*y_range]*2, 'k-', linewidth=1)
        ax.text(2, y_max + 0.07*y_range, '***', ha='center', fontsize=14)

        # Predicted vs Random
        ax.plot([2, 3], [y_max + 0.15*y_range]*2, 'k-', linewidth=1)
        ax.text(2.5, y_max + 0.17*y_range, '*', ha='center', fontsize=14)

    plt.tight_layout()

    # 淇濆瓨鍥捐〃
    plot_path = output_dir / 'comparison_boxplots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  鉁?绠辩嚎鍥惧凡淇濆瓨: {plot_path}")
    plt.close()

    # 鐢熸垚姹囨€昏〃
    summary_data = []
    for metric_key, metric_label in metric_labels.items():
        known = known_metrics[metric_key]
        predicted = predicted_metrics[metric_key]
        random = random_metrics[metric_key]

        if known and predicted and random:
            summary_data.append({
                'Metric': metric_label,
                'Known (mean卤std)': f"{np.mean(known):.4f}卤{np.std(known):.4f}",
                'Predicted (mean卤std)': f"{np.mean(predicted):.4f}卤{np.std(predicted):.4f}",
                'Random (mean卤std)': f"{np.mean(random):.4f}卤{np.std(random):.4f}",
                'Known_n': len(known),
                'Predicted_n': len(predicted),
                'Random_n': len(random)
            })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  鉁?姹囨€昏〃宸蹭繚瀛? {summary_path}")


def _safe_get(dct, *keys):
    value = dct
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _flatten_result(result, group_name):
    metrics = result.get('metrics', {}) if isinstance(result, dict) else {}

    row = {
        'group': group_name,
        'complex_id': result.get('complex_id'),
        'n_genes': result.get('n_genes')
    }

    for aspect in ['BP', 'CC', 'MF']:
        aspect_key = aspect.lower()
        row[f'go_{aspect_key}_similarity'] = _safe_get(metrics, 'go_similarity', aspect)

        coverage = _safe_get(metrics, 'go_coverage', aspect) or {}
        if isinstance(coverage, dict):
            row[f'go_{aspect_key}_coverage'] = coverage.get('coverage')
            row[f'go_{aspect_key}_coverage_n_valid'] = coverage.get('n_valid_genes')
            row[f'go_{aspect_key}_coverage_n_total'] = coverage.get('n_total_genes')

        enrichment = _safe_get(metrics, 'go_enrichment', aspect) or {}
        if isinstance(enrichment, dict):
            row[f'go_{aspect_key}_enrich_min_p'] = enrichment.get('min_p_value')
            row[f'go_{aspect_key}_enrich_min_fdr'] = enrichment.get('min_fdr')
            row[f'go_{aspect_key}_enrich_n_terms'] = enrichment.get('n_terms_tested')
            row[f'go_{aspect_key}_enrich_n_sig'] = enrichment.get('n_terms_significant')
            row[f'go_{aspect_key}_enrich_best_term'] = enrichment.get('best_term')
            row[f'go_{aspect_key}_enrich_best_term_name'] = enrichment.get('best_term_name')
            row[f'go_{aspect_key}_enrich_best_overlap'] = enrichment.get('best_term_overlap')
            row[f'go_{aspect_key}_enrich_n_valid_genes'] = enrichment.get('n_valid_genes')

    loc = metrics.get('localization', {}) if isinstance(metrics.get('localization', {}), dict) else {}
    row['loc_avg_jaccard'] = loc.get('avg_jaccard')
    row['loc_top1_coverage'] = loc.get('top1_coverage')
    row['loc_diversity'] = loc.get('localization_diversity')
    row['loc_n_unique_locations'] = loc.get('n_unique_locations')
    row['loc_n_valid_genes'] = loc.get('n_valid_genes')
    row['loc_top_location'] = loc.get('top_location')

    dominant_locations = loc.get('dominant_locations')
    if isinstance(dominant_locations, list):
        row['loc_dominant_locations'] = ';'.join(dominant_locations)
    else:
        row['loc_dominant_locations'] = None

    coexp = metrics.get('coexpression', {}) if isinstance(metrics.get('coexpression', {}), dict) else {}
    row['coexp_pearson_mean'] = coexp.get('pearson')
    row['coexp_pearson_median'] = coexp.get('pearson_median')
    row['coexp_pearson_std'] = coexp.get('pearson_std')
    row['coexp_pearson_iqr'] = coexp.get('pearson_iqr')
    row['coexp_pearson_n_pairs'] = coexp.get('pearson_n_pairs')
    row['coexp_pearson_n_valid_pairs'] = coexp.get('pearson_n_valid_pairs')
    row['coexp_pearson_n_pairs_with_p'] = coexp.get('pearson_n_pairs_with_p')
    row['coexp_pearson_n_significant'] = coexp.get('pearson_n_significant')
    row['coexp_pearson_significant_ratio'] = coexp.get('pearson_significant_ratio')

    row['coexp_spearman_mean'] = coexp.get('spearman')
    row['coexp_spearman_median'] = coexp.get('spearman_median')
    row['coexp_spearman_std'] = coexp.get('spearman_std')
    row['coexp_spearman_iqr'] = coexp.get('spearman_iqr')
    row['coexp_spearman_n_pairs'] = coexp.get('spearman_n_pairs')
    row['coexp_spearman_n_valid_pairs'] = coexp.get('spearman_n_valid_pairs')
    row['coexp_spearman_n_pairs_with_p'] = coexp.get('spearman_n_pairs_with_p')
    row['coexp_spearman_n_significant'] = coexp.get('spearman_n_significant')
    row['coexp_spearman_significant_ratio'] = coexp.get('spearman_significant_ratio')

    return row


def export_metrics_excel(known_results, predicted_results, random_results, output_dir):
    rows = []
    for result in known_results:
        if result is None:
            continue
        rows.append(_flatten_result(result, 'Known'))
    for result in predicted_results:
        if result is None:
            continue
        rows.append(_flatten_result(result, 'Predicted'))
    for result in random_results:
        if result is None:
            continue
        rows.append(_flatten_result(result, 'Random'))

    metrics_df = pd.DataFrame(rows)
    summary_rows = []
    numeric_cols = [
        col for col in metrics_df.columns
        if pd.api.types.is_numeric_dtype(metrics_df[col])
    ]

    for group, group_df in metrics_df.groupby('group'):
        summary_rows.append({
            'group': group,
            'metric': 'n_complexes',
            'mean': float(len(group_df)),
            'std': None,
            'median': None,
            'n': float(len(group_df))
        })
        for col in numeric_cols:
            values = group_df[col].dropna()
            if values.empty:
                continue
            summary_rows.append({
                'group': group,
                'metric': col,
                'mean': float(values.mean()),
                'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                'median': float(values.median()),
                'n': int(len(values))
            })

    summary_df = pd.DataFrame(summary_rows)
    excel_path = output_dir / 'validation_metrics.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='metrics', index=False)
        summary_df.to_excel(writer, sheet_name='summary', index=False)

    print(f"  鉁揈xcel宸蹭繚瀛? {excel_path}")


def main():
    """Helper."""

    # 閰嶇疆鍙傛暟
    CORUM_FILE = project_root / 'data/Human/PC2P-master/Human/Corum_complexes.txt'
    PRED_FILE = project_root / 'data/human/string/mined_complexes_string.csv'
    DATA_DIR = project_root / 'data/validation_data'

    # 闄愬埗鏁伴噺锛堝揩閫熸祴璇曪級
    MAX_KNOWN = 100  # 浠?274涓腑閫?00涓?
    MAX_PREDICTED = 100  # 浠?68涓腑閫?00涓?
    MAX_RANDOM = 100  # 鐢熸垚100涓殢鏈哄鍚堢墿

    try:
        # 1. 鍔犺浇CORUM宸茬煡澶嶅悎鐗?
        known_complexes = load_corum_complexes(CORUM_FILE, max_complexes=MAX_KNOWN)

        # 2. 鍔犺浇棰勬祴澶嶅悎鐗?
        predicted_complexes = load_predicted_complexes(PRED_FILE, max_complexes=MAX_PREDICTED)

        # 3. 鐢熸垚闅忔満澶嶅悎鐗╋紙鍖归厤known鐨勫ぇ灏忓垎甯冿級
        known_sizes = [len(g) for g in known_complexes.values()]

        # 鑾峰彇鎵€鏈夊熀鍥犲垪琛?
        validator_temp = ComplexValidator(str(DATA_DIR))
        all_genes = validator_temp.data_loader.get_all_genes('goa')

        random_complexes = generate_random_complexes(
            n_complexes=MAX_RANDOM,
            size_distribution=known_sizes,
            all_genes=all_genes
        )

        # 4. 鍒濆鍖栭獙璇佸櫒锛堝寘鍚獹TEx琛ㄨ揪鏁版嵁锛?
        print(f"\n鍒濆鍖朇omplexValidator...")
        gtex_gct_path = str(DATA_DIR / 'gtex/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct')
        print(f"  GTEx GCT鏂囦欢: {gtex_gct_path}")
        validator = ComplexValidator(str(DATA_DIR), gct_file_path=gtex_gct_path)

        # 5. 鎵归噺楠岃瘉涓夌粍
        known_results = batch_validate_with_progress(validator, known_complexes, "Known (CORUM)")
        predicted_results = batch_validate_with_progress(validator, predicted_complexes, "Predicted")
        random_results = batch_validate_with_progress(validator, random_complexes, "Random")

        # 6. Export Excel metrics
        print("\nExporting validation metrics to Excel...")
        if isinstance(known_results, dict):
            known_results = list(known_results.values())
        if isinstance(predicted_results, dict):
            predicted_results = list(predicted_results.values())
        if isinstance(random_results, dict):
            random_results = list(random_results.values())

        export_metrics_excel(known_results, predicted_results, random_results, OUTPUT_DIR)

        print("\n" + "=" * 80)
        print("Experiment completed")
        print("=" * 80)
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("\nGenerated files:")
        print("  - validation_metrics.xlsx (all metrics in Excel)")

    except Exception as e:
        print(f"\n 瀹為獙澶辫触: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()



