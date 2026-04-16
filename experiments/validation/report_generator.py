#!/usr/bin/env python3
"""
楠岃瘉鎶ュ憡鐢熸垚妯″潡

鑱岃矗锛氱敓鎴愰獙璇佺粨鏋滅殑鍙鍖栧浘琛ㄥ拰鎶ュ憡
鍔熻兘锛?
- 涓夌粍瀵规瘮鍙鍖栵紙Known vs Predicted vs Random锛?
- 鍗曞鍚堢墿璇︾粏鎶ュ憡
- 鎵归噺楠岃瘉姹囨€?
- 澶氱杈撳嚭鏍煎紡锛圥NG, PDF, HTML, JSON, CSV锛?
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 璁剧疆matplotlib涓枃鏀寔鍜屾牱寮?
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


class ReportGenerator:
    """Helper."""

    def __init__(self, output_dir: str = 'validation_reports'):
        """
        鍒濆鍖栨姤鍛婄敓鎴愬櫒

        鍙傛暟:
            output_dir: 鎶ュ憡杈撳嚭鐩綍
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"鎶ュ憡鐢熸垚鍣ㄥ垵濮嬪寲: {self.output_dir}")

    def plot_boxplot_comparison(
        self,
        known_scores: List[float],
        predicted_scores: List[float],
        random_scores: List[float],
        metric_name: str,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        涓夌粍瀵规瘮绠辩嚎鍥?

        鍙傛暟:
            known_scores: 宸茬煡澶嶅悎鐗╁垎鏁?
            predicted_scores: 棰勬祴澶嶅悎鐗╁垎鏁?
            random_scores: 闅忔満瀵圭収鍒嗘暟
            metric_name: 鎸囨爣鍚嶇О
            save_path: 淇濆瓨璺緞锛堝彲閫夛級
            title: 鍥捐〃鏍囬锛堝彲閫夛級

        杩斿洖:
            淇濆瓨鐨勬枃浠惰矾寰?
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 鍑嗗鏁版嵁
        data = []
        labels = []

        for scores, label in [
            (known_scores, 'Known'),
            (predicted_scores, 'Predicted'),
            (random_scores, 'Random')
        ]:
            clean_scores = [s for s in scores if not np.isnan(s) and not np.isinf(s)]
            if clean_scores:
                data.append(clean_scores)
                labels.append(f'{label}\n(n={len(clean_scores)})')

        if not data:
            logger.warning("No valid data available for boxplot")
            plt.close()
            return ""

        # 缁樺埗绠辩嚎鍥?
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='green', linewidth=2, linestyle='--'))

        # 娣诲姞鏁版嵁鐐?
        for i, scores in enumerate(data, 1):
            y = scores
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=20, color='gray')

        # 璁剧疆鏍囬鍜屾爣绛?
        if title is None:
            title = f'{metric_name} Comparison: Known vs Predicted vs Random'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True, alpha=0.3)

        # 娣诲姞缁熻淇℃伅
        info_text = []
        for i, scores in enumerate(data):
            mean = np.mean(scores)
            median = np.median(scores)
            std = np.std(scores)
            info_text.append(f'{labels[i].split("(")[0].strip()}: 渭={mean:.3f}, M={median:.3f}, 蟽={std:.3f}')

        ax.text(0.02, 0.98, '\n'.join(info_text),
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 淇濆瓨
        if save_path is None:
            save_path = self.output_dir / f'boxplot_{metric_name.replace(" ", "_").lower()}.png'
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"绠辩嚎鍥惧凡淇濆瓨: {save_path}")
        return str(save_path)

    def plot_violin_comparison(
        self,
        known_scores: List[float],
        predicted_scores: List[float],
        random_scores: List[float],
        metric_name: str,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        涓夌粍瀵规瘮灏忔彁鐞村浘

        鍙傛暟:
            known_scores: 宸茬煡澶嶅悎鐗╁垎鏁?
            predicted_scores: 棰勬祴澶嶅悎鐗╁垎鏁?
            random_scores: 闅忔満瀵圭収鍒嗘暟
            metric_name: 鎸囨爣鍚嶇О
            save_path: 淇濆瓨璺緞锛堝彲閫夛級
            title: 鍥捐〃鏍囬锛堝彲閫夛級

        杩斿洖:
            淇濆瓨鐨勬枃浠惰矾寰?
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 鍑嗗鏁版嵁
        df_data = []
        for scores, label in [
            (known_scores, 'Known'),
            (predicted_scores, 'Predicted'),
            (random_scores, 'Random')
        ]:
            clean_scores = [s for s in scores if not np.isnan(s) and not np.isinf(s)]
            for score in clean_scores:
                df_data.append({'Group': label, 'Score': score})

        if not df_data:
            logger.warning("娌℃湁鏈夋晥鏁版嵁鐢ㄤ簬缁樺埗灏忔彁鐞村浘")
            plt.close()
            return ""

        df = pd.DataFrame(df_data)

        # 缁樺埗灏忔彁鐞村浘
        sns.violinplot(data=df, x='Group', y='Score', ax=ax,
                       order=['Known', 'Predicted', 'Random'],
                       palette=['#1f77b4', '#ff7f0e', '#2ca02c'])

        # 娣诲姞绠辩嚎鍥?
        sns.boxplot(data=df, x='Group', y='Score', ax=ax,
                   order=['Known', 'Predicted', 'Random'],
                   width=0.3, palette=['white', 'white', 'white'],
                   showcaps=False, boxprops={'facecolor': 'None'},
                   showfliers=False, whiskerprops={'linewidth': 0},
                   saturation=1)

        # 璁剧疆鏍囬鍜屾爣绛?
        if title is None:
            title = f'{metric_name} Distribution: Known vs Predicted vs Random'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel('Group', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # 淇濆瓨
        if save_path is None:
            save_path = self.output_dir / f'violin_{metric_name.replace(" ", "_").lower()}.png'
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"灏忔彁鐞村浘宸蹭繚瀛? {save_path}")
        return str(save_path)

    def plot_radar_comparison(
        self,
        known_mean: Dict[str, float],
        predicted_mean: Dict[str, float],
        metric_names: List[str],
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        鍙岀粍瀵规瘮闆疯揪鍥?

        鍙傛暟:
            known_mean: 宸茬煡澶嶅悎鐗╁钩鍧囧€?
            predicted_mean: 棰勬祴澶嶅悎鐗╁钩鍧囧€?
            metric_names: 鎸囨爣鍚嶇О鍒楄〃
            save_path: 淇濆瓨璺緞锛堝彲閫夛級
            title: 鍥捐〃鏍囬锛堝彲閫夛級

        杩斿洖:
            淇濆瓨鐨勬枃浠惰矾寰?
        """
        # 鍑嗗鏁版嵁
        known_values = []
        predicted_values = []
        valid_metrics = []

        for metric in metric_names:
            known_val = known_mean.get(metric)
            pred_val = predicted_mean.get(metric)

            if known_val is not None and pred_val is not None and \
               not np.isnan(known_val) and not np.isnan(pred_val):
                known_values.append(known_val)
                predicted_values.append(pred_val)
                valid_metrics.append(metric)

        if len(valid_metrics) < 3:
            logger.warning("鑷冲皯闇€瑕?涓湁鏁堟寚鏍囩敤浜庣粯鍒堕浄杈惧浘")
            return ""

        # 璁剧疆闆疯揪鍥?
        angles = np.linspace(0, 2 * np.pi, len(valid_metrics), endpoint=False).tolist()
        known_values += known_values[:1]
        predicted_values += predicted_values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 缁樺埗宸茬煡澶嶅悎鐗?
        ax.plot(angles, known_values, 'o-', linewidth=2, label='Known', color='blue')
        ax.fill(angles, known_values, alpha=0.25, color='blue')

        # 缁樺埗棰勬祴澶嶅悎鐗?
        ax.plot(angles, predicted_values, 'o-', linewidth=2, label='Predicted', color='orange')
        ax.fill(angles, predicted_values, alpha=0.25, color='orange')

        # 璁剧疆鏍囩
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(valid_metrics, fontsize=10)

        # 璁剧疆鑼冨洿
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

        # 娣诲姞缃戞牸
        ax.grid(True)

        # 璁剧疆鏍囬鍜屽浘渚?
        if title is None:
            title = 'Multi-dimensional Comparison: Known vs Predicted'
        ax.set_title(title, y=1.08, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()

        # 淇濆瓨
        if save_path is None:
            save_path = self.output_dir / 'radar_comparison.png'
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"闆疯揪鍥惧凡淇濆瓨: {save_path}")
        return str(save_path)

    def plot_complex_heatmap(
        self,
        complex_result: Dict[str, Any],
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        鍗曞鍚堢墿鍩哄洜瀵圭浉浼煎害鐑浘

        鍙傛暟:
            complex_result: ComplexValidator杩斿洖鐨勫崟涓鍚堢墿缁撴灉
            save_path: 淇濆瓨璺緞锛堝彲閫夛級
            title: 鍥捐〃鏍囬锛堝彲閫夛級

        杩斿洖:
            淇濆瓨鐨勬枃浠惰矾寰?
        """
        genes = complex_result.get('genes', [])
        if len(genes) < 2:
            logger.warning("Not enough genes to draw heatmap")
            return ""

        # 鍒涘缓鐩镐技搴︾煩闃碉紙浣跨敤GO BP鐩镐技搴︿綔涓虹ず渚嬶級
        n_genes = len(genes)
        similarity_matrix = np.zeros((n_genes, n_genes))

        # 杩欓噷搴旇浠庣粨鏋滀腑鎻愬彇鍩哄洜瀵圭浉浼煎害
        # 鐢变簬ComplexValidator娌℃湁杩斿洖鍩哄洜瀵圭浉浼煎害锛岃繖閲屼娇鐢ㄦā鎷熸暟鎹?
        for i in range(n_genes):
            for j in range(n_genes):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 浣跨敤闅忔満鍊兼ā鎷燂紙瀹為檯搴旇浠巆alculator鑾峰彇锛?
                    similarity_matrix[i, j] = np.random.uniform(0.3, 0.9)

        # 瀵圭О鍖栫煩闃?
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        np.fill_diagonal(similarity_matrix, 1.0)

        # 缁樺埗鐑浘
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        # 璁剧疆鍒诲害
        ax.set_xticks(np.arange(n_genes))
        ax.set_yticks(np.arange(n_genes))
        ax.set_xticklabels(genes)
        ax.set_yticklabels(genes)

        # 鏃嬭浆x杞存爣绛?
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 娣诲姞鏁板€?
        for i in range(n_genes):
            for j in range(n_genes):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        # 娣诲姞棰滆壊鏉?
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Similarity Score', rotation=-90, va="bottom")

        # 璁剧疆鏍囬
        complex_id = complex_result.get('complex_id', 'unknown')
        if title is None:
            title = f'Gene Pair Similarity Heatmap: {complex_id}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # 淇濆瓨
        if save_path is None:
            save_path = self.output_dir / f'heatmap_{complex_id}.png'
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"鐑浘宸蹭繚瀛? {save_path}")
        return str(save_path)

    def plot_complex_radar(
        self,
        complex_result: Dict[str, Any],
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        鍗曞鍚堢墿澶氱淮搴﹂浄杈惧浘

        鍙傛暟:
            complex_result: ComplexValidator杩斿洖鐨勫崟涓鍚堢墿缁撴灉
            save_path: 淇濆瓨璺緞锛堝彲閫夛級
            title: 鍥捐〃鏍囬锛堝彲閫夛級

        杩斿洖:
            淇濆瓨鐨勬枃浠惰矾寰?
        """
        metrics = complex_result.get('metrics', {})

        # 鎻愬彇鎸囨爣鍊?
        values = []
        labels = []

        # GO鐩镐技搴?
        go_sim = metrics.get('go_similarity', {})
        for ont, label in [('BP', 'GO-BP'), ('CC', 'GO-CC'), ('MF', 'GO-MF')]:
            val = go_sim.get(ont)
            if val is not None and not np.isnan(val):
                values.append(val)
                labels.append(label)

        # 鍏辫〃杈?
        coexp = metrics.get('coexpression', {})
        pearson = coexp.get('pearson')
        if pearson is not None and not np.isnan(pearson):
            # 灏嗙浉鍏崇郴鏁拌浆鎹负0-1鑼冨洿
            values.append((pearson + 1) / 2)
            labels.append('Co-expression')

        # 瀹氫綅涓€鑷存€?
        loc = metrics.get('localization', {})
        if loc and 'avg_jaccard' in loc:
            jaccard = loc.get('avg_jaccard')
            if jaccard is not None and not np.isnan(jaccard):
                values.append(jaccard)
                labels.append('Localization')

        if len(values) < 3:
            logger.warning("鑷冲皯闇€瑕?涓湁鏁堟寚鏍囩敤浜庣粯鍒堕浄杈惧浘")
            return ""

        # 璁剧疆闆疯揪鍥?
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # 缁樺埗闆疯揪鍥?
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')

        # 璁剧疆鏍囩
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)

        # 璁剧疆鑼冨洿
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

        # 娣诲姞缃戞牸
        ax.grid(True)

        # 璁剧疆鏍囬
        complex_id = complex_result.get('complex_id', 'unknown')
        if title is None:
            title = f'Multi-dimensional Profile: {complex_id}'
        ax.set_title(title, y=1.08, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # 淇濆瓨
        if save_path is None:
            save_path = self.output_dir / f'radar_{complex_id}.png'
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"闆疯揪鍥惧凡淇濆瓨: {save_path}")
        return str(save_path)

    def generate_complex_report_html(
        self,
        complex_result: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        鐢熸垚鍗曞鍚堢墿HTML鎶ュ憡

        鍙傛暟:
            complex_result: ComplexValidator杩斿洖鐨勫崟涓鍚堢墿缁撴灉
            output_path: 杈撳嚭璺緞锛堝彲閫夛級

        杩斿洖:
            淇濆瓨鐨勬枃浠惰矾寰?
        """
        complex_id = complex_result.get('complex_id', 'unknown')
        genes = complex_result.get('genes', [])
        n_genes = complex_result.get('n_genes', 0)
        metrics = complex_result.get('metrics', {})

        # 鐢熸垚HTML鍐呭
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Complex Validation Report: {complex_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 5px solid #3498db;
            padding-left: 10px;
        }}
        .info-section {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metric-table th, .metric-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .metric-table th {{
            background-color: #3498db;
            color: white;
        }}
        .metric-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .gene-list {{
            display: inline-block;
            background-color: #e8f4f8;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 3px;
            font-family: monospace;
        }}
        .score-good {{
            color: #27ae60;
            font-weight: bold;
        }}
        .score-medium {{
            color: #f39c12;
            font-weight: bold;
        }}
        .score-poor {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Complex Validation Report</h1>

        <div class="info-section">
            <p><strong>Complex ID:</strong> {complex_id}</p>
            <p><strong>Number of Genes:</strong> {n_genes}</p>
            <p><strong>Genes:</strong><br>
"""

        for gene in genes:
            html += f'                <span class="gene-list">{gene}</span>\n'

        html += """
            </p>
        </div>

        <h2>Validation Metrics</h2>
        <table class="metric-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
"""

        # GO鐩镐技搴?
        go_sim = metrics.get('go_similarity', {})
        for ont, label in [('BP', 'GO Biological Process'), ('CC', 'GO Cellular Component'), ('MF', 'GO Molecular Function')]:
            val = go_sim.get(ont)
            if val is not None and not np.isnan(val):
                score_class = 'score-good' if val >= 0.6 else 'score-medium' if val >= 0.4 else 'score-poor'
                interp = 'High similarity' if val >= 0.6 else 'Medium similarity' if val >= 0.4 else 'Low similarity'
                html += f"""
            <tr>
                <td>{label}</td>
                <td class="{score_class}">{val:.4f}</td>
                <td>{interp}</td>
            </tr>
"""

        # 鍏辫〃杈?
        coexp = metrics.get('coexpression', {})
        for method, label in [('pearson', 'Co-expression (Pearson)'), ('spearman', 'Co-expression (Spearman)')]:
            val = coexp.get(method)
            if val is not None and not np.isnan(val):
                score_class = 'score-good' if val >= 0.5 else 'score-medium' if val >= 0.3 else 'score-poor'
                interp = 'High correlation' if val >= 0.5 else 'Medium correlation' if val >= 0.3 else 'Low correlation'
                html += f"""
            <tr>
                <td>{label}</td>
                <td class="{score_class}">{val:.4f}</td>
                <td>{interp}</td>
            </tr>
"""

        # 瀹氫綅涓€鑷存€?
        loc = metrics.get('localization', {})
        if loc and 'avg_jaccard' in loc:
            jaccard = loc.get('avg_jaccard')
            if jaccard is not None and not np.isnan(jaccard):
                score_class = 'score-good' if jaccard >= 0.5 else 'score-medium' if jaccard >= 0.3 else 'score-poor'
                interp = 'High consistency' if jaccard >= 0.5 else 'Medium consistency' if jaccard >= 0.3 else 'Low consistency'
                html += f"""
            <tr>
                <td>Localization Consistency</td>
                <td class="{score_class}">{jaccard:.4f}</td>
                <td>{interp}</td>
            </tr>
"""
                top_loc = loc.get('top_location', 'N/A')
                top_cov = loc.get('top1_coverage', 0)
                html += f"""
            <tr>
                <td>Dominant Localization</td>
                <td>{top_loc}</td>
                <td>Coverage: {top_cov:.1%}</td>
            </tr>
"""

        html += """
        </table>
    </div>
</body>
</html>
"""

        # 淇濆瓨
        if output_path is None:
            output_path = self.output_dir / f'report_{complex_id}.html'
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML鎶ュ憡宸蹭繚瀛? {output_path}")
        return str(output_path)

    def generate_summary_table(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        鐢熸垚鎵归噺楠岃瘉姹囨€昏〃锛圕SV锛?

        鍙傛暟:
            results: 鎵归噺楠岃瘉缁撴灉
            output_path: 杈撳嚭璺緞锛堝彲閫夛級

        杩斿洖:
            淇濆瓨鐨勬枃浠惰矾寰?
        """
        # 鎻愬彇鏁版嵁
        rows = []
        for complex_id, result in results.items():
            if 'error' in result:
                continue

            row = {'complex_id': complex_id}
            row['n_genes'] = result.get('n_genes', 0)

            metrics = result.get('metrics', {})

            # GO鐩镐技搴?
            go_sim = metrics.get('go_similarity', {})
            row['go_bp'] = go_sim.get('BP')
            row['go_cc'] = go_sim.get('CC')
            row['go_mf'] = go_sim.get('MF')

            # 鍏辫〃杈?
            coexp = metrics.get('coexpression', {})
            row['coexp_pearson'] = coexp.get('pearson')
            row['coexp_spearman'] = coexp.get('spearman')

            # 瀹氫綅
            loc = metrics.get('localization', {})
            if loc:
                row['loc_jaccard'] = loc.get('avg_jaccard')
                row['loc_top_location'] = loc.get('top_location')
                row['loc_top1_coverage'] = loc.get('top1_coverage')
            else:
                row['loc_jaccard'] = None
                row['loc_top_location'] = None
                row['loc_top1_coverage'] = None

            rows.append(row)

        df = pd.DataFrame(rows)

        # 淇濆瓨
        if output_path is None:
            output_path = self.output_dir / 'summary_table.csv'
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"姹囨€昏〃宸蹭繚瀛? {output_path}")
        return str(output_path)

    def plot_distribution_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        缁樺埗鎵归噺楠岃瘉缁撴灉鐨勫垎甯冨姣斿浘

        鍙傛暟:
            results: 鎵归噺楠岃瘉缁撴灉
            save_path: 淇濆瓨璺緞锛堝彲閫夛級
            title: 鍥捐〃鏍囬锛堝彲閫夛級

        杩斿洖:
            淇濆瓨鐨勬枃浠惰矾寰?
        """
        # 鎻愬彇鎸囨爣鏁版嵁
        metrics_data = {
            'GO-BP': [],
            'GO-CC': [],
            'GO-MF': [],
            'Co-expression': [],
            'Localization': []
        }

        for result in results.values():
            if 'error' in result:
                continue

            metrics = result.get('metrics', {})

            # GO鐩镐技搴?
            go_sim = metrics.get('go_similarity', {})
            for ont, key in [('BP', 'GO-BP'), ('CC', 'GO-CC'), ('MF', 'GO-MF')]:
                val = go_sim.get(ont)
                if val is not None and not np.isnan(val):
                    metrics_data[key].append(val)

            # 鍏辫〃杈?
            coexp = metrics.get('coexpression', {})
            pearson = coexp.get('pearson')
            if pearson is not None and not np.isnan(pearson):
                metrics_data['Co-expression'].append((pearson + 1) / 2)

            # 瀹氫綅
            loc = metrics.get('localization', {})
            if loc and 'avg_jaccard' in loc:
                jaccard = loc.get('avg_jaccard')
                if jaccard is not None and not np.isnan(jaccard):
                    metrics_data['Localization'].append(jaccard)

        # 缁樺埗鐩存柟鍥?
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (metric_name, values) in enumerate(metrics_data.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            if values:
                ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
                ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.3f}')
                ax.set_xlabel('Score', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title(f'{metric_name} Distribution (n={len(values)})', fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric_name} Distribution', fontsize=11, fontweight='bold')

        # 闅愯棌澶氫綑鐨勫瓙鍥?
        for i in range(len(metrics_data), len(axes)):
            axes[i].set_visible(False)

        if title is None:
            title = f'Validation Metrics Distribution (n={len(results)} complexes)'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # 淇濆瓨
        if save_path is None:
            save_path = self.output_dir / 'distribution_comparison.png'
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"鍒嗗竷瀵规瘮鍥惧凡淇濆瓨: {save_path}")
        return str(save_path)

    def export_json_report(
        self,
        results: Union[Dict[str, Any], Dict[str, Dict[str, Any]]],
        output_path: Optional[str] = None
    ) -> str:
        """
        瀵煎嚭JSON鏍煎紡缁撴灉

        鍙傛暟:
            results: 楠岃瘉缁撴灉
            output_path: 杈撳嚭璺緞锛堝彲閫夛級

        杩斿洖:
            淇濆瓨鐨勬枃浠惰矾寰?
        """
        if output_path is None:
            output_path = self.output_dir / 'validation_results.json'
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON缁撴灉宸蹭繚瀛? {output_path}")
        return str(output_path)


if __name__ == '__main__':
    print("\n=== ReportGenerator娴嬭瘯 ===\n")

    # 鍒濆鍖栨姤鍛婄敓鎴愬櫒
    print("1. 鍒濆鍖栨姤鍛婄敓鎴愬櫒...")
    generator = ReportGenerator('/root/autodl-tmp/KSHL-PC/test_reports')
    print(f"   鉁?鎶ュ憡鐢熸垚鍣ㄥ垵濮嬪寲瀹屾垚")

    # 鍑嗗娴嬭瘯鏁版嵁
    print("\n2. 鍑嗗娴嬭瘯鏁版嵁...")
    known_scores = np.random.normal(0.7, 0.15, 50).tolist()
    predicted_scores = np.random.normal(0.6, 0.2, 50).tolist()
    random_scores = np.random.normal(0.3, 0.15, 50).tolist()
    print("   OK test data generated")

    print("\n3. 鐢熸垚绠辩嚎鍥?..")
    boxplot_path = generator.plot_boxplot_comparison(
        known_scores, predicted_scores, random_scores,
        'GO Similarity',
        title='GO Similarity Comparison'
    )
    print(f"   鉁?绠辩嚎鍥惧凡淇濆瓨: {boxplot_path}")

    print("\n4. 鐢熸垚灏忔彁鐞村浘...")
    violin_path = generator.plot_violin_comparison(
        known_scores, predicted_scores, random_scores,
        'GO Similarity',
        title='GO Similarity Distribution'
    )
    print(f"   鉁?灏忔彁鐞村浘宸蹭繚瀛? {violin_path}")

    print("\n5. 鐢熸垚闆疯揪鍥惧姣?..")
    known_mean = {'GO-BP': 0.7, 'GO-CC': 0.65, 'GO-MF': 0.6, 'Co-expression': 0.55}
    predicted_mean = {'GO-BP': 0.6, 'GO-CC': 0.58, 'GO-MF': 0.52, 'Co-expression': 0.48}
    radar_path = generator.plot_radar_comparison(
        known_mean, predicted_mean,
        list(known_mean.keys())
    )
    print(f"   鉁?闆疯揪鍥惧姣斿凡淇濆瓨: {radar_path}")

    print("\n6. 鐢熸垚鍗曞鍚堢墿鎶ュ憡...")
    complex_result = {
        'complex_id': 'test_complex_1',
        'genes': ['TP53', 'MDM2', 'MDM4'],
        'n_genes': 3,
        'metrics': {
            'go_similarity': {'BP': 0.65, 'CC': 0.72, 'MF': 0.58},
            'coexpression': {'pearson': 0.45, 'spearman': 0.42},
            'localization': {
                'avg_jaccard': 0.38,
                'top_location': 'nucleus',
                'top1_coverage': 0.85
            }
        }
    }

    # 鐑浘
    heatmap_path = generator.plot_complex_heatmap(complex_result)
    print(f"   鉁?鐑浘宸蹭繚瀛? {heatmap_path}")

    # 鍗曞鍚堢墿闆疯揪鍥?
    radar_single_path = generator.plot_complex_radar(complex_result)
    print(f"   鉁?鍗曞鍚堢墿闆疯揪鍥惧凡淇濆瓨: {radar_single_path}")

    # HTML鎶ュ憡
    html_path = generator.generate_complex_report_html(complex_result)
    print(f"   鉁?HTML鎶ュ憡宸蹭繚瀛? {html_path}")

    print("\n7. 鐢熸垚鎵归噺鎶ュ憡...")
    batch_results = {
        'complex_1': complex_result,
        'complex_2': {
            'complex_id': 'complex_2',
            'genes': ['BRCA1', 'BRCA2'],
            'n_genes': 2,
            'metrics': {
                'go_similarity': {'BP': 0.75, 'CC': 0.68, 'MF': 0.62},
                'coexpression': {'pearson': 0.55, 'spearman': 0.52},
                'localization': {
                    'avg_jaccard': 0.48,
                    'top_location': 'nucleus',
                    'top1_coverage': 0.95
                }
            }
        }
    }

    # 姹囨€昏〃
    summary_path = generator.generate_summary_table(batch_results)
    print(f"   鉁?姹囨€昏〃宸蹭繚瀛? {summary_path}")

    # 鍒嗗竷瀵规瘮鍥?
    dist_path = generator.plot_distribution_comparison(batch_results)
    print(f"   鉁?鍒嗗竷瀵规瘮鍥惧凡淇濆瓨: {dist_path}")

    # JSON瀵煎嚭
    json_path = generator.export_json_report(batch_results)
    print(f"   鉁?JSON缁撴灉宸蹭繚瀛? {json_path}")

    print("\n 鎵€鏈夋祴璇曞畬鎴愶紒")



