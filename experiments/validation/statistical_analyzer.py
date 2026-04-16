#!/usr/bin/env python3
"""
缁熻鍒嗘瀽妯″潡

鑱岃矗锛氬楠岃瘉缁撴灉杩涜缁熻鏄捐憲鎬ф楠屽拰鏁堝簲閲忚绠?
鏂规硶锛?
- Wilcoxon绉╁拰妫€楠岋紙闈炲弬鏁版楠岋級
- Cohen's d鏁堝簲閲?
- FDR澶氶噸妫€楠屾牎姝?
- 闅忔満瀵圭収缁勭敓鎴?
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Helper."""

    def __init__(self, all_genes: List[str], random_seed: int = 42):
        """
        鍒濆鍖栫粺璁″垎鏋愬櫒

        鍙傛暟:
            all_genes: 鎵€鏈夊彲鐢ㄥ熀鍥犲垪琛紙鐢ㄤ簬鐢熸垚闅忔満瀵圭収锛?
            random_seed: 闅忔満绉嶅瓙
        """
        self.all_genes = all_genes
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        logger.info(f"缁熻鍒嗘瀽鍣ㄥ垵濮嬪寲: {len(self.all_genes):,} 涓熀鍥犳睜")

    def generate_random_pairs(self, n_pairs: int, exclude_pairs: Optional[set] = None) -> List[Tuple[str, str]]:
        """
        鐢熸垚闅忔満鍩哄洜瀵逛綔涓哄鐓х粍

        鍙傛暟:
            n_pairs: 鐢熸垚鐨勫熀鍥犲鏁伴噺
            exclude_pairs: 闇€瑕佹帓闄ょ殑鍩哄洜瀵归泦鍚?{(gene1, gene2), ...}

        杩斿洖:
            [(gene1, gene2), ...]
        """
        if exclude_pairs is None:
            exclude_pairs = set()

        random_pairs = []
        max_attempts = n_pairs * 10  # 闃叉鏃犻檺寰幆
        attempts = 0

        while len(random_pairs) < n_pairs and attempts < max_attempts:
            # 闅忔満閫夋嫨涓や釜鍩哄洜
            gene1, gene2 = random.sample(self.all_genes, 2)

            # 鏍囧噯鍖栭『搴?
            pair = tuple(sorted([gene1, gene2]))

            # 妫€鏌ユ槸鍚﹀凡瀛樺湪
            if pair not in exclude_pairs and pair not in random_pairs:
                random_pairs.append(pair)

            attempts += 1

        if len(random_pairs) < n_pairs:
            logger.warning(f"鍙敓鎴愪簡 {len(random_pairs)}/{n_pairs} 涓殢鏈哄熀鍥犲")

        return random_pairs

    def generate_random_complexes(self, n_complexes: int, complex_size: int) -> List[List[str]]:
        """
        鐢熸垚闅忔満铔嬬櫧璐ㄥ鍚堢墿浣滀负瀵圭収缁?

        鍙傛暟:
            n_complexes: 澶嶅悎鐗╂暟閲?
            complex_size: 姣忎釜澶嶅悎鐗╃殑鍩哄洜鏁?

        杩斿洖:
            [[gene1, gene2, ...], ...]
        """
        random_complexes = []

        for _ in range(n_complexes):
            # 闅忔満閫夋嫨鍩哄洜
            complex_genes = random.sample(self.all_genes, complex_size)
            random_complexes.append(complex_genes)

        return random_complexes

    def wilcoxon_test(self, group1: List[float], group2: List[float]) -> Dict:
        """
        Wilcoxon绉╁拰妫€楠岋紙Mann-Whitney U妫€楠岋級

        鐢ㄤ簬姣旇緝涓ょ粍鐙珛鏍锋湰鐨勫垎甯冩槸鍚︽湁鏄捐憲宸紓

        鍙傛暟:
            group1: 瀹為獙缁勬暟鎹?
            group2: 瀵圭収缁勬暟鎹?

        杩斿洖:
            {'statistic': ..., 'p_value': ..., 'significant': ...}
        """
        # 杩囨护NaN
        g1_clean = [x for x in group1 if not np.isnan(x) and not np.isinf(x)]
        g2_clean = [x for x in group2 if not np.isnan(x) and not np.isinf(x)]

        if len(g1_clean) < 3 or len(g2_clean) < 3:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': 'Insufficient data'
            }

        try:
            # Mann-Whitney U妫€楠岋紙Wilcoxon绉╁拰妫€楠岀殑涓ゆ牱鏈増鏈級
            statistic, p_value = stats.mannwhitneyu(g1_clean, g2_clean, alternative='two-sided')

            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'n_group1': len(g1_clean),
                'n_group2': len(g2_clean)
            }
        except Exception as e:
            logger.error(f"Wilcoxon妫€楠屽け璐? {e}")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'error': str(e)
            }

    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        璁＄畻Cohen's d鏁堝簲閲?

        d = (mean1 - mean2) / pooled_std

        鏁堝簲閲忚В閲婏細
        - |d| < 0.2: 灏忔晥搴?
        - 0.2 <= |d| < 0.5: 涓瓑鏁堝簲
        - 0.5 <= |d| < 0.8: 澶ф晥搴?
        - |d| >= 0.8: 闈炲父澶х殑鏁堝簲

        鍙傛暟:
            group1: 缁?鏁版嵁
            group2: 缁?鏁版嵁

        杩斿洖:
            Cohen's d鍊?
        """
        # 杩囨护NaN
        g1_clean = np.array([x for x in group1 if not np.isnan(x) and not np.isinf(x)])
        g2_clean = np.array([x for x in group2 if not np.isnan(x) and not np.isinf(x)])

        if len(g1_clean) < 2 or len(g2_clean) < 2:
            return np.nan

        # 璁＄畻鍧囧€?
        mean1 = np.mean(g1_clean)
        mean2 = np.mean(g2_clean)

        # 璁＄畻鏍囧噯宸?
        std1 = np.std(g1_clean, ddof=1)
        std2 = np.std(g2_clean, ddof=1)

        # 璁＄畻pooled鏍囧噯宸?
        n1 = len(g1_clean)
        n2 = len(g2_clean)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return np.nan

        # Cohen's d
        d = (mean1 - mean2) / pooled_std

        return float(d)

    def fdr_correction(self, p_values: List[float], alpha: float = 0.05, method: str = 'fdr_bh') -> Dict:
        """
        FDR澶氶噸妫€楠屾牎姝?

        鍙傛暟:
            p_values: p鍊煎垪琛?
            alpha: 鏄捐憲鎬ф按骞?
            method: 鏍℃鏂规硶 ('bonferroni', 'fdr_bh', 'fdr_by')

        杩斿洖:
            {
                'rejected': [True/False, ...],  # 鏄惁鎷掔粷鍘熷亣璁?
                'p_values_corrected': [...],     # 鏍℃鍚庣殑p鍊?
                'n_significant': int             # 鏄捐憲鐨勬暟閲?
            }
        """
        # 杩囨护NaN
        valid_indices = [i for i, p in enumerate(p_values) if not np.isnan(p) and not np.isinf(p)]
        valid_p_values = [p_values[i] for i in valid_indices]

        if not valid_p_values:
            return {
                'rejected': [False] * len(p_values),
                'p_values_corrected': [np.nan] * len(p_values),
                'n_significant': 0
            }

        try:
            p_vals = np.array(valid_p_values, dtype=float)
            n = len(p_vals)
            order = np.argsort(p_vals)
            ranked = p_vals[order]

            adj_sorted = ranked * n / np.arange(1, n + 1)
            adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
            adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

            p_adj = np.empty(n, dtype=float)
            p_adj[order] = adj_sorted
            rejected = p_adj <= alpha

            full_rejected = [False] * len(p_values)
            full_corrected = [np.nan] * len(p_values)

            for i, idx in enumerate(valid_indices):
                full_rejected[idx] = bool(rejected[i])
                full_corrected[idx] = float(p_adj[i])

            return {
                'rejected': full_rejected,
                'p_values_corrected': full_corrected,
                'n_significant': sum(full_rejected)
            }
        except Exception as e:
            logger.error(f"FDR????: {e}")
            return {
                'rejected': [False] * len(p_values),
                'p_values_corrected': [np.nan] * len(p_values),
                'n_significant': 0,
                'error': str(e)
            }
    def compare_groups(self, experimental_group: List[float], control_group: List[float]) -> Dict:
        """
        瀹屾暣鐨勭粍闂存瘮杈冨垎鏋?

        鍖呮嫭锛?
        - 鎻忚堪鎬х粺璁?
        - Wilcoxon妫€楠?
        - Cohen's d鏁堝簲閲?

        鍙傛暟:
            experimental_group: 瀹為獙缁勬暟鎹?
            control_group: 瀵圭収缁勬暟鎹?

        杩斿洖:
            瀹屾暣鐨勭粺璁″垎鏋愮粨鏋?
        """
        # 杩囨护NaN
        exp_clean = [x for x in experimental_group if not np.isnan(x) and not np.isinf(x)]
        ctrl_clean = [x for x in control_group if not np.isnan(x) and not np.isinf(x)]

        # 鎻忚堪鎬х粺璁?
        exp_stats = {
            'mean': float(np.mean(exp_clean)) if exp_clean else np.nan,
            'median': float(np.median(exp_clean)) if exp_clean else np.nan,
            'std': float(np.std(exp_clean)) if exp_clean else np.nan,
            'min': float(np.min(exp_clean)) if exp_clean else np.nan,
            'max': float(np.max(exp_clean)) if exp_clean else np.nan,
            'n': len(exp_clean)
        }

        ctrl_stats = {
            'mean': float(np.mean(ctrl_clean)) if ctrl_clean else np.nan,
            'median': float(np.median(ctrl_clean)) if ctrl_clean else np.nan,
            'std': float(np.std(ctrl_clean)) if ctrl_clean else np.nan,
            'min': float(np.min(ctrl_clean)) if ctrl_clean else np.nan,
            'max': float(np.max(ctrl_clean)) if ctrl_clean else np.nan,
            'n': len(ctrl_clean)
        }

        # 缁熻妫€楠?
        wilcoxon_result = self.wilcoxon_test(exp_clean, ctrl_clean)

        # 鏁堝簲閲?
        effect_size = self.cohens_d(exp_clean, ctrl_clean)

        # 鏁堝簲閲忚В閲?
        if not np.isnan(effect_size):
            abs_d = abs(effect_size)
            if abs_d < 0.2:
                effect_interpretation = 'negligible'
            elif abs_d < 0.5:
                effect_interpretation = 'small'
            elif abs_d < 0.8:
                effect_interpretation = 'medium'
            else:
                effect_interpretation = 'large'
        else:
            effect_interpretation = 'unknown'

        return {
            'experimental_stats': exp_stats,
            'control_stats': ctrl_stats,
            'wilcoxon_test': wilcoxon_result,
            'cohens_d': effect_size,
            'effect_interpretation': effect_interpretation
        }


if __name__ == '__main__':
    from data_loader import ValidationDataLoader

    print("\n=== 缁熻鍒嗘瀽鍣ㄦ祴璇?===\n")

    # 鍔犺浇鍩哄洜鍒楄〃
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    loader = ValidationDataLoader(project_root / 'data/validation_data')
    goa = loader.load_goa_index()
    all_genes = list(goa.keys())

    print("1. 鍒濆鍖栫粺璁″垎鏋愬櫒...")
    analyzer = StatisticalAnalyzer(all_genes)
    print(f"   OK gene pool: {len(all_genes):,} genes")

    print("\n2. 鐢熸垚闅忔満鍩哄洜瀵?..")
    random_pairs = analyzer.generate_random_pairs(10)
    print(f"   鉁?鐢熸垚 {len(random_pairs)} 涓殢鏈哄熀鍥犲")
    print(f"   绀轰緥: {random_pairs[:3]}")

    print("\n3. 鐢熸垚闅忔満澶嶅悎鐗?..")
    random_complexes = analyzer.generate_random_complexes(5, 4)
    print(f"   鉁?鐢熸垚 {len(random_complexes)} 涓殢鏈哄鍚堢墿")
    print(f"   绀轰緥: {random_complexes[0]}")

    print("\n4. 娴嬭瘯缁熻妫€楠?..")
    # 鐢熸垚妯℃嫙鏁版嵁锛氬疄楠岀粍骞冲潎鍊兼洿楂?
    np.random.seed(42)
    exp_group = np.random.normal(0.6, 0.2, 100).tolist()
    ctrl_group = np.random.normal(0.4, 0.2, 100).tolist()

    result = analyzer.compare_groups(exp_group, ctrl_group)

    print("   瀹為獙缁勭粺璁?")
    print(f"      鍧囧€? {result['experimental_stats']['mean']:.4f}")
    print(f"      涓綅鏁? {result['experimental_stats']['median']:.4f}")
    print(f"      鏍囧噯宸? {result['experimental_stats']['std']:.4f}")
    print(f"      鏍锋湰鏁? {result['experimental_stats']['n']}")

    print("\n   瀵圭収缁勭粺璁?")
    print(f"      鍧囧€? {result['control_stats']['mean']:.4f}")
    print(f"      涓綅鏁? {result['control_stats']['median']:.4f}")
    print(f"      鏍囧噯宸? {result['control_stats']['std']:.4f}")
    print(f"      鏍锋湰鏁? {result['control_stats']['n']}")

    print("\n   Wilcoxon妫€楠?")
    print(f"      缁熻閲? {result['wilcoxon_test']['statistic']:.2f}")
    print(f"      p鍊? {result['wilcoxon_test']['p_value']:.6f}")
    print(f"      Significant: {result['wilcoxon_test']['significant']}")

    print("\n   鏁堝簲閲?")
    print(f"      Cohen's d: {result['cohens_d']:.4f}")
    print(f"      瑙ｉ噴: {result['effect_interpretation']}")

    print("\n5. 娴嬭瘯FDR澶氶噸妫€楠屾牎姝?..")
    p_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.5, 0.8]
    fdr_result = analyzer.fdr_correction(p_values, alpha=0.05)

    print("   鍘熷p鍊?-> 鏍℃鍚巔鍊?-> 鏄惁鏄捐憲")
    for i, p in enumerate(p_values):
        p_corr = fdr_result['p_values_corrected'][i]
        rejected = fdr_result['rejected'][i]
        print(f"      {p:.4f} -> {p_corr:.4f} -> {rejected}")

    print(f"\n   鏄捐憲缁撴灉鏁? {fdr_result['n_significant']}/{len(p_values)}")

    print("\n 鎵€鏈夋祴璇曞畬鎴愶紒")



