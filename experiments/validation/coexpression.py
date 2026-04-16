#!/usr/bin/env python3
"""
鍏辫〃杈剧浉鍏虫€ц绠楁ā鍧?

鑱岃矗锛氬熀浜嶨TEx琛ㄨ揪鏁版嵁璁＄畻鍩哄洜瀵瑰拰铔嬬櫧璐ㄥ鍚堢墿鐨勫叡琛ㄨ揪鐩稿叧鎬?
绛栫暐锛氫紭鍏堝皾璇曞畬鏁村姞杞紾CT鏂囦欢锛屽鏋滃唴瀛樹笉瓒冲垯闄嶇骇浣跨敤缁熻鏁版嵁
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoexpressionCalculator:
    """鍏辫〃杈剧浉鍏虫€ц绠楀櫒"""

    def __init__(self, gtex_stats: Dict, gct_file_path: Optional[str] = None):
        """
        鍒濆鍖栧叡琛ㄨ揪璁＄畻鍣?

        鍙傛暟:
            gtex_stats: GTEx鍩哄洜缁熻鏁版嵁 {ensembl_id: {'mean': ..., 'std': ..., ...}}
            gct_file_path: GCT鏂囦欢璺緞锛堝彲閫夛紝鐢ㄤ簬瀹屾暣琛ㄨ揪鐭╅樀锛?
        """
        self.gtex_stats = gtex_stats
        self.gct_file_path = gct_file_path

        # 瀹屾暣琛ㄨ揪鐭╅樀锛堟噿鍔犺浇锛?
        self._expression_matrix: Optional[Dict[str, np.ndarray]] = None
        self._symbol_to_ensembl: Optional[Dict[str, str]] = None  # Gene Symbol -> Ensembl ID鏄犲皠
        self._use_full_matrix = False

        logger.info(f"鍏辫〃杈捐绠楀櫒鍒濆鍖? {len(self.gtex_stats):,} 鍩哄洜缁熻")

        # 灏濊瘯鍔犺浇瀹屾暣鐭╅樀
        if gct_file_path:
            self._try_load_full_matrix()

    def _try_load_full_matrix(self):
        """
        灏濊瘯鍔犺浇瀹屾暣GCT琛ㄨ揪鐭╅樀

        濡傛灉鏂囦欢澶ぇ鎴栧唴瀛樹笉瓒筹紝浼橀泤鍦伴檷绾у埌浣跨敤缁熻鏁版嵁
        """
        if not self.gct_file_path:
            return

        gct_path = Path(self.gct_file_path)
        if not gct_path.exists():
            logger.warning(f"GCT鏂囦欢涓嶅瓨鍦? {gct_path}")
            return

        # 妫€鏌ユ枃浠跺ぇ灏?
        file_size_gb = gct_path.stat().st_size / (1024**3)
        logger.info(f"GCT鏂囦欢澶у皬: {file_size_gb:.2f} GB")

        if file_size_gb > 5.0:
            logger.warning(f"GCT file is too large ({file_size_gb:.2f} GB); using summary-statistics mode")
            return

        try:
            logger.info("灏濊瘯鍔犺浇瀹屾暣琛ㄨ揪鐭╅樀...")
            self._expression_matrix = self._load_gct_file()
            self._use_full_matrix = True
            logger.info(f"鉁?瀹屾暣鐭╅樀鍔犺浇鎴愬姛: {len(self._expression_matrix):,} 鍩哄洜")
        except MemoryError:
            logger.warning("鍐呭瓨涓嶈冻锛岄檷绾у埌缁熻鏁版嵁妯″紡")
            self._expression_matrix = None
            self._use_full_matrix = False
        except Exception as e:
            logger.error(f"Failed to load GCT file: {e}; using summary-statistics mode")
            self._expression_matrix = None
            self._use_full_matrix = False

    def _load_gct_file(self) -> Dict[str, np.ndarray]:
        """
        鍔犺浇GCT鏂囦欢骞惰В鏋愯〃杈剧煩闃?

        杩斿洖:
            {ensembl_id: expression_array}
        """
        expression_matrix = {}
        symbol_to_ensembl = {}

        with open(self.gct_file_path, 'r') as f:
            # 璺宠繃鍓嶄袱琛屽厓鏁版嵁
            f.readline()  # #1.2
            f.readline()  # dimensions

            # 璇诲彇鏍锋湰鍚嶏紙绗笁琛岋級
            header = f.readline().strip().split('\t')
            n_samples = len(header) - 2  # 鍑忓幓Name鍜孌escription鍒?

            logger.info(f"GCT file samples: {n_samples}")

            # 閫愯璇诲彇鍩哄洜琛ㄨ揪鏁版嵁
            n_genes = 0
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue

                gene_id = parts[0]  # Ensembl ID
                gene_symbol = parts[1]  # Gene Symbol
                expression_values = np.array([float(x) for x in parts[2:]], dtype=np.float32)

                # 鍘婚櫎鐗堟湰鍙?
                gene_id_clean = gene_id.split('.')[0]

                expression_matrix[gene_id_clean] = expression_values

                # 鏋勫缓Symbol鍒癊nsembl鐨勬槧灏?
                if gene_symbol and gene_symbol != '':
                    symbol_to_ensembl[gene_symbol] = gene_id_clean

                n_genes += 1
                if n_genes % 10000 == 0:
                    logger.info(f"   宸插姞杞?{n_genes:,} 鍩哄洜...")

            logger.info(f"   瀹屾垚: {n_genes:,} 鍩哄洜")

        self._symbol_to_ensembl = symbol_to_ensembl
        logger.info(f"   Symbol mapping: {len(symbol_to_ensembl):,} genes")

        return expression_matrix

    def _calculate_correlation_from_matrix(self, gene1: str, gene2: str, method: str = 'pearson') -> Optional[float]:
        """
        浠庡畬鏁磋〃杈剧煩闃佃绠楃浉鍏崇郴鏁?

        鍙傛暟:
            gene1: 鍩哄洜1 (Ensembl ID)
            gene2: 鍩哄洜2 (Ensembl ID)
            method: 'pearson' 鎴?'spearman'

        杩斿洖:
            鐩稿叧绯绘暟鎴朜one
        """
        if not self._use_full_matrix or self._expression_matrix is None:
            return None

        # 鍘婚櫎鐗堟湰鍙?
        g1 = gene1.split('.')[0]
        g2 = gene2.split('.')[0]

        if g1 not in self._expression_matrix or g2 not in self._expression_matrix:
            return None

        expr1 = self._expression_matrix[g1]
        expr2 = self._expression_matrix[g2]

        # 杩囨护NaN
        mask = ~(np.isnan(expr1) | np.isnan(expr2))
        if mask.sum() < 3:  # 鑷冲皯闇€瑕?涓牱鏈?
            return None

        expr1_clean = expr1[mask]
        expr2_clean = expr2[mask]

        try:
            if method == 'pearson':
                corr, _ = pearsonr(expr1_clean, expr2_clean)
            elif method == 'spearman':
                corr, _ = spearmanr(expr1_clean, expr2_clean)
            else:
                raise ValueError(f"涓嶆敮鎸佺殑鐩稿叧鏂规硶: {method}")

            if np.isnan(corr) or np.isinf(corr):
                return None

            return float(corr)
        except Exception as e:
            logger.debug(f"璁＄畻鐩稿叧鎬уけ璐?{gene1} vs {gene2}: {e}")
            return None

    def _calculate_correlation_with_p(self, gene1: str, gene2: str, method: str = 'pearson') -> Tuple[Optional[float], Optional[float]]:
        if not self._use_full_matrix or self._expression_matrix is None:
            return None, None

        g1 = gene1.split('.')[0]
        g2 = gene2.split('.')[0]

        if g1 not in self._expression_matrix or g2 not in self._expression_matrix:
            return None, None

        expr1 = self._expression_matrix[g1]
        expr2 = self._expression_matrix[g2]

        mask = ~(np.isnan(expr1) | np.isnan(expr2))
        if mask.sum() < 3:
            return None, None

        expr1_clean = expr1[mask]
        expr2_clean = expr2[mask]

        try:
            if method == 'pearson':
                corr, p_value = pearsonr(expr1_clean, expr2_clean)
            elif method == 'spearman':
                corr, p_value = spearmanr(expr1_clean, expr2_clean)
            else:
                raise ValueError(f"Unsupported correlation method: {method}")

            if np.isnan(corr) or np.isinf(corr):
                return None, None

            if np.isnan(p_value) or np.isinf(p_value):
                p_value = None

            return float(corr), float(p_value) if p_value is not None else None
        except Exception as e:
            logger.debug(f"Correlation failed: {gene1} vs {gene2}: {e}")
            return None, None

    def _calculate_correlation_from_stats(self, gene1: str, gene2: str) -> Optional[float]:
        """
        浠庣粺璁℃暟鎹及绠楃浉鍏虫€э紙闄嶇骇鏂规锛?

        绛栫暐锛氫娇鐢ㄥ潎鍊煎拰鏍囧噯宸殑鐩镐技搴︿綔涓鸿繎浼?
        娉ㄦ剰锛氳繖涓嶆槸鐪熷疄鐨勭浉鍏崇郴鏁帮紝浠呯敤浜庣矖鐣ヤ及璁?

        鍙傛暟:
            gene1: 鍩哄洜1 (Ensembl ID)
            gene2: 鍩哄洜2 (Ensembl ID)

        杩斿洖:
            浼扮畻鐨勭浉浼煎害鎴朜one
        """
        g1 = gene1.split('.')[0]
        g2 = gene2.split('.')[0]

        if g1 not in self.gtex_stats or g2 not in self.gtex_stats:
            return None

        stats1 = self.gtex_stats[g1]
        stats2 = self.gtex_stats[g2]

        # 浣跨敤鍧囧€肩殑鐩镐技搴︼紙褰掍竴鍖栨姘忚窛绂荤殑鍊掓暟锛?
        mean1 = stats1.get('mean', 0)
        mean2 = stats2.get('mean', 0)
        std1 = stats1.get('std', 1)
        std2 = stats2.get('std', 1)

        # 褰掍竴鍖栧潎鍊煎樊寮?
        mean_diff = abs(mean1 - mean2) / (max(std1, std2) + 1e-6)

        # 杞崲涓虹浉浼煎害锛?-1锛?
        similarity = np.exp(-mean_diff)

        return float(similarity)

    def gene_pair_correlation(self, gene1: str, gene2: str, method: str = 'pearson') -> Optional[float]:
        """
        璁＄畻涓や釜鍩哄洜鐨勫叡琛ㄨ揪鐩稿叧鎬?

        鍙傛暟:
            gene1: 鍩哄洜1 (Ensembl ID鎴栧熀鍥犵鍙?
            gene2: 鍩哄洜2 (Ensembl ID鎴栧熀鍥犵鍙?
            method: 'pearson' 鎴?'spearman'

        杩斿洖:
            鐩稿叧绯绘暟鎴朜one
        """
        # 濡傛灉鏄疓ene Symbol锛屽皾璇曡浆鎹负Ensembl ID
        gene1_id = self._resolve_gene_id(gene1)
        gene2_id = self._resolve_gene_id(gene2)

        # 浼樺厛浣跨敤瀹屾暣鐭╅樀
        if self._use_full_matrix:
            corr = self._calculate_correlation_from_matrix(gene1_id, gene2_id, method)
            if corr is not None:
                return corr

        # 闄嶇骇鍒扮粺璁℃暟鎹?
        return self._calculate_correlation_from_stats(gene1_id, gene2_id)

    def gene_pair_correlation_with_p(self, gene1: str, gene2: str, method: str = 'pearson') -> Tuple[Optional[float], Optional[float]]:
        gene1_id = self._resolve_gene_id(gene1)
        gene2_id = self._resolve_gene_id(gene2)

        if self._use_full_matrix:
            corr, p_value = self._calculate_correlation_with_p(gene1_id, gene2_id, method)
            if corr is not None:
                return corr, p_value

        corr = self._calculate_correlation_from_stats(gene1_id, gene2_id)
        return corr, None

    def _resolve_gene_id(self, gene: str) -> str:
        """
        灏咷ene Symbol杞崲涓篍nsembl ID锛堝鏋滈渶瑕侊級

        鍙傛暟:
            gene: Gene Symbol鎴朎nsembl ID

        杩斿洖:
            Ensembl ID
        """
        # 濡傛灉宸茬粡鏄疎nsembl ID鏍煎紡锛圗NSG寮€澶达級锛岀洿鎺ヨ繑鍥?
        if gene.startswith('ENSG'):
            return gene.split('.')[0]  # 鍘婚櫎鐗堟湰鍙?

        # 灏濊瘯浠嶴ymbol鏄犲皠琛ㄦ煡鎵?
        if self._symbol_to_ensembl and gene in self._symbol_to_ensembl:
            return self._symbol_to_ensembl[gene]

        # 濡傛灉鎵句笉鍒帮紝杩斿洖鍘熷€硷紙鍙兘瀵艰嚧鍚庣画鏌ユ壘澶辫触锛屼絾涓嶆姏鍑哄紓甯革級
        return gene

    def complex_coexpression(self, complex_genes: List[str], method: str = 'pearson') -> Optional[float]:
        """
        璁＄畻铔嬬櫧璐ㄥ鍚堢墿鍐呭熀鍥犵殑骞冲潎鍏辫〃杈剧浉鍏虫€?

        鍙傛暟:
            complex_genes: 澶嶅悎鐗╀腑鐨勫熀鍥犲垪琛?(Ensembl IDs)
            method: 'pearson' 鎴?'spearman'

        杩斿洖:
            骞冲潎鐩稿叧绯绘暟鎴朜one
        """
        if len(complex_genes) < 2:
            return None

        # 璁＄畻鎵€鏈夊熀鍥犲鐨勭浉鍏虫€?
        correlations = []
        for i in range(len(complex_genes)):
            for j in range(i + 1, len(complex_genes)):
                corr = self.gene_pair_correlation(complex_genes[i], complex_genes[j], method)
                if corr is not None:
                    correlations.append(corr)

        if not correlations:
            return None

        return float(np.mean(correlations))

    def complex_coexpression_stats(
        self,
        complex_genes: List[str],
        method: str = 'pearson',
        alpha: float = 0.05
    ) -> Dict[str, Optional[float]]:
        if len(complex_genes) < 2:
            return {
                'mean': None,
                'median': None,
                'std': None,
                'iqr': None,
                'n_pairs': 0,
                'n_valid_pairs': 0,
                'n_pairs_with_p': 0,
                'n_significant': 0,
                'significant_ratio': None
            }

        correlations = []
        p_values = []
        n_pairs = 0

        for i in range(len(complex_genes)):
            for j in range(i + 1, len(complex_genes)):
                n_pairs += 1
                corr, p_value = self.gene_pair_correlation_with_p(complex_genes[i], complex_genes[j], method)
                if corr is not None:
                    correlations.append(corr)
                if p_value is not None:
                    p_values.append(p_value)

        if not correlations:
            return {
                'mean': None,
                'median': None,
                'std': None,
                'iqr': None,
                'n_pairs': n_pairs,
                'n_valid_pairs': 0,
                'n_pairs_with_p': len(p_values),
                'n_significant': 0,
                'significant_ratio': None
            }

        mean_val = float(np.mean(correlations))
        median_val = float(np.median(correlations))
        std_val = float(np.std(correlations, ddof=1)) if len(correlations) > 1 else 0.0
        iqr_val = float(np.percentile(correlations, 75) - np.percentile(correlations, 25))

        n_significant = 0
        if p_values:
            n_significant = int(np.sum(np.array(p_values) < alpha))
            sig_ratio = n_significant / len(p_values)
        else:
            sig_ratio = None

        return {
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'iqr': iqr_val,
            'n_pairs': n_pairs,
            'n_valid_pairs': len(correlations),
            'n_pairs_with_p': len(p_values),
            'n_significant': n_significant,
            'significant_ratio': float(sig_ratio) if sig_ratio is not None else None
        }

    def batch_gene_pair_correlation(self, gene_pairs: List[Tuple[str, str]], method: str = 'pearson') -> Dict[Tuple[str, str], Optional[float]]:
        """
        鎵归噺璁＄畻鍩哄洜瀵圭浉鍏虫€?

        鍙傛暟:
            gene_pairs: [(gene1, gene2), ...]
            method: 'pearson' 鎴?'spearman'

        杩斿洖:
            {(gene1, gene2): correlation}
        """
        results = {}
        for gene1, gene2 in gene_pairs:
            corr = self.gene_pair_correlation(gene1, gene2, method)
            results[(gene1, gene2)] = corr
        return results

    def get_statistics(self) -> Dict:
        """
        鑾峰彇璁＄畻鍣ㄧ粺璁′俊鎭?

        杩斿洖:
            缁熻淇℃伅瀛楀吀
        """
        stats = {
            'n_genes_stats': len(self.gtex_stats),
            'use_full_matrix': self._use_full_matrix
        }

        if self._use_full_matrix and self._expression_matrix:
            stats['n_genes_matrix'] = len(self._expression_matrix)
            # 鑾峰彇鏍锋湰鏁?
            first_gene = next(iter(self._expression_matrix.values()))
            stats['n_samples'] = len(first_gene)

        return stats


if __name__ == '__main__':
    from data_loader import ValidationDataLoader

    print("\n=== 鍏辫〃杈捐绠楀櫒娴嬭瘯 ===\n")

    # 鍔犺浇鏁版嵁
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    loader = ValidationDataLoader(project_root / 'data/validation_data')

    print("1. 鍔犺浇GTEx缁熻鏁版嵁...")
    gtex_stats = loader.load_gtex_stats()
    print(f"   鉁?{len(gtex_stats):,} 鍩哄洜缁熻")

    # 鍒濆鍖栬绠楀櫒锛堜笉鍔犺浇瀹屾暣鐭╅樀锛?
    print("\n2. 鍒濆鍖栬绠楀櫒锛堢粺璁℃ā寮忥級...")
    gct_path = project_root / 'data/validation_data/gtex/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct'
    calculator = CoexpressionCalculator(gtex_stats, gct_path)
    stats = calculator.get_statistics()
    print(f"   鉁?妯″紡: {'瀹屾暣鐭╅樀' if stats['use_full_matrix'] else '缁熻鏁版嵁'}")
    print(f"   鉁?{stats['n_genes_stats']:,} 鍩哄洜缁熻")

    if stats['use_full_matrix']:
        print(f"   鉁?{stats['n_genes_matrix']:,} 鍩哄洜琛ㄨ揪鐭╅樀")
        print(f"   鉁?{stats['n_samples']:,} 鏍锋湰")

    print("\n3. 娴嬭瘯鍩哄洜瀵圭浉鍏虫€?..")
    test_genes = list(gtex_stats.keys())[:10]
    print(f"   娴嬭瘯鍩哄洜: {test_genes[:3]}")

    for i in range(min(3, len(test_genes))):
        for j in range(i+1, min(3, len(test_genes))):
            gene1, gene2 = test_genes[i], test_genes[j]
            pearson_corr = calculator.gene_pair_correlation(gene1, gene2, 'pearson')
            spearman_corr = calculator.gene_pair_correlation(gene1, gene2, 'spearman')

            print(f"   {gene1} vs {gene2}:")
            pearson_str = f"{pearson_corr:.4f}" if pearson_corr is not None else "N/A"
            spearman_str = f"{spearman_corr:.4f}" if spearman_corr is not None else "N/A"
            print(f"      Pearson: {pearson_str}")
            print(f"      Spearman: {spearman_str}")

    print("\n4. 娴嬭瘯澶嶅悎鐗╁叡琛ㄨ揪...")
    complex_genes = test_genes[:5]
    corr = calculator.complex_coexpression(complex_genes, 'pearson')
    print(f"   澶嶅悎鐗╁熀鍥? {complex_genes}")
    corr_str = f"{corr:.4f}" if corr is not None else "N/A"
    print(f"   Pearson鐩稿叧: {corr_str}")

    # 鎬ц兘娴嬭瘯
    print("\n5. 鎬ц兘娴嬭瘯...")
    import time
    gene_pairs = [(test_genes[i], test_genes[j]) for i in range(10) for j in range(i+1, 10)]
    start = time.time()
    results = calculator.batch_gene_pair_correlation(gene_pairs[:20], 'pearson')
    elapsed = time.time() - start
    print(f"   Computed 20 gene pairs in {elapsed:.2f}s")
    print(f"   骞冲潎姣忓: {elapsed/20*1000:.1f}姣")

    print("\n 鎵€鏈夋祴璇曞畬鎴愶紒")



