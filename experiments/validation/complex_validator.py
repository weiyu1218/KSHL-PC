#!/usr/bin/env python3
"""
澶嶅悎鐗╅獙璇佺粺涓€鎺ュ彛

鑱岃矗锛氭暣鍚堟墍鏈夐獙璇佹ā鍧楋紝鎻愪緵缁熶竴鐨勫鍚堢墿楠岃瘉鎺ュ彛
鍔熻兘锛?
- 鍗曚釜澶嶅悎鐗╅獙璇侊紙GO鐩镐技搴︺€佸叡琛ㄨ揪銆佸畾浣嶄竴鑷存€э級
- 鎵归噺楠岃瘉
- 闅忔満瀵圭収缁熻姣旇緝
- 缁撴灉淇濆瓨
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .data_loader import ValidationDataLoader
from .go_similarity import GOSimilarityCalculator
from .coexpression import CoexpressionCalculator
from .localization import LocalizationCalculator
from .statistical_analyzer import StatisticalAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexValidator:
    """铔嬬櫧璐ㄥ鍚堢墿楠岃瘉鍣?- 缁熶竴鎺ュ彛"""

    def __init__(self, data_dir: str = 'data/validation_data', gct_file_path: Optional[str] = None):
        """
        鍒濆鍖栧鍚堢墿楠岃瘉鍣?

        鍙傛暟:
            data_dir: 楠岃瘉鏁版嵁鏍圭洰褰?
            gct_file_path: GTEx GCT鏂囦欢璺緞锛堝彲閫夛級
        """
        self.data_dir = data_dir
        self.gct_file_path = gct_file_path

        # 鍒濆鍖栨暟鎹姞杞藉櫒
        logger.info("鍒濆鍖栨暟鎹姞杞藉櫒...")
        self.data_loader = ValidationDataLoader(data_dir)

        # 鎳掑姞杞借绠楀櫒
        self._go_calculator: Optional[GOSimilarityCalculator] = None
        self._coexp_calculator: Optional[CoexpressionCalculator] = None
        self._loc_calculator: Optional[LocalizationCalculator] = None
        self._stat_analyzer: Optional[StatisticalAnalyzer] = None

        logger.info("ComplexValidator initialized")

    def _init_go_calculator(self) -> GOSimilarityCalculator:
        """鎳掑姞杞紾O鐩镐技搴﹁绠楀櫒"""
        if self._go_calculator is None:
            logger.info("鍔犺浇GO鐩镐技搴﹁绠楀櫒...")
            go_dag = self.data_loader.load_go_dag()
            goa_index = self.data_loader.load_goa_index()
            self._go_calculator = GOSimilarityCalculator(go_dag, goa_index)
        return self._go_calculator

    def _init_coexp_calculator(self) -> CoexpressionCalculator:
        """Helper."""
        if self._coexp_calculator is None:
            logger.info("鍔犺浇鍏辫〃杈捐绠楀櫒...")
            gtex_stats = self.data_loader.load_gtex_stats()
            self._coexp_calculator = CoexpressionCalculator(gtex_stats, self.gct_file_path)
        return self._coexp_calculator

    def _init_loc_calculator(self) -> LocalizationCalculator:
        """鎳掑姞杞藉畾浣嶈绠楀櫒"""
        if self._loc_calculator is None:
            logger.info("鍔犺浇瀹氫綅璁＄畻鍣?..")
            compartments = self.data_loader.load_compartments()
            self._loc_calculator = LocalizationCalculator(compartments)
        return self._loc_calculator

    def _init_stat_analyzer(self) -> StatisticalAnalyzer:
        """鎳掑姞杞界粺璁″垎鏋愬櫒"""
        if self._stat_analyzer is None:
            logger.info("鍒濆鍖栫粺璁″垎鏋愬櫒...")
            all_genes = self.data_loader.get_all_genes('goa')
            self._stat_analyzer = StatisticalAnalyzer(all_genes)
        return self._stat_analyzer

    def validate_complex(
        self,
        genes: List[str],
        complex_id: str = "unknown",
        compare_baseline: bool = True,
        n_baseline: int = 1000
    ) -> Dict[str, Any]:
        """
        楠岃瘉鍗曚釜铔嬬櫧璐ㄥ鍚堢墿

        鍙傛暟:
            genes: 鍩哄洜鍒楄〃
            complex_id: 澶嶅悎鐗㊣D
            compare_baseline: 鏄惁涓庨殢鏈哄鐓ф瘮杈?
            n_baseline: 闅忔満瀵圭収鏁伴噺

        杩斿洖:
            楠岃瘉缁撴灉瀛楀吀
        """
        if len(genes) < 2:
            return {
                'complex_id': complex_id,
                'n_genes': len(genes),
                'error': 'At least 2 genes required'
            }

        result = {
            'complex_id': complex_id,
            'genes': genes,
            'n_genes': len(genes),
            'metrics': {}
        }

        # 1. GO鐩镐技搴﹁绠?
        try:
            go_calc = self._init_go_calculator()

            go_bp = go_calc.complex_similarity(genes, 'BP')
            go_cc = go_calc.complex_similarity(genes, 'CC')
            go_mf = go_calc.complex_similarity(genes, 'MF')

            result['metrics']['go_similarity'] = {
                'BP': go_bp,
                'CC': go_cc,
                'MF': go_mf
            }

            result['metrics']['go_coverage'] = {
                'BP': go_calc.complex_annotation_coverage(genes, 'BP'),
                'CC': go_calc.complex_annotation_coverage(genes, 'CC'),
                'MF': go_calc.complex_annotation_coverage(genes, 'MF')
            }

            result['metrics']['go_enrichment'] = {
                'BP': go_calc.complex_enrichment(genes, 'BP'),
                'CC': go_calc.complex_enrichment(genes, 'CC'),
                'MF': go_calc.complex_enrichment(genes, 'MF')
            }
        except Exception as e:
            logger.error(f"GO鐩镐技搴﹁绠楀け璐?[{complex_id}]: {e}")
            result['metrics']['go_similarity'] = {
                'BP': None,
                'CC': None,
                'MF': None,
                'error': str(e)
            }
            result['metrics']['go_coverage'] = {
                'BP': None,
                'CC': None,
                'MF': None,
                'error': str(e)
            }
            result['metrics']['go_enrichment'] = {
                'BP': None,
                'CC': None,
                'MF': None,
                'error': str(e)
            }

        # 2. 鍏辫〃杈捐绠?
        try:
            coexp_calc = self._init_coexp_calculator()
            coexp_pearson_stats = coexp_calc.complex_coexpression_stats(genes, 'pearson')
            coexp_spearman_stats = coexp_calc.complex_coexpression_stats(genes, 'spearman')

            result['metrics']['coexpression'] = {
                'pearson': coexp_pearson_stats.get('mean'),
                'pearson_median': coexp_pearson_stats.get('median'),
                'pearson_std': coexp_pearson_stats.get('std'),
                'pearson_iqr': coexp_pearson_stats.get('iqr'),
                'pearson_n_pairs': coexp_pearson_stats.get('n_pairs'),
                'pearson_n_valid_pairs': coexp_pearson_stats.get('n_valid_pairs'),
                'pearson_n_pairs_with_p': coexp_pearson_stats.get('n_pairs_with_p'),
                'pearson_n_significant': coexp_pearson_stats.get('n_significant'),
                'pearson_significant_ratio': coexp_pearson_stats.get('significant_ratio'),
                'spearman': coexp_spearman_stats.get('mean'),
                'spearman_median': coexp_spearman_stats.get('median'),
                'spearman_std': coexp_spearman_stats.get('std'),
                'spearman_iqr': coexp_spearman_stats.get('iqr'),
                'spearman_n_pairs': coexp_spearman_stats.get('n_pairs'),
                'spearman_n_valid_pairs': coexp_spearman_stats.get('n_valid_pairs'),
                'spearman_n_pairs_with_p': coexp_spearman_stats.get('n_pairs_with_p'),
                'spearman_n_significant': coexp_spearman_stats.get('n_significant'),
                'spearman_significant_ratio': coexp_spearman_stats.get('significant_ratio')
            }
        except Exception as e:
            logger.error(f"鍏辫〃杈捐绠楀け璐?[{complex_id}]: {e}")
            result['metrics']['coexpression'] = {
                'pearson': None,
                'spearman': None,
                'error': str(e)
            }

        # 3. 瀹氫綅涓€鑷存€ц绠?
        try:
            loc_calc = self._init_loc_calculator()
            loc_result = loc_calc.complex_localization(genes)

            result['metrics']['localization'] = loc_result
        except Exception as e:
            logger.error(f"瀹氫綅涓€鑷存€ц绠楀け璐?[{complex_id}]: {e}")
            result['metrics']['localization'] = {
                'error': str(e)
            }

        # 4. 闅忔満瀵圭収姣旇緝锛堝彲閫夛級
        if compare_baseline:
            try:
                baseline_result = self._calculate_baseline(genes, n_baseline)
                result['baseline_comparison'] = baseline_result
            except Exception as e:
                logger.error(f"鍩虹嚎璁＄畻澶辫触 [{complex_id}]: {e}")
                result['baseline_comparison'] = {
                    'error': str(e)
                }

        return result

    def _calculate_baseline(self, genes: List[str], n_baseline: int) -> Dict[str, Any]:
        """
        璁＄畻闅忔満瀵圭収鍩虹嚎

        鍙傛暟:
            genes: 鍘熷鍚堢墿鍩哄洜鍒楄〃
            n_baseline: 闅忔満澶嶅悎鐗╂暟閲?

        杩斿洖:
            鍩虹嚎缁熻缁撴灉
        """
        stat_analyzer = self._init_stat_analyzer()
        complex_size = len(genes)

        # 鐢熸垚闅忔満澶嶅悎鐗?
        random_complexes = stat_analyzer.generate_random_complexes(n_baseline, complex_size)

        # 璁＄畻闅忔満澶嶅悎鐗╃殑鎸囨爣
        go_bp_baseline = []
        go_cc_baseline = []
        go_mf_baseline = []
        coexp_pearson_baseline = []
        loc_jaccard_baseline = []

        go_calc = self._init_go_calculator()
        coexp_calc = self._init_coexp_calculator()
        loc_calc = self._init_loc_calculator()

        for random_genes in random_complexes:
            # GO鐩镐技搴?
            bp_sim = go_calc.complex_similarity(random_genes, 'BP')
            if bp_sim is not None:
                go_bp_baseline.append(bp_sim)

            cc_sim = go_calc.complex_similarity(random_genes, 'CC')
            if cc_sim is not None:
                go_cc_baseline.append(cc_sim)

            mf_sim = go_calc.complex_similarity(random_genes, 'MF')
            if mf_sim is not None:
                go_mf_baseline.append(mf_sim)

            # 鍏辫〃杈?
            coexp = coexp_calc.complex_coexpression(random_genes, 'pearson')
            if coexp is not None:
                coexp_pearson_baseline.append(coexp)

            # 瀹氫綅
            loc_result = loc_calc.complex_localization(random_genes)
            if loc_result and 'avg_jaccard' in loc_result:
                loc_jaccard_baseline.append(loc_result['avg_jaccard'])

        # 缁熻姹囨€?
        baseline_result = {
            'n_random_complexes': n_baseline,
            'go_bp': self._summarize_baseline(go_bp_baseline),
            'go_cc': self._summarize_baseline(go_cc_baseline),
            'go_mf': self._summarize_baseline(go_mf_baseline),
            'coexp_pearson': self._summarize_baseline(coexp_pearson_baseline),
            'loc_jaccard': self._summarize_baseline(loc_jaccard_baseline)
        }

        return baseline_result

    def _summarize_baseline(self, values: List[float]) -> Dict[str, float]:
        """
        姹囨€诲熀绾跨粺璁℃暟鎹?

        鍙傛暟:
            values: 鏁板€煎垪琛?

        杩斿洖:
            缁熻姹囨€?
        """
        if not values:
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'n': 0
            }

        clean_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]

        if not clean_values:
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'n': 0
            }

        return {
            'mean': float(np.mean(clean_values)),
            'median': float(np.median(clean_values)),
            'std': float(np.std(clean_values)),
            'min': float(np.min(clean_values)),
            'max': float(np.max(clean_values)),
            'n': len(clean_values)
        }

    def batch_validate(
        self,
        complex_dict: Dict[str, List[str]],
        show_progress: bool = True,
        n_workers: int = 1,
        compare_baseline: bool = False,
        n_baseline: int = 1000
    ) -> Dict[str, Dict[str, Any]]:
        """
        鎵归噺楠岃瘉澶氫釜澶嶅悎鐗?

        鍙傛暟:
            complex_dict: {complex_id: [gene1, gene2, ...]}
            show_progress: 鏄惁鏄剧ず杩涘害鏉?
            n_workers: 骞惰宸ヤ綔杩涚▼鏁帮紙1琛ㄧず涓茶锛?
            compare_baseline: 鏄惁涓庨殢鏈哄鐓ф瘮杈?
            n_baseline: 闅忔満瀵圭収鏁伴噺

        杩斿洖:
            {complex_id: result}
        """
        results = {}

        if n_workers == 1:
            # 涓茶澶勭悊
            iterator = complex_dict.items()
            if show_progress:
                iterator = tqdm(iterator, total=len(complex_dict), desc="Validating complexes")

            for complex_id, genes in iterator:
                result = self.validate_complex(genes, complex_id, compare_baseline, n_baseline)
                results[complex_id] = result
        else:
            # 骞惰澶勭悊
            logger.info(f"浣跨敤 {n_workers} 涓繘绋嬪苟琛屽鐞?..")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        self.validate_complex,
                        genes,
                        complex_id,
                        compare_baseline,
                        n_baseline
                    ): complex_id
                    for complex_id, genes in complex_dict.items()
                }

                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(iterator, total=len(futures), desc="Validating complexes")

                for future in iterator:
                    complex_id = futures[future]
                    try:
                        result = future.result()
                        results[complex_id] = result
                    except Exception as e:
                        logger.error(f"澶嶅悎鐗?{complex_id} 楠岃瘉澶辫触: {e}")
                        results[complex_id] = {
                            'complex_id': complex_id,
                            'error': str(e)
                        }

        return results

    def save_results(
        self,
        results: Union[Dict[str, Any], Dict[str, Dict[str, Any]]],
        output_path: str,
        format: str = 'json'
    ):
        """
        淇濆瓨楠岃瘉缁撴灉

        鍙傛暟:
            results: 楠岃瘉缁撴灉锛堝崟涓垨鎵归噺锛?
            output_path: 杈撳嚭鏂囦欢璺緞
            format: 杈撳嚭鏍煎紡 ('json', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"缁撴灉宸蹭繚瀛樺埌: {output_path}")

        elif format == 'csv':
            # 杞崲涓篊SV鏍煎紡
            import csv

            # 鍒ゆ柇鏄崟涓粨鏋滆繕鏄壒閲忕粨鏋?
            if 'complex_id' in results:
                # 鍗曚釜缁撴灉
                results_list = [results]
            else:
                # 鎵归噺缁撴灉
                results_list = list(results.values())

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 鍐欏叆琛ㄥご
                writer.writerow([
                    'complex_id',
                    'n_genes',
                    'go_bp',
                    'go_cc',
                    'go_mf',
                    'coexp_pearson',
                    'coexp_spearman',
                    'loc_jaccard',
                    'loc_top_location',
                    'loc_top1_coverage'
                ])

                # 鍐欏叆鏁版嵁
                for result in results_list:
                    if 'error' in result:
                        continue

                    complex_id = result.get('complex_id', 'unknown')
                    n_genes = result.get('n_genes', 0)

                    metrics = result.get('metrics', {})

                    go_sim = metrics.get('go_similarity', {})
                    go_bp = go_sim.get('BP')
                    go_cc = go_sim.get('CC')
                    go_mf = go_sim.get('MF')

                    coexp = metrics.get('coexpression', {})
                    coexp_pearson = coexp.get('pearson')
                    coexp_spearman = coexp.get('spearman')

                    loc = metrics.get('localization', {})
                    loc_jaccard = loc.get('avg_jaccard') if loc else None
                    loc_top = loc.get('top_location') if loc else None
                    loc_top1_cov = loc.get('top1_coverage') if loc else None

                    writer.writerow([
                        complex_id,
                        n_genes,
                        go_bp,
                        go_cc,
                        go_mf,
                        coexp_pearson,
                        coexp_spearman,
                        loc_jaccard,
                        loc_top,
                        loc_top1_cov
                    ])

            logger.info(f"缁撴灉宸蹭繚瀛樺埌: {output_path}")

        else:
            raise ValueError(f"涓嶆敮鎸佺殑鏍煎紡: {format}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        鑾峰彇楠岃瘉鍣ㄧ粺璁′俊鎭?

        杩斿洖:
            缁熻淇℃伅瀛楀吀
        """
        stats = {
            'data_dir': str(self.data_dir),
            'calculators_loaded': {
                'go_similarity': self._go_calculator is not None,
                'coexpression': self._coexp_calculator is not None,
                'localization': self._loc_calculator is not None,
                'statistical_analyzer': self._stat_analyzer is not None
            }
        }

        # 娣诲姞鍚勮绠楀櫒鐨勭粺璁′俊鎭?
        if self._go_calculator:
            stats['go_calculator'] = self._go_calculator.get_statistics()

        if self._coexp_calculator:
            stats['coexp_calculator'] = self._coexp_calculator.get_statistics()

        if self._loc_calculator:
            stats['loc_calculator'] = self._loc_calculator.get_statistics()

        return stats


if __name__ == '__main__':
    print("\n=== ComplexValidator娴嬭瘯 ===\n")

    # 鍒濆鍖栭獙璇佸櫒
    print("1. 鍒濆鍖栭獙璇佸櫒...")
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    validator = ComplexValidator(project_root / 'data/validation_data')
    print("   鉁?楠岃瘉鍣ㄥ垵濮嬪寲瀹屾垚")

    print("\n2. 娴嬭瘯鍗曚釜澶嶅悎鐗╅獙璇?..")
    test_complex = ['TP53', 'MDM2', 'MDM4']
    print(f"   娴嬭瘯澶嶅悎鐗? {test_complex}")

    result = validator.validate_complex(
        genes=test_complex,
        complex_id='test_complex_1',
        compare_baseline=True,
        n_baseline=100
    )

    print(f"\n   缁撴灉:")
    print(f"   - 澶嶅悎鐗㊣D: {result['complex_id']}")
    print(f"   - 鍩哄洜鏁? {result['n_genes']}")

    metrics = result.get('metrics', {})

    # GO鐩镐技搴?
    go_sim = metrics.get('go_similarity', {})
    print(f"\n   GO鐩镐技搴?")
    for ont in ['BP', 'CC', 'MF']:
        val = go_sim.get(ont)
        val_str = f"{val:.4f}" if val is not None else "N/A"
        print(f"      {ont}: {val_str}")

    # 鍏辫〃杈?
    coexp = metrics.get('coexpression', {})
    print(f"\n   鍏辫〃杈?")
    pearson = coexp.get('pearson')
    spearman = coexp.get('spearman')
    pearson_str = f"{pearson:.4f}" if pearson is not None else "N/A"
    spearman_str = f"{spearman:.4f}" if spearman is not None else "N/A"
    print(f"      Pearson: {pearson_str}")
    print(f"      Spearman: {spearman_str}")

    # 瀹氫綅
    loc = metrics.get('localization')
    if loc and 'avg_jaccard' in loc:
        print(f"\n   瀹氫綅涓€鑷存€?")
        print(f"      骞冲潎Jaccard: {loc['avg_jaccard']:.4f}")
        print(f"      浼樺娍瀹氫綅: {loc.get('top_location', 'N/A')}")
        print(f"      Top1瑕嗙洊鐜? {loc.get('top1_coverage', 0):.2%}")

    # 鍩虹嚎姣旇緝
    if 'baseline_comparison' in result:
        baseline = result['baseline_comparison']
        print(f"\n   鍩虹嚎姣旇緝 (n={baseline.get('n_random_complexes', 0)}):")

        if 'go_bp' in baseline and baseline['go_bp']['n'] > 0:
            bp_mean = baseline['go_bp']['mean']
            print(f"      GO-BP闅忔満鍧囧€? {bp_mean:.4f}")

        if 'coexp_pearson' in baseline and baseline['coexp_pearson']['n'] > 0:
            coexp_mean = baseline['coexp_pearson']['mean']
            print(f"      鍏辫〃杈鹃殢鏈哄潎鍊? {coexp_mean:.4f}")

    print("\n3. 娴嬭瘯鎵归噺楠岃瘉...")
    complex_dict = {
        'complex_1': ['TP53', 'MDM2', 'MDM4'],
        'complex_2': ['BRCA1', 'BRCA2', 'PALB2'],
        'complex_3': ['EGFR', 'ERBB2', 'ERBB3']
    }

    batch_results = validator.batch_validate(
        complex_dict,
        show_progress=True,
        n_workers=1,
        compare_baseline=False
    )

    print(f"\n   楠岃瘉浜?{len(batch_results)} 涓鍚堢墿")

    print("\n4. 娴嬭瘯缁撴灉淇濆瓨...")

    # JSON鏍煎紡
    json_path = '/root/autodl-tmp/KSHL-PC/test_complex_validation.json'
    validator.save_results(batch_results, json_path, format='json')
    print(f"   鉁?JSON鏍煎紡宸蹭繚瀛? {json_path}")

    # CSV鏍煎紡
    csv_path = '/root/autodl-tmp/KSHL-PC/test_complex_validation.csv'
    validator.save_results(batch_results, csv_path, format='csv')
    print(f"   鉁?CSV鏍煎紡宸蹭繚瀛? {csv_path}")

    # 缁熻淇℃伅
    print("\n5. 楠岃瘉鍣ㄧ粺璁′俊鎭?..")
    stats = validator.get_statistics()
    print(json.dumps(stats, indent=2))

    print("\n 鎵€鏈夋祴璇曞畬鎴愶紒")



