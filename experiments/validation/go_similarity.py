#!/usr/bin/env python3
"""
GO璇箟鐩镐技搴﹁绠楁ā鍧?

鑱岃矗锛氬熀浜嶨O DAG缁撴瀯璁＄畻鍩哄洜瀵瑰拰铔嬬櫧璐ㄥ鍚堢墿鐨凣O璇箟鐩镐技搴?
鏂规硶锛歐ang鏂规硶 - 鑰冭檻GO term鐨勬嫇鎵戠粨鏋勫拰璇箟璐＄尞
"""

import logging
from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from scipy.stats import hypergeom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GOSimilarityCalculator:
    """GO璇箟鐩镐技搴﹁绠楀櫒"""

    def __init__(self, go_dag: Dict, goa_index: Dict, weight_is_a: float = 0.8, weight_part_of: float = 0.6):
        """
        鍒濆鍖朑O鐩镐技搴﹁绠楀櫒

        鍙傛暟:
            go_dag: GO DAG缁撴瀯 {'terms': {...}, 'parents': {...}, 'children': {...}}
            goa_index: GOA绱㈠紩 {gene: {'BP': [go_ids], 'CC': [...], 'MF': [...]}}
            weight_is_a: is_a鍏崇郴鏉冮噸
            weight_part_of: part_of鍏崇郴鏉冮噸
        """
        self.go_dag = go_dag
        self.goa_index = goa_index
        self.weight_is_a = weight_is_a
        self.weight_part_of = weight_part_of

        # 鎻愬彇鏍稿績鏁版嵁缁撴瀯
        self.terms = go_dag.get('terms', {})
        self.parents = go_dag.get('parents', {})
        self.children = go_dag.get('children', {})

        # 棰勮绠楃鍏堣妭鐐圭紦瀛?
        self._ancestor_cache: Dict[str, Dict[str, float]] = {}
        self._term_gene_cache: Dict[str, Dict[str, Set[str]]] = {}
        self._namespace_gene_cache: Dict[str, Set[str]] = {}

        logger.info(f"GO鐩镐技搴﹁绠楀櫒鍒濆鍖栧畬鎴? {len(self.terms):,} GO terms")

    def _fdr_correction(self, p_values: List[float], alpha: float = 0.05, method: str = 'fdr_bh') -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Hochberg FDR correction (no external dependency).
        Returns (rejected, p_adjusted).
        """
        if not p_values:
            return np.array([], dtype=bool), np.array([], dtype=float)

        p_vals = np.array(p_values, dtype=float)
        n = len(p_vals)
        order = np.argsort(p_vals)
        ranked = p_vals[order]

        adj = ranked * n / np.arange(1, n + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        adj = np.clip(adj, 0.0, 1.0)

        rejected = adj <= alpha

        p_adj = np.empty(n, dtype=float)
        rej = np.empty(n, dtype=bool)
        p_adj[order] = adj
        rej[order] = rejected

        return rej, p_adj

    def _build_namespace_cache(self, namespace: str) -> None:
        if namespace in self._term_gene_cache:
            return

        term_to_genes: Dict[str, Set[str]] = {}
        namespace_genes: Set[str] = set()

        for gene, ann in self.goa_index.items():
            terms = ann.get(namespace, [])
            if not terms:
                continue
            namespace_genes.add(gene)
            for term in terms:
                if term not in term_to_genes:
                    term_to_genes[term] = set()
                term_to_genes[term].add(gene)

        self._term_gene_cache[namespace] = term_to_genes
        self._namespace_gene_cache[namespace] = namespace_genes

    def complex_annotation_coverage(self, complex_genes: List[str], namespace: str = 'BP') -> Dict[str, Optional[float]]:
        if not complex_genes:
            return {
                'coverage': None,
                'n_valid_genes': 0,
                'n_total_genes': 0
            }

        genes = [g.upper() for g in complex_genes]
        valid_genes = [
            g for g in genes
            if g in self.goa_index and self.goa_index[g].get(namespace)
        ]

        total = len(genes)
        valid = len(valid_genes)
        coverage = valid / total if total > 0 else None

        return {
            'coverage': float(coverage) if coverage is not None else None,
            'n_valid_genes': valid,
            'n_total_genes': total
        }

    def complex_enrichment(
        self,
        complex_genes: List[str],
        namespace: str = 'BP',
        fdr_alpha: float = 0.05,
        fdr_method: str = 'fdr_bh'
    ) -> Dict[str, Optional[float]]:
        if len(complex_genes) < 2:
            return {
                'n_terms_tested': 0,
                'n_terms_significant': 0,
                'min_p_value': None,
                'min_fdr': None,
                'best_term': None,
                'best_term_name': None,
                'best_term_overlap': None,
                'n_valid_genes': 0,
                'n_background_genes': 0
            }

        self._build_namespace_cache(namespace)

        genes = [g.upper() for g in complex_genes]
        background_genes = self._namespace_gene_cache.get(namespace, set())
        valid_genes = [g for g in genes if g in background_genes]

        n = len(valid_genes)
        N = len(background_genes)
        if n < 2 or N == 0:
            return {
                'n_terms_tested': 0,
                'n_terms_significant': 0,
                'min_p_value': None,
                'min_fdr': None,
                'best_term': None,
                'best_term_name': None,
                'best_term_overlap': None,
                'n_valid_genes': n,
                'n_background_genes': N
            }

        term_to_genes = self._term_gene_cache.get(namespace, {})
        candidate_terms: Set[str] = set()
        for gene in valid_genes:
            candidate_terms.update(self.goa_index.get(gene, {}).get(namespace, []))

        terms = []
        p_values = []
        overlaps = []

        valid_set = set(valid_genes)
        for term in candidate_terms:
            term_genes = term_to_genes.get(term)
            if not term_genes:
                continue
            k = len(valid_set & term_genes)
            if k == 0:
                continue
            K = len(term_genes)
            p = hypergeom.sf(k - 1, N, K, n)
            terms.append(term)
            p_values.append(p)
            overlaps.append(k)

        if not p_values:
            return {
                'n_terms_tested': 0,
                'n_terms_significant': 0,
                'min_p_value': None,
                'min_fdr': None,
                'best_term': None,
                'best_term_name': None,
                'best_term_overlap': None,
                'n_valid_genes': n,
                'n_background_genes': N
            }

        rejected, p_adj = self._fdr_correction(p_values, alpha=fdr_alpha, method=fdr_method)
        min_idx = int(np.nanargmin(p_adj))

        best_term = terms[min_idx]
        best_term_name = self.terms.get(best_term, {}).get('name')

        return {
            'n_terms_tested': len(p_values),
            'n_terms_significant': int(np.sum(rejected)),
            'min_p_value': float(p_values[min_idx]),
            'min_fdr': float(p_adj[min_idx]),
            'best_term': best_term,
            'best_term_name': best_term_name,
            'best_term_overlap': int(overlaps[min_idx]),
            'n_valid_genes': n,
            'n_background_genes': N
        }

    def _get_ancestors_with_weights(self, go_id: str) -> Dict[str, float]:
        """
        鑾峰彇GO term鐨勬墍鏈夌鍏堣妭鐐瑰強鍏惰涔夎础鐚€?

        浣跨敤Wang鏂规硶璁＄畻璇箟璐＄尞锛?
        - 鑺傜偣鑷韩璐＄尞鍊间负1.0
        - 瀛愯妭鐐归€氳繃is_a鍏崇郴缁ф壙鐖惰妭鐐圭殑weight_is_a鍊嶈础鐚?
        - 瀛愯妭鐐归€氳繃part_of鍏崇郴缁ф壙鐖惰妭鐐圭殑weight_part_of鍊嶈础鐚?

        鍙傛暟:
            go_id: GO term ID

        杩斿洖:
            {ancestor_go_id: semantic_value}
        """
        # 妫€鏌ョ紦瀛?
        if go_id in self._ancestor_cache:
            return self._ancestor_cache[go_id]

        if go_id not in self.terms:
            return {}

        # 璇箟璐＄尞瀛楀吀
        sv = {go_id: 1.0}

        # BFS閬嶅巻绁栧厛鑺傜偣
        queue = [(go_id, 1.0)]
        visited = {go_id}

        while queue:
            current, current_value = queue.pop(0)

            # 鑾峰彇鐖惰妭鐐?
            parent_ids = self.parents.get(current, [])

            for parent_id in parent_ids:
                if parent_id not in self.terms:
                    continue

                # 璁＄畻鐖惰妭鐐圭殑璇箟璐＄尞
                # 绠€鍖栵細榛樿浣跨敤is_a鏉冮噸锛堝洜涓烘垜浠病鏈夋槑纭尯鍒嗗叧绯荤被鍨嬶級
                parent_value = current_value * self.weight_is_a

                # 鏇存柊鐖惰妭鐐圭殑鏈€澶ц涔夎础鐚?
                if parent_id not in sv:
                    sv[parent_id] = parent_value
                else:
                    sv[parent_id] = max(sv[parent_id], parent_value)

                # 娣诲姞鍒伴槦鍒楋紙濡傛灉鏈闂級
                if parent_id not in visited:
                    visited.add(parent_id)
                    queue.append((parent_id, sv[parent_id]))

        # 缂撳瓨缁撴灉
        self._ancestor_cache[go_id] = sv

        return sv

    def _calculate_term_similarity(self, go_id1: str, go_id2: str) -> float:
        """
        璁＄畻涓や釜GO term涔嬮棿鐨勭浉浼煎害锛圵ang鏂规硶锛?

        鍏紡锛?
        S(A,B) = 危(SV_A(t) + SV_B(t)) / (SV(A) + SV(B))

        鍏朵腑锛?
        - t 鏄疉鍜孊鐨勫叕鍏辩鍏?
        - SV_A(t) 鏄痶鍦ˋ鐨凞AG涓殑璇箟璐＄尞
        - SV(A) = 危 SV_A(t) 瀵规墍鏈塼

        鍙傛暟:
            go_id1: GO term 1
            go_id2: GO term 2

        杩斿洖:
            鐩镐技搴?(0-1)
        """
        if go_id1 == go_id2:
            return 1.0

        # 鑾峰彇涓や釜term鐨勭鍏堝強璇箟璐＄尞
        sv1 = self._get_ancestors_with_weights(go_id1)
        sv2 = self._get_ancestors_with_weights(go_id2)

        if not sv1 or not sv2:
            return 0.0

        # 鎵惧埌鍏叡绁栧厛
        common_ancestors = set(sv1.keys()) & set(sv2.keys())

        if not common_ancestors:
            return 0.0

        # 璁＄畻鍏叡绁栧厛鐨勮涔夎础鐚箣鍜?
        numerator = sum(sv1[t] + sv2[t] for t in common_ancestors)

        # 璁＄畻鍒嗘瘝
        denominator = sum(sv1.values()) + sum(sv2.values())

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def gene_pair_similarity(self, gene1: str, gene2: str, namespace: str = 'BP') -> Optional[float]:
        """
        璁＄畻涓や釜鍩哄洜涔嬮棿鐨凣O鐩镐技搴?

        绛栫暐锛?
        1. 鑾峰彇涓や釜鍩哄洜鐨凣O terms
        2. 璁＄畻鎵€鏈塆O term瀵圭殑鐩镐技搴?
        3. 浣跨敤BMA (Best Match Average) 绛栫暐鑱氬悎

        鍙傛暟:
            gene1: 鍩哄洜1
            gene2: 鍩哄洜2
            namespace: GO鏈綋绫诲瀷 ('BP', 'CC', 'MF')

        杩斿洖:
            鐩镐技搴?(0-1) 鎴?None锛堝鏋滄棤娉曡绠楋級
        """
        # 鏍囧噯鍖栧熀鍥犲悕
        gene1 = gene1.upper()
        gene2 = gene2.upper()

        # 鑾峰彇GO terms
        if gene1 not in self.goa_index or gene2 not in self.goa_index:
            return None

        go_terms1 = self.goa_index[gene1].get(namespace, [])
        go_terms2 = self.goa_index[gene2].get(namespace, [])

        if not go_terms1 or not go_terms2:
            return None

        # 璁＄畻鎵€鏈塼erm瀵圭殑鐩镐技搴︾煩闃?
        similarities = []

        # BMA绛栫暐锛氬浜巊ene1鐨勬瘡涓猼erm锛屾壘鍒颁笌gene2鏈€鐩镐技鐨則erm
        for go1 in go_terms1:
            max_sim = 0.0
            for go2 in go_terms2:
                sim = self._calculate_term_similarity(go1, go2)
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)

        # 瀵逛簬gene2鐨勬瘡涓猼erm锛屾壘鍒颁笌gene1鏈€鐩镐技鐨則erm
        for go2 in go_terms2:
            max_sim = 0.0
            for go1 in go_terms1:
                sim = self._calculate_term_similarity(go1, go2)
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)

        if not similarities:
            return None

        # 杩斿洖骞冲潎鐩镐技搴?
        return float(np.mean(similarities))

    def complex_similarity(self, complex_genes: List[str], namespace: str = 'BP') -> Optional[float]:
        """
        璁＄畻铔嬬櫧璐ㄥ鍚堢墿鍐呭熀鍥犵殑骞冲潎GO鐩镐技搴?

        鍙傛暟:
            complex_genes: 澶嶅悎鐗╀腑鐨勫熀鍥犲垪琛?
            namespace: GO鏈綋绫诲瀷

        杩斿洖:
            骞冲潎鐩镐技搴?(0-1) 鎴?None
        """
        if len(complex_genes) < 2:
            return None

        # 鏍囧噯鍖栧熀鍥犲悕
        genes = [g.upper() for g in complex_genes]

        # 杩囨护鎺夋病鏈塆O娉ㄩ噴鐨勫熀鍥?
        valid_genes = [g for g in genes if g in self.goa_index and self.goa_index[g].get(namespace)]

        if len(valid_genes) < 2:
            return None

        # 璁＄畻鎵€鏈夊熀鍥犲鐨勭浉浼煎害
        similarities = []
        for i in range(len(valid_genes)):
            for j in range(i + 1, len(valid_genes)):
                sim = self.gene_pair_similarity(valid_genes[i], valid_genes[j], namespace)
                if sim is not None:
                    similarities.append(sim)

        if not similarities:
            return None

        return float(np.mean(similarities))

    def batch_gene_pair_similarity(self, gene_pairs: List[Tuple[str, str]], namespace: str = 'BP') -> Dict[Tuple[str, str], Optional[float]]:
        """
        鎵归噺璁＄畻鍩哄洜瀵圭浉浼煎害

        鍙傛暟:
            gene_pairs: [(gene1, gene2), ...]
            namespace: GO鏈綋绫诲瀷

        杩斿洖:
            {(gene1, gene2): similarity}
        """
        results = {}
        for gene1, gene2 in gene_pairs:
            sim = self.gene_pair_similarity(gene1, gene2, namespace)
            results[(gene1, gene2)] = sim
        return results

    def get_statistics(self) -> Dict:
        """
        鑾峰彇璁＄畻鍣ㄧ粺璁′俊鎭?

        杩斿洖:
            缁熻淇℃伅瀛楀吀
        """
        return {
            'n_go_terms': len(self.terms),
            'n_genes': len(self.goa_index),
            'n_cached_ancestors': len(self._ancestor_cache),
            'weight_is_a': self.weight_is_a,
            'weight_part_of': self.weight_part_of
        }


if __name__ == '__main__':
    from data_loader import ValidationDataLoader

    print("\n=== GO鐩镐技搴﹁绠楀櫒娴嬭瘯 ===\n")

    # 鍔犺浇鏁版嵁
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    loader = ValidationDataLoader(project_root / 'data/validation_data')

    print("1. 鍔犺浇鏁版嵁...")
    go_dag = loader.load_go_dag()
    goa_index = loader.load_goa_index()
    print(f"   鉁?GO DAG: {len(go_dag['terms']):,} terms")
    print(f"   鉁?GOA: {len(goa_index):,} genes")

    # 鍒濆鍖栬绠楀櫒
    print("\n2. 鍒濆鍖栬绠楀櫒...")
    calculator = GOSimilarityCalculator(go_dag, goa_index)
    stats = calculator.get_statistics()
    print(f"   鉁?{stats['n_go_terms']:,} GO terms")
    print(f"   鉁?{stats['n_genes']:,} genes")

    print("\n3. 娴嬭瘯GO term鐩镐技搴?..")
    test_go1 = 'GO:0005515'  # protein binding
    test_go2 = 'GO:0005488'  # binding
    sim = calculator._calculate_term_similarity(test_go1, test_go2)
    print(f"   {test_go1} vs {test_go2}: {sim:.4f}")

    print("\n4. 娴嬭瘯鍩哄洜瀵圭浉浼煎害...")
    # 鎵句竴浜涙湁娉ㄩ噴鐨勫熀鍥?
    test_genes = list(goa_index.keys())[:10]
    print(f"   娴嬭瘯鍩哄洜: {test_genes[:3]}")

    for i in range(min(3, len(test_genes))):
        for j in range(i+1, min(3, len(test_genes))):
            gene1, gene2 = test_genes[i], test_genes[j]
            sim_bp = calculator.gene_pair_similarity(gene1, gene2, 'BP')
            sim_cc = calculator.gene_pair_similarity(gene1, gene2, 'CC')
            sim_mf = calculator.gene_pair_similarity(gene1, gene2, 'MF')

            print(f"   {gene1} vs {gene2}:")
            bp_str = f"{sim_bp:.4f}" if sim_bp is not None else "N/A"
            cc_str = f"{sim_cc:.4f}" if sim_cc is not None else "N/A"
            mf_str = f"{sim_mf:.4f}" if sim_mf is not None else "N/A"
            print(f"      BP: {bp_str}")
            print(f"      CC: {cc_str}")
            print(f"      MF: {mf_str}")

    print("\n5. 娴嬭瘯澶嶅悎鐗╃浉浼煎害...")
    complex_genes = test_genes[:5]
    sim = calculator.complex_similarity(complex_genes, 'BP')
    print(f"   澶嶅悎鐗╁熀鍥? {complex_genes}")
    sim_str = f"{sim:.4f}" if sim is not None else "N/A"
    print(f"   BP鐩镐技搴? {sim_str}")

    # 鎬ц兘娴嬭瘯
    print("\n6. 鎬ц兘娴嬭瘯...")
    import time
    gene_pairs = [(test_genes[i], test_genes[j]) for i in range(10) for j in range(i+1, 10)]
    start = time.time()
    results = calculator.batch_gene_pair_similarity(gene_pairs[:20], 'BP')
    elapsed = time.time() - start
    print(f"   Computed 20 gene pairs in {elapsed:.2f}s")
    print(f"   骞冲潎姣忓: {elapsed/20*1000:.1f}姣")

    print("\n 鎵€鏈夋祴璇曞畬鎴愶紒")



