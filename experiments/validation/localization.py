#!/usr/bin/env python3
"""
浜氱粏鑳炲畾浣嶄竴鑷存€ц绠楁ā鍧?

鑱岃矗锛氬熀浜嶤OMPARTMENTS鏁版嵁璁＄畻鍩哄洜瀵瑰拰铔嬬櫧璐ㄥ鍚堢墿鐨勫畾浣嶄竴鑷存€?
鏂规硶锛欽accard鐩镐技搴?+ 浼樺娍瀹氫綅鍒嗘瀽
"""

import logging
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalizationCalculator:
    """浜氱粏鑳炲畾浣嶄竴鑷存€ц绠楀櫒"""

    def __init__(self, compartments_data: Dict[str, List[str]]):
        """
        鍒濆鍖栧畾浣嶈绠楀櫒

        鍙傛暟:
            compartments_data: COMPARTMENTS鏁版嵁 {gene_symbol: [location1, location2, ...]}
        """
        self.compartments = compartments_data

        logger.info(f"瀹氫綅璁＄畻鍣ㄥ垵濮嬪寲: {len(self.compartments):,} 鍩哄洜")

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """
        璁＄畻Jaccard鐩镐技搴?

        鍙傛暟:
            set1: 闆嗗悎1
            set2: 闆嗗悎2

        杩斿洖:
            Jaccard鐩镐技搴?(0-1)
        """
        if not set1 and not set2:
            return 1.0

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def gene_pair_localization(self, gene1: str, gene2: str) -> Optional[float]:
        """
        璁＄畻涓や釜鍩哄洜鐨勫畾浣嶄竴鑷存€?

        鍙傛暟:
            gene1: 鍩哄洜1 (绗﹀彿)
            gene2: 鍩哄洜2 (绗﹀彿)

        杩斿洖:
            Jaccard鐩镐技搴?(0-1) 鎴?None
        """
        # 鏍囧噯鍖栧熀鍥犲悕
        g1 = gene1.upper()
        g2 = gene2.upper()

        if g1 not in self.compartments or g2 not in self.compartments:
            return None

        # 鑾峰彇瀹氫綅
        loc1 = set(self.compartments[g1])
        loc2 = set(self.compartments[g2])

        if not loc1 or not loc2:
            return None

        return self._jaccard_similarity(loc1, loc2)

    def complex_localization(self, complex_genes: List[str]) -> Optional[Dict]:
        """
        璁＄畻铔嬬櫧璐ㄥ鍚堢墿鐨勫畾浣嶄竴鑷存€?

        杩斿洖澶氫釜鎸囨爣锛?
        - avg_jaccard: 骞冲潎Jaccard鐩镐技搴?
        - dominant_locations: 浼樺娍瀹氫綅锛堝嚭鐜伴鐜囨渶楂樼殑浣嶇疆锛?
        - top1_coverage: 鏈€浼樺娍瀹氫綅瑕嗙洊鐨勫熀鍥犳瘮渚?
        - localization_diversity: 瀹氫綅澶氭牱鎬э紙Shannon鐔碉級

        鍙傛暟:
            complex_genes: 澶嶅悎鐗╀腑鐨勫熀鍥犲垪琛?

        杩斿洖:
            鎸囨爣瀛楀吀鎴朜one
        """
        if len(complex_genes) < 2:
            return None

        # 鏍囧噯鍖栧熀鍥犲悕
        genes = [g.upper() for g in complex_genes]

        # 杩囨护鎺夋病鏈夊畾浣嶆暟鎹殑鍩哄洜
        valid_genes = [g for g in genes if g in self.compartments and self.compartments[g]]

        if len(valid_genes) < 2:
            return None

        # 1. 璁＄畻骞冲潎Jaccard鐩镐技搴?
        jaccard_scores = []
        for i in range(len(valid_genes)):
            for j in range(i + 1, len(valid_genes)):
                score = self.gene_pair_localization(valid_genes[i], valid_genes[j])
                if score is not None:
                    jaccard_scores.append(score)

        if not jaccard_scores:
            return None

        avg_jaccard = float(np.mean(jaccard_scores))

        # 2. 鎵惧埌浼樺娍瀹氫綅
        all_locations = []
        for gene in valid_genes:
            all_locations.extend(self.compartments[gene])

        location_counts = Counter(all_locations)
        total_locations = len(all_locations)

        # 浼樺娍瀹氫綅锛堝嚭鐜伴鐜?> 闃堝€硷級
        dominant_threshold = 0.3
        dominant_locations = [
            loc for loc, count in location_counts.items()
            if count / total_locations >= dominant_threshold
        ]

        # 3. Top1瀹氫綅瑕嗙洊鐜?
        if location_counts:
            top_location, top_count = location_counts.most_common(1)[0]
            top1_coverage = sum(
                1 for gene in valid_genes
                if top_location in self.compartments[gene]
            ) / len(valid_genes)
        else:
            top_location = None
            top1_coverage = 0.0

        # 4. 瀹氫綅澶氭牱鎬э紙Shannon鐔碉級
        diversity = 0.0
        if location_counts:
            for count in location_counts.values():
                p = count / total_locations
                if p > 0:
                    diversity -= p * np.log2(p)

        return {
            'avg_jaccard': avg_jaccard,
            'dominant_locations': dominant_locations,
            'top_location': top_location,
            'top1_coverage': float(top1_coverage),
            'localization_diversity': float(diversity),
            'n_valid_genes': len(valid_genes),
            'n_unique_locations': len(location_counts)
        }

    def batch_gene_pair_localization(self, gene_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[float]]:
        """
        鎵归噺璁＄畻鍩哄洜瀵瑰畾浣嶄竴鑷存€?

        鍙傛暟:
            gene_pairs: [(gene1, gene2), ...]

        杩斿洖:
            {(gene1, gene2): jaccard_similarity}
        """
        results = {}
        for gene1, gene2 in gene_pairs:
            score = self.gene_pair_localization(gene1, gene2)
            results[(gene1, gene2)] = score
        return results

    def get_gene_locations(self, gene: str) -> Optional[List[str]]:
        """
        鑾峰彇鍩哄洜鐨勫畾浣嶅垪琛?

        鍙傛暟:
            gene: 鍩哄洜绗﹀彿

        杩斿洖:
            瀹氫綅鍒楄〃鎴朜one
        """
        g = gene.upper()
        if g not in self.compartments:
            return None
        return self.compartments[g]

    def get_statistics(self) -> Dict:
        """
        鑾峰彇璁＄畻鍣ㄧ粺璁′俊鎭?

        杩斿洖:
            缁熻淇℃伅瀛楀吀
        """
        # 缁熻瀹氫綅鏁版嵁
        n_genes_with_loc = sum(1 for locs in self.compartments.values() if locs)
        all_locs = [loc for locs in self.compartments.values() for loc in locs]
        unique_locs = set(all_locs)

        avg_locs_per_gene = len(all_locs) / max(n_genes_with_loc, 1)

        return {
            'n_genes': len(self.compartments),
            'n_genes_with_localization': n_genes_with_loc,
            'n_unique_locations': len(unique_locs),
            'avg_locations_per_gene': avg_locs_per_gene,
            'unique_locations': sorted(unique_locs)
        }


if __name__ == '__main__':
    from data_loader import ValidationDataLoader

    print("\n=== 瀹氫綅涓€鑷存€ц绠楀櫒娴嬭瘯 ===\n")

    # 鍔犺浇鏁版嵁
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    loader = ValidationDataLoader(project_root / 'data/validation_data')

    print("1. 鍔犺浇COMPARTMENTS鏁版嵁...")
    compartments = loader.load_compartments()
    print(f"   鉁?{len(compartments):,} 鍩哄洜")

    # 鍒濆鍖栬绠楀櫒
    print("\n2. 鍒濆鍖栬绠楀櫒...")
    calculator = LocalizationCalculator(compartments)
    stats = calculator.get_statistics()
    print(f"   鉁?{stats['n_genes']:,} 鍩哄洜")
    print(f"   OK {stats['n_genes_with_localization']:,} genes have localization data")
    print(f"   OK {stats['n_unique_locations']} unique locations")
    print(f"   OK average locations per gene: {stats['avg_locations_per_gene']:.2f}")

    print(f"\n   鍙敤瀹氫綅绫诲瀷: {', '.join(stats['unique_locations'][:10])}...")

    print("\n3. 娴嬭瘯鍩哄洜瀹氫綅鏌ヨ...")
    test_genes = [g for g in compartments.keys() if compartments[g]][:5]
    for gene in test_genes:
        locs = calculator.get_gene_locations(gene)
        if locs:
            print(f"   {gene}: {', '.join(locs)}")

    print("\n4. 娴嬭瘯鍩哄洜瀵瑰畾浣嶄竴鑷存€?..")
    for i in range(min(3, len(test_genes))):
        for j in range(i+1, min(3, len(test_genes))):
            gene1, gene2 = test_genes[i], test_genes[j]
            score = calculator.gene_pair_localization(gene1, gene2)

            print(f"   {gene1} vs {gene2}:")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"      Jaccard: {score_str}")

    print("\n5. 娴嬭瘯澶嶅悎鐗╁畾浣?..")
    complex_genes = test_genes[:5]
    result = calculator.complex_localization(complex_genes)

    print(f"   澶嶅悎鐗╁熀鍥? {complex_genes}")
    if result:
        print(f"   骞冲潎Jaccard: {result['avg_jaccard']:.4f}")
        print(f"   浼樺娍瀹氫綅: {', '.join(result['dominant_locations']) if result['dominant_locations'] else 'None'}")
        print(f"   Top1瀹氫綅: {result['top_location']}")
        print(f"   Top1瑕嗙洊鐜? {result['top1_coverage']:.2%}")
        print(f"   瀹氫綅澶氭牱鎬? {result['localization_diversity']:.4f}")
        print(f"   鏈夋晥鍩哄洜鏁? {result['n_valid_genes']}")
        print(f"   鍞竴瀹氫綅鏁? {result['n_unique_locations']}")
    else:
        print("   鏃犳硶璁＄畻")

    # 鎬ц兘娴嬭瘯
    print("\n6. 鎬ц兘娴嬭瘯...")
    import time
    gene_pairs = [(test_genes[i], test_genes[j]) for i in range(10) for j in range(i+1, min(10, len(test_genes)))]
    start = time.time()
    results = calculator.batch_gene_pair_localization(gene_pairs[:20])
    elapsed = time.time() - start
    print(f"   Computed 20 gene pairs in {elapsed:.2f}s")
    print(f"   骞冲潎姣忓: {elapsed/20*1000:.1f}姣")

    print("\n 鎵€鏈夋祴璇曞畬鎴愶紒")



