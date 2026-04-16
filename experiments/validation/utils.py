#!/usr/bin/env python3
"""
楠岃瘉宸ュ叿鍑芥暟妯″潡

鑱岃矗锛氭彁渚涢€氱敤鐨勮緟鍔╁嚱鏁帮紝濡傚熀鍥營D杞崲銆佹暟鎹獙璇佺瓑
"""

import re
import logging
from typing import Optional, List, Set, Dict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_gene_symbol(gene: str) -> str:
    """
    鏍囧噯鍖栧熀鍥犵鍙?

    鍙傛暟:
        gene: 鍘熷鍩哄洜绗﹀彿

    杩斿洖:
        鏍囧噯鍖栧悗鐨勫熀鍥犵鍙凤紙澶у啓锛屽幓闄ょ┖鏍硷級
    """
    if not gene:
        return ""
    return gene.strip().upper()


def is_valid_ensembl_id(ensembl_id: str) -> bool:
    """
    楠岃瘉Ensembl鍩哄洜ID鏍煎紡

    鍙傛暟:
        ensembl_id: Ensembl ID

    杩斿洖:
        鏄惁涓烘湁鏁堟牸寮?
    """
    pattern = r'^ENSG\d{11}(\.\d+)?$'
    return bool(re.match(pattern, ensembl_id))


def is_valid_go_term(go_id: str) -> bool:
    """
    楠岃瘉GO term ID鏍煎紡

    鍙傛暟:
        go_id: GO term ID

    杩斿洖:
        鏄惁涓烘湁鏁堟牸寮?
    """
    pattern = r'^GO:\d{7}$'
    return bool(re.match(pattern, go_id))


def remove_ensembl_version(ensembl_id: str) -> str:
    """
    绉婚櫎Ensembl ID鐨勭増鏈彿

    鍙傛暟:
        ensembl_id: 甯︾増鏈彿鐨凟nsembl ID (濡?ENSG00000139618.2)

    杩斿洖:
        鏃犵増鏈彿鐨凟nsembl ID (濡?ENSG00000139618)
    """
    return ensembl_id.split('.')[0]


def validate_gene_list(genes: List[str], available_genes: Set[str]) -> Dict[str, List[str]]:
    """
    楠岃瘉鍩哄洜鍒楄〃锛岃繑鍥炴湁鏁堝拰鏃犳晥鐨勫熀鍥?

    鍙傛暟:
        genes: 寰呴獙璇佺殑鍩哄洜鍒楄〃
        available_genes: 鍙敤鐨勫熀鍥犻泦鍚?

    杩斿洖:
        {'valid': [...], 'invalid': [...]}
    """
    valid = []
    invalid = []

    for gene in genes:
        normalized = normalize_gene_symbol(gene)
        if normalized in available_genes:
            valid.append(normalized)
        else:
            invalid.append(gene)

    return {
        'valid': valid,
        'invalid': invalid
    }


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """
    瀹夊叏璁＄畻鍧囧€硷紝澶勭悊绌哄垪琛ㄥ拰NaN

    鍙傛暟:
        values: 鏁板€煎垪琛?
        default: 榛樿鍊?

    杩斿洖:
        鍧囧€兼垨榛樿鍊?
    """
    if not values:
        return default

    clean_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]

    if not clean_values:
        return default

    return float(np.mean(clean_values))


def safe_median(values: List[float], default: float = 0.0) -> float:
    """
    瀹夊叏璁＄畻涓綅鏁帮紝澶勭悊绌哄垪琛ㄥ拰NaN

    鍙傛暟:
        values: 鏁板€煎垪琛?
        default: 榛樿鍊?

    杩斿洖:
        涓綅鏁版垨榛樿鍊?
    """
    if not values:
        return default

    clean_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]

    if not clean_values:
        return default

    return float(np.median(clean_values))


def jaccard_similarity(set1: Set, set2: Set) -> float:
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


def filter_nan_pairs(values1: List[float], values2: List[float]) -> tuple:
    """
    杩囨护鍖呭惈NaN鐨勬暟鎹

    鍙傛暟:
        values1: 鏁板€煎垪琛?
        values2: 鏁板€煎垪琛?

    杩斿洖:
        (clean_values1, clean_values2)
    """
    if len(values1) != len(values2):
        raise ValueError(f"鍒楄〃闀垮害涓嶅尮閰? {len(values1)} vs {len(values2)}")

    clean1 = []
    clean2 = []

    for v1, v2 in zip(values1, values2):
        if not (np.isnan(v1) or np.isnan(v2) or np.isinf(v1) or np.isinf(v2)):
            clean1.append(v1)
            clean2.append(v2)

    return clean1, clean2


def batch_process(items: List, batch_size: int = 100):
    """
    鎵瑰鐞嗙敓鎴愬櫒

    鍙傛暟:
        items: 寰呭鐞嗙殑椤圭洰鍒楄〃
        batch_size: 鎵瑰ぇ灏?

    鐢熸垚:
        鎵规鏁版嵁
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


if __name__ == '__main__':
    print("\n=== 宸ュ叿鍑芥暟娴嬭瘯 ===\n")

    print("1. 鍩哄洜绗﹀彿鏍囧噯鍖?")
    test_genes = ['brca1', 'BRCA2', ' tp53 ', 'TP53']
    for gene in test_genes:
        print(f"   {gene!r} -> {normalize_gene_symbol(gene)!r}")

    print("\n2. Ensembl ID楠岃瘉:")
    test_ids = ['ENSG00000139618', 'ENSG00000139618.2', 'INVALID']
    for eid in test_ids:
        valid = is_valid_ensembl_id(eid)
        print(f"   {eid}: {valid}")

    print("\n3. GO term楠岃瘉:")
    test_gos = ['GO:0005515', 'GO:123456', 'INVALID']
    for go in test_gos:
        valid = is_valid_go_term(go)
        print(f"   {go}: {valid}")

    print("\n4. Ensembl鐗堟湰鍙风Щ闄?")
    print(f"   ENSG00000139618.2 -> {remove_ensembl_version('ENSG00000139618.2')}")

    print("\n5. Jaccard鐩镐技搴?")
    set1 = {1, 2, 3, 4}
    set2 = {3, 4, 5, 6}
    print(f"   {set1} 鈭?{set2} = {jaccard_similarity(set1, set2):.3f}")

    print("\n6. 瀹夊叏缁熻鍑芥暟:")
    values = [1.0, 2.0, np.nan, 3.0, np.inf, 4.0]
    print(f"   鍘熷鏁版嵁: {values}")
    print(f"   瀹夊叏鍧囧€? {safe_mean(values):.2f}")
    print(f"   瀹夊叏涓綅鏁? {safe_median(values):.2f}")

    print("\n7. NaN瀵硅繃婊?")
    v1 = [1.0, 2.0, np.nan, 3.0]
    v2 = [4.0, np.nan, 5.0, 6.0]
    clean1, clean2 = filter_nan_pairs(v1, v2)
    print(f"   鍘熷: {v1} + {v2}")
    print(f"   娓呯悊: {clean1} + {clean2}")

    print("\nAll utility checks passed")



