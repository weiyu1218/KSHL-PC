#!/usr/bin/env python3
"""
鏁版嵁鍔犺浇妯″潡

鑱岃矗锛氬姞杞藉拰缂撳瓨棰勫鐞嗘暟鎹紝鎻愪緵缁熶竴鐨勬暟鎹闂帴鍙?
"""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationDataLoader:
    """Helper."""

    def __init__(self, data_dir: str = 'data/validation_data'):
        """
        鍒濆鍖栨暟鎹姞杞藉櫒

        鍙傛暟:
            data_dir: 楠岃瘉鏁版嵁鏍圭洰褰?
        """
        self.data_dir = Path(data_dir)

        # 楠岃瘉鏁版嵁鐩綍瀛樺湪
        if not self.data_dir.exists():
            raise FileNotFoundError(f"鏁版嵁鐩綍涓嶅瓨鍦? {self.data_dir}")

        # 缂撳瓨鍙橀噺锛堟噿鍔犺浇锛?
        self._goa_index: Optional[Dict] = None
        self._go_dag: Optional[Dict] = None
        self._gtex_stats: Optional[Dict] = None
        self._compartments: Optional[Dict] = None

        logger.info(f"ValidationDataLoader 鍒濆鍖栧畬鎴愶紝鏁版嵁鐩綍: {self.data_dir}")

    def load_goa_index(self, use_pickle: bool = True) -> Dict[str, Dict[str, List[str]]]:
        """
        鍔犺浇GOA鍩哄洜-GO娉ㄩ噴绱㈠紩

        鍙傛暟:
            use_pickle: 鏄惁浣跨敤pickle鏍煎紡锛堥€熷害鏇村揩锛?

        杩斿洖:
            {gene_symbol: {'BP': [go_ids], 'CC': [go_ids], 'MF': [go_ids]}}
        """
        if self._goa_index is not None:
            return self._goa_index

        if use_pickle:
            pkl_path = self.data_dir / 'goa/processed/goa_gene_go_index.pkl'
            logger.info(f"鍔犺浇GOA绱㈠紩: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                self._goa_index = pickle.load(f)
        else:
            json_path = self.data_dir / 'goa/processed/goa_gene_go_index.json'
            logger.info(f"鍔犺浇GOA绱㈠紩: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                self._goa_index = json.load(f)

        logger.info(f"GOA index loaded: {len(self._goa_index):,} genes")
        return self._goa_index

    def load_go_dag(self, use_pickle: bool = True) -> Dict[str, Any]:
        """
        鍔犺浇GO DAG缁撴瀯

        鍙傛暟:
            use_pickle: 鏄惁浣跨敤pickle鏍煎紡

        杩斿洖:
            {
                'terms': {go_id: term_dict},
                'edges': {'is_a': [...], 'part_of': [...]},
                'parents': {go_id: [parent_ids]},
                'children': {go_id: [child_ids]},
                'namespaces': {'biological_process': [...], ...}
            }
        """
        if self._go_dag is not None:
            return self._go_dag

        if use_pickle:
            pkl_path = self.data_dir / 'go/processed/go_dag_structure.pkl'
            logger.info(f"鍔犺浇GO DAG: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                self._go_dag = pickle.load(f)
        else:
            json_path = self.data_dir / 'go/processed/go_dag_structure.json'
            logger.info(f"鍔犺浇GO DAG: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                self._go_dag = json.load(f)

        n_terms = len(self._go_dag.get('terms', {}))
        logger.info(f"GO DAG鍔犺浇瀹屾垚: {n_terms:,} 涓狦O terms")
        return self._go_dag

    def load_gtex_stats(self, use_pickle: bool = True) -> Dict[str, Dict[str, float]]:
        """
        鍔犺浇GTEx鍩哄洜琛ㄨ揪缁熻

        鍙傛暟:
            use_pickle: 鏄惁浣跨敤pickle鏍煎紡

        杩斿洖:
            {ensembl_id: {'mean': float, 'std': float, 'median': float, 'max': float, 'min': float, 'n_samples': int}}
        """
        if self._gtex_stats is not None:
            return self._gtex_stats

        if use_pickle:
            pkl_path = self.data_dir / 'gtex/processed/gtex_gene_stats.pkl'
            logger.info(f"鍔犺浇GTEx缁熻: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                self._gtex_stats = pickle.load(f)
        else:
            json_path = self.data_dir / 'gtex/processed/gtex_gene_stats.json'
            logger.info(f"鍔犺浇GTEx缁熻: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                self._gtex_stats = json.load(f)

        logger.info(f"GTEx statistics loaded: {len(self._gtex_stats):,} genes")
        return self._gtex_stats

    def load_compartments(self, use_pickle: bool = True, use_simple: bool = True) -> Dict[str, List[str]]:
        """
        鍔犺浇COMPARTMENTS瀹氫綅鏁版嵁

        鍙傛暟:
            use_pickle: 鏄惁浣跨敤pickle鏍煎紡
            use_simple: 鏄惁浣跨敤绠€鍖栫増锛堟帹鑽愶紝鏂囦欢鏇村皬锛?

        杩斿洖:
            {gene_symbol: [location1, location2, ...]}
        """
        if self._compartments is not None:
            return self._compartments

        suffix = 'simple' if use_simple else 'detailed'

        if use_pickle:
            pkl_path = self.data_dir / f'localization/processed/compartments_gene_location_{suffix}.pkl'
            logger.info(f"鍔犺浇COMPARTMENTS鏁版嵁: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                self._compartments = pickle.load(f)
        else:
            json_path = self.data_dir / f'localization/processed/compartments_gene_location_{suffix}.json'
            logger.info(f"鍔犺浇COMPARTMENTS鏁版嵁: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                self._compartments = json.load(f)

        logger.info(f"COMPARTMENTS data loaded: {len(self._compartments):,} genes")
        return self._compartments

    def get_gene_go_terms(self, gene_symbol: str, namespace: Optional[str] = None) -> List[str]:
        """
        鑾峰彇鍩哄洜鐨凣O terms

        鍙傛暟:
            gene_symbol: 鍩哄洜绗﹀彿
            namespace: 鏈綋绫诲瀷 ('BP', 'CC', 'MF')锛孨one琛ㄧず鍏ㄩ儴

        杩斿洖:
            GO term鍒楄〃
        """
        goa_index = self.load_goa_index()

        if gene_symbol not in goa_index:
            return []

        if namespace:
            return goa_index[gene_symbol].get(namespace, [])

        # 杩斿洖鎵€鏈夋湰浣撶殑terms
        all_terms = []
        for ont_terms in goa_index[gene_symbol].values():
            all_terms.extend(ont_terms)
        return all_terms

    def get_gene_locations(self, gene_symbol: str) -> List[str]:
        """
        鑾峰彇鍩哄洜鐨勪簹缁嗚優瀹氫綅

        鍙傛暟:
            gene_symbol: 鍩哄洜绗﹀彿

        杩斿洖:
            瀹氫綅鍒楄〃
        """
        compartments = self.load_compartments()
        return compartments.get(gene_symbol, [])

    def get_gene_expression_stats(self, ensembl_id: str) -> Optional[Dict[str, float]]:
        """
        鑾峰彇鍩哄洜鐨勮〃杈剧粺璁′俊鎭?

        鍙傛暟:
            ensembl_id: Ensembl鍩哄洜ID

        杩斿洖:
            缁熻淇℃伅瀛楀吀鎴朜one
        """
        gtex_stats = self.load_gtex_stats()
        return gtex_stats.get(ensembl_id)

    def get_all_genes(self, source: str = 'goa') -> List[str]:
        """
        鑾峰彇鎵€鏈夊熀鍥犲垪琛紙鐢ㄤ簬鐢熸垚闅忔満瀵圭収锛?

        鍙傛暟:
            source: 鏁版嵁婧?('goa', 'gtex', 'compartments')

        杩斿洖:
            鍩哄洜鍒楄〃
        """
        if source == 'goa':
            goa_index = self.load_goa_index()
            return list(goa_index.keys())
        elif source == 'gtex':
            gtex_stats = self.load_gtex_stats()
            return list(gtex_stats.keys())
        elif source == 'compartments':
            compartments = self.load_compartments()
            return list(compartments.keys())
        else:
            raise ValueError(f"涓嶆敮鎸佺殑鏁版嵁婧? {source}")

    def clear_cache(self):
        """Helper."""
        self._goa_index = None
        self._go_dag = None
        self._gtex_stats = None
        self._compartments = None
        logger.info("Cache cleared")

    def get_data_info(self) -> Dict[str, Any]:
        """
        鑾峰彇鏁版嵁闆嗕俊鎭?

        杩斿洖:
            鏁版嵁闆嗙粺璁′俊鎭?
        """
        info = {}

        if self._goa_index is not None or self.data_dir.joinpath('goa/processed/goa_gene_go_index.pkl').exists():
            goa = self.load_goa_index()
            info['goa'] = {
                'n_genes': len(goa),
                'n_bp_annotations': sum(len(g.get('BP', [])) for g in goa.values()),
                'n_cc_annotations': sum(len(g.get('CC', [])) for g in goa.values()),
                'n_mf_annotations': sum(len(g.get('MF', [])) for g in goa.values())
            }

        if self._go_dag is not None or self.data_dir.joinpath('go/processed/go_dag_structure.pkl').exists():
            go_dag = self.load_go_dag()
            terms = go_dag.get('terms', {})
            namespaces = go_dag.get('namespaces', {})
            info['go_dag'] = {
                'n_terms': len(terms),
                'n_bp_terms': len(namespaces.get('biological_process', [])),
                'n_cc_terms': len(namespaces.get('cellular_component', [])),
                'n_mf_terms': len(namespaces.get('molecular_function', []))
            }

        if self._gtex_stats is not None or self.data_dir.joinpath('gtex/processed/gtex_gene_stats.pkl').exists():
            gtex = self.load_gtex_stats()
            info['gtex'] = {
                'n_genes': len(gtex)
            }

        if self._compartments is not None or self.data_dir.joinpath('localization/processed/compartments_gene_location_simple.pkl').exists():
            compartments = self.load_compartments()
            info['compartments'] = {
                'n_genes': len(compartments)
            }

        return info
