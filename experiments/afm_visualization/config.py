"""
AFM澶嶅悎鐗╁彲瑙嗗寲閰嶇疆鏂囦欢
"""

import os

# 鍩虹璺緞
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'results', 'case_tasks', 'afm_visualization')

# 娓叉煋鍙傛暟
RENDER_PARAMS = {
    'width': 1200,
    'height': 1200,
    'dpi': 300
}

# Chain棰滆壊锛圧GB锛?
CHAIN_COLORS_RGB = {
    'A': (0, 176, 240),      # 闈掕壊
    'B': (112, 48, 160),     # 绱壊
    'C': (255, 192, 0),      # 姗欒壊
    'D': (255, 0, 255),      # 鍝佺孩
    'E': (146, 208, 80),     # 缁胯壊
}

# 澶嶅悎鐗╀俊鎭?
COMPLEX_INFO = {
    'HSP90complex_5f951': {
        'short_name': 'HSP90-CDC37',
        'avg_plddt': 79.57,
        'chains': {
            'A': 'CDC37',
            'B': 'HSP90AA1',
            'C': 'HSP90AB1',
            'D': 'STIP1'
        },
        'integrated_view': '婕旂ず鏂囩1_01.jpg'
    },
    'SCFE3_57587': {
        'short_name': 'SCF-SKP2',
        'avg_plddt': 81.43,
        'chains': {
            'A': 'CUL1',
            'B': 'RBX1',
            'C': 'SKP1',
            'D': 'SKP2'
        },
        'integrated_view': '婕旂ず鏂囩1_04.jpg'
    },
    'ost_complex_b8f13': {
        'short_name': 'OST-B Core',
        'avg_plddt': 85.30,
        'chains': {
            'A': 'DAD1',
            'B': 'DDOST',
            'C': 'MAGT1',
            'D': 'STT3B'
        },
        'integrated_view': '婕旂ず鏂囩1_02(1).jpg'
    },
    'Pol伪primase_4bcf3': {
        'short_name': 'Pol 伪-POLD3',
        'avg_plddt': 70.54,
        'chains': {
            'A': 'POLA1',
            'B': 'POLD3',
            'C': 'PRIM1',
            'D': 'PRIM2'
        },
        'integrated_view': '婕旂ず鏂囩1_03.jpg'
    }
}

# 杈撳嚭鏂囦欢鍚?
OUTPUT_FILENAMES = {
    'view1': 'view1_chain_colored.png',
    'view2': 'view2.png',
    'view3': 'view3.png',
    'combined_png': 'combined.png',
    'combined_tiff': 'combined.tiff',
    'all_combined': 'all_complexes_combined.png'
}

# 鏍囨敞鍙傛暟
ANNOTATION_PARAMS = {
    'font_family': 'arial.ttf',
    'font_size': 24
}


def get_complex_info(complex_id):
    """Helper."""
    return COMPLEX_INFO.get(complex_id)


def get_output_dir(complex_id):
    """鑾峰彇澶嶅悎鐗╃殑杈撳嚭鐩綍"""
    return os.path.join(OUTPUT_DIR, complex_id)


def list_all_complexes():
    """鍒楀嚭鎵€鏈夊鍚堢墿ID"""
    return list(COMPLEX_INFO.keys())


def validate_config():
    """楠岃瘉閰嶇疆"""
    errors = []

    # 妫€鏌ヨ緭鍑虹洰褰?
    if not os.path.exists(OUTPUT_DIR):
        errors.append(f"杈撳嚭鐩綍涓嶅瓨鍦? {OUTPUT_DIR}")

    # 妫€鏌ユ瘡涓鍚堢墿
    for complex_id in COMPLEX_INFO:
        complex_dir = get_output_dir(complex_id)
        if not os.path.exists(complex_dir):
            errors.append(f"澶嶅悎鐗╃洰褰曚笉瀛樺湪: {complex_dir}")

    return (len(errors) == 0, errors)



