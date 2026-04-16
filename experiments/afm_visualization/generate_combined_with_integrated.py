"""
浣跨敤涓夎鍥剧敓鎴愬鍚堢墿鍚堝苟鍥惧儚
"""

import os
import sys

# 娣诲姞褰撳墠鐩綍鍒癙ython璺緞
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from image_combiner import ImageCombiner


def generate_single_complex(complex_id):
    """
    鐢熸垚鍗曚釜澶嶅悎鐗╃殑鍚堝苟鍥惧儚

    Args:
        complex_id: 澶嶅悎鐗㊣D
    """
    print(f"\n澶勭悊澶嶅悎鐗? {complex_id}")

    # 鑾峰彇澶嶅悎鐗╀俊鎭?
    info = config.get_complex_info(complex_id)
    complex_dir = config.get_output_dir(complex_id)

    # 鑾峰彇鏂囦欢璺緞
    view1_path = os.path.join(complex_dir, config.OUTPUT_FILENAMES['view1'])
    view2_path = os.path.join(complex_dir, config.OUTPUT_FILENAMES['view2'])
    view3_path = os.path.join(complex_dir, config.OUTPUT_FILENAMES['view3'])

    print(f"  view1: {view1_path}")
    print(f"  view2: {view2_path}")
    print(f"  view3: {view3_path}")

    # 妫€鏌ユ枃浠舵槸鍚﹀瓨鍦?
    if not os.path.exists(view1_path):
        print(f"  ERROR: view1 file not found: {view1_path}")
        return None

    if not os.path.exists(view2_path):
        print(f"  ERROR: view2 file not found: {view2_path}")
        return None

    if not os.path.exists(view3_path):
        print(f"  ERROR: view3 file not found: {view3_path}")
        return None

    # 鍒涘缓鍥惧儚缁勫悎鍣?
    combiner = ImageCombiner(complex_id)

    # 缁勫悎涓変釜瑙嗗浘
    output_paths = combiner.combine_three_views(
        view1_path,
        view2_path,
        view3_path,
        add_annotations=True,
        add_colorbar=False
    )

    print(f"  鐢熸垚: {output_paths['png']}")

    return output_paths


def generate_all_complexes():
    """
    鐢熸垚鎵€鏈夊鍚堢墿鐨勫悎骞跺浘鍍忓苟鍒涘缓鎬昏鍥?
    """
    print("寮€濮嬬敓鎴愬鍚堢墿鍚堝苟鍥惧儚...\n")

    # 鑾峰彇鎵€鏈夊鍚堢墿ID
    complex_ids = config.list_all_complexes()

    # 鐢熸垚姣忎釜澶嶅悎鐗╃殑鍥惧儚
    results = []
    for complex_id in complex_ids:
        output_paths = generate_single_complex(complex_id)
        if output_paths:
            results.append((complex_id, output_paths))

    if not results:
        print("\n閿欒: 娌℃湁鎴愬姛鐢熸垚浠讳綍鍥惧儚")
        return

    print(f"\n鎴愬姛鐢熸垚 {len(results)}/{len(complex_ids)} 涓鍚堢墿鍥惧儚")

    # 鐢熸垚鎬昏鍥?
    print("\n鐢熸垚鎬昏鍥?..")

    complex_ids_success = [r[0] for r in results]
    combined_png_paths = [r[1]['png'] for r in results]

    # 浣跨敤绗竴涓鍚堢墿鐨刢ombiner鏉ョ敓鎴愭€昏鍥?
    combiner = ImageCombiner(complex_ids_success[0])
    overview_path = combiner.combine_all_complexes(complex_ids_success, combined_png_paths)

    print(f"\n鎬昏鍥惧凡鐢熸垚: {overview_path}")
    print("\n瀹屾垚!")


if __name__ == "__main__":
    generate_all_complexes()



