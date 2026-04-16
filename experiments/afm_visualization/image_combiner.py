"""
鍥惧儚缁勫悎鍜屾爣娉ㄦā鍧?

璐熻矗锛?
1. 缁勫悎涓変釜瑙嗗浘涓哄崟寮犲浘鍍?
2. 娣诲姞铔嬬櫧鍚嶇О鏍囨敞
3. 娣诲姞pLDDT鍒嗘暟鏍囨敞
4. 鐢熸垚pLDDT鑹叉爣鏉?
5. 淇濆瓨PNG鍜孴IFF鏍煎紡
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageColor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

import config


class ImageCombiner:
    """鍥惧儚缁勫悎鍣ㄧ被"""

    def __init__(self, complex_id):
        """
        鍒濆鍖栧浘鍍忕粍鍚堝櫒

        Args:
            complex_id: 澶嶅悎鐗㊣D
        """
        self.complex_id = complex_id
        self.complex_info = config.get_complex_info(complex_id)
        self.output_dir = config.get_output_dir(complex_id)

        # 纭繚杈撳嚭鐩綍瀛樺湪
        os.makedirs(self.output_dir, exist_ok=True)

    def create_plddt_colorbar(self, width=1200, height=80):
        """
        鐢熸垚pLDDT鑹叉爣鏉?

        Args:
            width: 鑹叉爣瀹藉害
            height: 鑹叉爣楂樺害

        Returns:
            PIL.Image: 鑹叉爣鍥惧儚
        """
        # 鍒涘缓figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor('white')

        # 鍒涘缓棰滆壊鏄犲皠锛圓lphaFold鏍囧噯锛氱孩->榛?>闈?>钃濓級
        # 浣跨敤RdYlBu (涓嶅弽杞?锛?=绾㈣壊(浣?, 100=钃濊壊(楂?
        cmap = plt.cm.RdYlBu
        norm = mcolors.Normalize(vmin=0, vmax=100)

        # 鍒涘缓color bar
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation='horizontal'
        )

        # 璁剧疆鏍囩鍜屽埢搴?
        cb.set_label('Per-residue confidence (pLDDT)',
                    fontsize=14, fontweight='bold')
        cb.set_ticks([0, 25, 50, 75, 100])
        cb.ax.tick_params(labelsize=12)

        # 淇濆瓨鍒颁复鏃舵枃浠?
        temp_path = os.path.join(self.output_dir, '_temp_colorbar.png')
        plt.savefig(temp_path, dpi=100, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close()

        # 鍔犺浇涓篜IL鍥惧儚
        colorbar_img = Image.open(temp_path)

        # 澶嶅埗鍥惧儚浠ヤ究鍏抽棴鏂囦欢
        colorbar_copy = colorbar_img.copy()
        colorbar_img.close()

        # 鍒犻櫎涓存椂鏂囦欢
        try:
            os.remove(temp_path)
        except:
            pass

        return colorbar_copy

    def add_text_with_background(self, draw, text, position, font,
                                 text_color=(0, 0, 0),
                                 bg_color=(255, 255, 255, 230),
                                 padding=10, anchor='lt'):
        """
        娣诲姞甯﹁儗鏅鐨勬枃鏈?

        Args:
            draw: ImageDraw瀵硅薄
            text: 鏂囨湰鍐呭
            position: 鏂囨湰浣嶇疆 (x, y)
            font: 瀛椾綋瀵硅薄
            text_color: 鏂囨湰棰滆壊
            bg_color: 鑳屾櫙棰滆壊锛圧GBA锛?
            padding: 鑳屾櫙妗嗗唴杈硅窛
            anchor: 鏂囨湰閿氱偣 ('lt'=宸︿笂, 'rt'=鍙充笂, 'lb'=宸︿笅, 'rb'=鍙充笅)
        """
        # 妫€鏌ユ槸鍚︽槸澶氳鏂囨湰
        is_multiline = '\n' in text

        # 鑾峰彇鏂囨湰杈圭晫妗?
        if is_multiline:
            # 澶氳鏂囨湰锛氶渶瑕佹墜鍔ㄨ绠椾綅缃?
            bbox = draw.multiline_textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 鏍规嵁anchor璋冩暣浣嶇疆
            x, y = position
            if 'r' in anchor:  # 鍙冲榻?
                x = x - text_width
            if 'b' in anchor:  # 搴曢儴瀵归綈
                y = y - text_height

            # 璁＄畻瀹為檯缁樺埗浣嶇疆
            actual_pos = (x, y)
            bbox = (x, y, x + text_width, y + text_height)
        else:
            bbox = draw.textbbox(position, text, font=font, anchor=anchor)
            actual_pos = position

        x0, y0, x1, y1 = bbox

        # 娣诲姞padding
        bg_bbox = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)

        # 缁樺埗鍗婇€忔槑鑳屾櫙妗?
        base_img = draw._image if hasattr(draw, '_image') else draw.im
        bg_img = Image.new('RGBA', base_img.size, (255, 255, 255, 0))
        bg_draw = ImageDraw.Draw(bg_img)
        bg_draw.rectangle(bg_bbox, fill=bg_color, outline=(0, 0, 0, 255), width=2)

        # 灏嗚儗鏅鍚堝苟鍒板師鍥?
        base_img.paste(bg_img, (0, 0), bg_img)

        # 缁樺埗鏂囨湰
        if is_multiline:
            draw.multiline_text(actual_pos, text, fill=text_color, font=font)
        else:
            draw.text(position, text, fill=text_color, font=font, anchor=anchor)

    def get_font(self, size=None):
        """
        鑾峰彇瀛椾綋瀵硅薄

        Args:
            size: 瀛椾綋澶у皬锛屽鏋滀负None鍒欎娇鐢ㄩ厤缃腑鐨勯粯璁ゅ€?

        Returns:
            ImageFont瀵硅薄
        """
        if size is None:
            size = config.ANNOTATION_PARAMS['font_size']

        try:
            # 灏濊瘯浣跨敤閰嶇疆鐨勫瓧浣?
            font = ImageFont.truetype(config.ANNOTATION_PARAMS['font_family'], size)
        except:
            try:
                # 灏濊瘯浣跨敤DejaVuSans
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size)
            except:
                try:
                    # 灏濊瘯浣跨敤Liberation瀛椾綋
                    font = ImageFont.truetype('/usr/share/fonts/liberation/LiberationSans-Bold.ttf', size)
                except:
                    # 浣跨敤榛樿瀛椾綋
                    print("Warning: Using default font. Text may not display correctly.")
                    font = ImageFont.load_default()

        return font

    def draw_interface_box(self, img, box_position='center', box_color=(255, 100, 100), box_width=8):
        """
        鍦ㄦ暣浣撶粨鏋勫浘涓婄粯鍒剁晫闈㈠尯鍩熸爣娉ㄦ柟妗?

        Args:
            img: PIL Image瀵硅薄
            box_position: 鏂规浣嶇疆 ('center', 'custom')
            box_color: 鏂规棰滆壊RGB
            box_width: 鏂规绾垮

        Returns:
            娣诲姞鏂规鍚庣殑PIL Image瀵硅薄
        """
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        # 璁＄畻鏂规浣嶇疆锛堝湪鍥惧儚涓績鍖哄煙锛?
        if box_position == 'center':
            box_size_ratio = 0.35
            box_w = int(img_width * box_size_ratio)
            box_h = int(img_height * box_size_ratio)

            x0 = (img_width - box_w) // 2
            y0 = (img_height - box_h) // 2
            x1 = x0 + box_w
            y1 = y0 + box_h

        # 缁樺埗鐭╁舰妗?
        draw.rectangle([(x0, y0), (x1, y1)], outline=box_color, width=box_width)

        # 鍦ㄦ鐨勫彸涓婅娣诲姞灏忕澶存爣璁?
        arrow_size = 40
        arrow_x = x1 + 10
        arrow_y = y0
        # 缁樺埗涓€涓畝鍗曠殑绠ご鎸囧悜鍙充晶
        arrow_points = [
            (arrow_x, arrow_y),
            (arrow_x + arrow_size, arrow_y + arrow_size // 2),
            (arrow_x, arrow_y + arrow_size)
        ]
        draw.polygon(arrow_points, fill=box_color)

        return img

    def draw_connection_line(self, combined_img, panel1_width, box_position_ratio=0.5):
        """
        鍦ㄧ粍鍚堝浘涓婄粯鍒惰繛鎺ョ嚎锛堜粠宸anel鐨勬柟妗嗗埌鍙硃anel锛?

        Args:
            combined_img: 缁勫悎鍚庣殑PIL Image瀵硅薄
            panel1_width: 宸anel鐨勫搴?
            box_position_ratio: 鏂规浣嶇疆姣斾緥

        Returns:
            娣诲姞杩炴帴绾垮悗鐨凱IL Image瀵硅薄
        """
        if combined_img.mode != 'RGBA':
            combined_img = combined_img.convert('RGBA')

        draw = ImageDraw.Draw(combined_img)
        img_width, img_height = combined_img.size

        # 璁＄畻宸anel鏂规鐨勫彸杈圭紭浣嶇疆
        box_size_ratio = 0.35
        box_w = int(panel1_width * box_size_ratio)
        box_h = int(img_height * box_size_ratio)

        x0 = (panel1_width - box_w) // 2
        y0 = (img_height - box_h) // 2
        x1 = x0 + box_w
        y1 = y0 + box_h

        # 鍙硃anel鐨勫乏杈圭紭
        right_panel_x = panel1_width

        # 缁樺埗铏氱嚎杩炴帴锛堜粠鏂规鍙充笂瑙掑埌鍙硃anel宸︿笂瑙掞級
        line_color = (255, 100, 100)
        line_width = 4

        # 涓婅竟杩炴帴绾?
        draw.line([(x1, y0), (right_panel_x, y0)], fill=line_color, width=line_width)

        # 涓嬭竟杩炴帴绾?
        draw.line([(x1, y1), (right_panel_x, y1)], fill=line_color, width=line_width)

        return combined_img

    def add_protein_labels(self, img, position='top-left'):
        """
        鍦ㄥ浘鍍忎笂娣诲姞铔嬬櫧鍚嶇О鏍囨敞锛堟瘡涓簹鍩轰娇鐢ㄥ搴旂殑chain棰滆壊锛?

        Args:
            img: PIL Image瀵硅薄
            position: 鏍囨敞浣嶇疆 ('top-left', 'top-right', 'bottom-left', 'bottom-right')

        Returns:
            娣诲姞鏍囨敞鍚庣殑PIL Image瀵硅薄
        """
        # 杞崲涓篟GBA妯″紡浠ユ敮鎸佸崐閫忔槑
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        draw = ImageDraw.Draw(img)
        font = self.get_font(24)

        # 鏋勫缓鏍囨敞鏂囨湰鍒楄〃
        chains = self.complex_info['chains']
        labels = []
        colors = []
        for chain_id in sorted(chains.keys()):
            protein_name = chains[chain_id]
            color_rgb = config.CHAIN_COLORS_RGB[chain_id]
            labels.append(f"鈼?{protein_name}")
            colors.append(color_rgb)

        # 纭畾浣嶇疆
        img_width, img_height = img.size
        margin = 30
        line_spacing = 32

        # 璁＄畻璧峰浣嶇疆
        if position == 'bottom-left':
            # 浠庡簳閮ㄥ線涓婄粯鍒?
            x = margin
            y = img_height - margin - len(labels) * line_spacing
            anchor = 'lt'
        elif position == 'top-left':
            x = margin
            y = margin
            anchor = 'lt'
        elif position == 'top-right':
            x = img_width - margin
            y = margin
            anchor = 'rt'
        else:  # bottom-right
            x = img_width - margin
            y = img_height - margin - len(labels) * line_spacing
            anchor = 'rt'

        # 閫愯缁樺埗锛屾瘡琛屼娇鐢ㄥ搴旂殑棰滆壊
        for i, (label, color) in enumerate(zip(labels, colors)):
            line_y = y + i * line_spacing
            pos = (x, line_y)

            # 娣诲姞甯﹁儗鏅殑褰╄壊鏂囨湰
            self.add_text_with_background(
                draw, label, pos, font,
                text_color=color,  # 浣跨敤chain瀵瑰簲鐨勯鑹?
                bg_color=(255, 255, 255, 230),
                padding=8,
                anchor=anchor
            )

        return img

    def add_plddt_score(self, img, position='bottom-right'):
        """
        鍦ㄥ浘鍍忎笂娣诲姞骞冲潎pLDDT鍒嗘暟

        Args:
            img: PIL Image瀵硅薄
            position: 鏍囨敞浣嶇疆

        Returns:
            娣诲姞鏍囨敞鍚庣殑PIL Image瀵硅薄
        """
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        draw = ImageDraw.Draw(img)
        font = self.get_font(40)

        # 鑾峰彇pLDDT鍒嗘暟
        plddt = self.complex_info['avg_plddt']
        score_text = f"pLDDT: {plddt:.1f}"

        # 纭畾浣嶇疆
        img_width, img_height = img.size
        margin = 50

        if position == 'bottom-right':
            pos = (img_width - margin, img_height - margin)
            anchor = 'rb'
        elif position == 'bottom-left':
            pos = (margin, img_height - margin)
            anchor = 'lb'
        elif position == 'top-right':
            pos = (img_width - margin, margin)
            anchor = 'rt'
        else:  # top-left
            pos = (margin, margin)
            anchor = 'lt'

        # 娣诲姞甯﹁儗鏅殑鏂囨湰
        self.add_text_with_background(
            draw, score_text, pos, font,
            text_color=(0, 0, 0),
            bg_color=(255, 255, 255, 230),
            padding=15,
            anchor=anchor
        )

        return img

    def add_view_label(self, img, label, position='top-center'):
        """
        鍦ㄥ浘鍍忎笂娣诲姞瑙嗗浘鏍囩锛堝"Chain coloring", "180掳 view"绛夛級

        Args:
            img: PIL Image瀵硅薄
            label: 鏍囩鏂囨湰
            position: 鏍囩浣嶇疆

        Returns:
            娣诲姞鏍囩鍚庣殑PIL Image瀵硅薄
        """
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        draw = ImageDraw.Draw(img)
        font = self.get_font(28)

        img_width, img_height = img.size
        margin = 30

        if position == 'top-center':
            pos = (img_width // 2, margin)
            anchor = 'mt'
        elif position == 'bottom-center':
            pos = (img_width // 2, img_height - margin)
            anchor = 'mb'
        else:
            pos = (margin, margin)
            anchor = 'lt'

        self.add_text_with_background(
            draw, label, pos, font,
            text_color=(0, 0, 0),
            bg_color=(255, 255, 255, 200),
            padding=10,
            anchor=anchor
        )

        return img

    def combine_three_views(self, view1_path, view2_path, view3_path,
                          add_annotations=True, add_colorbar=False):
        """
        缁勫悎涓変釜瑙嗗浘涓轰竴寮犲浘鍍?

        Args:
            view1_path: 宸﹁鍥捐矾寰勶紙chain鐫€鑹诧級
            view2_path: 涓鍥捐矾寰勶紙pLDDT 180掳锛?
            view3_path: 鍙宠鍥捐矾寰勶紙pLDDT 姝ｉ潰锛?
            add_annotations: 鏄惁娣诲姞鏍囨敞
            add_colorbar: 鏄惁娣诲姞鑹叉爣锛堥粯璁alse锛屽彧鍦ㄦ€昏鍥炬坊鍔狅級

        Returns:
            output_paths: dict锛屽寘鍚?png'鍜?tiff'鐨勮緭鍑鸿矾寰?
        """
        # 鍔犺浇涓変釜瑙嗗浘
        view1 = Image.open(view1_path)
        view2 = Image.open(view2_path)
        view3 = Image.open(view3_path)

        # 璁剧疆鐩爣灏哄
        base_height = config.RENDER_PARAMS['height']
        view1_width = config.RENDER_PARAMS['width']

        # view1淇濇寔鏍囧噯灏哄
        view1 = view1.resize((view1_width, base_height), Image.Resampling.LANCZOS)

        # view2澧炲ぇ灏哄
        view2_width = int(view1_width * 1.25)  # 澧炲ぇ25%
        view2 = view2.resize((view2_width, base_height), Image.Resampling.LANCZOS)

        # view3缂╁皬灏哄锛屼繚鎸佸師濮嬬旱妯瘮
        view3_target_height = int(base_height * 0.8)  # 缂╁皬鍒?0%楂樺害
        view3_aspect = view3.width / view3.height
        view3_width = int(view3_target_height * view3_aspect)
        view3 = view3.resize((view3_width, view3_target_height), Image.Resampling.LANCZOS)

        # 娣诲姞鏍囨敞
        if add_annotations:
            # view1: Chain coloring + 铔嬬櫧璐ㄥ浘渚?
            view1 = self.add_protein_labels(view1, 'bottom-left')
            view1 = self.add_view_label(view1, 'Chain coloring', 'bottom-center')

        # 鍒涘缓缁勫悎鐢诲竷
        combined_width = view1_width + view2_width + view3_width
        combined_height = base_height

        final_img = Image.new('RGB', (combined_width, combined_height), 'white')

        # 绮樿创涓変釜瑙嗗浘
        final_img.paste(view1.convert('RGB'), (0, 0))
        final_img.paste(view2.convert('RGB'), (view1_width, 0))
        # view3鍨傜洿灞呬腑
        view3_y_offset = (base_height - view3_target_height) // 2
        final_img.paste(view3.convert('RGB'), (view1_width + view2_width, view3_y_offset))

        # 缁檝iew3娣诲姞榛戣壊杈规
        draw_temp = ImageDraw.Draw(final_img)
        view3_x = view1_width + view2_width
        view3_y = view3_y_offset
        border_width = 3
        draw_temp.rectangle(
            [(view3_x, view3_y), (view3_x + view3_width - 1, view3_y + view3_target_height - 1)],
            outline=(0, 0, 0),
            width=border_width
        )

        # 娣诲姞鏍囨敞鍒版渶缁堝浘鍍?
        if add_annotations:
            final_img = final_img.convert('RGBA')
            draw = ImageDraw.Draw(final_img)

            # Interface residues鏍囬锛氭斁鍦╲iew2鍜寁iew3涔嬮棿
            interface_x = view1_width + view2_width
            font = self.get_font(28)
            label_text = 'Interface residues'
            # 鑾峰彇鏂囨湰瀹藉害鏉ュ眳涓?
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            # 鏍囬灞呬腑鍦╲iew2鍜寁iew3浜ょ晫澶?
            label_x = interface_x - text_width // 2
            label_y = combined_height - 30
            self.add_text_with_background(
                draw, label_text, (label_x, label_y), font,
                text_color=(0, 0, 0),
                bg_color=(255, 255, 255, 200),
                padding=10,
                anchor='lb'
            )

            # pLDDT鍒嗘暟锛氱粺涓€鏀惧湪鍙充笅瑙?
            plddt = self.complex_info['avg_plddt']
            score_text = f"pLDDT: {plddt:.1f}"
            score_font = self.get_font(40)
            margin = 50
            score_pos = (combined_width - margin, combined_height - margin)
            self.add_text_with_background(
                draw, score_text, score_pos, score_font,
                text_color=(0, 0, 0),
                bg_color=(255, 255, 255, 230),
                padding=15,
                anchor='rb'
            )

            final_img = final_img.convert('RGB')

        # 濡傛灉闇€瑕佹坊鍔犺壊鏍?
        if add_colorbar:
            colorbar = self.create_plddt_colorbar(width=combined_width - 200, height=60)
            final_height = combined_height + colorbar.height + 20
            final_with_colorbar = Image.new('RGB', (combined_width, final_height), 'white')

            final_with_colorbar.paste(final_img, (0, 0))
            colorbar_x = (combined_width - colorbar.width) // 2
            colorbar_y = combined_height + 10
            final_with_colorbar.paste(colorbar, (colorbar_x, colorbar_y))

            final_img = final_with_colorbar

        # 淇濆瓨PNG
        png_path = os.path.join(self.output_dir, config.OUTPUT_FILENAMES['combined_png'])
        final_img.save(png_path, 'PNG', dpi=(config.RENDER_PARAMS['dpi'],
                                              config.RENDER_PARAMS['dpi']))

        # 淇濆瓨TIFF
        tiff_path = os.path.join(self.output_dir, config.OUTPUT_FILENAMES['combined_tiff'])
        final_img.save(tiff_path, 'TIFF', dpi=(config.RENDER_PARAMS['dpi'],
                                                config.RENDER_PARAMS['dpi']),
                      compression='lzw')

        return {'png': png_path, 'tiff': tiff_path}

    def combine_with_integrated_view(self, view1_path, integrated_view_path,
                                     add_annotations=True, add_colorbar=False):
        """
        缁勫悎view1鍜屾暣鍚堣鍥句负涓€寮犲浘鍍忥紙鐢ㄤ簬鏂扮殑甯冨眬锛?

        Args:
            view1_path: 宸﹁鍥捐矾寰勶紙chain鐫€鑹诧級
            integrated_view_path: 鍙充晶鏁村悎瑙嗗浘璺緞锛堟紨绀烘枃绋縅PG锛?
            add_annotations: 鏄惁娣诲姞鏍囨敞
            add_colorbar: 鏄惁娣诲姞鑹叉爣锛堥粯璁alse锛屽彧鍦ㄦ€昏鍥炬坊鍔狅級

        Returns:
            output_paths: dict锛屽寘鍚?png'鍜?tiff'鐨勮緭鍑鸿矾寰?
        """
        # 鍔犺浇涓や釜瑙嗗浘
        view1 = Image.open(view1_path)
        integrated_view = Image.open(integrated_view_path)

        # 璁剧疆鐩爣楂樺害
        target_height = config.RENDER_PARAMS['height']

        # 璋冩暣view1灏哄锛堜繚鎸佸楂樻瘮锛?
        view1_aspect = view1.width / view1.height
        view1_width = int(target_height * view1_aspect)
        view1_resized = view1.resize((view1_width, target_height), Image.Resampling.LANCZOS)

        # 璋冩暣鏁村悎瑙嗗浘灏哄锛堜繚鎸佸楂樻瘮锛?
        integrated_aspect = integrated_view.width / integrated_view.height
        integrated_width = int(target_height * integrated_aspect)
        integrated_resized = integrated_view.resize((integrated_width, target_height), Image.Resampling.LANCZOS)

        # 娣诲姞鏍囨敞
        if add_annotations:
            view1_resized = self.add_protein_labels(view1_resized, 'bottom-left')
            view1_resized = self.add_view_label(view1_resized, 'Chain coloring', 'bottom-center')

            integrated_resized = self.add_plddt_score(integrated_resized, 'bottom-right')
            integrated_resized = self.add_view_label(integrated_resized, 'Interface detail', 'bottom-center')

        # 鍒涘缓缁勫悎鐢诲竷锛?涓鍥炬í鍚戞帓鍒楋級
        combined_width = view1_width + integrated_width
        combined_height = target_height

        # 濡傛灉闇€瑕佹坊鍔犺壊鏍囷紝鍒欏鍔犵敾甯冮珮搴?
        if add_colorbar:
            colorbar = self.create_plddt_colorbar(width=combined_width - 200, height=60)
            final_height = combined_height + colorbar.height + 20
            final_img = Image.new('RGB', (combined_width, final_height), 'white')

            # 鍏堝垱寤轰复鏃剁粍鍚堝浘
            combined = Image.new('RGB', (combined_width, combined_height), 'white')
            combined.paste(view1_resized.convert('RGB'), (0, 0))
            combined.paste(integrated_resized.convert('RGB'), (view1_width, 0))

            # 绮樿创缁勫悎鍥惧拰鑹叉爣
            final_img.paste(combined, (0, 0))
            colorbar_x = (combined_width - colorbar.width) // 2
            colorbar_y = combined_height + 10
            final_img.paste(colorbar, (colorbar_x, colorbar_y))
        else:
            # 涓嶆坊鍔犺壊鏍囷紝鐩存帴鍒涘缓缁勫悎鍥?
            final_img = Image.new('RGB', (combined_width, combined_height), 'white')
            final_img.paste(view1_resized.convert('RGB'), (0, 0))
            final_img.paste(integrated_resized.convert('RGB'), (view1_width, 0))

        # 淇濆瓨PNG
        png_path = os.path.join(self.output_dir, config.OUTPUT_FILENAMES['combined_png'])
        final_img.save(png_path, 'PNG', dpi=(config.RENDER_PARAMS['dpi'],
                                              config.RENDER_PARAMS['dpi']))

        # 淇濆瓨TIFF
        tiff_path = os.path.join(self.output_dir, config.OUTPUT_FILENAMES['combined_tiff'])
        final_img.save(tiff_path, 'TIFF', dpi=(config.RENDER_PARAMS['dpi'],
                                                config.RENDER_PARAMS['dpi']),
                      compression='lzw')

        return {'png': png_path, 'tiff': tiff_path}

    def combine_two_views(self, overview_path, interface_path, add_annotations=True, add_box=True):
        """
        缁勫悎涓や釜瑙嗗浘涓轰竴寮犲浘鍍忥紙鏁翠綋缁撴瀯 + 鐣岄潰鏀惧ぇ锛?

        Args:
            overview_path: 鏁翠綋缁撴瀯瑙嗗浘璺緞
            interface_path: 鐣岄潰鏀惧ぇ瑙嗗浘璺緞
            add_annotations: 鏄惁娣诲姞鏍囨敞
            add_box: 鏄惁娣诲姞鏂规鍜岃繛鎺ョ嚎

        Returns:
            output_paths: dict锛屽寘鍚?png'鍜?tiff'鐨勮緭鍑鸿矾寰?
        """
        # 鍔犺浇涓や釜瑙嗗浘
        overview = Image.open(overview_path)
        interface = Image.open(interface_path)

        # 纭繚鎵€鏈夊浘鍍忓昂瀵镐竴鑷?
        width = config.RENDER_PARAMS['width']
        height = config.RENDER_PARAMS['height']

        overview = overview.resize((width, height), Image.Resampling.LANCZOS)
        interface = interface.resize((width, height), Image.Resampling.LANCZOS)

        # 娣诲姞鏂规鏍囨敞
        if add_box:
            overview = self.draw_interface_box(overview, box_position='center')

        # 娣诲姞鏍囨敞
        if add_annotations:
            overview = self.add_protein_labels(overview, 'bottom-left')
            overview = self.add_view_label(overview, 'Overall Structure', 'bottom-center')

            interface = self.add_plddt_score(interface, 'bottom-right')
            interface = self.add_view_label(interface, 'Interface Detail', 'bottom-center')

        # 鍒涘缓缁勫悎鐢诲竷锛?涓鍥炬í鍚戞帓鍒楋級
        combined_width = width * 2
        combined_height = height

        final_img = Image.new('RGB', (combined_width, combined_height), 'white')
        final_img.paste(overview.convert('RGB'), (0, 0))
        final_img.paste(interface.convert('RGB'), (width, 0))

        # 娣诲姞杩炴帴绾?
        if add_box:
            final_img = final_img.convert('RGBA')
            final_img = self.draw_connection_line(final_img, width)
            final_img = final_img.convert('RGB')

        # 淇濆瓨PNG
        png_path = os.path.join(self.output_dir, 'combined_2panel.png')
        final_img.save(png_path, 'PNG', dpi=(config.RENDER_PARAMS['dpi'],
                                              config.RENDER_PARAMS['dpi']))

        # 淇濆瓨TIFF
        tiff_path = os.path.join(self.output_dir, 'combined_2panel.tiff')
        final_img.save(tiff_path, 'TIFF', dpi=(config.RENDER_PARAMS['dpi'],
                                                config.RENDER_PARAMS['dpi']),
                      compression='lzw')

        return {'png': png_path, 'tiff': tiff_path}

    def combine_all_complexes(self, complex_ids, combined_images_paths):
        """
        灏嗗涓鍚堢墿鐨勭粍鍚堝浘绾靛悜鎺掑垪涓烘€昏鍥撅紝骞跺湪搴曢儴娣诲姞鑹叉爣

        Args:
            complex_ids: 澶嶅悎鐗㊣D鍒楄〃
            combined_images_paths: 瀵瑰簲鐨勭粍鍚堝浘璺緞鍒楄〃

        Returns:
            鎬昏鍥捐矾寰?
        """
        # 鍔犺浇鎵€鏈夌粍鍚堝浘
        images = [Image.open(path) for path in combined_images_paths]

        # 鑾峰彇灏哄
        max_width = max(img.width for img in images)
        images_height = sum(img.height for img in images) + (len(images) - 1) * 20  # 20px闂磋窛

        # 鍒涘缓鑹叉爣
        colorbar = self.create_plddt_colorbar(width=max_width - 200, height=80)

        # 璁＄畻鎬婚珮搴︼紙鍥惧儚 + 鑹叉爣 + 闂磋窛锛?
        total_height = images_height + colorbar.height + 40  # 40px闂磋窛

        # 鍒涘缓鎬昏鐢诲竷
        overview = Image.new('RGB', (max_width, total_height), 'white')

        # 渚濇绮樿创姣忎釜澶嶅悎鐗╃殑鍥惧儚
        y_offset = 0
        for i, (img, complex_id) in enumerate(zip(images, complex_ids)):
            overview.paste(img, (0, y_offset))

            # 娣诲姞澶嶅悎鐗╂爣绛?
            draw = ImageDraw.Draw(overview)
            font = self.get_font(48)
            info = config.get_complex_info(complex_id)
            label = f"({chr(65+i)}) {info['short_name']}"  # (A), (B), (C), (D)

            # 鍦ㄥ浘鍍忓乏涓婅娣诲姞鏍囩
            self.add_text_with_background(
                draw, label, (50, y_offset + 50), font,
                text_color=(0, 0, 0),
                bg_color=(255, 255, 255, 230),
                padding=20,
                anchor='lt'
            )

            y_offset += img.height + 20  # 娣诲姞闂磋窛

        # 鍦ㄥ簳閮ㄦ坊鍔犺壊鏍?
        colorbar_x = (max_width - colorbar.width) // 2
        colorbar_y = images_height + 20
        overview.paste(colorbar, (colorbar_x, colorbar_y))

        # 淇濆瓨鎬昏鍥?
        output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAMES['all_combined'])
        overview.save(output_path, 'PNG', dpi=(config.RENDER_PARAMS['dpi'],
                                               config.RENDER_PARAMS['dpi']))

        return output_path


def test_image_combiner():
    """Helper."""
    print("娴嬭瘯鍥惧儚缁勫悎鍣?..")

    # 鍒涘缓娴嬭瘯鍥惧儚锛堢函鑹茬煩褰級
    test_complex_id = 'HSP90complex_5f951'
    combiner = ImageCombiner(test_complex_id)

    # 鍒涘缓涓変釜娴嬭瘯鍥惧儚
    width = config.RENDER_PARAMS['width']
    height = config.RENDER_PARAMS['height']

    test_view1 = Image.new('RGB', (width, height), (255, 200, 200))  # 娣＄孩鑹?
    test_view2 = Image.new('RGB', (width, height), (200, 255, 200))  # 娣＄豢鑹?
    test_view3 = Image.new('RGB', (width, height), (200, 200, 255))  # 娣¤摑鑹?

    # 淇濆瓨娴嬭瘯鍥惧儚
    test_dir = combiner.output_dir
    test_view1.save(os.path.join(test_dir, 'test_view1.png'))
    test_view2.save(os.path.join(test_dir, 'test_view2.png'))
    test_view3.save(os.path.join(test_dir, 'test_view3.png'))

    # 娴嬭瘯缁勫悎
    output_paths = combiner.combine_three_views(
        os.path.join(test_dir, 'test_view1.png'),
        os.path.join(test_dir, 'test_view2.png'),
        os.path.join(test_dir, 'test_view3.png'),
        add_annotations=True
    )

    print(f"鉁?缁勫悎鍥惧凡鐢熸垚:")
    print(f"  PNG: {output_paths['png']}")
    print(f"  TIFF: {output_paths['tiff']}")

    # 娓呯悊娴嬭瘯鏂囦欢
    os.remove(os.path.join(test_dir, 'test_view1.png'))
    os.remove(os.path.join(test_dir, 'test_view2.png'))
    os.remove(os.path.join(test_dir, 'test_view3.png'))


if __name__ == "__main__":
    test_image_combiner()



