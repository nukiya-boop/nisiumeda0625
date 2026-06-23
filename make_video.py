from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import os
import unicodedata
import math

BASE      = '/home/user/nisiumeda0625/images/'
OUT_RAW   = '/home/user/nisiumeda0625/output_raw.mp4'
OUT_FINAL = '/home/user/nisiumeda0625/output_video_h264.mp4'
FFMPEG    = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'

FONT_PATH      = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'
FONT_PATH_BOLD = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'

W, H   = 1080, 1920
FPS    = 30
TRANS  = FPS          # 1秒クロスフェード
TELOP_FADE = int(FPS * 0.6)  # テロップフェードイン

IMAGE_FILES = [
    'イメージ_調理イメージ0066.jpg',
    'イメージ_調理イメージ0083.jpg',
    'イメージ_調理イメージ0094 (1).jpg',
    'イメージ_調理イメージ0096.jpg',
    '単体_懐石仕上（鰻穴子重）_0005.jpg',
    '単体_懐石造里（鮮魚４種）_0007.jpg',
    '単体_長物尽くし懐石（ふうりん）_0005.jpg',
]

# duration=シーン秒数, zoom=Ken Burnsズーム量(0.0〜0.15)
SCENES = [
    {
        'img': 0,
        'telops': [
            {'text': '季節限定',                     'style': 'badge',    'y_ratio': 0.04},
            {'text': '職　人　の　技',                'style': 'main',     'y_ratio': 0.12},
            {'text': 'ふうりん',                      'style': 'accent',   'y_ratio': 0.72},
            {'text': '── 2026 Summer ──',           'style': 'caption',  'y_ratio': 0.79},
        ],
        'duration': 5, 'zoom': 0.06,
    },
    {
        'img': 4,
        'telops': [
            {'text': '鰻　の　焼　き',                'style': 'main',     'y_ratio': 0.10},
            {'text': 'Grilled Eel',                  'style': 'caption',  'y_ratio': 0.18},
            {'text': 'じっくり、丁寧に焼き上げた',      'style': 'sub',      'y_ratio': 0.74},
            {'text': '職人の一品',                    'style': 'accent',   'y_ratio': 0.80},
        ],
        'duration': 5, 'zoom': 0.05,
    },
    {
        'img': 1,
        'telops': [
            {'text': '鱧　の　湯　引　き',             'style': 'main',     'y_ratio': 0.10},
            {'text': 'Blanched Daggertooth Pike',    'style': 'caption',  'y_ratio': 0.18},
            {'text': '繊細な包丁さばきで仕上げる',      'style': 'sub',      'y_ratio': 0.74},
            {'text': '夏の風物詩',                    'style': 'accent',   'y_ratio': 0.80},
        ],
        'duration': 5, 'zoom': 0.05,
    },
    {
        'img': 5,
        'telops': [
            {'text': '職人の目利き',                  'style': 'main',     'y_ratio': 0.10},
            {'text': "Chef's Selection",              'style': 'caption',  'y_ratio': 0.18},
            {'text': '市場で選び抜いた旬の鮮魚',        'style': 'sub',      'y_ratio': 0.74},
            {'text': '四種盛り合わせ',                 'style': 'accent',   'y_ratio': 0.80},
        ],
        'duration': 5, 'zoom': 0.05,
    },
    {
        'img': 6,
        'telops': [
            {'text': '旬のコース料理',                 'style': 'main',     'y_ratio': 0.10},
            {'text': 'Seasonal Course',               'style': 'caption',  'y_ratio': 0.18},
            {'text': '職人の技と目利きで選んだ',        'style': 'sub',      'y_ratio': 0.72},
            {'text': '"長物"にこだわったコース料理',     'style': 'sub',      'y_ratio': 0.78},
        ],
        'duration': 5, 'zoom': 0.05,
    },
    {
        'img': 2,
        'telops': [
            {'text': '季節限定',                     'style': 'badge',    'y_ratio': 0.04},
            {'text': '一期一会のひと皿を',             'style': 'main',     'y_ratio': 0.12},
            {'text': 'ぜひご賞味ください',             'style': 'sub',      'y_ratio': 0.72},
            {'text': '── ご予約お待ちしております ──', 'style': 'caption',  'y_ratio': 0.79},
        ],
        'duration': 5, 'zoom': 0.06,
    },
]


# ---- 画像読み込み（切り抜きなし） ----
def load_img_fullfit(path):
    """元画像を切り取らず縦型フレームにフィット。上下/左右はぼかし背景で埋める"""
    src = Image.open(path).convert('RGB')
    iw, ih = src.size

    # ぼかし背景：縦型フレームにFill
    bg_ratio = W / H
    src_ratio = iw / ih
    if src_ratio > bg_ratio:
        bg_h = H
        bg_w = int(H * src_ratio)
    else:
        bg_w = W
        bg_h = int(W / src_ratio)
    bg = src.resize((bg_w, bg_h), Image.LANCZOS)
    # 中央クロップ→ぼかし
    bx = (bg_w - W) // 2
    by = (bg_h - H) // 2
    bg = bg.crop((bx, by, bx + W, by + H))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
    # 暗め
    dark = Image.new('RGB', (W, H), (0, 0, 0))
    bg = Image.blend(bg, dark, 0.45)

    # メイン画像：縦横比を保ったままフィット
    scale = min(W / iw, H / ih)
    fw = int(iw * scale)
    fh = int(ih * scale)
    front = src.resize((fw, fh), Image.LANCZOS)
    fx = (W - fw) // 2
    fy = (H - fh) // 2
    bg.paste(front, (fx, fy))
    return bg, (fx, fy, fw, fh)  # 背景合成済み画像、前景位置


def ken_burns(base_img, front_rect, progress, zoom=0.08):
    """Ken Burnsエフェクト：ゆっくりズームイン"""
    scale = 1.0 + zoom * progress
    nw = int(W * scale)
    nh = int(H * scale)
    img = base_img.resize((nw, nh), Image.LANCZOS)
    x = (nw - W) // 2
    y = (nh - H) // 2
    return img.crop((x, y, x + W, y + H))


# ---- テロップ描画 ----
def draw_telop_frame(base_img, telops, telop_alpha):
    img = base_img.copy().convert('RGBA')

    for t in telops:
        text  = t['text']
        style = t['style']
        yr    = t['y_ratio']

        if style == 'main':
            size = 72
            font = ImageFont.truetype(FONT_PATH_BOLD, size)
            color = (255, 248, 220)   # クリーム
            shadow = (60, 30, 0)
            draw_text_with_line(img, text, font, yr, color, shadow, telop_alpha,
                                line_color=(200, 160, 80), line_thickness=2)

        elif style == 'sub':
            size = 46
            font = ImageFont.truetype(FONT_PATH, size)
            color = (240, 240, 240)
            draw_text_simple(img, text, font, yr, color, (0, 0, 0), telop_alpha)

        elif style == 'caption':
            size = 28
            font = ImageFont.truetype(FONT_PATH, size)
            color = (180, 160, 100)
            draw_text_simple(img, text, font, yr, color, (0, 0, 0), telop_alpha)

        elif style == 'badge':
            size = 52
            font = ImageFont.truetype(FONT_PATH_BOLD, size)
            draw_badge(img, text, font, yr, telop_alpha)

        elif style == 'accent':
            size = 68
            font = ImageFont.truetype(FONT_PATH_BOLD, size)
            color = (255, 220, 80)
            draw_text_with_line(img, text, font, yr, color, (60, 30, 0), telop_alpha,
                                line_color=(200, 160, 80), line_thickness=2)

    return img.convert('RGB')


def draw_text_with_line(img, text, font, yr, color, shadow_color, alpha, line_color, line_thickness):
    tmp = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    x = (W - tw) // 2
    y = int(H * yr)

    # 横ライン
    lpad = 40
    ly = y + th // 2
    d.line([(x - lpad - 80, ly), (x - lpad, ly)], fill=(*line_color, int(200 * alpha)), width=line_thickness)
    d.line([(x + tw + lpad, ly), (x + tw + lpad + 80, ly)], fill=(*line_color, int(200 * alpha)), width=line_thickness)

    # 影
    d.text((x + 3, y + 3), text, font=font, fill=(*shadow_color, int(180 * alpha)))
    # 本文
    d.text((x, y), text, font=font, fill=(*color, int(255 * alpha)))
    img.alpha_composite(tmp)


def draw_text_simple(img, text, font, yr, color, shadow_color, alpha):
    tmp = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    x = (W - tw) // 2
    y = int(H * yr)
    d.text((x + 2, y + 2), text, font=font, fill=(*shadow_color, int(160 * alpha)))
    d.text((x, y), text, font=font, fill=(*color, int(255 * alpha)))
    img.alpha_composite(tmp)


def draw_badge(img, text, font, yr, alpha):
    tmp = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    x = (W - tw) // 2
    y = int(H * yr)
    pad = 18
    # 枠背景（ダークレッド）
    d.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad],
                fill=(140, 10, 10, int(220 * alpha)))
    # 細枠
    d.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad],
                outline=(255, 200, 100, int(200 * alpha)), width=2)
    d.text((x, y), text, font=font, fill=(255, 255, 255, int(255 * alpha)))
    img.alpha_composite(tmp)


def img_to_frame(pil_img):
    arr = np.array(pil_img.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---- メイン ----
raw_files = [f for f in os.listdir(BASE) if f.endswith('.jpg')]
nfc_map   = {unicodedata.normalize('NFC', f): os.path.join(BASE, f) for f in raw_files}

def get_path(name):
    return nfc_map[unicodedata.normalize('NFC', name)]

print("画像読み込み中...")
loaded = [load_img_fullfit(get_path(f)) for f in IMAGE_FILES]

print("動画生成中...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_RAW, fourcc, FPS, (W, H))

prev_frames = None

for si, scene in enumerate(SCENES):
    base_img, front_rect = loaded[scene['img']]
    total_frames = int(scene['duration'] * FPS)
    scene_frames = []

    for f in range(total_frames):
        progress = f / total_frames
        kb_img = ken_burns(base_img, front_rect, progress, scene['zoom'])

        # テロップのフェードイン（開始0.5秒後から）
        fade_start = int(FPS * 0.5)
        if f < fade_start:
            talpha = 0.0
        elif f < fade_start + TELOP_FADE:
            talpha = (f - fade_start) / TELOP_FADE
        else:
            talpha = 1.0

        composed = draw_telop_frame(kb_img, scene['telops'], talpha)
        scene_frames.append(composed)

    # クロスフェード書き出し
    for f, frm in enumerate(scene_frames):
        if f < TRANS and prev_frames is not None and f < len(prev_frames):
            alpha = f / TRANS
            blended = Image.blend(prev_frames[-(TRANS - f)], frm, alpha)
            writer.write(img_to_frame(blended))
        else:
            writer.write(img_to_frame(frm))

    prev_frames = scene_frames

writer.release()
print("H.264に変換中...")
os.system(f'{FFMPEG} -i {OUT_RAW} -vcodec libx264 -pix_fmt yuv420p -preset fast -crf 22 {OUT_FINAL} -y 2>/dev/null')
os.remove(OUT_RAW)
print(f"完成: {OUT_FINAL}  (30秒 / {W}x{H})")
