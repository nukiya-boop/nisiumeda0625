from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import unicodedata

BASE = '/home/user/nisiumeda0625/images/'
OUT  = '/home/user/nisiumeda0625/output_video.mp4'

FONT_PATH = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'

# Instagram縦型
W, H = 1080, 1920
FPS  = 30
TOTAL_SEC = 30

# 使用画像（料理系のみ）
IMAGE_FILES = [
    'イメージ_調理イメージ0066.jpg',
    'イメージ_調理イメージ0083.jpg',
    'イメージ_調理イメージ0094 (1).jpg',
    'イメージ_調理イメージ0096.jpg',
    '単体_懐石仕上（鰻穴子重）_0005.jpg',
    '単体_懐石造里（鮮魚４種）_0007.jpg',
    '単体_長物尽くし懐石（ふうりん）_0005.jpg',
]

# 各シーン：画像index / テロップ[(text, size, pos)] / 表示秒
# 計30秒になるよう設定
SCENES = [
    {
        'img': 0,
        'telops': [
            ('季節限定', 80, 'tag'),
            ('職人の技', 130, 'center_big'),
            ('鰻・鱧　夏の傑作コース', 52, 'center_sub'),
        ],
        'duration': 5,
    },
    {
        'img': 4,  # 鰻穴子重
        'telops': [
            ('鰻の焼き', 90, 'tag'),
            ('炭火でじっくり', 70, 'bottom_line1'),
            ('丁寧に焼き上げた職人の一品', 52, 'bottom_line2'),
        ],
        'duration': 5,
    },
    {
        'img': 1,
        'telops': [
            ('鱧の湯引き', 90, 'tag'),
            ('繊細な包丁さばきで仕上げる', 55, 'bottom_line1'),
            ('夏の風物詩', 70, 'bottom_line2'),
        ],
        'duration': 5,
    },
    {
        'img': 5,  # 懐石造里
        'telops': [
            ('職人の目利き', 90, 'tag'),
            ('市場で選び抜いた', 65, 'bottom_line1'),
            ('旬の鮮魚を余すことなく', 55, 'bottom_line2'),
        ],
        'duration': 5,
    },
    {
        'img': 6,  # 長物尽くし懐石
        'telops': [
            ('旬のコース料理', 90, 'tag'),
            ('職人の技と目利きで選んだ', 55, 'bottom_line1'),
            ('旬のものを使ったコース料理です', 52, 'bottom_line2'),
        ],
        'duration': 5,
    },
    {
        'img': 2,
        'telops': [
            ('季節限定', 80, 'tag'),
            ('一期一会のひと皿を', 70, 'bottom_line1'),
            ('ぜひご賞味ください', 60, 'bottom_line2'),
        ],
        'duration': 5,
    },
]

TRANSITION_FRAMES = FPS  # 1秒クロスフェード


def load_img(path):
    img = Image.open(path).convert('RGB')
    iw, ih = img.size
    target_ratio = W / H
    img_ratio = iw / ih
    # 縦型にトリミング（中央）
    if img_ratio > target_ratio:
        new_w = int(ih * target_ratio)
        x = (iw - new_w) // 2
        img = img.crop((x, 0, x + new_w, ih))
    else:
        new_h = int(iw / target_ratio)
        y = (ih - new_h) // 2
        img = img.crop((0, y, iw, y + new_h))
    img = img.resize((W, H), Image.LANCZOS)
    return img


def alpha_overlay(img, rect, color_rgba):
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle(rect, fill=color_rgba)
    return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')


def draw_telop(base_img, telops):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    for text, size, pos in telops:
        font = ImageFont.truetype(FONT_PATH, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = (W - tw) // 2

        if pos == 'tag':
            # 上部の赤帯タグ
            pad_x, pad_y = 30, 15
            y = 80
            img = alpha_overlay(img,
                [cx - pad_x, y - pad_y, cx + tw + pad_x, y + th + pad_y],
                (180, 20, 20, 220))
            draw = ImageDraw.Draw(img)
            draw.text((cx, y), text, font=font, fill=(255, 255, 255))

        elif pos == 'center_big':
            # 画面中央の大テキスト
            y = H // 2 - th - 10
            # 縦帯
            img = alpha_overlay(img,
                [0, y - 30, W, y + th + 30],
                (0, 0, 0, 170))
            draw = ImageDraw.Draw(img)
            draw.text((cx + 3, y + 3), text, font=font, fill=(0, 0, 0))
            draw.text((cx, y), text, font=font, fill=(255, 240, 180))

        elif pos == 'center_sub':
            y = H // 2 + 20
            draw = ImageDraw.Draw(img)
            draw.text((cx + 2, y + 2), text, font=font, fill=(0, 0, 0))
            draw.text((cx, y), text, font=font, fill=(255, 255, 255))

        elif pos == 'bottom_line1':
            y = H - 260
            img = alpha_overlay(img,
                [0, H - 320, W, H],
                (0, 0, 0, 160))
            draw = ImageDraw.Draw(img)
            draw.text((cx, y), text, font=font, fill=(255, 255, 255))

        elif pos == 'bottom_line2':
            y = H - 150
            draw = ImageDraw.Draw(img)
            draw.text((cx + 2, y + 2), text, font=font, fill=(0, 0, 0))
            draw.text((cx, y), text, font=font, fill=(255, 240, 180))

    return img


def img_to_frame(pil_img):
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---- メイン ----
raw_files = [f for f in os.listdir(BASE) if f.endswith('.jpg')]
nfc_map = {unicodedata.normalize('NFC', f): os.path.join(BASE, f) for f in raw_files}

def get_path(name):
    return nfc_map[unicodedata.normalize('NFC', name)]

print("画像読み込み中...")
images = [load_img(get_path(f)) for f in IMAGE_FILES]

print("動画生成中...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))

prev_composed = None

for scene in SCENES:
    composed = draw_telop(images[scene['img']], scene['telops'])
    total_frames = int(scene['duration'] * FPS)

    for f in range(total_frames):
        if f < TRANSITION_FRAMES and prev_composed is not None:
            alpha = f / TRANSITION_FRAMES
            blended = Image.blend(prev_composed, composed, alpha)
            frame = img_to_frame(blended)
        else:
            frame = img_to_frame(composed)
        writer.write(frame)

    prev_composed = composed

writer.release()
total = sum(s['duration'] for s in SCENES)
print(f"完成: {OUT}  ({total}秒 / {W}x{H})")
