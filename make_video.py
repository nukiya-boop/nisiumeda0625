from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import os

BASE = '/home/user/nisiumeda0625/images/'
OUT  = '/home/user/nisiumeda0625/output_video.mp4'

FONT_PATH = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf'

W, H = 1920, 1080
FPS  = 30

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

# 各シーンの設定 [画像index, テロップ行リスト, 表示時間(秒)]
SCENES = [
    {
        'img': 0,
        'telops': [
            ('季節限定', 90, 'top'),
            ('職人の技', 120, 'bottom_big'),
        ],
        'duration': 4,
    },
    {
        'img': 4,  # 鰻穴子重
        'telops': [
            ('鰻の焼き', 80, 'top'),
            ('炭火でじっくり、丁寧に焼き上げた', 50, 'bottom'),
            ('職人の一品', 70, 'bottom2'),
        ],
        'duration': 4,
    },
    {
        'img': 1,
        'telops': [
            ('鱧の湯引き', 80, 'top'),
            ('繊細な包丁さばきで仕上げる', 50, 'bottom'),
            ('夏の風物詩', 70, 'bottom2'),
        ],
        'duration': 4,
    },
    {
        'img': 5,  # 懐石造里
        'telops': [
            ('職人の目利き', 80, 'top'),
            ('旬の素材を厳選', 60, 'bottom'),
        ],
        'duration': 4,
    },
    {
        'img': 6,  # 長物尽くし懐石
        'telops': [
            ('旬のコース料理', 80, 'top'),
            ('職人の技、職人の目利きで選んだ', 48, 'bottom'),
            ('旬のものを使ったコース料理です', 48, 'bottom2'),
        ],
        'duration': 5,
    },
    {
        'img': 2,
        'telops': [
            ('季節限定', 90, 'top'),
            ('一期一会のひと皿を', 60, 'bottom'),
        ],
        'duration': 4,
    },
]

TRANSITION_FRAMES = FPS  # 1秒クロスフェード


def load_img(path):
    img = Image.open(path).convert('RGB')
    # クロップ中央 16:9
    iw, ih = img.size
    target_ratio = W / H
    img_ratio = iw / ih
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


def draw_telop(base_img, telops):
    """テロップ描画"""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    for text, size, pos in telops:
        font = ImageFont.truetype(FONT_PATH, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if pos == 'top':
            x = (W - tw) // 2
            y = 60

            # 帯背景
            pad = 20
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rectangle(
                [x - pad, y - pad, x + tw + pad, y + th + pad],
                fill=(180, 20, 20, 200)
            )
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)
            draw.text((x, y), text, font=font, fill=(255, 255, 255))

        elif pos == 'bottom_big':
            size2 = size
            font2 = ImageFont.truetype(FONT_PATH, size2)
            bbox2 = draw.textbbox((0, 0), text, font=font2)
            tw2 = bbox2[2] - bbox2[0]
            th2 = bbox2[3] - bbox2[1]
            x = (W - tw2) // 2
            y = H - th2 - 80

            # 縦書き風の装飾ライン
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rectangle([0, y - 30, W, H], fill=(0, 0, 0, 160))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)
            # 影
            draw.text((x + 3, y + 3), text, font=font2, fill=(0, 0, 0, 180))
            draw.text((x, y), text, font=font2, fill=(255, 240, 180))

        elif pos == 'bottom':
            x = (W - tw) // 2
            y = H - th - 130
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rectangle([0, H - 200, W, H], fill=(0, 0, 0, 150))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)
            draw.text((x, y), text, font=font, fill=(255, 255, 255))

        elif pos == 'bottom2':
            x = (W - tw) // 2
            y = H - th - 50
            draw.text((x, y), text, font=font, fill=(255, 240, 180))

    return img


def img_to_frames(pil_img):
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---- メイン ----
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))

import os, unicodedata
raw_files = [f for f in os.listdir(BASE) if f.endswith('.jpg')]
# NFCに正規化してマッチング
nfc_map = {unicodedata.normalize('NFC', f): os.path.join(BASE, f) for f in raw_files}
def get_path(name):
    key = unicodedata.normalize('NFC', name)
    return nfc_map[key]
images = [load_img(get_path(IMAGE_FILES[i])) for i in range(len(IMAGE_FILES))]

prev_frame_img = None

for scene_idx, scene in enumerate(SCENES):
    img_base = images[scene['img']]
    composed = draw_telop(img_base, scene['telops'])

    total_frames = int(scene['duration'] * FPS)

    for f in range(total_frames):
        # フェードイン
        if f < TRANSITION_FRAMES and prev_frame_img is not None:
            alpha = f / TRANSITION_FRAMES
            blended = Image.blend(prev_frame_img, composed, alpha)
            frame = img_to_frames(blended)
        else:
            frame = img_to_frames(composed)
        writer.write(frame)

    prev_frame_img = composed

writer.release()
print(f"Done: {OUT}")
