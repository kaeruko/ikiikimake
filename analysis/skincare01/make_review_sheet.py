from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
font = ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc', 21)
small = ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc', 16)
items = [
    (2, '引き：塗布前候補', '顔は小さく、口元に手が近い'),
    (91, '引き：1回塗布後候補', '手の遮りは少ない／完成後の静止画ではない'),
    (14, '寄り：塗布前候補', '頬が見える瞬間／手が近く、口が開いている'),
    (64, '寄り：塗布中の参考', '手が離れた瞬間／仕上がりのafterとは分ける'),
]
canvas = Image.new('RGB', (1280, 860), '#20242b')
draw = ImageDraw.Draw(canvas)
for index, (num, title, note) in enumerate(items):
    x, y = (index % 2) * 640, (index // 2) * 430
    draw.text((x + 14, y + 6), f'{title}  frame_{num:05d}', fill='white', font=font)
    with Image.open(ROOT / 'skincare01' / f'frame_{num:05d}.png') as source:
        canvas.paste(source.convert('RGB'), (x, y + 39))
    draw.text((x + 14, y + 403), note, fill='#cbd5e1', font=small)
canvas.save(OUT / 'candidates.png')
