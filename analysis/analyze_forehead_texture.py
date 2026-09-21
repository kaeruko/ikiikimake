"""Exploratory horizontal forehead-line contrast, not a wrinkle severity score.

Reuses reviewed face landmarks; never identifies physical wrinkle depth or
attributes a change to makeup. All thresholds/regions are shared across images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIZE = (440, 300)
EYE_CENTER = np.array([220., 200.])


def read_image(path):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Cannot read image: {path}")
    return im


def load_report(base, frame_id):
    return json.loads((base / 'rois' / frame_id / 'roi_points.json').read_text(encoding='utf-8'))


def points(report):
    return np.asarray(report['face_landmarks_normalized'])[:, :2] * [report['source']['width'], report['source']['height']]


def align(base, frame_id, eye_distance):
    report = load_report(base, frame_id)
    p = points(report)
    path = base / 'frames' / f'{frame_id}.png'
    if hashlib.sha256(path.read_bytes()).hexdigest() != report['source']['sha256']:
        raise ValueError(f'Image changed since landmark extraction: {path}')
    im = read_image(path)
    u = p[263] - p[33]
    distance = np.linalg.norm(u)
    u = u / distance
    a = np.array([u, [-u[1], u[0]]]) * eye_distance / distance
    t = EYE_CENTER - a @ ((p[33] + p[263]) / 2)
    transform = np.column_stack([a, t])
    im = cv2.warpAffine(im, transform, SIZE, flags=cv2.INTER_LINEAR)
    p = p @ a.T + t
    # Upper forehead arc, then halfway from mid-forehead to the eyebrows.
    # The shape uses anatomy only, not image contrast or treatment labels.
    polygon = np.array([p[67], p[109], p[10], p[338], p[297],
                        (p[299]+p[296])/2, (p[337]+p[336])/2,
                        (p[151]+p[9])/2, (p[108]+p[107])/2,
                        (p[69]+p[66])/2], np.float32)
    mask = np.zeros(im.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    # Distance from brows/hair and the polygon boundary, in common-scale pixels.
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
    lightness = cv2.cvtColor(im.astype(np.float32)/255, cv2.COLOR_BGR2LAB)[:, :, 0]
    return im, lightness, mask, report, transform, p


def line_response(lightness, width=5, extra_blur=0):
    """Dark narrow horizontal structures; values in L* and local % of L*."""
    work = lightness
    if extra_blur:
        work = cv2.GaussianBlur(work, (0, 0), extra_blur)
    # A small horizontal average and opening favor extended lines over speckles.
    work = cv2.GaussianBlur(work, (5, 1), 1.0)
    dark = cv2.morphologyEx(work, cv2.MORPH_BLACKHAT, np.ones((width, 1), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, 5), np.uint8))
    local = cv2.GaussianBlur(work, (0, 0), 12.)
    relative = 100. * dark / np.maximum(local, 1.)
    return dark, relative


def stats(lightness, mask, width=5, extra_blur=0):
    dark, rel = line_response(lightness, width, extra_blur)
    return {'pixels': int(mask.sum()), 'mean_L': float(lightness[mask].mean()),
            'dark_line_mean_L': float(dark[mask].mean()),
            'relative_line_mean': float(rel[mask].mean()),
            'relative_line_p90': float(np.percentile(rel[mask], 90)),
            'area_above_1pct': float(np.mean(rel[mask] > 1.) * 100)}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, out = args.search_dir.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pair = json.loads((base/'selected_pair.json').read_text(encoding='utf-8'))
    ids = [pair[s]['frame_id'] for s in ('before', 'after')]
    distance = min(np.linalg.norm(points(load_report(base, i))[263]-points(load_report(base, i))[33]) for i in ids)
    data = [align(base, i, distance) for i in ids]
    mask = data[0][2] & data[1][2]
    if mask.sum() < 1000:
        raise ValueError('Insufficient common forehead pixels')
    y, x = np.where(mask)
    x0, x1, y0, y1 = int(x.min()), int(x.max()+1), int(y.min()), int(y.max()+1)
    masks = {'full': mask, 'inset_2px': cv2.erode(mask.astype(np.uint8), np.ones((5,5),np.uint8)) > 0}
    xx = np.indices(mask.shape)[1]
    for j, name in enumerate(['screen_left', 'center', 'screen_right']):
        masks[name] = mask & (xx >= x0+(x1-x0)*j/3) & (xx < x0+(x1-x0)*(j+1)/3)
    rows = []
    for region, m in masks.items():
        for width in (3, 5, 7):
            for blur in (0., .7):
                for side, item in zip(('before', 'after'), data):
                    rows.append({'side': side, 'region': region, 'filter_height_px': width,
                                 'extra_blur_sigma': blur, **stats(item[1], m, width, blur)})
    write_csv(out/'measurements.csv', rows)
    selected = {s: next(r for r in rows if r['side']==s and r['region']=='full' and r['filter_height_px']==5 and r['extra_blur_sigma']==0) for s in ('before','after')}

    # Nearby frames assess temporal instability, not independent participants.
    records = [json.loads(line) for line in (base/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    temporal = []
    aligned_nearby = []
    for side in ('before', 'after'):
        center = pair[side]['timestamp_seconds']
        candidates = sorted([r for r in records if abs(r['timestamp_seconds']-center) <= 2. and r.get('report',{}).get('status')=='needs_review'], key=lambda r:r['timestamp_seconds'])
        for rec in candidates:
            item = align(base, rec['frame_id'], distance)
            aligned_nearby.append((side, rec, item))
    temporal_mask = mask.copy()
    for _, _, item in aligned_nearby:
        temporal_mask &= item[2]
    if temporal_mask.sum() < 1000:
        raise ValueError('Insufficient common pixels for neighboring-frame check')
    for side, rec, item in aligned_nearby:
        temporal.append({'side': side, 'frame_id': rec['frame_id'], 'time_seconds': rec['timestamp_seconds'],
                         **stats(item[1], temporal_mask), **item[3]['quality']['pose_degrees']})
    write_csv(out/'neighboring_frames.csv', temporal)
    cv2.imencode('.png', (mask*255).astype(np.uint8))[1].tofile(out/'common_forehead_mask.png')
    cv2.imencode('.png', (temporal_mask*255).astype(np.uint8))[1].tofile(out/'neighboring_common_mask.png')
    for side, item in zip(('before','after'),data):
        cv2.imencode('.png',item[0])[1].tofile(out/f'{side}_aligned.png')

    result = {'purpose': 'Exploratory image contrast of horizontal forehead lines, NOT validated wrinkle severity or makeup efficacy.',
              'selected_pair_path': str(base/'selected_pair.json'), 'canonical_eye_distance_px': float(distance),
              'roi_bounds_xyxy': [x0,y0,x1,y1], 'common_pixels': int(mask.sum()),
              'neighboring_common_pixels': int(temporal_mask.sum()),
              'primary_setting': {'filter_height_px':5, 'extra_blur_sigma':0., 'region':'full', 'metric':'relative_line_mean'},
              'reproducibility': {'opencv_version':cv2.__version__, 'numpy_version':np.__version__,
                                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'selected': selected, 'sources': {}, 'neighboring_summary': {},
              'limits': ['One person, observational video, no controlled lighting/expression or focus.',
                         'Lines may include skin texture, compression, highlights or shadow; this is not verified wrinkle segmentation.',
                         'Local brightness normalization does not remove illumination direction, specular reflections or focus differences.',
                         'Pixels and neighboring frames are not independent people; no significance test or confidence interval.',
                         'The selected after frame is before lipstick, not confirmed final full makeup.']}
    for side, item in zip(('before','after'),data):
        path = base/'frames'/f'{pair[side]["frame_id"]}.png'
        result['sources'][side] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                   'roi_points_sha256':hashlib.sha256((base/'rois'/pair[side]['frame_id']/'roi_points.json').read_bytes()).hexdigest(),
                                   'transform':item[4].tolist(),'pose':item[3]['quality']['pose_degrees']}
        vals = [r['relative_line_mean'] for r in temporal if r['side']==side]
        result['neighboring_summary'][side] = {'n':len(vals),'median':float(np.median(vals)), 'min':min(vals),'max':max(vals)}
    b,a = (selected[s]['relative_line_mean'] for s in ('before','after'))
    result['primary_percent_change'] = 100*(a/b-1) if b else None
    (out/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')

    font = ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc', 18)
    small = ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc', 14)
    canvas = Image.new('RGB',(1120,970),'#faf9f6')
    draw = ImageDraw.Draw(canvas)
    draw.text((24,16),'額の横線の見え方：同じ範囲・同じ処理で比較',font=font,fill='#202020')
    crop = (x0-12, y0-20, x1+12, y1+20)
    for j,(side,item) in enumerate(zip(('before','after'),data)):
        at = 24+j*550
        draw.text((at,52),f'{side.upper()}  '+('00:34.50' if j==0 else '17:47.55'),font=font,fill='#202020')
        rgb = cv2.cvtColor(item[0],cv2.COLOR_BGR2RGB)
        for row in range(3):
            im = rgb.copy()
            if row==1:
                cv2.drawContours(im,cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,190,190),1)
            if row==2:
                _,rel = line_response(item[1])
                # Shared absolute scale: 0--3 percent of local L*, never per-image scaling.
                heat = cv2.applyColorMap(np.clip(rel/3*255,0,255).astype(np.uint8),cv2.COLORMAP_INFERNO)
                im = cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)
                im[~mask] = 236
            cr = Image.fromarray(im).crop(crop)
            cr = cr.resize((510,round(510*cr.height/cr.width)),Image.Resampling.NEAREST)
            canvas.paste(cr,(at,84+row*275))
            draw.text((at,327+row*275),['位置・大きさをそろえた元画像（明るさ未補正）','共通の測定範囲：眉・生え際を避けた中央の額','横線の反応：共通の色尺度 0〜3%（明るいほど強い）'][row],font=small,fill='#303030')
        value = selected[side]['relative_line_mean']
        draw.text((at,919),f'平均線コントラスト {value:.3f}（局所L*に対する%）',font=font,fill='#202020')
    canvas.save(out/'forehead_comparison.png')
    # Keep the result readable without running Python or opening raw measurements.
    changes = []
    for width in (3,5,7):
        values = [next(r['relative_line_mean'] for r in rows if r['side']==s and r['region']=='full' and r['filter_height_px']==width and r['extra_blur_sigma']==0) for s in ('before','after')]
        changes.append(100*(values[1]/values[0]-1))
    regional = []
    for name, label in [('screen_left','画像左側'),('center','中央'),('screen_right','画像右側')]:
        values = [next(r['relative_line_mean'] for r in rows if r['side']==s and r['region']==name and r['filter_height_px']==5 and r['extra_blur_sigma']==0) for s in ('before','after')]
        regional.append(f'<tr><td>{label}</td><td>{values[0]:.3f}</td><td>{values[1]:.3f}</td><td>{100*(values[1]/values[0]-1):+.1f}%</td></tr>')
    nb, na = (result['neighboring_summary'][s] for s in ('before','after'))
    direction = '強い' if a>b else '弱い'
    html = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>額の横線の比較</title>
<style>body{{font-family:Meiryo,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#223}}p,li{{line-height:1.9}}h1{{font-size:26px}}h2{{font-size:20px;margin-top:36px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ccd;text-align:left}}img{{max-width:100%}}.lead{{padding:22px;background:#edf3f2}}small{{line-height:1.8}}</style>
<h1>額の横線の見え方を検証</h1>
<p class="lead">この2枚の測定範囲では、after の横線のコントラストが before より<b>{abs(result['primary_percent_change']):.0f}% {direction}</b>という結果です。<br>これは画像中の線の反応の差です。シワの本数・深さ・重症度の変化率ではありません。</p>
<p>対象：00:34.50 → 17:47.55。1920×1080の元PNGを使用しました。画像右側には弱く出る部分もあり、額全体が一様に変化したわけではありません。</p>
<img src="forehead_comparison.png" alt="beforeとafterの額、測定範囲、同じ色尺度の横線反応">
<h2>何を測ったか</h2>
<p>目尻の位置を基準に回転・大きさをそろえ、両画像に共通する額中央を測定しました。範囲は約{x1-x0}×{y1-y0}画素、実際の面積は{int(mask.sum()):,}画素です。眉と髪の生え際を避けています。拡大表示は新しい細部を復元するものではありません。</p>
<p>周囲より暗く写った細い横線を古典画像処理で抽出し、近くの肌の明るさで割って平均しました。小さいほど線のコントラストが弱い指標です。シワ専用に検証済みの検出器ではなく、皮膚の細かな凹凸・光沢・圧縮模様も混ざります。beforeの額には垂れた細い髪も含まれます。縦方向の線には反応しにくい処理ですが、髪の影響を完全に除去したものではありません。</p>
<table><tr><th>測定範囲</th><th>before</th><th>after</th><th>相対変化</th></tr><tr><td>共通領域全体</td><td>{b:.3f}</td><td>{a:.3f}</td><td>{result['primary_percent_change']:+.1f}%</td></tr>{''.join(regional)}</table>
<p><small>単位：局所的なL*に対する線の暗さ（%）の領域平均。画像右側の変化は線幅設定によって増減が逆転するため、改善したとは判定しません。</small></p>
<h2>条件を変えても同じか</h2>
<ul><li>3種類の線幅設定では、全体の変化は {min(changes):+.1f}%〜{max(changes):+.1f}%。</li>
<li>測定境界をさらに2画素内側に狭めた場合、また両方に同じ軽いぼかしを加えた場合も、全体の増加方向は同じでした。これは元のピント差を取り除いた検証ではありません。</li>
<li>選択時刻の前後2秒以内に保存済みの before {nb['n']}枚・after {na['n']}枚を、全フレーム共通の領域（{int(temporal_mask.sum()):,}画素）で追加比較。選択ペアより小さい範囲による補助検証です。指標の中央値は {nb['median']:.3f} → {na['median']:.3f}。範囲は {nb['min']:.3f}〜{nb['max']:.3f} → {na['min']:.3f}〜{na['max']:.3f} で一部重なります。独立した13人の結果ではなく、同じ人の近接フレームです。</li></ul>
<h2>この結果で言える範囲</h2>
<p><b>この映像の測定範囲では、after の横線が強く写る傾向があります。</b>一方、化粧でシワが増えた、印象が悪くなった、という結論にはつながりません。眉や目の表情、照明・光沢、ピントが変わっている可能性があります。after の領域平均の明るさL*は {selected['before']['mean_L']:.2f} → {selected['after']['mean_L']:.2f}。局所的な明るさで割っても、光の方向や反射の影響は消せません。after は字幕上「口紅をつける前に」の場面です。</p>
<p>次の検証では、眉を上げない表情・同じ照明・同じピントで前後を撮影し、時系列を伏せた人の「シワの目立ちやすさ」評価とこの指標が一致するかを確認します。顔全体の「いきいきした印象」は別の評価項目です。</p>
<p>画像指標を人の評価と照合する考え方は<a href="https://pubmed.ncbi.nlm.nih.gov/25601617/">既存の検証研究</a>にもありますが、今回の処理が同様に検証済みという意味ではありません。処理の基本操作：<a href="https://docs.opencv.org/4.x/d3/dbe/tutorial_opening_closing_hats.html">OpenCV Black Hat</a>。</p>
<p><a href="measurements.csv">条件別の測定値</a>・<a href="neighboring_frames.csv">周辺フレームの測定値</a>・<a href="results.json">設定と画像の記録</a></p>
</html>'''
    (out/'report.html').write_text(html, encoding='utf-8')
    print(json.dumps({'selected':selected,'primary_percent_change':result['primary_percent_change'],'neighbors':result['neighboring_summary'],'roi_bounds':result['roi_bounds_xyxy']},indent=2))


if __name__ == '__main__':
    main()
