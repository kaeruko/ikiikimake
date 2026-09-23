"""Describe a selected pair with image features; never infer a beauty score.

The original pixels are measured independently at their native resolution.
Only display crops are resized. Landmarks define experimental regions requiring
visual inspection; a passed face/hand check cannot approve these new regions.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analysis.analyze_cheek_lab import load_image, load_masks
from analysis.appearance_features import build_feature_masks, measure_features


APPEARANCE_VERSION = 'selected-appearance-v2-eye-texture'
PHASES = ('before', 'after')
ARTIFACTS = ('feature_summary.json', 'feature_deltas.csv', 'appearance_rois.png',
             'region_samples.png', 'feature_changes.png', 'report.html',
             'before_feature_masks.npz', 'after_feature_masks.npz')
REGION_LABELS = {
    'screen_left_brow': '画面左の眉', 'screen_right_brow': '画面右の眉',
    'screen_left_upper_lid': '画面左の上まぶた',
    'screen_right_upper_lid': '画面右の上まぶた',
    'lips': '唇', 'screen_left_lower_eye_skin': '画面左の目の下',
    'screen_right_lower_eye_skin': '画面右の目の下',
    'left_cheek': '画面左の頬', 'right_cheek': '画面右の頬', 'forehead': '額',
}
COLORS = {
    'brow': (205, 120, 255), 'upper_lid': (0, 205, 240),
    'lower_eye': (80, 220, 220), 'lips': (255, 90, 155), 'left_cheek': (50, 215, 80),
    'right_cheek': (65, 150, 255), 'forehead': (240, 190, 30),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _checked(path: Path, expected: str, label: str) -> Path:
    path = path.resolve()
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f'{label}: SHA-256 does not match the selected pair: {path}')
    return path


def _read_inputs(selected_path: Path):
    selected_path = selected_path.resolve()
    selected = json.loads(selected_path.read_text(encoding='utf-8'))
    frames, records = {}, {}
    for phase in PHASES:
        item = selected[phase]
        image_path = _checked(Path(item['image_path']), item['image_sha256'], f'{phase} image')
        roi_dir = Path(item['roi_dir'])
        paths = {
            name: _checked(roi_dir / name, item[key], f'{phase} {name}')
            for name, key in [('roi_masks.npz', 'roi_masks_sha256'),
                              ('roi_points.json', 'roi_points_sha256'),
                              ('roi_overlay.png', 'roi_overlay_sha256')]
        }
        image = load_image(image_path)
        report = json.loads(paths['roi_points.json'].read_text(encoding='utf-8'))
        if report.get('errors') != [] or report.get('status') not in ('needs_review', 'approved'):
            raise ValueError(f'{phase}: ROI extraction has errors or an invalid status')
        source = report['source']
        if source['sha256'] != item['image_sha256']:
            raise ValueError(f'{phase}: landmarks refer to a different image')
        if (source['height'], source['width']) != image.shape[:2]:
            raise ValueError(f'{phase}: image dimensions do not match the landmarks')
        landmarks = np.asarray(report['face_landmarks_normalized'], dtype=np.float64)
        if (landmarks.ndim != 2 or landmarks.shape[0] < 468 or landmarks.shape[1] != 3
                or not np.isfinite(landmarks).all()):
            raise ValueError(f'{phase}: invalid face landmarks')
        points = landmarks[:, :2] * [image.shape[1], image.shape[0]]
        base_masks = load_masks(paths['roi_masks.npz'], image.shape)
        masks = build_feature_masks(image.shape, points, base_masks)
        frames[phase] = dict(image=image, points=points, masks=masks, report=report)
        records[phase] = {
            'image_path': str(image_path), 'image_sha256': item['image_sha256'],
            'roi_masks_sha256': item['roi_masks_sha256'],
            'roi_points_sha256': item['roi_points_sha256'],
            'roi_overlay_sha256': item['roi_overlay_sha256'],
            'timestamp_seconds': item['timestamp_seconds'],
        }
    return selected, frames, records


def compare_measurements(before: list[dict], after: list[dict]) -> list[dict]:
    """Keep missing values missing; changes never imply improvement."""
    b, a = ({r['id']: r for r in rows} for rows in (before, after))
    if len(b) != len(before) or len(a) != len(after) or set(b) != set(a):
        raise ValueError('Feature IDs are duplicate or inconsistent')
    result = []
    for key, first in b.items():
        second = a[key]
        if any(first[k] != second[k] for k in ('region', 'unit', 'label')):
            raise ValueError(f'{key}: feature definitions differ')
        if any(r['value'] is not None and not np.isfinite(r['value']) for r in (first, second)):
            raise ValueError(f'{key}: non-finite metric')
        valid = all(r['status'] == 'ok' and r['value'] is not None for r in (first, second))
        result.append({
            'id': key, 'region': first['region'], 'label': first['label'],
            'unit': first['unit'], 'before': first['value'], 'after': second['value'],
            'delta': float(second['value'] - first['value']) if valid else None,
            'before_pixels': first['pixels'], 'after_pixels': second['pixels'],
            'status': 'ok' if valid else 'insufficient_pixels',
            'note': first['note'],
        })
    return result


def _quality(frames: dict) -> dict:
    dimensions, widths, pose = {}, {}, {}
    warnings = [
        '目・眉・唇のROIは新規の自動生成領域です。重ね合わせ画像で位置と遮蔽物を確認してください。',
        '視線、表情、照明、ピント、動画圧縮は自動的に補正していません。前後の工程も画像で確認してください。',
        '同じ人の2枚の記述比較です。総合的な好印象、化粧の効果、統計的な有意差は判定しません。',
        '目の下の高周波指標は細かな線・凹凸・粒状感の見え方を拾いますが、乾燥・シワの診断ではありません。ピント、照明、圧縮、メイク境界も混入します。',
    ]
    for phase, frame in frames.items():
        dimensions[phase] = list(frame['image'].shape[1::-1])
        widths[phase] = float(np.linalg.norm(frame['points'][454] - frame['points'][234]))
        pose[phase] = frame['report']['quality']['pose_degrees']
        if widths[phase] < 250:
            warnings.append(f'{phase}: 顔幅は約{widths[phase]:.0f}画素。細かなシワ・肌理の改善はこの処理の評価対象外です。')
    gaps = {axis: abs(float(pose['after'][axis]) - float(pose['before'][axis]))
            for axis in ('yaw', 'pitch', 'roll')}
    if any(v > 5 for v in gaps.values()):
        warnings.append('顔向きの差が5度を超える軸があります。部位の写り方に影響します（確認用の目安）。')
    ratio = max(widths.values()) / max(min(widths.values()), 1e-6)
    if ratio > 1.1:
        warnings.append('顔の大きさが10%以上異なります。解像感の差に注意してください（確認用の目安）。')
    return {'image_dimensions': dimensions, 'face_width_px': widths,
            'pose_gaps_degrees': gaps, 'face_scale_ratio': ratio, 'warnings': warnings}


def _color(name):
    for k, color in COLORS.items():
        if k in name:
            return color
    return (240, 190, 30) if name == 'lip_skin' else (180, 180, 180)


def _visible_masks(frame):
    rgb = cv2.cvtColor(frame['image'], cv2.COLOR_BGR2RGB)
    tinted = rgb.copy()
    for name, mask in frame['masks'].items():
        color = (255, 255, 255) if name.endswith('_skin') else _color(name)
        tinted[mask] = np.round(.65 * rgb[mask] + .35 * np.array(color)).astype(np.uint8)
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(tinted, contours, -1, color, 1)
    return rgb, tinted


def _save_visuals(output_dir: Path, frames: dict, deltas: list[dict]):
    font = {'family': 'Meiryo'}
    fig = Figure(figsize=(11, 9), facecolor='white')
    FigureCanvasAgg(fig)
    for col, phase in enumerate(PHASES):
        frame = frames[phase]
        rgb, overlay = _visible_masks(frame)
        points = frame['points']
        x0, y0 = np.maximum(np.floor(points.min(axis=0) - 10).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(points.max(axis=0) + 10).astype(int), rgb.shape[1::-1])
        for row, im in enumerate((rgb, overlay)):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1)
            ax.imshow(im[y0:y1, x0:x1], interpolation='nearest')
            ax.set_title(f'{phase}: ' + ('元画像' if row == 0 else '測定範囲（白線＝比較用の肌）'), fontdict=font)
            ax.axis('off')
    fig.suptitle('目視確認：同じルールの領域・色補正なし', fontfamily='Meiryo')
    fig.tight_layout()
    fig.savefig(output_dir/'appearance_rois.png', dpi=140)
    fig.clear()

    # Individual crops make the very small eyelid/lip regions reviewable.
    fig = Figure(figsize=(10, 17), facecolor='white')
    FigureCanvasAgg(fig)
    for row, (region, label) in enumerate(REGION_LABELS.items()):
        for col, phase in enumerate(PHASES):
            frame = frames[phase]
            rgb, _ = _visible_masks(frame)
            mask = frame['masks'][region]
            ref_name = 'lip_skin' if region == 'lips' else region + '_skin'
            ref = frame['masks'].get(ref_name, np.zeros(mask.shape, bool))
            union = mask | ref
            ax = fig.add_subplot(len(REGION_LABELS), 2, row * 2 + col + 1)
            if union.any():
                ys, xs = np.where(union)
                x0, x1 = max(0, xs.min()-2), min(rgb.shape[1], xs.max()+3)
                y0, y1 = max(0, ys.min()-2), min(rgb.shape[0], ys.max()+3)
                crop = np.full_like(rgb, 35)
                crop[mask] = rgb[mask]
                crop[ref] = rgb[ref]
                cv2.drawContours(crop, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, _color(region), 1)
                ax.imshow(crop[y0:y1, x0:x1], interpolation='nearest')
            ax.set_title(f'{phase} / {label} / {int(mask.sum())} px', fontdict=font, fontsize=10)
            ax.axis('off')
    fig.suptitle('拡大した測定画素：輪郭内が対象、周囲が比較用の肌', fontfamily='Meiryo')
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(output_dir/'region_samples.png', dpi=120)
    fig.clear()

    rows = [r for r in deltas if r['region'] != 'forehead' and not r['id'].endswith(('_L_median', '_b_median'))]
    fig = Figure(figsize=(11, 2.7 * int(np.ceil(len(rows)/3))), facecolor='white')
    FigureCanvasAgg(fig)
    for i, r in enumerate(rows):
        ax = fig.add_subplot(int(np.ceil(len(rows)/3)), 3, i+1)
        if r['status'] == 'ok':
            values = [r['before'], r['after']]
            bars = ax.bar(['before', 'after'], values, color=['#8d98a3', '#387ca8'])
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
            lower, upper = min(0., min(values)), max(0., max(values))
            span = max(upper-lower, 1.)
            ax.set_ylim(lower-.15*span if lower < 0 else 0, upper+.18*span)
            ax.axhline(0, color='#cccccc', linewidth=.7)
        else:
            ax.text(.5, .5, '画素不足：判定不可', ha='center', fontfamily='Meiryo', transform=ax.transAxes)
        ax.set_title(r['label'], fontdict=font, fontsize=10)
        ax.set_ylabel(r['unit'], fontfamily='Meiryo', fontsize=9)
    fig.suptitle('部位別の画像指標（増減は良し悪しを意味しません）', fontfamily='Meiryo')
    fig.tight_layout()
    fig.savefig(output_dir/'feature_changes.png', dpi=140)
    fig.clear()


def summary_html(summary: dict, *, include_images=False) -> str:
    """HTML fragment suitable for both a notebook and the saved report."""
    def e(value):
        return html.escape(str(value))
    def num(value):
        return '—' if value is None else f'{value:.2f}'
    warning_list = ''.join(f'<li>{e(w)}</li>' for w in summary['quality']['warnings'])
    rows = []
    for r in summary['deltas']:
        delta = '—' if r['delta'] is None else f'{r["delta"]:+.2f}'
        rows.append('<tr>' + ''.join(f'<td>{e(s)}</td>' for s in
                    [r['label'], r['unit'], num(r['before']), num(r['after']), delta,
                     '測定値あり・要目視' if r['status']=='ok' else '画素不足：判定不可']) + '</tr>')
    definitions = ''.join(f'<li><b>{e(r["label"])}</b>：{e(r["note"])} '
                          f'対象画素 {r["before_pixels"]} → {r["after_pixels"]}</li>' for r in summary['deltas'])
    pictures = ('<img src="appearance_rois.png" alt="元画像と測定領域">'
                '<img src="feature_changes.png" alt="部位別の前後比較">'
                '<details><summary>小さい領域を拡大して確認</summary><img src="region_samples.png" alt="測定画素"></details>') if include_images else ''
    def time_label(seconds):
        return f'{int(seconds//60):02d}:{seconds%60:05.2f}'
    inputs = summary['input_specification']['inputs']
    timing = ' → '.join(time_label(inputs[p]['timestamp_seconds']) for p in PHASES)
    return f'''<section class="appearance"><h2>部位別の見た目の変化</h2>
<p>比較時刻：{timing} ／ 左右は画面上の位置です。</p>
<p>目元・眉・唇のコントラスト、目の下の細かな質感コントラスト、頬の色づき・色むら・明部率を同じルールで比較します。
<b>「好印象になったか」の総合点は付けません。</b>自動値の増減を、人の印象評価の理由と照合するための試作です。</p>
<p>総合印象・いきいき感・自然さは人が評価する項目です。ノートブック末尾で任意の目視記録を保存できます。</p>
<ul>{warning_list}</ul>{pictures}
<table><thead><tr><th>指標</th><th>単位</th><th>before</th><th>after</th><th>差</th><th>状態</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>差は after − before。赤み・明暗差や高周波コントラストが増えても、それだけで好ましい・悪いとは判定しません。
明部率は光沢の候補であり、照明や皮膚色の影響を含みます。目の下の高周波指標は乾燥やシワの診断ではありません。</p>
<details><summary>測定ルールと画素数</summary><ul>{definitions}</ul></details></section>'''


def analyze_selected_appearance(selected_path: Path, output_dir: Path) -> dict:
    """Verify immutable inputs and store each input/implementation in its own run.

    Content-addressed run folders preserve previous measurements and any user
    notes. A matching run is reused only after all its saved artifacts are checked.
    """
    selected_path, output_root = Path(selected_path).resolve(), Path(output_dir).resolve()
    selected, frames, records = _read_inputs(selected_path)
    spec = {
        'version': APPEARANCE_VERSION, 'inputs': records,
        'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ('analyze_appearance.py', 'appearance_features.py')},
        'runtime': {'opencv': cv2.__version__, 'numpy': np.__version__},
    }
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode('utf-8')).hexdigest()
    output_dir = output_root / fingerprint[:16]
    manifest_path = output_dir/'artifact_manifest.json'
    if output_dir.exists():
        if not manifest_path.is_file():
            raise ValueError(f'Incomplete saved appearance run; choose a new output directory: {output_dir}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('fingerprint') != fingerprint or set(manifest.get('files', {})) != set(ARTIFACTS):
            raise ValueError('Saved appearance run has an invalid artifact manifest')
        for name, digest in manifest['files'].items():
            _checked(output_dir/name, digest, f'saved {name}')
        return json.loads((output_dir/'feature_summary.json').read_text(encoding='utf-8'))

    measured = {phase: measure_features(frame['image'], frame['masks']) for phase, frame in frames.items()}
    deltas = compare_measurements(measured['before'], measured['after'])
    summary = {
        'schema_version': 1, 'version': APPEARANCE_VERSION, 'fingerprint': fingerprint,
        'selected_pair_path': str(selected_path), 'selected_pair_sha256': sha256_file(selected_path),
        'input_specification': spec, 'output_dir': str(output_dir),
        'overall_impression': None, 'interpretation_status': 'descriptive_features_only_requires_visual_review',
        'quality': _quality(frames), 'measurements': measured, 'deltas': deltas,
        'color_conversion': 'OpenCV float32 BGR/255 to CIELAB (L* 0..100)',
        'pixel_correspondence': False, 'native_pixels_only': True,
        'artifacts': {name: str(output_dir/name) for name in ARTIFACTS},
    }
    # Reject malformed measurements before creating any output.
    document = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/'feature_summary.json').write_text(document, encoding='utf-8')
    with (output_dir/'feature_deltas.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)
    for phase in PHASES:
        np.savez_compressed(output_dir/f'{phase}_feature_masks.npz', **frames[phase]['masks'])
    _save_visuals(output_dir, frames, deltas)
    css = 'body{font-family:Meiryo,sans-serif;max-width:1150px;margin:32px auto;padding:0 24px;line-height:1.8;color:#24313a}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ccd;text-align:left}th{background:#eef3f6}details{margin:24px 0}li{margin:6px 0}'
    report = ('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
              '<title>部位別の見た目の変化</title><style>'+css+'</style><h1>選択したbefore / afterの比較</h1>'
              +summary_html(summary, include_images=True)
              +'<p><a href="feature_deltas.csv">測定値CSV</a> · <a href="feature_summary.json">設定と結果JSON</a></p></html>')
    (output_dir/'report.html').write_text(report, encoding='utf-8')
    manifest_path.write_text(json.dumps({'fingerprint': fingerprint,
                                        'files': {name: sha256_file(output_dir/name) for name in ARTIFACTS}},
                                       indent=2), encoding='utf-8')
    return summary
