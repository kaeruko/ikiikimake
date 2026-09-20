"""Static Japanese reports for geometry-ranked video frame candidates.

This module only presents candidate frames. It never approves an ROI, invokes
Lab analysis, infers identity, or converts matching distances into beauty scores.
"""

from __future__ import annotations

from collections import Counter
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _number(value, digits: int = 2) -> str:
    try:
        value = float(value)
        return f"{value:.{digits}f}" if math.isfinite(value) else "—"
    except (TypeError, ValueError):
        return "—"


def _time(value) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(seconds) or seconds < 0:
        return "—"
    centiseconds = round(seconds * 100)
    minutes, remainder = divmod(centiseconds, 6000)
    hours, minutes = divmod(minutes, 60)
    body = f"{minutes:02d}:{remainder / 100:05.2f}"
    return f"{hours:02d}:{body}" if hours else body


def _uri(path: Path, output: Path) -> str:
    path = path.resolve()
    try:
        relative = os.path.relpath(path, output.resolve()).replace("\\", "/")
        return quote(relative, safe="/.")
    except ValueError:  # Different Windows drives cannot have a relative path.
        return path.as_uri()


def _link(path: Path, output: Path, label: str) -> str:
    if not path.is_file():
        return ""
    return f'<a href="{_escape(_uri(path, output))}">{_escape(label)}</a>'


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError(f"Cannot decode candidate frame: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode report image: {path}")
    encoded.tofile(path)


def _face_crop(image: np.ndarray, report: dict) -> np.ndarray:
    height, width = image.shape[:2]
    points = []
    landmarks = report.get("face_landmarks_normalized", [])
    try:
        landmarks = np.asarray(landmarks, dtype=float)
        if landmarks.ndim == 2 and landmarks.shape[1] >= 2:
            xy = landmarks[:, :2]
            xy = xy[np.isfinite(xy).all(axis=1)]
            points = (xy * [width, height]).tolist()
    except (ValueError, TypeError):
        pass
    if not points:
        for roi in report.get("rois", {}).values():
            for point in roi.get("polygon_px", []):
                if len(point) >= 2 and all(math.isfinite(float(value)) for value in point[:2]):
                    points.append(point[:2])
    if not points:
        return image
    xy = np.asarray(points, dtype=float)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    padding = max(x1 - x0, y1 - y0) * 0.16
    x0, y0 = max(0, int(math.floor(x0 - padding))), max(0, int(math.floor(y0 - padding)))
    x1, y1 = min(width, int(math.ceil(x1 + padding))), min(height, int(math.ceil(y1 + padding)))
    return image[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else image


def _pair_image(records: tuple[dict, dict], *, crop: bool, overlay: bool = False) -> np.ndarray:
    panels = []
    for phase, record in zip(("BEFORE candidate", "AFTER candidate"), records):
        image_path = Path(record["roi_dir"]) / "roi_overlay.png" if overlay else Path(record["image_path"])
        image = _read_image(image_path)
        if crop:
            image = _face_crop(image, record.get("report", {}))
            scale = min(600 / image.shape[1], 520 / image.shape[0])
        else:
            scale = min(1.0, 960 / image.shape[1], 540 / image.shape[0])
        if abs(scale - 1.0) > 1e-6:
            size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        panel_width = max(image.shape[1], 380)
        panel = np.full((image.shape[0] + 64, panel_width, 3), 28, dtype=np.uint8)
        offset = (panel_width - image.shape[1]) // 2
        panel[64:, offset:offset + image.shape[1]] = image
        cv2.putText(panel, phase, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(panel, f"time {_time(record['timestamp_seconds'])}", (12, 49),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 205, 205), 1, cv2.LINE_AA)
        panels.append(panel)
    canvas = np.full((max(panel.shape[0] for panel in panels), sum(panel.shape[1] for panel in panels) + 16, 3),
                     28, dtype=np.uint8)
    x = 0
    for panel in panels:
        canvas[:panel.shape[0], x:x + panel.shape[1]] = panel
        x += panel.shape[1] + 16
    return canvas


def _failure_category(error: str) -> str:
    lower = error.lower()
    categories = (
        (("hand occlusion", "hand overlap"), "手が領域に重なる疑い"),
        (("too small",), "顔が小さい"),
        (("face yaw", "face pitch", "face roll"), "顔向き・傾きが制限外"),
        (("too few pixels",), "領域の画素数不足"),
        (("outside image",), "領域が画像外"),
        (("clipped pixels",), "白飛び・黒つぶれが多い"),
        (("exactly one face", "multiple face"), "顔が1人に確定しない"),
        (("no face", "face not", "face detected"), "顔を検出できない"),
        (("overlap",), "領域同士が重なる"),
        (("invalid polygon",), "領域の形が不正"),
    )
    for needles, label in categories:
        if any(needle in lower for needle in needles):
            return label
    return error


def _frame_links(record: dict, output: Path) -> str:
    links = [
        _link(Path(record["image_path"]), output, "元画像 PNG"),
        _link(Path(record["roi_dir"]) / "roi_overlay.png", output, "ROI 重ね合わせ"),
        _link(Path(record["roi_dir"]) / "roi_points.json", output, "領域・診断 JSON"),
    ]
    return " · ".join(link for link in links if link)


def _frame_card(record: dict, phase: str, output: Path) -> str:
    overlay = Path(record["roi_dir"]) / "roi_overlay.png"
    source = Path(record["image_path"])
    visible = overlay if overlay.is_file() else source
    image = (f'<a href="{_escape(_uri(visible, output))}"><img loading="lazy" src="{_escape(_uri(visible, output))}" '
             f'alt="{_escape(phase)} {_escape(record["frame_id"])} の ROI 候補"></a>') if visible.is_file() else "<p>画像なし</p>"
    pose = record.get("report", {}).get("quality", {}).get("pose_degrees", {})
    return (f'<section><h3>{_escape(phase)}候補 {_time(record["timestamp_seconds"])}</h3>{image}'
            f'<p class="small">フレーム {_escape(record["frame_id"])} / '
            f'yaw {_number(pose.get("yaw"))}° · pitch {_number(pose.get("pitch"))}° · roll {_number(pose.get("roll"))}°</p>'
            f'<p>{_frame_links(record, output)}</p></section>')


def write_video_report(output: Path, manifest: dict, matching: dict) -> dict[str, str]:
    """Render existing frame extraction/matching results without altering them."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = manifest.get("records", [])
    by_id = {record["frame_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Video report contains duplicate frame IDs")
    pairs = matching.get("ranked_pairs", [])[:10]
    for pair in pairs:
        if pair["before_id"] not in by_id or pair["after_id"] not in by_id:
            raise ValueError("Ranked pair refers to an unknown frame ID")
    artifacts = {"html": "report.html"}
    top_html = "<p class=notice>比較候補は見つかりませんでした。下の停止理由と抽出範囲を確認してください。条件を満たさない画像の解析は行っていません。</p>"
    if pairs:
        best = (by_id[pairs[0]["before_id"]], by_id[pairs[0]["after_id"]])
        for name, crop in (("best_pair", False), ("best_pair_faces", True)):
            filename = f"{name}.png"
            _write_image(output / filename, _pair_image(best, crop=crop))
            artifacts[name] = filename
        overlay_html = ""
        if all((Path(record["roi_dir"]) / "roi_overlay.png").is_file() for record in best):
            artifacts["best_pair_roi_overlay"] = "best_pair_roi_overlay.png"
            _write_image(output / artifacts["best_pair_roi_overlay"], _pair_image(best, crop=False, overlay=True))
            overlay_html = ('<p>ROI の位置と時刻</p><a href="best_pair_roi_overlay.png">'
                            '<img class="hero" src="best_pair_roi_overlay.png" alt="先頭候補の ROI 重ね合わせと時刻"></a>')
        top_html = ('<h2>今回の選択範囲内の先頭候補</h2>' + overlay_html
                    + '<p>元画像の全景</p><a href="best_pair.png"><img class="hero" src="best_pair.png" alt="先頭候補の全景と時刻"></a>'
                    '<p>顔の拡大表示</p><a href="best_pair_faces.png"><img class="hero" src="best_pair_faces.png" alt="先頭候補の顔を拡大した画像"></a>'
                    '<p class="small">顔画像は見やすい寸法に拡大・縮小した表示用画像です。色補正は行っていません。'
                    '今後の解析には、リンク先の元画像と元解像度のマスクを使用します。</p>')
    passed = [record for record in records if record.get("report", {}).get("status") == "needs_review"
              and not record.get("report", {}).get("errors")]
    failure_counts = Counter()
    for record in records:
        report = record.get("report", {})
        reasons = {_failure_category(str(error)) for error in report.get("errors", [])}
        if report.get("status") != "needs_review" and not reasons:
            reasons.add("処理未完了・診断情報なし")
        failure_counts.update(reasons)
    reason_rows = "".join(f"<tr><td>{_escape(reason)}</td><td>{count}</td></tr>" for reason, count in failure_counts.most_common())
    failures_html = (f'<table><thead><tr><th>自動チェックで停止した理由</th><th>画像数</th></tr></thead><tbody>{reason_rows}</tbody></table>'
                     '<p class="small">1枚に複数の理由があるため、理由別の合計は停止画像数と一致しない場合があります。</p>') if reason_rows else "<p>記録された停止理由はありません。</p>"
    cards = []
    for rank, pair in enumerate(pairs, 1):
        before, after = by_id[pair["before_id"]], by_id[pair["after_id"]]
        detail = json.dumps({"terms": pair.get("terms", {}), "diagnostics": pair.get("diagnostics", {})}, ensure_ascii=False, indent=2)
        cards.append(f'<article><h2>候補 {rank} <span class="distance">幾何差 {_number(pair.get("score"), 4)}</span></h2>'
                     '<p class="small">値が小さいほど撮影条件の差が小さい候補です。品質・本人一致・メイク効果の確率ではありません。</p>'
                     f'<div class="pair">{_frame_card(before, "before", output)}{_frame_card(after, "after", output)}</div>'
                     f'<details><summary>比較条件の内訳</summary><pre>{_escape(detail)}</pre></details></article>')
    sample_rows = []
    for record in sorted(records, key=lambda item: item["timestamp_seconds"]):
        report = record.get("report", {})
        ok = report.get("status") == "needs_review" and not report.get("errors")
        status = "自動チェック通過・要目視" if ok else "停止"
        errors = " / ".join(str(error) for error in report.get("errors", []))
        sample_rows.append(f'<tr><td>{_escape(record["frame_id"])}</td><td>{_time(record["timestamp_seconds"])}</td>'
                           f'<td>{status}</td><td>{_escape(errors)}</td><td>{_frame_links(record, output)}</td></tr>')
    video, config = manifest.get("video", {}), manifest.get("config", {})
    notes = manifest.get("notes", [])
    notes_html = "".join(f"<li>{_escape(note)}</li>" for note in notes)
    scope_html = f"<h2>候補を選んだ範囲・前提</h2><ul>{notes_html}</ul>" if notes_html else ""
    if manifest.get("selected_ranges") is not None:
        scope_html += ('<p>今回の選択範囲（以下の範囲内で候補を比較）：</p><pre>'
                       + _escape(json.dumps(manifest["selected_ranges"], ensure_ascii=False, indent=2)) + "</pre>")
    config_rows = [
        ("動画", Path(video.get("path", "")).name),
        ("長さ", _time(video.get("duration_seconds"))),
        ("元動画の寸法・フレームレート", f'{video.get("width", "—")} × {video.get("height", "—")} / {_number(video.get("fps"))} fps'),
        ("抽出間隔", f'{_number(config.get("sample_interval_seconds"))} 秒'),
        ("仮の before / after 境界", _time(config.get("split_seconds"))),
        ("ペアの最小時間間隔", f'{_number(config.get("min_gap_seconds"))} 秒'),
    ]
    for key, label in (("before_range", "before 抽出範囲（秒）"), ("after_range", "after 抽出範囲（秒）")):
        if config.get(key) is not None:
            config_rows.append((label, json.dumps(config[key], ensure_ascii=False)))
    configuration = "".join(f'<tr><th>{_escape(label)}</th><td>{_escape(value)}</td></tr>' for label, value in config_rows)
    data_links = " · ".join(_link(output / name, output, label) for name, label in (
        ("scan_manifest.json", "全フレーム記録 JSON"), ("manifest.json", "記録 JSON"), ("matching.json", "ペア比較記録 JSON"),
        ("frames.csv", "全フレーム CSV"), ("samples.csv", "全サンプル CSV"),
    ) if (output / name).is_file())
    stats = _escape(json.dumps(matching.get("stats", {}), ensure_ascii=False, indent=2))
    document = f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>動画の before / after ROI 候補</title><style>
:root{{color-scheme:light}}body{{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f3f5f4;color:#20302a}}
main{{max-width:1440px;margin:auto;padding:24px}}h1{{font-size:1.8rem}}h2{{font-size:1.35rem}}h3{{font-size:1.05rem}}
a{{color:#11624e}}.notice{{background:#fff0c6;border-left:5px solid #d49e27;padding:16px}}
.metrics{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}}.metric{{flex:1;min-width:150px;background:white;padding:16px;border-radius:8px}}
.metric strong{{display:block;font-size:1.9rem}}.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
article{{background:white;border:1px solid #d8e1dc;border-radius:10px;padding:20px;margin:20px 0}}
img{{max-width:100%;height:auto;border-radius:4px}}.hero{{display:block;max-height:640px;margin:auto}}.pair img{{width:100%;object-fit:contain;background:#202522}}
.small{{font-size:.9rem;color:#50655b}}.distance{{font-size:.85rem;font-weight:normal;margin-left:12px}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:10px;border:1px solid #d8e1dc;vertical-align:top}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#e9efeb;padding:12px;font-size:.85rem}}details{{margin:16px 0}}summary{{cursor:pointer;font-weight:600}}
.scroll{{overflow:auto}}@media(max-width:760px){{main{{padding:14px}}.pair{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>動画から抽出した before / after ROI 候補</h1>
<p class="notice">ここに示すのは、時間帯と顔の幾何条件から選んだ比較候補です。before / after は仮の時系列ラベルで、
メイク開始前・完成後や同一人物であることを保証しません。自動チェック通過後も、手・髪・道具・影・表情・領域の位置を目視してください。
この処理では ROI の承認・Lab 解析・メイクの採点は行っていません。</p>
<div class="metrics"><div class="metric">抽出画像<strong>{len(records)}</strong></div>
<div class="metric">自動チェック通過・要目視<strong>{len(passed)}</strong></div>
<div class="metric">自動チェックで停止<strong>{len(records) - len(passed)}</strong></div>
<div class="metric">表示中の候補ペア<strong>{len(pairs)}</strong></div></div>
<p>候補順位は顔向き・サイズなどの幾何条件を使い、肌色の改善量では選んでいません。額にもメイクが加わる可能性があるため、額の変化を露出補正とは扱いません。</p>{scope_html}
{top_html}<h2>比較候補（上位 {len(pairs)} 組）</h2>
<p>ROI の色：画面左頬＝緑、画面右頬＝青、額＝黄。左右は画像上の位置です。</p>{''.join(cards)}
<h2>抽出結果と停止理由</h2>{failures_html}<h2>今回の条件</h2><table>{configuration}</table><p>{data_links}</p>
<details><summary>ペア選択の集計</summary><pre>{stats}</pre></details>
<details><summary>全 {len(records)} フレームの時刻・状態・ファイル</summary><div class="scroll"><table><thead>
<tr><th>フレーム</th><th>時刻</th><th>状態</th><th>停止理由</th><th>ファイル</th></tr></thead><tbody>{''.join(sample_rows)}</tbody></table></div></details>
</main></body></html>'''
    (output / "report.html").write_text(document, encoding="utf-8")
    return artifacts
