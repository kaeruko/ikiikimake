"""Build a readable Japanese HTML report for five-point eyebrow metrics."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from statistics import mean
from urllib.parse import quote


VARIANT_LABELS = {
    "baseline": ("基準", "抽出した実眉レイヤーの基準形状"),
    "soft_arch": ("柔らかいアーチ", "P2〜P5を局所的に持ち上げ、P4を最も高くした形状"),
    "flat": ("平坦化", "P1とP5を結ぶ直線上へ中間点を並べた形状"),
    "tail_up": ("眉尻上げ", "P3〜P5を段階的に持ち上げた形状"),
    "tail_down": ("眉尻下げ", "P3〜P5を段階的に下げた形状"),
}

METRICS = (
    ("length_eye_widths", "眉の長さ", "目幅"),
    ("effective_area_eye_width2", "面積", "目幅²"),
    ("average_width_eye_widths", "平均幅", "目幅"),
    ("apex_arc_fraction", "眉山位置 / 眉全長", "比"),
    ("max_chord_distance_eye_widths", "弦からの最大距離", "目幅"),
    ("head_to_tail_slope_degrees", "眉頭―眉尻角度", "°"),
    ("tail_to_head_thickness_ratio", "眉尻 / 眉頭の太さ", "比"),
)


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _number(value, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _relative(path: Path, output: Path) -> str:
    return quote(os.path.relpath(path.resolve(), output.parent.resolve()).replace("\\", "/"), safe="/._-+")


def _side_values(condition: dict) -> tuple[dict, dict]:
    sides = condition["shape_metrics"]["sides"]
    return sides["screen_left"], sides["screen_right"]


def _average(condition: dict, key: str) -> float:
    left, right = _side_values(condition)
    return mean((float(left[key]), float(right[key])))


def _point_pair(condition: dict, point: str, coordinate: str) -> str:
    left, right = _side_values(condition)
    return f"{float(left[point][coordinate]):.3f} / {float(right[point][coordinate]):.3f}"


def _metric_table(conditions: list[dict]) -> str:
    header = "".join(f"<th>{_e(label)}<small>{_e(unit)}</small></th>" for _, label, unit in METRICS)
    rows = []
    for condition in conditions:
        label = VARIANT_LABELS.get(condition["variant"], (condition["variant"], ""))[0]
        cells = "".join(
            f'<td class="num">{_number(_average(condition, key), 2 if unit == "°" else 3)}</td>'
            for key, _, unit in METRICS
        )
        rows.append(f"<tr><th>{_e(label)}</th>{cells}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr><th>条件</th>' + header
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _asymmetry_table(conditions: list[dict]) -> str:
    keys = (
        ("length_eye_widths", "長さ"),
        ("effective_area_eye_width2", "面積"),
        ("average_width_eye_widths", "平均幅"),
        ("apex_arc_fraction", "眉山位置"),
        ("max_chord_distance_eye_widths", "曲率"),
        ("head_to_tail_slope_degrees", "角度"),
        ("tail_to_head_thickness_ratio", "太さ比"),
    )
    rows = []
    for condition in conditions:
        label = VARIANT_LABELS.get(condition["variant"], (condition["variant"], ""))[0]
        differences = condition["shape_metrics"]["left_right_differences"]
        cells = "".join(
            f'<td class="num">{_number(differences[key]["absolute_difference"], 2 if key.endswith("degrees") else 3)}</td>'
            for key, _ in keys
        )
        rows.append(f"<tr><th>{_e(label)}</th>{cells}</tr>")
    header = "".join(f"<th>{_e(label)}</th>" for _, label in keys)
    return (
        '<div class="table-wrap"><table><thead><tr><th>条件</th>' + header
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _bars(conditions: list[dict], key: str, label: str, unit: str) -> str:
    values = [_average(condition, key) for condition in conditions]
    low, high = min(values), max(values)
    span = high - low
    rows = []
    for condition, value in zip(conditions, values):
        title = VARIANT_LABELS.get(condition["variant"], (condition["variant"], ""))[0]
        width = 50.0 if span < 1e-12 else 8.0 + 92.0 * (value - low) / span
        digits = 1 if unit == "°" else 3
        rows.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{_e(title)}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{width:.2f}%"></span></span>'
            f'<strong>{_number(value, digits)}</strong>'
            "</div>"
        )
    return (
        '<section class="mini-chart" aria-label="' + _e(label) + '">'
        f'<h3>{_e(label)} <small>{_e(unit)}</small></h3>'
        + "".join(rows) + "</section>"
    )


def _variant_cards(conditions: list[dict], directory: Path, output: Path) -> str:
    cards = []
    for condition in conditions:
        variant = condition["variant"]
        title, description = VARIANT_LABELS.get(variant, (variant, ""))
        left, right = _side_values(condition)
        apex = f'P{left["apex_control_point"]} / P{right["apex_control_point"]}'
        fraction = f'{left["apex_arc_fraction"]:.3f} / {right["apex_arc_fraction"]:.3f}'
        image = _relative(directory / condition["file"], output)
        cards.append(
            '<article class="variant-card">'
            f'<img src="{image}" alt="{_e(title)}の眉形状" loading="lazy">'
            '<div class="variant-copy">'
            f'<h3>{_e(title)}</h3><p>{_e(description)}</p>'
            '<dl>'
            f'<div><dt>眉山</dt><dd>{_e(apex)}</dd></div>'
            f'<div><dt>眉山弧長比</dt><dd>{_e(fraction)}</dd></div>'
            f'<div><dt>変更画素</dt><dd>{int(condition["changed_pixels"]):,}</dd></div>'
            '</dl></div></article>'
        )
    return '<div class="variant-grid">' + "".join(cards) + "</div>"


def _detail_rows(conditions: list[dict]) -> str:
    rows = []
    for condition in conditions:
        title = VARIANT_LABELS.get(condition["variant"], (condition["variant"], ""))[0]
        for side_name, side_label in (("screen_left", "画面左"), ("screen_right", "画面右")):
            values = condition["shape_metrics"]["sides"][side_name]
            rows.append(
                "<tr>"
                f"<th>{_e(title)}</th><td>{_e(side_label)}</td>"
                f'<td class="num">{values["length_px"]:.1f}</td>'
                f'<td class="num">{values["effective_area_px2"]:.1f}</td>'
                f'<td class="num">{values["average_width_px"]:.1f}</td>'
                f'<td class="num">P{values["apex_control_point"]}</td>'
                f'<td class="num">{values["apex_arc_fraction"]:.3f}</td>'
                f'<td class="num">{values["max_chord_distance_px"]:.1f}</td>'
                f'<td class="num">{values["head_to_tail_slope_degrees"]:.1f}°</td>'
                f'<td class="num">{values["tail_to_head_thickness_ratio"]:.3f}</td>'
                "</tr>"
            )
    return "".join(rows)


def build_report(directory: Path, output: Path) -> None:
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    conditions = report["conditions"]
    required = [directory / "contact_sheet.png", directory / "five_control_points.png",
                directory / "shape_metrics.csv", *[directory / row["file"] for row in conditions]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing report artifact(s): " + ", ".join(missing))

    registration = report["registration"]
    outside_max = max(int(row["outside_support_max_abs_delta"]) for row in conditions)
    flat = next(row for row in conditions if row["variant"] == "flat")
    nonflat = [row for row in conditions if row["variant"] != "flat"]
    flat_apex = [side["apex_control_point"] for side in _side_values(flat)]
    normal_apex = sorted({side["apex_control_point"] for row in nonflat for side in _side_values(row)})

    chart_specs = (
        ("length_eye_widths", "眉の長さ", "目幅比"),
        ("effective_area_eye_width2", "眉面積", "目幅²比"),
        ("max_chord_distance_eye_widths", "曲率：弦からの最大距離", "目幅比"),
        ("head_to_tail_slope_degrees", "眉頭―眉尻角度", "度"),
        ("tail_to_head_thickness_ratio", "眉尻 / 眉頭の太さ", "比"),
    )
    charts = "".join(_bars(conditions, *spec) for spec in chart_specs)
    contact = _relative(directory / "contact_sheet.png", output)
    controls = _relative(directory / "five_control_points.png", output)
    csv_link = _relative(directory / "shape_metrics.csv", output)
    json_link = _relative(directory / "report.json", output)

    css = """
:root{color-scheme:light;--ink:#24201e;--muted:#6d625d;--paper:#fffdfb;--panel:#f6f0ec;--line:#ded3cc;--accent:#a64b62;--accent-soft:#ead0d7;--warn:#8b4a20}*{box-sizing:border-box}body{margin:0;background:#eee8e4;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Yu Gothic UI","Hiragino Sans",sans-serif;line-height:1.65}main{max-width:1180px;margin:0 auto;background:var(--paper);padding:40px 48px 72px;box-shadow:0 0 24px #5b47401c}h1,h2,h3,p{margin-top:0}h1{font-size:2rem;line-height:1.3;margin-bottom:8px}h2{font-size:1.35rem;margin:42px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--ink)}h3{font-size:1rem;margin-bottom:8px}small,.muted{color:var(--muted);font-weight:400}.lead{font-size:1.05rem;max-width:850px}.summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:24px 0}.summary div{background:var(--panel);padding:16px 18px;border-radius:8px}.summary dt{font-size:.82rem;color:var(--muted)}.summary dd{margin:2px 0 0;font-size:1.35rem;font-weight:600;font-variant-numeric:tabular-nums}.hero{width:100%;display:block;background:#1e1d1c;border-radius:8px}.note{background:#fff4e8;border-left:5px solid var(--warn);padding:14px 18px;margin:18px 0}.control-layout{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(280px,.7fr);gap:24px;align-items:start}.control-layout img{width:100%;display:block;border-radius:8px}.control-layout ul{margin-top:0;padding-left:1.25rem}.variant-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.variant-card{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff}.variant-card img{width:100%;display:block}.variant-copy{padding:14px 16px}.variant-copy p{color:var(--muted);min-height:3.2em}.variant-copy dl{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:0}.variant-copy dl div{background:var(--panel);padding:8px;border-radius:6px}.variant-copy dt{font-size:.76rem;color:var(--muted)}.variant-copy dd{margin:0;font-variant-numeric:tabular-nums}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.mini-chart{border-top:1px solid var(--line);padding-top:12px}.bar-row{display:grid;grid-template-columns:7.5em 1fr 4.5em;gap:9px;align-items:center;margin:7px 0;font-size:.88rem}.bar-track{display:block;height:12px;background:#eee6e1;border-radius:6px;overflow:hidden}.bar-fill{display:block;height:100%;background:var(--accent);border-radius:6px}.bar-row strong{text-align:right;font-variant-numeric:tabular-nums}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:.86rem}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}thead th{background:var(--panel);white-space:nowrap}thead small{display:block}tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}details{margin-top:16px}summary{font-weight:600}.definitions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.definitions div{padding:12px 14px;background:var(--panel);border-radius:8px}.definitions dt{font-weight:600}.definitions dd{margin:2px 0 0;color:var(--muted)}code{background:var(--panel);padding:.12em .35em;border-radius:4px}.links{display:flex;gap:16px;flex-wrap:wrap}.links a{color:var(--accent);font-weight:600}@media(max-width:760px){main{padding:24px 18px 48px}.summary,.charts,.variant-grid,.control-layout,.definitions{grid-template-columns:1fr}.variant-copy p{min-height:0}.bar-row{grid-template-columns:6.7em 1fr 4em}}
"""
    document = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>眉5点曲線・形状指標レポート</title><style>{css}</style></head><body><main>
<header><p class="muted">Issue #5「A. 眉の形状」実装結果</p><h1>眉5点曲線・形状指標レポート</h1>
<p class="lead">実眉テクスチャをP1（眉頭）からP5（眉尻）の5点曲線として局所変形し、形状を左右別に測定した。数値は形状の記述であり、美しさや自然さの得点ではない。</p></header>
<dl class="summary"><div><dt>位置合わせ誤差・中央値</dt><dd>{registration['median_control_error_px']:.2f} px</dd></div>
<div><dt>曲線制御点</dt><dd>左右 各5点</dd></div><div><dt>編集領域外の最大画素差</dt><dd>{outside_max}</dd></div></dl>
<h2>全条件の見た目</h2><img class="hero" src="{contact}" alt="元画像と5種類の眉曲線条件の比較">
<div class="note"><strong>今回の主な差：</strong>平坦化だけは左右とも眉山がP{flat_apex[0]}、眉山弧長比0になった。ほかの条件は眉山が{', '.join('P'+str(x) for x in normal_apex)}に残り、弧長比はおよそ0.61〜0.67だった。</div>
<h2>5点曲線の読み方</h2><div class="control-layout"><img src="{controls}" alt="左右眉のP1からP5までの制御点">
<div><ul><li><strong>P1</strong>：眉頭</li><li><strong>P2</strong>：内側中間</li><li><strong>P3</strong>：中央</li><li><strong>P4</strong>：外側中間</li><li><strong>P5</strong>：眉尻</li></ul>
<p>左右は鏡像化した意味座標で比較する。Xは目頭から外側、Yは目の中心から額方向を正とし、各目の幅で正規化する。</p></div></div>
<h2>条件別</h2>{_variant_cards(conditions, directory, output)}
<h2>指標の相対比較</h2><p class="muted">棒は各指標内の最小〜最大を相対表示する。値は左右眉の平均。</p><div class="charts">{charts}</div>
<h2>左右平均の数値</h2>{_metric_table(conditions)}
<h2>左右差</h2><p class="muted">すべて絶対差。角度以外は目幅などで正規化した値。</p>{_asymmetry_table(conditions)}
<details><summary>左右別の実画素値を表示</summary><div class="table-wrap"><table><thead><tr><th>条件</th><th>側</th><th>長さ px</th><th>面積 px²</th><th>平均幅 px</th><th>眉山</th><th>眉山比</th><th>曲率 px</th><th>角度</th><th>眉尻/眉頭太さ</th></tr></thead><tbody>{_detail_rows(conditions)}</tbody></table></div></details>
<h2>指標の定義</h2><dl class="definitions">
<div><dt>眉の長さ</dt><dd>P1〜P5を結ぶ折れ線の弧長。</dd></div><div><dt>面積</dt><dd>抽出眉レイヤーのα値合計。半透明境界も有効面積へ含める。</dd></div>
<div><dt>平均幅</dt><dd>有効面積 ÷ 眉の長さ。</dd></div><div><dt>眉頭・眉山・眉尻</dt><dd>P1、額方向へ最も高い制御点、P5の局所目座標。</dd></div>
<div><dt>眉山位置 / 眉全長</dt><dd>P1から眉山までの弧長 ÷ P1からP5までの弧長。</dd></div><div><dt>曲率</dt><dd>P1―P5の弦から5点曲線までの最大垂直距離。</dd></div>
<div><dt>眉頭―眉尻角度</dt><dd>目頭から目尻方向を基準としたP1―P5の角度。正値は眉尻が高い。</dd></div><div><dt>太さの減衰</dt><dd>眉を5区間に分けた有効幅から、眉尻/眉頭比と正規化傾向を算出。</dd></div>
<div><dt>左右差</dt><dd>鏡像化した意味座標上で、画面左と画面右の各指標の絶対差。</dd></div></dl>
<h2>解釈上の注意</h2><ul><li>このレポートは1人・1動画の探索結果で、一般的な理想眉を示さない。</li><li>元眉除去部分には局所補間を含む。</li><li>面積と太さはTPS補間およびα閾値の影響を受ける。</li><li>「自然」「若々しい」「元気」などは別途、人による印象評価が必要。</li></ul>
<h2>再現用データ</h2><div class="links"><a href="{csv_link}">shape_metrics.csv</a><a href="{json_link}">report.json</a></div>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.analysis_dir / "shape_report.html"
    build_report(args.analysis_dir, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
