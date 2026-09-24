"""Upper-cheek makeup appearance review: unevenness vs dryness-like texture.

This module reuses the saved multi-rank selections without changing candidate
matching. It separates broad color/lightness unevenness from fine texture and
dark-line appearance. Metrics are image-space appearance features, not clinical
dryness, hydration, or physical wrinkle depth.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np

from analysis.analyze_selected_regions import _load_phase
from analysis.appearance_features import (
    TEXTURE_BLUR_SIGMA,
    measure_skin_appearance_features,
)


REPORT_VERSION = "upper-cheek-dryness-review-v1"

METRIC_ORDER = (
    "lowfreq_L_mad",
    "lowfreq_ab_mad",
    "highpass_median_pct",
    "highpass_p90_pct",
    "fine_dark_line_p95_pct",
)

METRIC_LABELS = {
    "lowfreq_L_mad": "低周波明度ムラ（L* MAD）",
    "lowfreq_ab_mad": "低周波色ムラ（ab MAD）",
    "highpass_median_pct": "細かな質感コントラスト（中央値）",
    "highpass_p90_pct": "細かな質感コントラスト（p90）",
    "fine_dark_line_p95_pct": "細線候補・暗さコントラスト（p95）",
}


def build_upper_cheek_mask(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    side: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    candidate_name = f"{side}_nasolabial_candidate"
    control_name = f"{side}_nasolabial_outer_control"
    for name in (candidate_name, control_name):
        if name not in masks:
            raise KeyError(f"required ROI is missing: {name}")

    candidate = np.asarray(masks[candidate_name], dtype=bool)
    control = np.asarray(masks[control_name], dtype=bool)
    if candidate.shape != image.shape[:2] or control.shape != image.shape[:2]:
        raise ValueError(f"{side}: nasolabial masks do not match image shape")
    union = candidate | control

    if side == "screen_left":
        cheek_name = "left_cheek"
    elif side == "screen_right":
        cheek_name = "right_cheek"
    else:
        raise ValueError(f"unknown side: {side}")
    if cheek_name not in masks:
        raise KeyError(f"required cheek ROI is missing: {cheek_name}")
    cheek = np.asarray(masks[cheek_name], dtype=bool)
    if cheek.shape != image.shape[:2]:
        raise ValueError(f"{side}: cheek mask does not match image shape")

    ys, xs = np.where(union)
    if len(xs) == 0:
        raise ValueError(f"{side}: nasolabial union is empty")

    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    roi_w = max(18, round(0.95 * width))
    roi_h = max(18, round(1.10 * height))
    gap = max(4, round(0.10 * width))
    top = max(0, int(round(y0 - 0.15 * height)))
    bottom = min(image.shape[0], top + roi_h)

    if side == "screen_left":
        right = max(0, int(x0 - gap))
        left = max(0, right - roi_w)
    else:
        left = min(image.shape[1], int(x1 + gap))
        right = min(image.shape[1], left + roi_w)

    rect = np.zeros(image.shape[:2], dtype=bool)
    rect[top:bottom, left:right] = True
    mask = rect & cheek & ~union
    pixels = int(mask.sum())
    if pixels < 100:
        raise ValueError(f"{side}: upper-cheek ROI is too small ({pixels} px)")
    return mask, (left, top, right, bottom)


def measure_highpass(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    image = np.asarray(image_bgr)
    mask = np.asarray(mask, dtype=bool)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or mask.shape != image.shape[:2]
        or not mask.any()
    ):
        raise ValueError("invalid image or upper-cheek mask")

    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float64)
    low_frequency = cv2.GaussianBlur(lightness, (0, 0), TEXTURE_BLUR_SIGMA)
    highpass_abs = np.abs(lightness - low_frequency)
    base_l = float(np.median(lightness[mask]))
    if not math.isfinite(base_l) or base_l <= 1e-6:
        raise ValueError("upper-cheek median L* is invalid")
    values = highpass_abs[mask]
    return {
        "highpass_median_pct": float(100.0 * np.percentile(values, 50) / base_l),
        "highpass_p90_pct": float(100.0 * np.percentile(values, 90) / base_l),
    }


def measure_upper_cheek(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    highpass = measure_highpass(image_bgr, mask)
    appearance_rows = measure_skin_appearance_features(
        image_bgr,
        {"upper_cheek": np.asarray(mask, dtype=bool)},
        regions=(("upper_cheek", "上頬"),),
    )
    by_id = {row["id"]: row for row in appearance_rows}
    expected = {
        "upper_cheek_lowfreq_L_mad",
        "upper_cheek_lowfreq_ab_mad",
        "upper_cheek_fine_dark_line_p95_pct",
    }
    if set(by_id) != expected:
        raise RuntimeError(f"upper-cheek appearance metrics mismatch: {set(by_id)}")
    for key in expected:
        if by_id[key]["status"] != "ok" or by_id[key]["value"] is None:
            raise RuntimeError(
                f"upper-cheek appearance metric unavailable: {key} "
                f"status={by_id[key]['status']!r}"
            )
    return {
        "lowfreq_L_mad": float(by_id["upper_cheek_lowfreq_L_mad"]["value"]),
        "lowfreq_ab_mad": float(by_id["upper_cheek_lowfreq_ab_mad"]["value"]),
        **highpass,
        "fine_dark_line_p95_pct": float(
            by_id["upper_cheek_fine_dark_line_p95_pct"]["value"]
        ),
    }


def _crop_uri(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> str:
    x0, y0, x1, y1 = box
    roi_w = max(1, x1 - x0)
    roi_h = max(1, y1 - y0)
    pad = max(18, round(max(roi_w, roi_h) * 0.35))
    left = max(0, x0 - pad)
    top = max(0, y0 - pad)
    right = min(image_bgr.shape[1], x1 + pad)
    bottom = min(image_bgr.shape[0], y1 + pad)
    if left >= right or top >= bottom:
        raise ValueError(f"invalid upper-cheek crop: {(left, top, right, bottom)}")

    crop = image_bgr[top:bottom, left:right].copy()
    local_mask = np.asarray(mask[top:bottom, left:right], dtype=bool)
    if not local_mask.any():
        raise ValueError("upper-cheek local mask is empty")
    fill = crop.copy()
    fill[local_mask] = (0, 175, 255)
    overlay = cv2.addWeighted(fill, 0.26, crop, 0.74, 0)
    contours = cv2.findContours(
        local_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    cv2.drawContours(overlay, contours, -1, (0, 140, 255), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("failed to encode upper-cheek crop")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _fmt(value: float, signed: bool = False) -> str:
    return f"{float(value):+.3f}" if signed else f"{float(value):.3f}"


def _render_html(rows: list[dict], review: list[dict]) -> str:
    lookup = {(item["rank"], item["side_label"]): item for item in review}
    ranks = sorted({row["rank"] for row in rows})
    parts = [
        "<div class='upper-cheek-report'>",
        f"<h1>上頬 ROI：ムラと乾燥感候補を分けて確認（{len(ranks)}ペア）</h1>",
        (
            "<p>オレンジ色が計測ROIです。低周波の明度・色ムラと、"
            "高周波の細かな質感、細い暗線候補を別々に表示します。"
            "細線候補は物理的なシワ深さではなく、画像上の暗い細線の目立ちです。</p>"
        ),
    ]

    for rank in ranks:
        first = lookup[(rank, "画面左")]
        parts.append(
            f"<section><h2>rank {rank} &nbsp; "
            f"before {first['before_t']:.2f}s → after {first['after_t']:.2f}s</h2>"
        )
        for side_label in ("画面左", "画面右"):
            item = lookup[(rank, side_label)]
            before_uri = _crop_uri(
                item["before_img"], item["before_roi"], item["before_box"]
            )
            after_uri = _crop_uri(
                item["after_img"], item["after_roi"], item["after_box"]
            )
            side_rows = {
                row["metric"]: row
                for row in rows
                if row["rank"] == rank and row["side"] == side_label
            }
            if set(side_rows) != set(METRIC_ORDER):
                raise RuntimeError(
                    f"rank {rank} {side_label}: metrics mismatch: {set(side_rows)}"
                )
            table_rows = []
            for metric in METRIC_ORDER:
                row = side_rows[metric]
                table_rows.append(
                    "<tr>"
                    f"<td>{html.escape(METRIC_LABELS[metric])}</td>"
                    f"<td>{_fmt(row['before'])}</td>"
                    f"<td>{_fmt(row['after'])}</td>"
                    f"<td>{_fmt(row['delta_after_minus_before'], signed=True)}</td>"
                    "</tr>"
                )
            parts.append(
                f"<div class='side-card'><h3>{side_label}</h3>"
                "<div class='roi-pair'>"
                f"<figure><figcaption>before</figcaption><img src='{before_uri}'></figure>"
                f"<figure><figcaption>after</figcaption><img src='{after_uri}'></figure>"
                "</div>"
                "<table><thead><tr><th>指標</th><th>before</th><th>after</th>"
                "<th>after - before</th></tr></thead>"
                f"<tbody>{''.join(table_rows)}</tbody></table></div>"
            )
        parts.append("</section>")
    parts.append("</div>")

    style = """
<style>
.upper-cheek-report{background:#fff!important;color:#17221d!important;color-scheme:light!important;padding:20px;font:15px/1.6 system-ui,sans-serif}
.upper-cheek-report section{background:#fff!important;border:1px solid #cfd8d3;border-radius:12px;padding:18px;margin:24px 0}
.upper-cheek-report .side-card{background:#fafcfb!important;border:1px solid #dce4df;border-radius:10px;padding:14px;margin:16px 0}
.upper-cheek-report h1,.upper-cheek-report h2,.upper-cheek-report h3,.upper-cheek-report p,.upper-cheek-report th,.upper-cheek-report td,.upper-cheek-report figcaption{color:#17221d!important}
.upper-cheek-report .roi-pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}
.upper-cheek-report figure{margin:0;background:#fff!important}
.upper-cheek-report figcaption{font-weight:700;margin:0 0 6px}
.upper-cheek-report img{display:block;width:100%;max-height:420px;object-fit:contain;background:#fff!important;border:1px solid #dfe6e2;border-radius:8px}
.upper-cheek-report table{width:100%;border-collapse:collapse;margin-top:12px;background:#fff!important}
.upper-cheek-report th{background:#eef3f0!important;font-weight:700}
.upper-cheek-report th,.upper-cheek-report td{padding:8px 10px;border-bottom:1px solid #d7ded9;text-align:right}
.upper-cheek-report th:first-child,.upper-cheek-report td:first-child{text-align:left}
@media(max-width:800px){.upper-cheek-report .roi-pair{grid-template-columns:1fr}}
</style>
"""
    return style + "".join(parts)


def analyze_upper_cheek_review(
    rank_summary_path: Path,
    output_html: Path | None = None,
    output_csv: Path | None = None,
) -> dict:
    rank_summary_path = Path(rank_summary_path)
    if not rank_summary_path.is_file():
        raise FileNotFoundError(rank_summary_path)
    summary = json.loads(rank_summary_path.read_text(encoding="utf-8"))
    pairs = summary.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("rank summary has no pairs")

    if output_html is None:
        output_html = rank_summary_path.parent / "upper_cheek_dryness_report.html"
    if output_csv is None:
        output_csv = rank_summary_path.parent / "upper_cheek_dryness_metrics.csv"
    output_html = Path(output_html)
    output_csv = Path(output_csv)

    rows = []
    review = []
    for pair in pairs:
        rank = pair.get("rank")
        selection = pair.get("selection")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"invalid rank: {rank!r}")
        if not isinstance(selection, dict):
            raise ValueError(f"rank {rank}: selection is missing")
        before_img, before_masks, _ = _load_phase(selection["before"], "eye_texture")
        after_img, after_masks, _ = _load_phase(selection["after"], "eye_texture")

        for side, side_label in (
            ("screen_left", "画面左"),
            ("screen_right", "画面右"),
        ):
            before_roi, before_box = build_upper_cheek_mask(
                before_img, before_masks, side
            )
            after_roi, after_box = build_upper_cheek_mask(
                after_img, after_masks, side
            )
            before_values = measure_upper_cheek(before_img, before_roi)
            after_values = measure_upper_cheek(after_img, after_roi)
            if set(before_values) != set(METRIC_ORDER) or set(after_values) != set(METRIC_ORDER):
                raise RuntimeError(f"rank {rank} {side}: metric set mismatch")

            for metric in METRIC_ORDER:
                before_value = before_values[metric]
                after_value = after_values[metric]
                rows.append({
                    "rank": rank,
                    "side": side_label,
                    "metric": metric,
                    "before": before_value,
                    "after": after_value,
                    "delta_after_minus_before": after_value - before_value,
                })
            review.append({
                "rank": rank,
                "side": side,
                "side_label": side_label,
                "before_t": float(selection["before"]["timestamp_seconds"]),
                "after_t": float(selection["after"]["timestamp_seconds"]),
                "before_img": before_img,
                "after_img": after_img,
                "before_roi": before_roi,
                "after_roi": after_roi,
                "before_box": before_box,
                "after_box": after_box,
            })

    expected_rows = len(pairs) * 2 * len(METRIC_ORDER)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"upper-cheek rows mismatch: expected={expected_rows}, actual={len(rows)}"
        )

    output_html.write_text(_render_html(rows, review), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rank", "side", "metric", "before", "after",
                "delta_after_minus_before",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    sign_summary = []
    for metric in METRIC_ORDER:
        for side_label in ("画面左", "画面右"):
            values = [
                row["delta_after_minus_before"]
                for row in rows
                if row["metric"] == metric and row["side"] == side_label
            ]
            sign_summary.append({
                "metric": metric,
                "side": side_label,
                "n": len(values),
                "positive": sum(value > 0 for value in values),
                "negative": sum(value < 0 for value in values),
                "zero": sum(value == 0 for value in values),
                "median_delta": float(np.median(values)),
            })

    return {
        "schema_version": 1,
        "version": REPORT_VERSION,
        "rank_summary_path": str(rank_summary_path.resolve()),
        "pairs": len(pairs),
        "metrics": list(METRIC_ORDER),
        "output_html": str(output_html.resolve()),
        "output_csv": str(output_csv.resolve()),
        "sign_summary": sign_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rank_summary", type=Path)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    result = analyze_upper_cheek_review(args.rank_summary, args.html, args.csv)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
