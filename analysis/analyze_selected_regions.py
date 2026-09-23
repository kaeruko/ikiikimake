"""Analyze descriptive appearance features on approved region-specific video pairs.

The input is ``selected_region_pairs.json`` produced by
``notebooks/video_region_pair_candidates.ipynb``. Each selected region keeps its
own before/after frame. Saved image/ROI hashes are verified before measurement.
No region is substituted and no missing selection is inferred.

These are descriptive image features, not diagnoses or beauty scores.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np

from analysis.analyze_cheek_lab import load_image, load_masks
from analysis.appearance_features import build_feature_masks, measure_features


ANALYSIS_VERSION = "selected-region-appearance-v1"

REGION_METRIC_IDS = {
    "eye_texture": (
        "screen_left_lower_eye_skin_highpass_median_pct",
        "screen_left_lower_eye_skin_highpass_p90_pct",
        "screen_right_lower_eye_skin_highpass_median_pct",
        "screen_right_lower_eye_skin_highpass_p90_pct",
    ),
    "lips": (
        "lips_relative_a",
        "lips_skin_delta_e76",
    ),
    "cheeks": (
        "left_cheek_L_median",
        "left_cheek_a_median",
        "left_cheek_b_median",
        "left_cheek_ab_mad",
        "left_cheek_highlight_percent",
        "right_cheek_L_median",
        "right_cheek_a_median",
        "right_cheek_b_median",
        "right_cheek_ab_mad",
        "right_cheek_highlight_percent",
    ),
}

REGION_MASK_NAMES = {
    "eye_texture": ("screen_left_lower_eye_skin", "screen_right_lower_eye_skin"),
    "lips": ("lips",),
    "cheeks": ("left_cheek", "right_cheek"),
}

REGION_LABELS = {
    "eye_texture": "目の下の質感",
    "lips": "唇",
    "cheeks": "頬",
}


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_phase(entry: dict, region: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict]]:
    if not isinstance(entry, dict):
        raise ValueError(f"{region}: selected frame entry must be an object")
    required = (
        "image_path", "image_sha256", "roi_dir",
        "roi_masks_sha256", "roi_points_sha256", "roi_overlay_sha256",
    )
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"{region}: selected frame is missing {missing}")

    image_path = Path(entry["image_path"])
    roi_dir = Path(entry["roi_dir"])
    roi_masks = roi_dir / "roi_masks.npz"
    roi_points = roi_dir / "roi_points.json"
    roi_overlay = roi_dir / "roi_overlay.png"
    checks = {
        image_path: entry["image_sha256"],
        roi_masks: entry["roi_masks_sha256"],
        roi_points: entry["roi_points_sha256"],
        roi_overlay: entry["roi_overlay_sha256"],
    }
    for path, expected in checks.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"{region}: saved input changed: {path}; expected={expected} actual={actual}"
            )

    image = load_image(image_path)
    report = json.loads(roi_points.read_text(encoding="utf-8"))
    if report.get("status") != "needs_review" or report.get("errors") != []:
        raise RuntimeError(f"{region}: ROI report is not reviewable: {roi_points}")
    source = report.get("source", {})
    if source.get("width") != image.shape[1] or source.get("height") != image.shape[0]:
        raise RuntimeError(f"{region}: ROI report dimensions do not match image")
    if source.get("sha256") != entry["image_sha256"]:
        raise RuntimeError(f"{region}: ROI report image hash does not match selection")

    landmarks = np.asarray(report.get("face_landmarks_normalized"), dtype=np.float64)
    if (landmarks.ndim != 2 or landmarks.shape[0] < 468 or landmarks.shape[1] != 3
            or not np.isfinite(landmarks).all()):
        raise ValueError(f"{region}: invalid face landmarks")
    points = landmarks[:, :2] * [image.shape[1], image.shape[0]]
    base_masks = load_masks(roi_masks, image.shape)
    masks = build_feature_masks(image.shape, points, base_masks)
    measured = {row["id"]: row for row in measure_features(image, masks)}
    needed = REGION_METRIC_IDS[region]
    absent = [identifier for identifier in needed if identifier not in measured]
    if absent:
        raise RuntimeError(f"{region}: expected metrics are missing: {absent}")
    return image, masks, measured


def _overlay(image: np.ndarray, masks: dict[str, np.ndarray], region: str) -> np.ndarray:
    canvas = image.copy()
    union = np.zeros(image.shape[:2], dtype=bool)
    for name in REGION_MASK_NAMES[region]:
        if name not in masks:
            raise RuntimeError(f"{region}: expected mask is missing: {name}")
        union |= masks[name].astype(bool)
    if not union.any():
        raise RuntimeError(f"{region}: selected region mask is empty")
    fill = canvas.copy()
    fill[union] = (50, 220, 240)
    canvas = cv2.addWeighted(fill, 0.35, canvas, 0.65, 0)
    contours = cv2.findContours(union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(canvas, contours, -1, (0, 200, 255), 2, cv2.LINE_AA)
    return canvas


def _save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "region", "id", "label", "unit", "before", "after", "delta",
        "before_pixels", "after_pixels", "status", "note",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report_html(summary: dict) -> str:
    sections = []
    for region, result in summary["regions"].items():
        rows = []
        for row in result["deltas"]:
            before = "—" if row["before"] is None else f'{row["before"]:.3f}'
            after = "—" if row["after"] is None else f'{row["after"]:.3f}'
            delta = "—" if row["delta"] is None else f'{row["delta"]:+.3f}'
            rows.append(
                "<tr>"
                f"<td>{html.escape(row['label'])}</td>"
                f"<td>{html.escape(row['unit'])}</td>"
                f"<td>{before}</td><td>{after}</td><td>{delta}</td>"
                f"<td>{html.escape(row['status'])}</td>"
                "</tr>"
            )
        before_t = result["selection"]["before"]["timestamp_seconds"]
        after_t = result["selection"]["after"]["timestamp_seconds"]
        scale = result["selection"]["region_gate"].get("max_face_scale_ratio")
        sections.append(
            f"<section><h2>{html.escape(REGION_LABELS[region])}</h2>"
            f"<p>選択rank {result['selection']['rank']} ／ before {before_t:.2f}s → after {after_t:.2f}s</p>"
            f"<p>候補選定時の顔サイズ上限 {scale:.3f}。画像自体のサイズ補正・登録・平滑化は行っていません。</p>"
            f"<div class='pair'><img src='{region}_before.png'><img src='{region}_after.png'></div>"
            "<table><thead><tr><th>指標</th><th>単位</th><th>before</th><th>after</th><th>差</th><th>状態</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )

    return f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>部位別 selected pair 解析</title>
<style>
body{{font:16px/1.7 system-ui,sans-serif;max-width:1300px;margin:28px auto;padding:0 22px;color:#25322d}}
section{{border:1px solid #d6dfda;border-radius:10px;padding:18px;margin:22px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;margin-top:16px}}th,td{{padding:8px;border-bottom:1px solid #d7ded9;text-align:left}}
.notice{{background:#fff2c8;padding:14px;border-left:5px solid #d79b20}}
@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}
</style><body><h1>承認済み部位別 before / after の記述解析</h1>
<p class="notice">この結果は画像上の記述的特徴です。乾燥・シワの診断、メイク効果の因果推定、美しさの採点ではありません。
目の下の高周波指標にはピント・照明・圧縮・まつげ・メイク境界などが混入し得ます。</p>
{"".join(sections)}
</body></html>"""


def analyze_selected_regions(selected_path: Path, output_root: Path) -> dict:
    selected_path = Path(selected_path)
    output_root = Path(output_root)
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selections = selected.get("selections")
    if not isinstance(selections, dict) or not selections:
        raise ValueError("selected_region_pairs.json has no selections")
    unknown = set(selections) - set(REGION_METRIC_IDS)
    if unknown:
        raise ValueError(f"Unknown selected regions: {sorted(unknown)}")

    implementation = sha256_file(Path(__file__))
    feature_impl = sha256_file(Path(__file__).with_name("appearance_features.py"))
    fingerprint_spec = {
        "version": ANALYSIS_VERSION,
        "selected_region_pairs_sha256": sha256_file(selected_path),
        "implementation_sha256": implementation,
        "appearance_features_sha256": feature_impl,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_spec, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_dir = output_root / fingerprint[:16]

    if output_dir.exists():
        summary_path = output_dir / "summary.json"
        report_path = output_dir / "report.html"
        csv_path = output_dir / "feature_deltas.csv"
        if not all(path.is_file() for path in (summary_path, report_path, csv_path)):
            raise FileExistsError(f"Incomplete existing output: {output_dir}")
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint or saved.get("fingerprint_spec") != fingerprint_spec:
            raise RuntimeError(f"Existing output fingerprint mismatch: {output_dir}")
        return saved

    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "version": ANALYSIS_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_spec": fingerprint_spec,
        "selected_region_pairs_path": str(selected_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "regions": {},
    }
    all_delta_rows = []

    for region, selection in selections.items():
        before_image, before_masks, before_rows = _load_phase(selection["before"], region)
        after_image, after_masks, after_rows = _load_phase(selection["after"], region)
        if before_image.shape != after_image.shape:
            raise RuntimeError(
                f"{region}: source image dimensions differ: "
                f"{before_image.shape} vs {after_image.shape}"
            )

        deltas = []
        for identifier in REGION_METRIC_IDS[region]:
            before = before_rows[identifier]
            after = after_rows[identifier]
            if before["status"] != after["status"]:
                status = f'before={before["status"]}; after={after["status"]}'
            else:
                status = before["status"]
            before_value, after_value = before["value"], after["value"]
            delta = (
                float(after_value - before_value)
                if before_value is not None and after_value is not None else None
            )
            row = {
                "region": region,
                "id": identifier,
                "label": before["label"],
                "unit": before["unit"],
                "before": before_value,
                "after": after_value,
                "delta": delta,
                "before_pixels": before["pixels"],
                "after_pixels": after["pixels"],
                "status": status,
                "note": before["note"],
            }
            deltas.append(row)
            all_delta_rows.append(row)

        _save_png(output_dir / f"{region}_before.png", _overlay(before_image, before_masks, region))
        _save_png(output_dir / f"{region}_after.png", _overlay(after_image, after_masks, region))
        summary["regions"][region] = {
            "selection": selection,
            "deltas": deltas,
        }

    _write_csv(output_dir / "feature_deltas.csv", all_delta_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.html").write_text(_report_html(summary), encoding="utf-8")
    return summary
