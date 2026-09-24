"""Region-specific before/after candidate ranking for video makeup analysis.

Geometry ranking remains identical to :mod:`analysis.match_video_rois`, but each
analysis region adds explicit gates for face scale and detected-hand overlap.
No fallback is used when a region has no eligible pair. Hand detection is a
rejection aid, not proof that an unoccluded region is truly clean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from analysis.analyze_cheek_lab import load_image, load_masks
from analysis.appearance_features import build_feature_masks
from analysis.extract_face_rois import RoiConfig, hand_coverage
from analysis.match_video_rois import _frame_features, _pair


REGION_MATCH_VERSION = "region-specific-v3-unique-endpoints"


@dataclass(frozen=True)
class RegionRule:
    mask_names: tuple[str, ...]
    max_face_scale_ratio: float
    max_hand_overlap_ratio: float

    def __post_init__(self):
        if not self.mask_names or any(not isinstance(name, str) or not name for name in self.mask_names):
            raise ValueError("mask_names must contain nonempty strings")
        if (not math.isfinite(self.max_face_scale_ratio)
                or self.max_face_scale_ratio < 1.0):
            raise ValueError("max_face_scale_ratio must be finite and >= 1")
        if (not math.isfinite(self.max_hand_overlap_ratio)
                or not 0 <= self.max_hand_overlap_ratio <= 1):
            raise ValueError("max_hand_overlap_ratio must be inside [0, 1]")


REGION_RULES = {
    "eye_texture": RegionRule(
        ("screen_left_lower_eye_skin", "screen_right_lower_eye_skin"),
        max_face_scale_ratio=1.03,
        max_hand_overlap_ratio=0.01,
    ),
    "lips": RegionRule(
        ("lips", "lip_skin"),
        max_face_scale_ratio=1.10,
        max_hand_overlap_ratio=0.01,
    ),
    "cheeks": RegionRule(
        ("left_cheek", "right_cheek"),
        max_face_scale_ratio=1.10,
        max_hand_overlap_ratio=0.01,
    ),
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


def _feature_masks_for_record(record: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    report = record.get("report")
    if not isinstance(report, dict) or report.get("status") != "needs_review" or report.get("errors") != []:
        raise ValueError(f"{record.get('frame_id')}: ROI report is not reviewable")
    image_path = Path(record["image_path"])
    source = report.get("source", {})
    expected_sha = source.get("sha256")
    if not isinstance(expected_sha, str) or sha256_file(image_path) != expected_sha:
        raise ValueError(f"{record.get('frame_id')}: source image hash mismatch")
    image = load_image(image_path)
    if (source.get("height"), source.get("width")) != image.shape[:2]:
        raise ValueError(f"{record.get('frame_id')}: source dimensions mismatch")
    landmarks = np.asarray(report.get("face_landmarks_normalized"), dtype=np.float64)
    if (landmarks.ndim != 2 or landmarks.shape[0] < 468 or landmarks.shape[1] != 3
            or not np.isfinite(landmarks).all()):
        raise ValueError(f"{record.get('frame_id')}: invalid face landmarks")
    points = landmarks[:, :2] * [image.shape[1], image.shape[0]]
    base_masks = load_masks(Path(record["roi_dir"]) / "roi_masks.npz", image.shape)
    return image, build_feature_masks(image.shape, points, base_masks)


def _region_union(masks: dict[str, np.ndarray], rule: RegionRule, label: str) -> np.ndarray:
    missing = [name for name in rule.mask_names if name not in masks]
    if missing:
        raise ValueError(f"{label}: missing masks: {', '.join(missing)}")
    union = np.zeros(next(iter(masks.values())).shape, dtype=bool)
    for name in rule.mask_names:
        union |= np.asarray(masks[name], dtype=bool)
    if not union.any():
        raise ValueError(f"{label}: region mask is empty")
    return union


def scan_region_occlusion(records: list[dict], model_dir: Path,
                          rules: dict[str, RegionRule] | None = None) -> dict:
    """Re-run hand detection on saved source frames and measure region overlap.

    Existing scans only persist hand overlap for cheek/forehead base ROIs, so
    eye/lip overlap must be computed from the saved frame itself. This function
    never edits the frame extraction or silently treats missing hand metadata as
    zero overlap.
    """
    rules = REGION_RULES if rules is None else rules
    if set(rules) != set(REGION_RULES):
        raise ValueError(f"rules must define exactly: {sorted(REGION_RULES)}")
    model_dir = Path(model_dir)
    model_path = model_dir / "hand_landmarker.task"
    if not model_path.is_file():
        raise FileNotFoundError(f"Hand model missing: {model_path}")

    import mediapipe as mp

    config = RoiConfig()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=4,
        min_hand_detection_confidence=config.min_hand_detection_confidence,
        min_hand_presence_confidence=config.min_hand_presence_confidence,
    )
    entries = {}
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for record in records:
            report = record.get("report", {}) if isinstance(record, dict) else {}
            if report.get("status") != "needs_review" or report.get("errors") != []:
                continue
            frame_id = record.get("frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                raise ValueError("reviewable record is missing frame_id")
            if frame_id in entries:
                raise ValueError(f"duplicate frame_id: {frame_id}")
            image, masks = _feature_masks_for_record(record)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            hands = [np.asarray([[p.x * image.shape[1], p.y * image.shape[0]]
                                 for p in hand], dtype=float)
                     for hand in result.hand_landmarks]
            face_width = float(report["quality"]["face_width_px"])
            if not math.isfinite(face_width) or face_width <= 0:
                raise ValueError(f"{frame_id}: invalid face_width_px")
            margin_px = max(1, round(face_width * config.hand_margin_face_fraction))
            hand_mask, hulls = hand_coverage(hands, image.shape[:2], margin_px)
            region_data = {}
            for region, rule in rules.items():
                union = _region_union(masks, rule, f"{frame_id}/{region}")
                region_data[region] = {
                    "pixels": int(union.sum()),
                    "hand_overlap_ratio": float(np.mean(hand_mask[union] != 0)),
                }
            entries[frame_id] = {
                "timestamp_seconds": float(record["timestamp_seconds"]),
                "detected_hands": len(hands),
                "hand_margin_px": margin_px,
                "hand_hulls_px": [hull.reshape(-1, 2).astype(int).tolist() for hull in hulls],
                "regions": region_data,
            }

    return {
        "schema_version": 1,
        "version": REGION_MATCH_VERSION,
        "hand_model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "mediapipe_version": version("mediapipe"),
            "settings": {
                "min_hand_detection_confidence": config.min_hand_detection_confidence,
                "min_hand_presence_confidence": config.min_hand_presence_confidence,
                "hand_margin_face_fraction": config.hand_margin_face_fraction,
            },
        },
        "rules": {name: asdict(rule) for name, rule in rules.items()},
        "frames": entries,
        "notice": (
            "Detected-hand overlap is a rejection heuristic. A zero overlap does not "
            "prove absence of a hand, tool, shadow, hair, or other occlusion."
        ),
    }


def _all_geometry_pairs(records: list[dict], split_seconds: float,
                        min_gap_seconds: float) -> list[dict]:
    if not math.isfinite(split_seconds) or split_seconds < 0:
        raise ValueError("split_seconds must be finite and nonnegative")
    if not math.isfinite(min_gap_seconds) or min_gap_seconds < 0:
        raise ValueError("min_gap_seconds must be finite and nonnegative")
    frames = []
    for record in records:
        try:
            frames.append(_frame_features(record))
        except (TypeError, ValueError, KeyError, OverflowError, np.linalg.LinAlgError):
            continue
    before = [frame for frame in frames if frame.timestamp < split_seconds]
    after = [frame for frame in frames if frame.timestamp >= split_seconds]
    pairs = []
    for first in before:
        for second in after:
            if second.timestamp - first.timestamp < min_gap_seconds:
                continue
            pair, errors = _pair(first, second)
            if not errors:
                pairs.append(pair)
    pairs.sort(key=lambda item: (
        round(item["score"], 12), item["before_time"], item["after_time"],
        item["before_id"], item["after_id"],
    ))
    return pairs


def filter_region_pairs(geometry_pairs: list[dict], occlusion: dict,
                        rules: dict[str, RegionRule] | None = None,
                        top_k: int = 10, diversity_seconds: float = 15.0) -> dict:
    """Apply region-specific gates with unique endpoints and pair diversity.

    A before frame or after frame can appear in at most one selected pair.
    Separately, when diversity_seconds > 0, candidates whose before and after
    timestamps are both close to an already selected pair are suppressed.
    This prevents duplicate-frame vote inflation without requiring every
    distinct endpoint to be several seconds apart.
    """
    rules = REGION_RULES if rules is None else rules
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(diversity_seconds) or diversity_seconds < 0:
        raise ValueError("diversity_seconds must be finite and nonnegative")
    frames = occlusion.get("frames") if isinstance(occlusion, dict) else None
    if not isinstance(frames, dict):
        raise ValueError("occlusion result is missing frames")

    results = {}
    for region, rule in rules.items():
        eligible, rejected = [], {"face_scale_ratio": 0, "hand_overlap": 0}
        for pair in geometry_pairs:
            before_id, after_id = pair["before_id"], pair["after_id"]
            if before_id not in frames or after_id not in frames:
                raise ValueError(
                    f"{region}: missing hand-occlusion result for {before_id} or {after_id}"
                )
            scale_ratio = float(pair["terms"]["face_scale_ratio"])
            if scale_ratio > rule.max_face_scale_ratio:
                rejected["face_scale_ratio"] += 1
                continue
            overlaps = []
            for frame_id in (before_id, after_id):
                region_entry = frames[frame_id].get("regions", {}).get(region)
                if not isinstance(region_entry, dict) or "hand_overlap_ratio" not in region_entry:
                    raise ValueError(f"{region}: missing overlap for {frame_id}")
                overlap = float(region_entry["hand_overlap_ratio"])
                if not math.isfinite(overlap) or not 0 <= overlap <= 1:
                    raise ValueError(f"{region}: invalid overlap for {frame_id}")
                overlaps.append(overlap)
            if any(value > rule.max_hand_overlap_ratio for value in overlaps):
                rejected["hand_overlap"] += 1
                continue
            candidate = json.loads(json.dumps(pair))
            candidate["region_gate"] = {
                "region": region,
                "max_face_scale_ratio": rule.max_face_scale_ratio,
                "max_hand_overlap_ratio": rule.max_hand_overlap_ratio,
                "before_hand_overlap_ratio": overlaps[0],
                "after_hand_overlap_ratio": overlaps[1],
            }
            eligible.append(candidate)

        selected, diversity_skipped = [], 0
        for pair in eligible:
            if any(
                pair["before_id"] == prior["before_id"]
                or pair["after_id"] == prior["after_id"]
                for prior in selected
            ):
                diversity_skipped += 1
                continue
            if diversity_seconds > 0 and any(
                abs(pair["before_time"] - prior["before_time"]) < diversity_seconds
                and abs(pair["after_time"] - prior["after_time"]) < diversity_seconds
                for prior in selected
            ):
                diversity_skipped += 1
                continue
            if len(selected) >= top_k:
                continue
            selected.append(pair)
        results[region] = {
            "label": REGION_LABELS[region],
            "rule": asdict(rule),
            "ranked_pairs": selected,
            "eligible_before_diversity": len(eligible),
            "diversity_skipped": diversity_skipped,
            "rejected": rejected,
        }

    return {
        "schema_version": 1,
        "version": REGION_MATCH_VERSION,
        "top_k": top_k,
        "diversity_seconds": diversity_seconds,
        "diversity_mode": "unique_endpoints_joint_pair_time",
        "regions": results,
    }


def rank_region_pairs(records: list[dict], split_seconds: float,
                      min_gap_seconds: float, occlusion: dict,
                      rules: dict[str, RegionRule] | None = None,
                      top_k: int = 10, diversity_seconds: float = 15.0) -> dict:
    geometry_pairs = _all_geometry_pairs(records, split_seconds, min_gap_seconds)
    result = filter_region_pairs(
        geometry_pairs, occlusion, rules=rules,
        top_k=top_k, diversity_seconds=diversity_seconds,
    )
    result["geometry_eligible_pairs"] = len(geometry_pairs)
    result["settings"] = {
        "split_seconds": split_seconds,
        "min_gap_seconds": min_gap_seconds,
        "color_used_for_ranking": False,
        "region_hand_overlap_used_as_gate": True,
        "region_scale_used_as_gate": True,
        "thresholds_validated": False,
    }
    return result


def _uri(path: Path, output: Path) -> str:
    path = Path(path).resolve()
    try:
        relative = os.path.relpath(path, output.resolve()).replace("\\", "/")
        return quote(relative, safe="/.")
    except ValueError:
        return path.as_uri()


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _time(seconds: float) -> str:
    centiseconds = round(float(seconds) * 100)
    minutes, remainder = divmod(centiseconds, 6000)
    hours, minutes = divmod(minutes, 60)
    body = f"{minutes:02d}:{remainder / 100:05.2f}"
    return f"{hours:02d}:{body}" if hours else body


def _review_overlay(record: dict, region: str, rule: RegionRule,
                    occlusion_entry: dict) -> np.ndarray:
    image, masks = _feature_masks_for_record(record)
    union = _region_union(masks, rule, region)
    overlay = image.copy()
    fill = overlay.copy()
    fill[union] = (50, 220, 240)
    overlay = cv2.addWeighted(fill, 0.35, overlay, 0.65, 0)
    contours = cv2.findContours(union.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(overlay, contours, -1, (0, 200, 255), 2, cv2.LINE_AA)
    for points in occlusion_entry.get("hand_hulls_px", []):
        hull = np.asarray(points, dtype=np.int32)
        if hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 3:
            raise ValueError(f"{record['frame_id']}: invalid saved hand hull")
        cv2.polylines(overlay, [hull], True, (40, 40, 255), 2, cv2.LINE_AA)
    return overlay


def write_region_report(output_dir: Path, records: list[dict], matching: dict,
                        occlusion: dict, rules: dict[str, RegionRule] | None = None) -> Path:
    """Write a new review report. Existing nonempty output is never overwritten."""
    rules = REGION_RULES if rules is None else rules
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Region report output already exists and is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {record["frame_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate frame IDs in report records")
    frames = occlusion["frames"]

    sections = []
    for region, data in matching["regions"].items():
        rule = rules[region]
        cards = []
        for rank, pair in enumerate(data["ranked_pairs"], 1):
            panel_html = []
            for phase, frame_id in (("before", pair["before_id"]), ("after", pair["after_id"])):
                record = by_id.get(frame_id)
                if record is None:
                    raise ValueError(f"Unknown frame in region pair: {frame_id}")
                filename = f"{region}_rank{rank}_{phase}.png"
                image = _review_overlay(record, region, rule, frames[frame_id])
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise RuntimeError(f"Cannot encode report overlay: {filename}")
                encoded.tofile(output_dir / filename)
                overlap = pair["region_gate"][f"{phase}_hand_overlap_ratio"]
                panel_html.append(
                    f'<div><h4>{phase} {_time(record["timestamp_seconds"])}</h4>'
                    f'<a href="{_escape(filename)}"><img src="{_escape(filename)}" '
                    f'alt="{_escape(region)} {phase} rank {rank}"></a>'
                    f'<p>手の重なり {overlap:.2%}</p></div>'
                )
            scale = pair["terms"]["face_scale_ratio"]
            cards.append(
                f'<article><h3>候補 {rank} ／ 幾何差 {pair["score"]:.4f}</h3>'
                f'<p>顔サイズ比 {scale:.4f}</p>'
                f'<div class="pair">{"".join(panel_html)}</div></article>'
            )
        empty = (
            '<p class="warn">この条件では候補がありません。自動的に閾値を緩めていません。</p>'
            if not cards else ""
        )
        rule_text = (
            f'顔サイズ比 ≤ {rule.max_face_scale_ratio:.3f} ／ '
            f'領域内の検出手重なり ≤ {rule.max_hand_overlap_ratio:.1%}'
        )
        sections.append(
            f'<section><h2>{_escape(data["label"])}</h2><p>{_escape(rule_text)}</p>'
            f'<p>幾何条件通過後の地域候補 {data["eligible_before_diversity"]} 組。'
            f'サイズで除外 {data["rejected"]["face_scale_ratio"]}、'
            f'手重なりで除外 {data["rejected"]["hand_overlap"]}。</p>'
            f'{empty}{"".join(cards)}</section>'
        )

    document = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>部位別 before / after 候補</title>
<style>
body{{font:16px/1.65 system-ui,sans-serif;max-width:1400px;margin:24px auto;padding:0 20px;background:#f5f6f5;color:#24302b}}
section,article{{background:white;border:1px solid #d7ded9;border-radius:10px;padding:18px;margin:18px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{max-width:100%;height:auto}}
.warn{{background:#fff0c8;padding:12px;border-left:5px solid #d79b20}}
.note{{background:#eaf3ef;padding:14px}}@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>部位別 before / after 候補</h1>
<p class="note">黄色の半透明領域が今回比較する部位、赤線が検出手の凸包です。
手検出は見逃し得るため、0%でも手・道具・影が無いことを保証しません。
候補が0件でも閾値を自動的に緩めません。色や美しさは順位に使っていません。</p>
{"".join(sections)}
</body></html>'''
    path = output_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
