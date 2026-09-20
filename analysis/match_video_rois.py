"""Rank reviewable video frames by geometry; never by cosmetic color changes.

This module does not read images or RGB/Lab fields. It does not establish identity,
that a frame is genuinely before/after treatment, or that an ROI is unobstructed.
Callers must first select the semantic time windows, and people must review the
resulting candidates. Existing extraction status/errors supply hand/clipping gates.

All limits and score scales below are experimental, adjustable module constants,
not calibrated confidence levels. The score is a sum of seven nonnegative terms;
one unit means one gate limit for pose, scale, or expression, or ROI shape RMS .05.
The forehead participates only as geometry. Its Lab value is neither a rank term
nor an eligibility gate because the forehead may itself receive makeup.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from analysis.extract_face_rois import ROI_INDICES, polygons_from_landmarks


MATCH_VERSION = "geometry-only-v1"
POSE_GAP_LIMITS_DEGREES = {"yaw": 10.0, "pitch": 10.0, "roll": 12.0}
MAX_FACE_SCALE_RATIO = 1.5
MAX_EYE_APERTURE_GAP = 0.04
MAX_MOUTH_OPENING_GAP = 0.08
ROI_SHAPE_SCORE_SCALE = 0.05
ROI_NAMES = ("left_cheek", "right_cheek", "forehead")
NOTICE = (
    "Geometry-only ranking; lower is more similar, not higher confidence. "
    "RGB, Lab, forehead brightness and cosmetic color changes are not used for "
    "ranking or pair gates. Frame status/errors retain extraction hand/clipping "
    "checks. Landmark shape cannot establish that two frames show the same person. "
    "Treatment timing, identity and unobstructed ROI placement require human "
    "review. Limits and score weights are experimental."
)


def _finite_number(value, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def similarity_procrustes_rms(before: np.ndarray, after: np.ndarray) -> float:
    """RMS after translation/rotation/uniform-scale alignment, no reflections.

    Each centered cloud is normalized to RMS radius one. Point order is retained,
    so this does not silently find new correspondences or permute polygon vertices.
    """
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if (before.ndim != 2 or before.shape[1] != 2 or before.shape != after.shape
            or len(before) < 3 or not np.isfinite(before).all() or not np.isfinite(after).all()):
        raise ValueError("Procrustes requires matching finite Nx2 arrays with at least 3 points")
    normalized = []
    for points in (before, after):
        centered = points - points.mean(axis=0)
        scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        if scale <= 1e-9 or np.linalg.matrix_rank(centered) < 2:
            raise ValueError("Degenerate ROI geometry")
        normalized.append(centered / scale)
    source, target = normalized
    u, _, vt = np.linalg.svd(source.T @ target)
    correction = np.eye(2)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ correction @ vt
    residual = float(np.sqrt(np.mean(np.sum((source @ rotation - target) ** 2, axis=1))))
    return 0.0 if residual < 1e-12 else residual


@dataclass
class _Frame:
    frame_id: str
    timestamp: float
    pose: dict[str, float]
    face_width: float
    roi_points: np.ndarray
    eye_apertures: tuple[float, float]
    mouth_opening: float
    canonical_cheek_a_on_left: bool


def _frame_features(record: dict) -> _Frame:
    if not isinstance(record, dict):
        raise ValueError("record is not an object")
    frame_id = record.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("missing or invalid frame_id")
    timestamp = _finite_number(record.get("timestamp_seconds"), "timestamp_seconds")
    if timestamp < 0:
        raise ValueError("timestamp_seconds must be nonnegative")
    report = record.get("report")
    if not isinstance(report, dict):
        raise ValueError("missing or invalid ROI report")
    if report.get("status") != "needs_review":
        raise ValueError("ROI report status is not needs_review")
    if report.get("errors") != []:
        raise ValueError("ROI report has errors or is missing its errors list")
    source, quality = report.get("source"), report.get("quality")
    if not isinstance(source, dict) or not isinstance(quality, dict):
        raise ValueError("missing source dimensions or quality metadata")
    width = _finite_number(source.get("width"), "image width", positive=True)
    height = _finite_number(source.get("height"), "image height", positive=True)
    if width != int(width) or height != int(height):
        raise ValueError("image dimensions must be integers")
    landmarks = np.asarray(report.get("face_landmarks_normalized"), dtype=float)
    if (landmarks.ndim != 2 or landmarks.shape[0] < 468 or landmarks.shape[1] != 3
            or not np.isfinite(landmarks).all()):
        raise ValueError("missing or malformed finite Nx3 face landmarks")
    points = landmarks[:, :2] * (width, height)
    face_width = _finite_number(quality.get("face_width_px"), "face_width_px", positive=True)
    measured_width = float(np.linalg.norm(points[454] - points[234]))
    if measured_width <= 1e-6 or not math.isclose(face_width, measured_width, rel_tol=1e-5, abs_tol=1e-4):
        raise ValueError("face_width_px is inconsistent with landmarks")
    raw_pose = quality.get("pose_degrees")
    if not isinstance(raw_pose, dict):
        raise ValueError("missing pose_degrees")
    pose = {axis: _finite_number(raw_pose.get(axis), f"pose {axis}") for axis in POSE_GAP_LIMITS_DEGREES}
    if any(abs(value) > 180 for value in pose.values()):
        raise ValueError("pose angle outside [-180, 180]")
    config = report.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("invalid extraction config")
    polygon_scale = _finite_number(config.get("polygon_scale", 0.88), "polygon_scale", positive=True)
    if polygon_scale > 1:
        raise ValueError("polygon_scale must be at most one")
    derived = polygons_from_landmarks(points, polygon_scale)
    rois = report.get("rois")
    if not isinstance(rois, dict):
        raise ValueError("missing ROI polygons")
    for name in ROI_NAMES:
        roi = rois.get(name)
        if not isinstance(roi, dict):
            raise ValueError(f"missing {name} polygon")
        saved = np.asarray(roi.get("polygon_px"), dtype=float)
        if (saved.shape != derived[name].shape or not np.isfinite(saved).all()
                or not np.allclose(saved, derived[name], rtol=1e-6, atol=1e-3)):
            raise ValueError(f"{name} polygon is malformed or inconsistent with landmarks")
        if np.any(saved < 0) or np.any(saved[:, 0] >= width) or np.any(saved[:, 1] >= height):
            raise ValueError(f"{name} polygon is outside the image")
        x, y = saved.T
        if abs(float(x @ np.roll(y, 1) - y @ np.roll(x, 1))) < 2:
            raise ValueError(f"{name} polygon is degenerate")
    cloud = np.vstack([derived[name] for name in ROI_NAMES])
    # Validate point-cloud rank once per frame, not in every candidate pairing.
    if np.linalg.matrix_rank(cloud - cloud.mean(axis=0)) < 2:
        raise ValueError("Degenerate ROI geometry")
    eyes = []
    for corners, lids in (
        ((33, 133), ((159, 145), (158, 153))),
        ((362, 263), ((386, 374), (385, 380))),
    ):
        if np.linalg.norm(points[corners[0]] - points[corners[1]]) < 1e-6:
            raise ValueError("Degenerate eye landmarks")
        aperture = float(np.mean([np.linalg.norm(points[top] - points[bottom]) for top, bottom in lids])) / face_width
        center_x = float(points[list(corners), 0].mean())
        eyes.append((center_x, aperture))
    eyes.sort(key=lambda item: item[0])
    cheek_a_x = float(points[list(ROI_INDICES["cheek_a"]), 0].mean())
    cheek_b_x = float(points[list(ROI_INDICES["cheek_b"]), 0].mean())
    return _Frame(
        frame_id, timestamp, pose, face_width, cloud,
        (eyes[0][1], eyes[1][1]), float(np.linalg.norm(points[13] - points[14])) / face_width,
        cheek_a_x < cheek_b_x,
    )


def _pair(before: _Frame, after: _Frame) -> tuple[dict | None, list[str]]:
    gaps = {axis: abs(after.pose[axis] - before.pose[axis]) for axis in POSE_GAP_LIMITS_DEGREES}
    scale_ratio = max(after.face_width / before.face_width, before.face_width / after.face_width)
    eye_gaps = [abs(a - b) for a, b in zip(before.eye_apertures, after.eye_apertures)]
    eye_gap = max(eye_gaps)
    mouth_gap = abs(after.mouth_opening - before.mouth_opening)
    mirrored = before.canonical_cheek_a_on_left != after.canonical_cheek_a_on_left
    reasons = [f"{axis}_gap" for axis, limit in POSE_GAP_LIMITS_DEGREES.items() if gaps[axis] > limit]
    if scale_ratio > MAX_FACE_SCALE_RATIO:
        reasons.append("face_scale_ratio")
    if eye_gap > MAX_EYE_APERTURE_GAP:
        reasons.append("eye_aperture_gap")
    if mouth_gap > MAX_MOUTH_OPENING_GAP:
        reasons.append("mouth_opening_gap")
    if mirrored:
        reasons.append("mirrored_landmark_order_changed")
    if reasons:
        return None, reasons
    shape_rms = similarity_procrustes_rms(before.roi_points, after.roi_points)
    log_scale = math.log(scale_ratio)
    contributions = {axis: gaps[axis] / limit for axis, limit in POSE_GAP_LIMITS_DEGREES.items()}
    contributions.update(
        face_scale=log_scale / math.log(MAX_FACE_SCALE_RATIO),
        roi_shape=shape_rms / ROI_SHAPE_SCORE_SCALE,
        eye_aperture=eye_gap / MAX_EYE_APERTURE_GAP,
        mouth_opening=mouth_gap / MAX_MOUTH_OPENING_GAP,
    )
    return {
        "before_id": before.frame_id,
        "after_id": after.frame_id,
        "before_time": before.timestamp,
        "after_time": after.timestamp,
        "score": float(sum(contributions.values())),
        "terms": {
            **{f"{axis}_gap_degrees": value for axis, value in gaps.items()},
            "face_scale_ratio": scale_ratio,
            "log_face_scale_ratio": log_scale,
            "roi_procrustes_rms": shape_rms,
            "eye_aperture_gap": eye_gap,
            "mouth_opening_gap": mouth_gap,
            "contributions": contributions,
        },
        "diagnostics": {
            "time_gap_seconds": after.timestamp - before.timestamp,
            "before_expression": {"screen_eye_apertures": list(before.eye_apertures), "mouth_opening": before.mouth_opening},
            "after_expression": {"screen_eye_apertures": list(after.eye_apertures), "mouth_opening": after.mouth_opening},
            "screen_eye_aperture_gaps": eye_gaps,
            "mirrored_landmark_order_changed": mirrored,
            "alignment_allows_reflection": False,
            "needs_human_review": True,
            "identity_verified": False,
        },
    }, []


def rank_pairs(
    records: list[dict],
    split_seconds: float,
    min_gap_seconds: float,
    top_k: int = 10,
    diversity_seconds: float = 15,
) -> dict:
    """Rank before (< split) / after (>= split) review candidates by geometry.

    A candidate is redundant only when *both* timestamps are closer than
    ``diversity_seconds`` to an already selected pair. Zero disables suppression.
    Ties sort by time and frame IDs. Input records and reports are never modified.
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    split_seconds = _finite_number(split_seconds, "split_seconds")
    min_gap_seconds = _finite_number(min_gap_seconds, "min_gap_seconds")
    diversity_seconds = _finite_number(diversity_seconds, "diversity_seconds")
    if split_seconds < 0 or min_gap_seconds < 0 or diversity_seconds < 0:
        raise ValueError("time settings must be nonnegative")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    identifiers = Counter(record.get("frame_id") for record in records
                          if isinstance(record, dict) and isinstance(record.get("frame_id"), str))
    frames, rejected_frames = [], []
    for index, record in enumerate(records):
        try:
            if isinstance(record, dict) and isinstance(record.get("frame_id"), str) and identifiers[record["frame_id"]] > 1:
                raise ValueError("duplicate frame_id")
            frames.append(_frame_features(record))
        except (TypeError, ValueError, KeyError, OverflowError, np.linalg.LinAlgError) as exc:
            identifier = record.get("frame_id") if isinstance(record, dict) else None
            rejected_frames.append({"record_index": index, "frame_id": identifier if isinstance(identifier, str) else None,
                                    "reason": str(exc)})
    before_frames = [frame for frame in frames if frame.timestamp < split_seconds]
    after_frames = [frame for frame in frames if frame.timestamp >= split_seconds]
    ranked, rejected_pairs, reasons = [], 0, Counter()
    for before in before_frames:
        for after in after_frames:
            if after.timestamp - before.timestamp < min_gap_seconds:
                rejected_pairs += 1
                reasons["minimum_time_gap"] += 1
                continue
            pair, errors = _pair(before, after)
            if errors:
                rejected_pairs += 1
                reasons.update(errors)
            else:
                ranked.append(pair)
    ranked.sort(key=lambda pair: (round(pair["score"], 12), pair["before_time"], pair["after_time"], pair["before_id"], pair["after_id"]))
    selected, diversity_skipped, top_k_omitted = [], 0, 0
    for pair in ranked:
        if diversity_seconds > 0 and any(
            abs(pair["before_time"] - prior["before_time"]) < diversity_seconds
            and abs(pair["after_time"] - prior["after_time"]) < diversity_seconds
            for prior in selected
        ):
            diversity_skipped += 1
        elif len(selected) >= top_k:
            top_k_omitted += 1
        else:
            selected.append(pair)
    return {
        "schema_version": 1,
        "method": MATCH_VERSION,
        "notice": NOTICE,
        "settings": {
            "split_seconds": split_seconds,
            "before_rule": "timestamp < split_seconds",
            "after_rule": "timestamp >= split_seconds",
            "min_gap_seconds": min_gap_seconds,
            "top_k": top_k,
            "diversity_seconds": diversity_seconds,
            "pose_gap_limits_degrees": dict(POSE_GAP_LIMITS_DEGREES),
            "max_face_scale_ratio": MAX_FACE_SCALE_RATIO,
            "max_eye_aperture_gap_face_width": MAX_EYE_APERTURE_GAP,
            "max_mouth_opening_gap_face_width": MAX_MOUTH_OPENING_GAP,
            "roi_shape_score_scale": ROI_SHAPE_SCORE_SCALE,
            "shape_rms_definition": "RMS distance after centering, unit RMS radius normalization, and proper rotation",
            "expression_units": "pixel distances divided by face width in pixels",
            "color_used_for_ranking": False,
            "forehead_L_used_for_pair_gate": False,
            "reflection_alignment_allowed": False,
            "reject_mirrored_landmark_order_change": True,
            "thresholds_validated": False,
        },
        "ranked_pairs": selected,
        "stats": {
            "total_frames": len(records),
            "accepted_frames": len(frames),
            "rejected_frames": rejected_frames,
            "rejected_frame_count": len(rejected_frames),
            "before_frames": len(before_frames),
            "after_frames": len(after_frames),
            "candidate_pairs": len(before_frames) * len(after_frames),
            "rejected_pairs": rejected_pairs,
            "pair_rejection_counts": dict(sorted(reasons.items())),
            "eligible_pairs": len(ranked),
            "diversity_skipped_pairs": diversity_skipped,
            "top_k_omitted_pairs": top_k_omitted,
            "selected_pairs": len(selected),
        },
    }
