"""Descriptive eyebrow-shape metrics for five-point curve stimuli.

These metrics operationalize section ``A. 眉の形状`` of issue #5. They are
geometry and mask measurements, not a naturalness, attractiveness, or beauty
score. Screen-left/right values use a mirrored semantic eye frame: X increases
from the inner eye corner outward, and Y increases toward the forehead.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from analysis.appearance_features import UPPER_EYE_INDICES


SIDE_NAMES = ("screen_left", "screen_right")


def _unit(vector: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        raise ValueError(f"Degenerate {name}")
    return vector / length, length


def centerline_metrics(centerline: np.ndarray, eye_inner: np.ndarray,
                       eye_outer: np.ndarray, face_up: np.ndarray) -> dict:
    """Measure one P1-to-P5 centreline in a semantic eye coordinate frame."""
    curve = np.asarray(centerline, dtype=np.float64)
    eye_inner = np.asarray(eye_inner, dtype=np.float64)
    eye_outer = np.asarray(eye_outer, dtype=np.float64)
    if curve.shape != (5, 2) or not np.isfinite(curve).all():
        raise ValueError("Expected five finite eyebrow centre points")
    outward, eye_width = _unit(eye_outer - eye_inner, "eye width")
    up, _ = _unit(np.asarray(face_up, dtype=np.float64), "face-up direction")
    eye_mid = (eye_inner + eye_outer) / 2.0

    segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    arc_length = float(cumulative[-1])
    if arc_length < 1e-6:
        raise ValueError("Degenerate eyebrow centreline")

    def local(point: np.ndarray) -> dict:
        return {
            "image_x_px": float(point[0]),
            "image_y_px": float(point[1]),
            "outward_x_eye_widths": float(np.dot(point - eye_inner, outward) / eye_width),
            "up_y_eye_widths": float(np.dot(point - eye_mid, up) / eye_width),
        }

    local_up = np.array([np.dot(point - eye_mid, up) for point in curve])
    apex_index = int(np.argmax(local_up))
    chord = curve[-1] - curve[0]
    chord_unit, chord_length = _unit(chord, "brow chord")
    relative = curve - curve[0]
    perpendicular = relative - np.outer(relative @ chord_unit, chord_unit)
    max_chord_distance = float(np.linalg.norm(perpendicular, axis=1).max())
    outward_delta = float(np.dot(chord, outward))
    upward_delta = float(np.dot(chord, up))
    return {
        "length_px": arc_length,
        "length_eye_widths": arc_length / eye_width,
        "chord_length_px": chord_length,
        "chord_length_eye_widths": chord_length / eye_width,
        "head": local(curve[0]),
        "apex": local(curve[apex_index]),
        "tail": local(curve[-1]),
        "apex_control_point": apex_index + 1,
        "apex_arc_fraction": float(cumulative[apex_index] / arc_length),
        "max_chord_distance_px": max_chord_distance,
        "max_chord_distance_eye_widths": max_chord_distance / eye_width,
        "head_to_tail_slope_degrees": float(np.degrees(np.arctan2(upward_delta, outward_delta))),
        "eye_width_px": eye_width,
    }


def mask_metrics(alpha: np.ndarray, centerline: np.ndarray, eye_inner: np.ndarray,
                 eye_outer: np.ndarray, face_width: float) -> dict:
    """Measure effective area, average width, and head-to-tail taper."""
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 2 or not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("Eyebrow alpha must be a finite 2-D array in [0, 1]")
    outward, eye_width = _unit(np.asarray(eye_outer) - np.asarray(eye_inner), "eye width")
    length = float(np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())
    effective_area = float(alpha.sum())
    average_width = effective_area / max(length, 1e-6)

    ys, xs = np.nonzero(alpha > 0)
    weights = alpha[ys, xs]
    pixels = np.column_stack((xs, ys)).astype(np.float64)
    head = np.asarray(centerline[0], dtype=np.float64)
    longitudinal_extent = float(np.dot(np.asarray(centerline[-1]) - head, outward))
    if longitudinal_extent <= 1e-6:
        raise ValueError("Eyebrow tail is not outward from its head")
    t = ((pixels - head) @ outward) / longitudinal_extent
    bin_widths = []
    for low, high in zip(np.linspace(0, 0.8, 5), np.linspace(0.2, 1.0, 5)):
        selected = (t >= low) & (t < high if high < 1.0 else t <= high)
        area = float(weights[selected].sum())
        bin_widths.append(area / max(longitudinal_extent * (high - low), 1e-6))
    head_thickness = float(np.mean(bin_widths[:2]))
    tail_thickness = float(np.mean(bin_widths[-2:]))
    mean_bin_width = float(np.mean(bin_widths))
    normalized_slope = float(np.polyfit(np.linspace(0.1, 0.9, 5), bin_widths, 1)[0]
                             / max(mean_bin_width, 1e-6))
    tail_head_ratio = tail_thickness / max(head_thickness, 1e-6)
    return {
        "effective_area_px2": effective_area,
        "effective_area_eye_width2": effective_area / (eye_width * eye_width),
        "effective_area_face_width2": effective_area / (face_width * face_width),
        "average_width_px": average_width,
        "average_width_eye_widths": average_width / eye_width,
        "average_width_face_widths": average_width / face_width,
        "thickness_bins_px_head_to_tail": [float(value) for value in bin_widths],
        "head_thickness_px": head_thickness,
        "tail_thickness_px": tail_thickness,
        "tail_to_head_thickness_ratio": tail_head_ratio,
        "head_to_tail_thickness_decay_fraction": 1.0 - tail_head_ratio,
        "normalized_thickness_trend_per_brow": normalized_slope,
        "binary_area_px2_at_alpha_0_5": int((alpha >= 0.5).sum()),
    }


def measure_brow_shapes(alpha: np.ndarray, geometry: list[dict], landmarks: np.ndarray,
                        face_width: float) -> dict:
    """Measure both eyebrows and return mirrored-semantic left/right differences."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 363:
        raise ValueError("Expected a full 2-D face landmark array")
    face_up, _ = _unit(points[10] - points[152], "face vertical")
    face_center_x = float(np.median(points[:, 0]))
    grid_x = np.arange(alpha.shape[1])[None, :]
    sides = {}
    by_group = {int(row["group"]): row for row in geometry}
    for group_number, eye_indices in enumerate(UPPER_EYE_INDICES):
        row = by_group[group_number]
        side_name = row["side"]
        centreline = np.asarray(row["destination_points_head_to_tail"], dtype=np.float64)
        corners = points[[eye_indices[0], eye_indices[-1]]]
        distances = np.abs(corners[:, 0] - face_center_x)
        eye_inner = corners[int(np.argmin(distances))]
        eye_outer = corners[int(np.argmax(distances))]
        side_selector = ((grid_x < face_center_x) if side_name == "screen_left"
                         else (grid_x >= face_center_x))
        side_alpha = np.where(side_selector, alpha, 0.0)
        sides[side_name] = {
            **centerline_metrics(centreline, eye_inner, eye_outer, face_up),
            **mask_metrics(side_alpha, centreline, eye_inner, eye_outer, face_width),
        }
    if set(sides) != set(SIDE_NAMES):
        raise ValueError("Expected one screen-left and one screen-right eyebrow")

    left, right = (sides[name] for name in SIDE_NAMES)
    scalar_keys = (
        "length_eye_widths", "effective_area_eye_width2", "average_width_eye_widths",
        "apex_arc_fraction", "max_chord_distance_eye_widths",
        "head_to_tail_slope_degrees", "tail_to_head_thickness_ratio",
        "head_to_tail_thickness_decay_fraction", "normalized_thickness_trend_per_brow",
    )
    differences = {
        key: {
            "absolute_difference": abs(float(left[key]) - float(right[key])),
            "signed_screen_left_minus_right": float(left[key]) - float(right[key]),
        }
        for key in scalar_keys
    }
    for point_name in ("head", "apex", "tail"):
        for coordinate in ("outward_x_eye_widths", "up_y_eye_widths"):
            key = f"{point_name}_{coordinate}"
            differences[key] = {
                "absolute_difference": abs(float(left[point_name][coordinate])
                                           - float(right[point_name][coordinate])),
                "signed_screen_left_minus_right": float(left[point_name][coordinate])
                                                   - float(right[point_name][coordinate]),
            }
    return {
        "coordinate_definition": (
            "Per-eye mirrored frame: outward X starts at the inner eye corner; upward Y starts "
            "at the eye midpoint. Both are normalized by that eye's corner-to-corner width."
        ),
        "mask_definition": "Effective area uses alpha sum; binary review area uses alpha >= 0.5.",
        "sides": sides,
        "left_right_differences": differences,
    }


def write_shape_csv(path: Path, rows: list[tuple[str, dict]]) -> None:
    """Write one flat row per variant and screen side."""
    fields = (
        "variant", "side", "length_px", "length_eye_widths", "effective_area_px2",
        "effective_area_eye_width2", "average_width_px", "average_width_eye_widths",
        "head_x_eye_widths", "head_y_eye_widths", "apex_x_eye_widths", "apex_y_eye_widths",
        "tail_x_eye_widths", "tail_y_eye_widths", "apex_control_point", "apex_arc_fraction",
        "max_chord_distance_px", "max_chord_distance_eye_widths",
        "head_to_tail_slope_degrees", "head_thickness_px", "tail_thickness_px",
        "tail_to_head_thickness_ratio", "head_to_tail_thickness_decay_fraction",
        "normalized_thickness_trend_per_brow",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant, measurement in rows:
            for side, values in measurement["sides"].items():
                writer.writerow({
                    "variant": variant, "side": side,
                    "length_px": values["length_px"],
                    "length_eye_widths": values["length_eye_widths"],
                    "effective_area_px2": values["effective_area_px2"],
                    "effective_area_eye_width2": values["effective_area_eye_width2"],
                    "average_width_px": values["average_width_px"],
                    "average_width_eye_widths": values["average_width_eye_widths"],
                    "head_x_eye_widths": values["head"]["outward_x_eye_widths"],
                    "head_y_eye_widths": values["head"]["up_y_eye_widths"],
                    "apex_x_eye_widths": values["apex"]["outward_x_eye_widths"],
                    "apex_y_eye_widths": values["apex"]["up_y_eye_widths"],
                    "tail_x_eye_widths": values["tail"]["outward_x_eye_widths"],
                    "tail_y_eye_widths": values["tail"]["up_y_eye_widths"],
                    "apex_control_point": values["apex_control_point"],
                    "apex_arc_fraction": values["apex_arc_fraction"],
                    "max_chord_distance_px": values["max_chord_distance_px"],
                    "max_chord_distance_eye_widths": values["max_chord_distance_eye_widths"],
                    "head_to_tail_slope_degrees": values["head_to_tail_slope_degrees"],
                    "head_thickness_px": values["head_thickness_px"],
                    "tail_thickness_px": values["tail_thickness_px"],
                    "tail_to_head_thickness_ratio": values["tail_to_head_thickness_ratio"],
                    "head_to_tail_thickness_decay_fraction": values[
                        "head_to_tail_thickness_decay_fraction"],
                    "normalized_thickness_trend_per_brow": values[
                        "normalized_thickness_trend_per_brow"],
                })
