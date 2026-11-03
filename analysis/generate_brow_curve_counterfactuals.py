"""Generate five-control-point eyebrow curve counterfactuals.

The script reuses the registered real-brow layer extracted by
``generate_brow_counterfactuals``.  Each eyebrow is represented by five ordered
centre points (head to tail), then warped with a local thin-plate spline (TPS).
Only the eyebrow edit support is changed; the rest of the before image remains
bit-identical.  Variant names describe geometry, not aesthetic quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analysis.appearance_features import BROW_INDICES
from analysis.brow_shape_metrics import measure_brow_shapes, write_shape_csv
from analysis.generate_brow_counterfactuals import (
    composite_brow_layer,
    estimate_registration,
    extract_brow_layer,
    face_crop_bounds,
    load_landmarks,
    make_contact_sheet,
    read_image,
    remove_original_brow,
    sha256_file,
    write_png,
)


VARIANT_ORDER = ("baseline", "soft_arch", "flat", "tail_up", "tail_down")


def five_point_centerline(points: np.ndarray, indices: tuple[int, ...],
                          face_center_x: float) -> np.ndarray:
    """Pair upper/lower brow landmarks into five centres ordered head-to-tail."""
    brow = points[list(indices)].astype(np.float64)
    order = np.argsort(brow[:, 0])
    pairs = brow[order].reshape(5, 2, 2)
    centres = pairs.mean(axis=1)
    # The head is the endpoint nearest the face centre.
    if abs(centres[-1, 0] - face_center_x) < abs(centres[0, 0] - face_center_x):
        centres = centres[::-1]
    return centres


def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    return r2 * np.log(r2 + 1e-12)


def fit_tps_displacement(control_points: np.ndarray, displacements: np.ndarray,
                         smoothing: float = 1e-6) -> dict:
    """Fit a normalized 2-D TPS that predicts pixel displacement."""
    controls = np.asarray(control_points, dtype=np.float64)
    values = np.asarray(displacements, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != 2 or controls.shape != values.shape:
        raise ValueError("TPS controls and displacements must both be Nx2")
    if len(controls) < 3 or not np.isfinite(controls).all() or not np.isfinite(values).all():
        raise ValueError("TPS requires at least three finite controls")
    centre = controls.mean(axis=0)
    scale = float(max(np.ptp(controls[:, 0]), np.ptp(controls[:, 1]), 1.0))
    normalized = (controls - centre) / scale
    delta = normalized[:, None, :] - normalized[None, :, :]
    kernel = _tps_kernel(np.sum(delta * delta, axis=2))
    kernel.flat[::len(controls) + 1] += smoothing
    affine = np.column_stack((np.ones(len(controls)), normalized))
    system = np.block([
        [kernel, affine],
        [affine.T, np.zeros((3, 3), dtype=np.float64)],
    ])
    targets = np.vstack((values, np.zeros((3, 2), dtype=np.float64)))
    try:
        coefficients = np.linalg.solve(system, targets)
    except np.linalg.LinAlgError as error:
        raise ValueError("TPS controls are degenerate") from error
    return {
        "centre": centre,
        "scale": scale,
        "controls": normalized,
        "weights": coefficients[:len(controls)],
        "affine": coefficients[len(controls):],
    }


def evaluate_tps(model: dict, query_points: np.ndarray) -> np.ndarray:
    queries = np.asarray(query_points, dtype=np.float64)
    original_shape = queries.shape
    if queries.shape[-1] != 2:
        raise ValueError("TPS query points must end in an XY pair")
    flat = queries.reshape(-1, 2)
    normalized = (flat - model["centre"]) / model["scale"]
    delta = normalized[:, None, :] - model["controls"][None, :, :]
    radial = _tps_kernel(np.sum(delta * delta, axis=2)) @ model["weights"]
    affine = np.column_stack((np.ones(len(flat)), normalized)) @ model["affine"]
    return (radial + affine).reshape(original_shape)


def variant_destination(source: np.ndarray, variant: str, amplitude: float) -> np.ndarray:
    """Return P1..P5 destinations; negative image Y raises a point."""
    destination = source.copy().astype(np.float64)
    if variant == "baseline":
        return destination
    if variant == "soft_arch":
        destination[:, 1] += amplitude * np.array([0.0, -0.20, -0.55, -1.00, -0.35])
    elif variant == "flat":
        destination[:, 1] = np.linspace(source[0, 1], source[-1, 1], 5)
    elif variant == "tail_up":
        destination[:, 1] += amplitude * np.array([0.0, 0.0, -0.15, -0.55, -1.00])
    elif variant == "tail_down":
        destination[:, 1] += amplitude * np.array([0.0, 0.0, 0.15, 0.55, 1.00])
    else:
        raise ValueError(f"Unknown curve variant: {variant}")
    return destination


def warp_brow_layer(layer: np.ndarray, alpha: np.ndarray, points: np.ndarray,
                    variant: str, amplitude: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Warp both brows independently with fixed crop-corner TPS anchors."""
    height, width = alpha.shape
    face_center_x = float(np.median(points[:, 0]))
    warped_layer = np.zeros_like(layer)
    warped_alpha = np.zeros_like(alpha)
    geometry = []
    grid_x = np.arange(width)[None, :]
    for group_number, indices in enumerate(BROW_INDICES):
        source = five_point_centerline(points, indices, face_center_x)
        destination = variant_destination(source, variant, amplitude)
        group_is_left = float(source[:, 0].mean()) < face_center_x
        side = (grid_x < face_center_x) if group_is_left else (grid_x >= face_center_x)
        support = (alpha >= (0.5 / 255.0)) & side
        ys, xs = np.nonzero(support)
        if not len(xs):
            raise ValueError(f"No extracted brow pixels for group {group_number}")
        padding = max(16, int(np.ceil(amplitude * 2.5)))
        x0, x1 = max(0, int(xs.min()) - padding), min(width, int(xs.max()) + padding + 1)
        y0, y1 = max(0, int(ys.min()) - padding), min(height, int(ys.max()) + padding + 1)
        anchors = np.array([
            [x0, y0], [x1 - 1, y0], [x0, y1 - 1], [x1 - 1, y1 - 1],
        ], dtype=np.float64)
        # cv2.remap needs an inverse map: destination coordinates -> source.
        inverse_controls = np.vstack((destination, anchors))
        inverse_displacements = np.vstack((source - destination, np.zeros((4, 2))))
        model = fit_tps_displacement(inverse_controls, inverse_displacements)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        query = np.column_stack((xx.ravel(), yy.ravel()))
        inverse = query + evaluate_tps(model, query)
        map_x = (inverse[:, 0] - x0).reshape(yy.shape).astype(np.float32)
        map_y = (inverse[:, 1] - y0).reshape(yy.shape).astype(np.float32)
        local_layer = cv2.remap(
            layer[y0:y1, x0:x1], map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        local_alpha = cv2.remap(
            alpha[y0:y1, x0:x1], map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # The two padded crops do not normally overlap; max-alpha selection also
        # makes the behavior deterministic if they ever do.
        target = warped_alpha[y0:y1, x0:x1]
        replace = local_alpha > target
        warped_layer[y0:y1, x0:x1][replace] = local_layer[replace]
        target[replace] = local_alpha[replace]
        geometry.append({
            "group": group_number,
            "side": "screen_left" if group_is_left else "screen_right",
            "source_points_head_to_tail": source.tolist(),
            "destination_points_head_to_tail": destination.tolist(),
            "delta_points": (destination - source).tolist(),
        })
    return warped_layer, np.clip(warped_alpha, 0.0, 1.0), geometry


def control_overlay(image: np.ndarray, geometry: list[dict]) -> np.ndarray:
    overlay = image.copy()
    colors = ((30, 220, 255), (255, 170, 30))
    for group, color in zip(geometry, colors):
        points = np.rint(group["source_points_head_to_tail"]).astype(np.int32)
        cv2.polylines(overlay, [points], False, color, 2, cv2.LINE_AA)
        for index, point in enumerate(points, 1):
            cv2.circle(overlay, tuple(point), 5, color, -1, cv2.LINE_AA)
            cv2.putText(overlay, f"P{index}", (int(point[0] + 6), int(point[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return overlay


def generate(before_path: Path, after_path: Path, before_roi_path: Path,
             after_roi_path: Path, output_dir: Path) -> dict:
    before = read_image(before_path)
    after = read_image(after_path)
    if before.shape != after.shape:
        raise ValueError("Before and after images must have identical dimensions")
    before_points, before_report = load_landmarks(before_roi_path, before.shape)
    after_points, after_report = load_landmarks(after_roi_path, after.shape)
    transform, registration = estimate_registration(after_points, before_points)
    height, width = before.shape[:2]
    aligned_after = cv2.warpAffine(after, transform, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
    layer, alpha, original_brow_mask, extraction = extract_brow_layer(
        before, aligned_after, before_points)
    fill_radius = max(2, round(extraction["face_width_px"] * 0.008))
    clean_base, erase_support = remove_original_brow(before, original_brow_mask, fill_radius)
    amplitude = max(4.0, round(extraction["face_width_px"] * 0.020))

    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = []
    shape_rows = []
    sheet_images = [("original before", before)]
    baseline_geometry = None
    for variant in VARIANT_ORDER:
        warped_layer, warped_alpha, geometry = warp_brow_layer(
            layer, alpha, before_points, variant, amplitude)
        result, edit_support, outside_delta = composite_brow_layer(
            before, clean_base, erase_support, warped_layer, warped_alpha, 0)
        shape_metrics = measure_brow_shapes(
            warped_alpha, geometry, before_points, extraction["face_width_px"])
        filename = f"curve_{variant}.png"
        write_png(output_dir / filename, result)
        changed = np.any(result != before, axis=2)
        conditions.append({
            "variant": variant,
            "file": filename,
            "geometry": geometry,
            "shape_metrics": shape_metrics,
            "edit_support_pixels": int(edit_support.sum()),
            "changed_pixels": int(changed.sum()),
            "outside_support_max_abs_delta": outside_delta,
        })
        sheet_images.append((variant.replace("_", " "), result))
        if variant == "baseline":
            baseline_geometry = geometry
        shape_rows.append((variant, shape_metrics))
    if baseline_geometry is None:
        raise AssertionError("Missing baseline geometry")
    write_png(output_dir / "five_control_points.png", control_overlay(before, baseline_geometry))
    crop = face_crop_bounds(before_points, before.shape[:2])
    write_png(output_dir / "contact_sheet.png", make_contact_sheet(sheet_images, crop))
    write_shape_csv(output_dir / "shape_metrics.csv", shape_rows)

    report = {
        "schema_version": 1,
        "method": "registered-real-brow-five-point-local-tps-v1",
        "interpretation": (
            "Exploratory geometry counterfactuals. Variant names describe curve operations and "
            "do not imply an aesthetic ranking. P1 is the brow head and P5 the tail."
        ),
        "sources": {
            "before": {"path": str(before_path.resolve()), "sha256": sha256_file(before_path),
                       "roi_path": str(before_roi_path.resolve()), "roi_sha256": sha256_file(before_roi_path),
                       "roi_status": before_report.get("status")},
            "after": {"path": str(after_path.resolve()), "sha256": sha256_file(after_path),
                      "roi_path": str(after_roi_path.resolve()), "roi_sha256": sha256_file(after_roi_path),
                      "roi_status": after_report.get("status")},
        },
        "registration": {"after_to_before_affine": transform.tolist(), **registration},
        "extraction": extraction,
        "curve": {
            "control_points_per_brow": 5,
            "order": "P1 brow head to P5 brow tail",
            "amplitude_px": amplitude,
            "amplitude_face_width_fraction": amplitude / extraction["face_width_px"],
            "warp": "local inverse thin-plate spline with four fixed crop-corner anchors",
        },
        "conditions": conditions,
        "implementation_sha256": sha256_file(Path(__file__)),
        "artifacts": {},
    }
    for path in sorted(output_dir.glob("*.png")):
        report["artifacts"][path.name] = sha256_file(path)
    report["artifacts"]["shape_metrics.csv"] = sha256_file(output_dir / "shape_metrics.csv")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-image", type=Path, required=True)
    parser.add_argument("--after-image", type=Path, required=True)
    parser.add_argument("--before-roi", type=Path, required=True)
    parser.add_argument("--after-roi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate(args.before_image, args.after_image, args.before_roi,
                      args.after_roi, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "registration": report["registration"],
        "curve": report["curve"],
        "conditions": [{key: row[key] for key in (
            "variant", "file", "changed_pixels", "outside_support_max_abs_delta")}
            for row in report["conditions"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
