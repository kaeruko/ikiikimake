"""Create deterministic eyebrow-height counterfactuals from a real before/after pair.

This is deliberately not a generative model.  The after frame is registered to
the before frame and locally colour matched.  The old brow is removed with local
boundary inpainting, then one extracted real-brow layer is placed at the requested
height.  Every pixel outside the explicit edit support remains bit-identical.

The result is an exploratory stimulus, not a claim that anatomy or perceived
attractiveness has been isolated.  In particular, a large vertical move can show
the faint natural brow underneath the transferred makeup layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from analysis.appearance_features import BROW_INDICES, FACE_OVAL_INDICES


REGISTRATION_INDICES = (
    1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 168, 195, 197,
    278, 327, 33, 133, 263, 362, 234, 454, 10, 152,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode PNG: {path}")
    encoded.tofile(path)


def load_landmarks(path: Path, image_shape: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    height, width = image_shape[:2]
    source = report.get("source", {})
    if source.get("width") != width or source.get("height") != height:
        raise ValueError(f"ROI dimensions do not match image: {path}")
    points = np.asarray(report.get("face_landmarks_normalized"), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] < 2:
        raise ValueError(f"Expected at least 468 face landmarks: {path}")
    points = points[:, :2] * np.array([width, height], dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite face landmarks: {path}")
    return points, report


def estimate_registration(after_points: np.ndarray, before_points: np.ndarray) -> tuple[np.ndarray, dict]:
    indices = np.asarray(REGISTRATION_INDICES, dtype=np.int32)
    transform, inliers = cv2.estimateAffinePartial2D(
        after_points[indices], before_points[indices], method=cv2.RANSAC,
        ransacReprojThreshold=4.0, maxIters=5000, confidence=0.999,
        refineIters=25,
    )
    if transform is None or not np.isfinite(transform).all():
        raise ValueError("Could not estimate after-to-before facial registration")
    aligned = cv2.transform(after_points[None].astype(np.float32), transform)[0]
    errors = np.linalg.norm(aligned[indices] - before_points[indices], axis=1)
    inlier_values = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(indices), bool)
    metrics = {
        "control_points": int(len(indices)),
        "ransac_inliers": int(inlier_values.sum()),
        "median_control_error_px": float(np.median(errors)),
        "p90_control_error_px": float(np.percentile(errors, 90)),
    }
    return transform, metrics


def _face_width(points: np.ndarray) -> float:
    return float(np.max(points[list(FACE_OVAL_INDICES), 0]) - np.min(points[list(FACE_OVAL_INDICES), 0]))


def brow_band_mask(points: np.ndarray, shape: tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for indices in BROW_INDICES:
        vertices = np.rint(points[list(indices)]).astype(np.int32)
        cv2.fillPoly(mask, [vertices], 255)
        cv2.polylines(mask, [vertices], True, 255, max(1, radius), cv2.LINE_AA)
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def robust_colour_match(after_bgr: np.ndarray, before_bgr: np.ndarray,
                        reference_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Match local skin colour in Lab with robust percentile spans."""
    if int(reference_mask.sum()) < 100:
        raise ValueError("Too few reference skin pixels for local colour matching")
    after_lab = cv2.cvtColor(after_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    before_lab = cv2.cvtColor(before_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    corrected = after_lab.copy()
    channels = []
    for channel in range(3):
        source = after_lab[:, :, channel][reference_mask]
        target = before_lab[:, :, channel][reference_mask]
        source_span = float(np.percentile(source, 90) - np.percentile(source, 10))
        target_span = float(np.percentile(target, 90) - np.percentile(target, 10))
        gain = float(np.clip(target_span / max(source_span, 1e-4), 0.5, 2.0))
        offset = float(np.median(target) - gain * np.median(source))
        corrected[:, :, channel] = corrected[:, :, channel] * gain + offset
        channels.append({"gain": gain, "offset": offset})
    corrected[:, :, 0] = np.clip(corrected[:, :, 0], 0.0, 100.0)
    corrected[:, :, 1:] = np.clip(corrected[:, :, 1:], -127.0, 127.0)
    corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
    return np.clip(corrected_bgr * 255.0, 0, 255).astype(np.float32), {"lab_channels": channels}


def detect_original_brow(before_bgr: np.ndarray, points: np.ndarray,
                         face_width: float) -> tuple[np.ndarray, dict]:
    """Detect only dark natural-brow pixels inside the landmark brow band."""
    height, width = before_bgr.shape[:2]
    search_radius = max(2, round(face_width * 0.008))
    search = brow_band_mask(points, (height, width), search_radius)
    lightness = cv2.cvtColor(before_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)[:, :, 0]
    # A broad inpaint is used only as a skin-colour estimate for detection; it
    # is never copied into the result.  This avoids treating nearby eye/hair
    # pixels as the noise distribution, which made the threshold too strict.
    estimate_radius = max(3, round(face_width * 0.012))
    estimated_skin = cv2.inpaint(before_bgr, search.astype(np.uint8) * 255,
                                 float(estimate_radius), cv2.INPAINT_TELEA)
    estimated_lightness = cv2.cvtColor(
        estimated_skin.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)[:, :, 0]
    darkness = estimated_lightness - lightness
    threshold = 6.0
    detected = ((darkness >= threshold) & search).astype(np.uint8)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    detected = cv2.morphologyEx(detected, cv2.MORPH_CLOSE, small, iterations=1)
    detected = cv2.dilate(detected, small, iterations=1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(detected, 8)
    cleaned = np.zeros_like(detected)
    kept = 0
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= 6:
            cleaned[labels == label] = 1
            kept += 1
    if not cleaned.any():
        raise ValueError("Could not detect original brow pixels for replacement")
    return cleaned.astype(bool), {
        "original_brow_search_pixels": int(search.sum()),
        "original_brow_darkness_threshold_L": threshold,
        "original_brow_detection_estimate_radius_px": estimate_radius,
        "original_brow_components": kept,
    }


def extract_brow_layer(before_bgr: np.ndarray, aligned_after_bgr: np.ndarray,
                       before_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return a real after-brow layer, its alpha, and the original-brow mask."""
    height, width = before_bgr.shape[:2]
    face_width = _face_width(before_points)
    band_radius = max(3, round(face_width * 0.012))
    band = brow_band_mask(before_points, (height, width), band_radius)
    ring_outer_radius = max(12, round(face_width * 0.055))
    ring_inner_radius = max(band_radius + 2, round(face_width * 0.022))
    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_outer_radius + 1, 2 * ring_outer_radius + 1))
    inner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_inner_radius + 1, 2 * ring_inner_radius + 1))
    reference = (
        cv2.dilate(band.astype(np.uint8), outer_kernel).astype(bool)
        & ~cv2.dilate(band.astype(np.uint8), inner_kernel).astype(bool)
    )
    corrected_after, colour_metrics = robust_colour_match(aligned_after_bgr, before_bgr, reference)
    before_float = before_bgr.astype(np.float32)
    before_lab = cv2.cvtColor(before_float / 255.0, cv2.COLOR_BGR2LAB)
    after_lab = cv2.cvtColor(corrected_after / 255.0, cv2.COLOR_BGR2LAB)
    darkening = before_lab[:, :, 0] - after_lab[:, :, 0]
    noise = darkening[reference]
    noise_median = float(np.median(noise))
    noise_mad = float(np.median(np.abs(noise - noise_median)))
    threshold = max(1.5, noise_median + 3.0 * 1.4826 * noise_mad)
    detected = (darkening >= threshold) & band
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    detected = cv2.morphologyEx(detected.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    detected = cv2.dilate(detected, kernel, iterations=1)

    # Remove tiny registration/noise islands without imposing a learned brow shape.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(detected, 8)
    minimum_area = max(10, round(face_width * face_width * 0.00012))
    cleaned = np.zeros_like(detected)
    kept_components = 0
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            cleaned[labels == label] = 1
            kept_components += 1
    if not cleaned.any():
        raise ValueError("No brow-darkening residual survived thresholding")

    feather_sigma = max(0.8, face_width * 0.003)
    alpha = cv2.GaussianBlur(cleaned.astype(np.float32), (0, 0), feather_sigma)
    alpha = np.clip(alpha, 0.0, 1.0)
    layer = corrected_after * alpha[:, :, None]
    support = alpha >= (0.5 / 255.0)
    # Remove only the actually dark natural-brow pixels, rather than inpainting
    # the entire landmark polygon (which leaves a visible smooth patch).
    original_brow_mask, original_metrics = detect_original_brow(
        before_bgr, before_points, face_width)
    metrics = {
        "face_width_px": face_width,
        "band_radius_px": band_radius,
        "reference_pixels": int(reference.sum()),
        "darkening_noise_median_L": noise_median,
        "darkening_noise_mad_L": noise_mad,
        "darkening_threshold_L": threshold,
        "detected_pixels": int(cleaned.sum()),
        "support_pixels": int(support.sum()),
        "original_brow_erase_pixels": int(original_brow_mask.sum()),
        **original_metrics,
        "kept_components": kept_components,
        "colour_match": colour_metrics,
    }
    return layer, alpha, original_brow_mask, metrics


def translate_array(array: np.ndarray, dy: int) -> np.ndarray:
    """Translate an array vertically without interpolation or wraparound."""
    shifted = np.zeros_like(array)
    height = array.shape[0]
    if abs(dy) >= height:
        return shifted
    if dy >= 0:
        shifted[dy:] = array[:height - dy] if dy else array
    else:
        shifted[:height + dy] = array[-dy:]
    return shifted


def composite_residual(before_bgr: np.ndarray, residual: np.ndarray,
                       support: np.ndarray, dy: int) -> tuple[np.ndarray, np.ndarray, int]:
    shifted_residual = translate_array(residual, dy)
    shifted_support = translate_array(support, dy)
    result = before_bgr.copy()
    values = np.clip(before_bgr.astype(np.float32) + shifted_residual, 0, 255).astype(np.uint8)
    result[shifted_support] = values[shifted_support]
    outside_delta = np.abs(result.astype(np.int16) - before_bgr.astype(np.int16))[~shifted_support]
    maximum = int(outside_delta.max()) if outside_delta.size else 0
    return result, shifted_support, maximum


def remove_original_brow(before_bgr: np.ndarray, brow_mask: np.ndarray,
                         radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove the detected old brow with boundary-propagating NS inpainting."""
    support = brow_mask.astype(bool)
    values = cv2.inpaint(before_bgr, support.astype(np.uint8) * 255,
                         float(radius), cv2.INPAINT_NS)
    clean = before_bgr.copy()
    clean[support] = values[support]
    return clean, support


def composite_brow_layer(before_bgr: np.ndarray, clean_base: np.ndarray,
                         original_brow_mask: np.ndarray, layer: np.ndarray,
                         alpha: np.ndarray, dy: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Erase the old brow and place exactly one shifted real-brow layer."""
    shifted_layer = translate_array(layer, dy)
    shifted_alpha = translate_array(alpha, dy)
    shifted_support = shifted_alpha >= (0.5 / 255.0)
    edit_support = original_brow_mask | shifted_support
    result_float = clean_base.astype(np.float32)
    result_float = result_float * (1.0 - shifted_alpha[:, :, None]) + shifted_layer
    result = before_bgr.copy()
    values = np.clip(result_float, 0, 255).astype(np.uint8)
    result[edit_support] = values[edit_support]
    outside_delta = np.abs(result.astype(np.int16) - before_bgr.astype(np.int16))[~edit_support]
    maximum = int(outside_delta.max()) if outside_delta.size else 0
    return result, edit_support, maximum


def face_crop_bounds(points: np.ndarray, shape: tuple[int, int], padding: float = 0.18) -> tuple[int, int, int, int]:
    height, width = shape
    face = points[list(FACE_OVAL_INDICES)]
    x0, y0 = face.min(axis=0)
    x1, y1 = face.max(axis=0)
    margin = max(x1 - x0, y1 - y0) * padding
    return (
        max(0, int(np.floor(x0 - margin))), max(0, int(np.floor(y0 - margin))),
        min(width, int(np.ceil(x1 + margin))), min(height, int(np.ceil(y1 + margin))),
    )


def make_contact_sheet(images: list[tuple[str, np.ndarray]], crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop
    panels = []
    for label, image in images:
        view = image[y0:y1, x0:x1]
        scale = min(1.0, 360.0 / max(1, view.shape[1]))
        if scale < 1.0:
            view = cv2.resize(view, (round(view.shape[1] * scale), round(view.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
        panel = np.full((view.shape[0] + 48, view.shape[1], 3), 28, np.uint8)
        panel[48:] = view
        cv2.putText(panel, label, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(panel)
    gap = 10
    canvas = np.full((max(p.shape[0] for p in panels),
                      sum(p.shape[1] for p in panels) + gap * (len(panels) - 1), 3), 28, np.uint8)
    x = 0
    for panel in panels:
        canvas[:panel.shape[0], x:x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    return canvas


def generate(before_path: Path, after_path: Path, before_roi_path: Path,
             after_roi_path: Path, output_dir: Path, offsets: list[int]) -> dict:
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
    clean_base, erase_support = remove_original_brow(
        before, original_brow_mask, fill_radius)
    extraction["original_brow_fill_method"] = "OpenCV Navier-Stokes boundary inpainting"
    extraction["original_brow_fill_radius_px"] = fill_radius
    extraction["original_brow_fill_support_pixels"] = int(erase_support.sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    write_png(output_dir / "aligned_after.png", aligned_after)
    preview = before.copy()
    layer_support = alpha >= (0.5 / 255.0)
    preview[layer_support] = np.clip(
        0.45 * before[layer_support].astype(np.float32) + np.array([0, 0, 140], np.float32), 0, 255,
    ).astype(np.uint8)
    write_png(output_dir / "extracted_residual_overlay.png", preview)
    write_png(output_dir / "original_brow_removed.png", clean_base)

    conditions = []
    sheet_images = [("original before", before)]
    for dy in offsets:
        result, edit_support, outside_delta = composite_brow_layer(
            before, clean_base, erase_support, layer, alpha, dy)
        filename = f"brow_height_{dy:+d}px.png"
        write_png(output_dir / filename, result)
        changed = np.any(result != before, axis=2)
        condition = {
            "dy_px": dy,
            "direction": "higher" if dy < 0 else "lower" if dy > 0 else "registered height",
            "file": filename,
            "support_pixels": int(edit_support.sum()),
            "changed_pixels": int(changed.sum()),
            "outside_support_max_abs_delta": outside_delta,
        }
        conditions.append(condition)
        sheet_images.append((f"dy {dy:+d}px", result))
    crop = face_crop_bounds(before_points, before.shape[:2])
    write_png(output_dir / "contact_sheet.png", make_contact_sheet(sheet_images, crop))

    report = {
        "schema_version": 1,
        "method": "registered-real-brow-layer-with-source-replacement-v2",
        "interpretation": (
            "Exploratory brow-position counterfactual. It removes the detected original brow "
            "with local boundary inpainting and moves a brow layer extracted from the registered "
            "after frame."
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
        "conditions": conditions,
        "artifacts": {},
    }
    for path in sorted(output_dir.glob("*.png")):
        report["artifacts"][path.name] = sha256_file(path)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-image", type=Path, required=True)
    parser.add_argument("--after-image", type=Path, required=True)
    parser.add_argument("--before-roi", type=Path, required=True)
    parser.add_argument("--after-roi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=[-12, -6, 0, 6, 12])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate(args.before_image, args.after_image, args.before_roi,
                      args.after_roi, args.output_dir, args.offsets)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "registration": report["registration"],
        "extraction": report["extraction"],
        "conditions": report["conditions"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
