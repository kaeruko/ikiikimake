"""Generate conservative facial polygons; never declare them human-approved.

Screen-left/right names refer to the displayed image, not anatomical sides.
The face/hand ML models only supply landmarks; masks use fixed geometry.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROI_VERSION = "mediapipe-cheek-forehead-v1"
ROI_INDICES = {
    "cheek_a": (116, 117, 118, 101, 205, 207, 187, 123),
    "cheek_b": (345, 346, 347, 330, 425, 427, 411, 352),
    "forehead": (109, 10, 338, 337, 151, 108),
}
COLORS = {"left_cheek": (50, 220, 50), "right_cheek": (255, 110, 45), "forehead": (0, 220, 255)}


@dataclass(frozen=True)
class RoiConfig:
    min_face_detection_confidence: float = 0.6
    min_face_presence_confidence: float = 0.6
    min_hand_detection_confidence: float = 0.35
    min_hand_presence_confidence: float = 0.35
    max_yaw_degrees: float = 25.0
    max_pitch_degrees: float = 25.0
    max_roll_degrees: float = 20.0
    min_face_width_px: float = 120.0
    min_roi_pixels: int = 100
    polygon_scale: float = 0.88
    hand_margin_face_fraction: float = 0.025
    max_hand_overlap_ratio: float = 0.02
    max_clipped_pixel_ratio: float = 0.10

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_face_detection_confidence", "min_face_presence_confidence",
                     "min_hand_detection_confidence", "min_hand_presence_confidence",
                     "polygon_scale", "max_hand_overlap_ratio", "max_clipped_pixel_ratio"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")


class RoiGenerationError(RuntimeError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__("ROI generation failed: " + "; ".join(report["errors"]))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_image(path: Path) -> np.ndarray:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def pose_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Missing or invalid face transformation matrix")
    # Remove any scale/shear before estimating Euler angles (Rz * Ry * Rx).
    u, singular, vt = np.linalg.svd(matrix[:3, :3])
    if singular.min() < 1e-6:
        raise ValueError("Degenerate face transformation matrix")
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise ValueError("Reflected face transformation matrix")
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return dict(zip(("pitch", "yaw", "roll"), map(math.degrees, (pitch, yaw, roll))))


def polygons_from_landmarks(points: np.ndarray, scale: float = 0.88) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Invalid face landmarks: expected at least 468 finite xy points")
    polygons = {}
    for name, indices in ROI_INDICES.items():
        polygon = points[list(indices)]
        center = polygon.mean(axis=0)
        polygons[name] = center + scale * (polygon - center)
    # Sorting by displayed x also handles mirrored input without changing labels.
    cheeks = sorted((polygons["cheek_a"], polygons["cheek_b"]), key=lambda p: p[:, 0].mean())
    return {"left_cheek": cheeks[0], "right_cheek": cheeks[1], "forehead": polygons["forehead"]}


def rasterize_polygons(polygons: dict[str, np.ndarray], shape: tuple[int, int], min_pixels: int) -> tuple[dict, list[str]]:
    height, width = shape
    masks, errors = {}, []
    occupied = np.zeros(shape, dtype=bool)
    for name, polygon in polygons.items():
        polygon = np.asarray(polygon)
        if not np.isfinite(polygon).all() or polygon.ndim != 2 or polygon.shape[1] != 2:
            errors.append(f"{name}: invalid polygon")
            continue
        if (polygon[:, 0] < 0).any() or (polygon[:, 0] >= width).any() or (polygon[:, 1] < 0).any() or (polygon[:, 1] >= height).any():
            errors.append(f"{name}: polygon outside image (not clipped)")
            continue
        vertices = np.rint(polygon).astype(np.int32)
        if (vertices[:, 0] >= width).any() or (vertices[:, 1] >= height).any():
            errors.append(f"{name}: rounded polygon outside image")
            continue
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 1)
        if cv2.contourArea(vertices) < 1 or np.count_nonzero(mask) < min_pixels:
            errors.append(f"{name}: too few pixels ({np.count_nonzero(mask)} < {min_pixels})")
        if np.any(occupied & (mask != 0)):
            errors.append(f"{name}: ROI polygons overlap")
        occupied |= mask != 0
        masks[name] = mask
    return masks, errors


def hand_coverage(hand_points: list[np.ndarray], shape: tuple[int, int], margin_px: int) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = np.zeros(shape, dtype=np.uint8)
    hulls = []
    for points in hand_points:
        if np.asarray(points).shape != (21, 2) or not np.isfinite(points).all():
            raise ValueError("Invalid hand landmarks")
        hull = cv2.convexHull(np.rint(points).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
        hulls.append(hull)
    # Hulls deliberately overestimate fingers/gaps; this is a rejection heuristic,
    # not segmentation or proof that an undetected hand is absent.
    if margin_px > 0 and hulls:
        size = 2 * margin_px + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return mask, hulls


def assess_geometry(image: np.ndarray, points: np.ndarray, matrix: np.ndarray,
                    hands: list[np.ndarray], config: RoiConfig) -> tuple[dict, dict, dict, list[str], list[str], list]:
    polygons = polygons_from_landmarks(points, config.polygon_scale)
    points = np.asarray(points, dtype=float)
    pose = pose_from_matrix(matrix)
    face_width = float(np.linalg.norm(points[454] - points[234]))
    quality = {"pose_degrees": pose, "pose_method": "MediaPipe matrix, SVD rotation, RzRyRx Euler",
               "face_width_px": face_width, "detected_hands": len(hands),
               "landmark_confidence": None,
               "hand_detection_note": "Zero detected hands does not establish absence of hands; partial hands and held tools can be missed.",
               "confidence_note": "Face detection/presence thresholds are internal; no per-landmark confidence is returned."}
    errors, warnings = [], ["Human review required: hair, tools, text, reflections and undetected hands are not fully screened."]
    for axis in ("yaw", "pitch", "roll"):
        limit = getattr(config, f"max_{axis}_degrees")
        if abs(pose[axis]) > limit:
            errors.append(f"face {axis} too large: {pose[axis]:.1f} deg (limit {limit:g})")
    if face_width < config.min_face_width_px:
        errors.append(f"face too small: {face_width:.1f} px (minimum {config.min_face_width_px:g})")
    masks, polygon_errors = rasterize_polygons(polygons, image.shape[:2], config.min_roi_pixels)
    errors.extend(polygon_errors)
    hand_mask, hulls = hand_coverage(hands, image.shape[:2], max(1, round(face_width * config.hand_margin_face_fraction)))
    quality["hand_overlap_ratios"], quality["clipped_pixel_ratios"] = {}, {}
    for name, mask in masks.items():
        selection = mask != 0
        if not selection.any():
            continue
        overlap = float(np.mean(hand_mask[selection] != 0))
        quality["hand_overlap_ratios"][name] = overlap
        if overlap > config.max_hand_overlap_ratio:
            errors.append(f"{name}: suspected hand occlusion {overlap:.1%} (limit {config.max_hand_overlap_ratio:.1%})")
        elif overlap > 0:
            warnings.append(f"{name}: hand close to ROI; inspect shadow/reflection")
        pixels = image[selection]
        clipped = float(np.mean(np.any(pixels == 255, axis=1) | np.all(pixels <= 2, axis=1)))
        quality["clipped_pixel_ratios"][name] = clipped
        if clipped > config.max_clipped_pixel_ratio:
            errors.append(f"{name}: excessive clipped pixels {clipped:.1%}")
    if "forehead" in masks and np.any(masks["forehead"]):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        quality["forehead_L_median"] = float(np.median(lab[:, :, 0][masks["forehead"] != 0]) * 100 / 255)
    return polygons, masks, quality, errors, warnings, hulls


def overlay_image(image: np.ndarray, polygons: dict, hulls: list, failed: bool) -> np.ndarray:
    overlay = image.copy()
    for name, points in polygons.items():
        vertices = np.rint(points).astype(np.int32)
        fill = overlay.copy()
        cv2.fillPoly(fill, [vertices], COLORS[name])
        overlay = cv2.addWeighted(fill, 0.22, overlay, 0.78, 0)
        cv2.polylines(overlay, [vertices], True, COLORS[name], 2, cv2.LINE_AA)
    for hull in hulls:
        cv2.polylines(overlay, [hull], True, (50, 50, 255), 1, cv2.LINE_AA)
    banner = np.full((58, image.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(banner, "REJECTED - diagnostic only" if failed else "REVIEW REQUIRED - no Lab analysis yet",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255) if failed else (235, 235, 235), 1, cv2.LINE_AA)
    for x, name, label in ((10, "left_cheek", "Screen left"), (175, "right_cheek", "Screen right"), (350, "forehead", "Forehead")):
        cv2.putText(banner, label, (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS[name], 1, cv2.LINE_AA)
    return np.vstack((banner, overlay))


class RoiExtractor:
    def __init__(self, model_dir: Path = ROOT / "models", config: RoiConfig | None = None):
        self.model_dir, self.config = Path(model_dir), config or RoiConfig()
        self._stack = None

    def __enter__(self):
        import mediapipe as mp
        for filename in ("face_landmarker.task", "hand_landmarker.task"):
            if not (self.model_dir / filename).is_file():
                raise FileNotFoundError(f"Model missing: {self.model_dir / filename}. Run analysis/setup_roi_models.py first.")
        self._stack = ExitStack()
        try:
            self.face = self._stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_dir / "face_landmarker.task")),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
                    min_face_detection_confidence=self.config.min_face_detection_confidence,
                    min_face_presence_confidence=self.config.min_face_presence_confidence,
                    output_facial_transformation_matrixes=True)))
            self.hand = self._stack.enter_context(mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_dir / "hand_landmarker.task")),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=4,
                    min_hand_detection_confidence=self.config.min_hand_detection_confidence,
                    min_hand_presence_confidence=self.config.min_hand_presence_confidence)))
        except Exception:
            self._stack.close()
            raise
        self.mp = mp
        return self

    def __exit__(self, *args):
        if self._stack:
            self._stack.close()

    def extract(self, image_path: Path, output_dir: Path) -> dict:
        image_path, output_dir = Path(image_path).resolve(), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        report = {"schema_version": 1, "roi_version": ROI_VERSION, "status": "failed",
                  "source": {"path": str(image_path)}, "config": asdict(self.config),
                  "quality": {}, "rois": {}, "errors": [], "warnings": []}
        image, polygons, masks, hulls = None, {}, {}, []
        try:
            image = load_image(image_path)
            height, width = image.shape[:2]
            report["source"].update(sha256=sha256_file(image_path), width=width, height=height)
            report["models"] = {name: {"sha256": sha256_file(self.model_dir / name)}
                                for name in ("face_landmarker.task", "hand_landmarker.task")}
            report["mediapipe_version"] = self.mp.__version__
            report["runtime_versions"] = {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__}
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            faces = self.face.detect(mp_image)
            report["quality"]["detected_faces"] = len(faces.face_landmarks)
            if len(faces.face_landmarks) != 1:
                raise ValueError(f"Expected exactly one face passing confidence thresholds, found {len(faces.face_landmarks)}")
            hand_result = self.hand.detect(mp_image)
            points = np.array([[p.x * width, p.y * height] for p in faces.face_landmarks[0]])
            hands = [np.array([[p.x * width, p.y * height] for p in hand]) for hand in hand_result.hand_landmarks]
            if not faces.facial_transformation_matrixes:
                raise ValueError("Face pose is unavailable")
            polygons, masks, quality, errors, warnings, hulls = assess_geometry(
                image, points, faces.facial_transformation_matrixes[0], hands, self.config)
            report["quality"].update(quality)
            report["errors"].extend(errors)
            report["warnings"].extend(warnings)
            report["face_landmarks_normalized"] = [[p.x, p.y, p.z] for p in faces.face_landmarks[0]]
            for name, polygon in polygons.items():
                report["rois"][name] = {"polygon_px": polygon.tolist(),
                                       "polygon_normalized": (polygon / [width, height]).tolist(),
                                       "pixel_count": int(np.count_nonzero(masks.get(name, [])))}
            if not report["errors"]:
                report["status"] = "needs_review"
        except (ValueError, RuntimeError, OSError, cv2.error) as error:
            report["errors"].append(str(error))
        if image is not None:
            save_image(output_dir / "roi_overlay.png", overlay_image(image, polygons, hulls, bool(report["errors"])))
        if not report["errors"]:
            np.savez_compressed(output_dir / "roi_masks.npz", **masks)
            for name, mask in masks.items():
                save_image(output_dir / f"{name}_mask.png", mask * 255)
        (output_dir / "roi_points.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if report["errors"]:
            raise RoiGenerationError(report)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are not overwritten")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    try:
        with RoiExtractor(args.model_dir) as extractor:
            extractor.extract(args.image, args.output)
    except (RoiGenerationError, OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"Review required: {args.output / 'roi_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
