"""Landmark alignment and inverse thin-plate-spline sampling in RGB coordinates.

The paper uses a proprietary tracker. This implementation uses MediaPipe's
public 478-point tracker and conservative geometric masks in its place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REGIONS = ("eye", "lip", "cheek")
EYE_LEFT = (33, 160, 158, 133, 153, 144)
EYE_RIGHT = (362, 385, 387, 263, 373, 380)
BROW_LEFT = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
BROW_RIGHT = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
LIP_OUTER = (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91)
LIP_INNER = (78, 81, 13, 311, 308, 402, 14, 178)
CHEEK_LEFT = (116, 117, 118, 101, 205, 207, 187, 123)
CHEEK_RIGHT = (345, 346, 347, 330, 425, 427, 411, 352)
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
             288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
             150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
REGION_INDICES = {
    "eye": EYE_LEFT + EYE_RIGHT + BROW_LEFT + BROW_RIGHT,
    "lip": LIP_OUTER + LIP_INNER,
    "cheek": CHEEK_LEFT + CHEEK_RIGHT,
}
CONTROL_INDICES = tuple(dict.fromkeys(FACE_OVAL[::2] + REGION_INDICES["eye"]
                                    + REGION_INDICES["lip"] + REGION_INDICES["cheek"] + (1, 6, 168)))
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def _points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected at least 468 finite MediaPipe xy landmarks")
    return points


def read_rgb(path: str | Path) -> np.ndarray:
    """Unicode-path-safe image loading on Windows."""
    path = Path(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FaceDetector:
    def __init__(self, model_path: str | Path | None = None):
        import mediapipe as mp
        path = Path(model_path or DEFAULT_MODEL)
        if not path.is_file():
            raise FileNotFoundError(f"Face landmark model is missing: {path}")
        self.mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2, min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("FaceDetector expects an HWC uint8 RGB image")
        result = self.detector.detect(self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if len(result.face_landmarks) != 1:
            return None
        height, width = rgb.shape[:2]
        return np.array([(p.x * width, p.y * height) for p in result.face_landmarks[0]], dtype=np.float32)

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def alignment_anchors(landmarks: np.ndarray) -> np.ndarray:
    points = _points(landmarks)
    return np.array([points[list(EYE_LEFT)].mean(0), points[list(EYE_RIGHT)].mean(0),
                     points[list(LIP_OUTER)].mean(0)], dtype=np.float32)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(points) @ matrix[:, :2].T + matrix[:, 2]).astype(np.float32)


def _boxes(points: np.ndarray, size: int) -> dict[str, tuple[int, int, int, int]]:
    boxes = {}
    for region, indices in REGION_INDICES.items():
        roi = points[list(indices)]
        low, high = roi.min(0), roi.max(0)
        extent = np.maximum(high - low, size * 0.025)
        # A little surrounding skin remains visible to the extraction network.
        margin = extent * ({"eye": (0.12, 0.35), "lip": (0.25, 0.4), "cheek": (0.12, 0.35)}[region])
        low = np.floor(np.maximum(low - margin, 0)).astype(int)
        high = np.ceil(np.minimum(high + margin, size)).astype(int)
        boxes[region] = (int(low[0]), int(low[1]), int(high[0]), int(high[1]))
    return boxes


@dataclass
class CanonicalGeometry:
    landmarks: np.ndarray
    canvas_size: int
    boxes: dict[str, tuple[int, int, int, int]]

    def save(self, path: str | Path):
        np.savez_compressed(path, landmarks=self.landmarks, canvas_size=self.canvas_size,
                            **{f"box_{region}": self.boxes[region] for region in REGIONS})

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalGeometry":
        with np.load(path, allow_pickle=False) as data:
            return cls(_points(data["landmarks"]), int(data["canvas_size"]),
                       {region: tuple(map(int, data[f"box_{region}"])) for region in REGIONS})


def build_canonical(landmark_sets: list[np.ndarray], image_sizes=None,
                    canvas_size: int = 512) -> CanonicalGeometry:
    """Average aligned TRAINING faces only; callers must exclude held-out faces."""
    if not landmark_sets:
        raise ValueError("At least one training face is needed for canonical geometry")
    if canvas_size < 32:
        raise ValueError("canvas_size must be at least 32")
    target = np.array(((0.34, 0.37), (0.66, 0.37), (0.5, 0.68)), dtype=np.float32) * canvas_size
    aligned = []
    for landmarks in landmark_sets:
        points = _points(landmarks)
        source = alignment_anchors(points)
        if abs(cv2.contourArea(source)) < 1:
            raise ValueError("Degenerate landmark anchors")
        matrix = cv2.getAffineTransform(source, target)
        aligned.append(transform_points(points[:468], matrix))
    average = np.mean(aligned, axis=0).astype(np.float32)
    return CanonicalGeometry(average, canvas_size, _boxes(average, canvas_size))


def affine_align(rgb: np.ndarray, landmarks: np.ndarray,
                 geometry: CanonicalGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = alignment_anchors(landmarks)
    if abs(cv2.contourArea(source)) < 1:
        raise ValueError("Degenerate landmark anchors")
    matrix = cv2.getAffineTransform(source, alignment_anchors(geometry.landmarks))
    aligned = cv2.warpAffine(rgb, matrix, (geometry.canvas_size, geometry.canvas_size),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return aligned, transform_points(landmarks, matrix), matrix


def region_landmarks(landmarks: np.ndarray, region: str | None = None) -> np.ndarray:
    points = _points(landmarks)
    indices = CONTROL_INDICES if region is None else tuple(dict.fromkeys(
        REGION_INDICES[region] + FACE_OVAL[::4] + (1, 6, 168)))
    return points[list(indices)]


def _kernel(squared_distance: np.ndarray) -> np.ndarray:
    return squared_distance * np.log(np.maximum(squared_distance, 1e-12))


def tps_maps(source_points: np.ndarray, target_points: np.ndarray,
             output_shape: tuple[int, int], grid_size: int | None = 128,
             regularization: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Build inverse TPS maps: output target coordinates -> input source pixels.

    TPS is evaluated on a bounded grid, then bilinearly upsampled. Pass
    grid_size=None for dense exact evaluation. Duplicate target landmarks are
    merged before solving, preventing singular systems on closed mouths.
    """
    source, target = np.asarray(source_points, float), np.asarray(target_points, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("TPS needs matching Nx2 point arrays with N >= 3")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("TPS coordinates must be finite")
    height, width = map(int, output_shape)
    if min(height, width) < 1:
        raise ValueError("TPS output dimensions must be positive")
    scale = max(float(np.ptp(target, axis=0).max()), 1.0)
    center = target.mean(0)
    target = (target - center) / scale
    _, first = np.unique(np.round(target, 6), axis=0, return_index=True)
    source, target = source[first], target[first]
    if len(source) < 3 or np.linalg.matrix_rank(np.column_stack((np.ones(len(target)), target))) < 3:
        raise ValueError("TPS target control points must be non-collinear")
    distance = np.sum((target[:, None] - target[None, :]) ** 2, axis=2)
    kernel = _kernel(distance) + regularization * np.eye(len(target))
    affine = np.column_stack((np.ones(len(target)), target))
    system = np.block([[kernel, affine], [affine.T, np.zeros((3, 3))]])
    rhs = np.vstack((source, np.zeros((3, 2))))
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, rhs, rcond=None)[0]
    gh, gw = (height, width) if grid_size is None else (min(height, grid_size), min(width, grid_size))
    xx, yy = np.meshgrid(np.linspace(0, width - 1, gw), np.linspace(0, height - 1, gh))
    query = (np.column_stack((xx.ravel(), yy.ravel())) - center) / scale
    mapped = np.empty_like(query)
    for start in range(0, len(query), 8192):
        chunk = query[start:start + 8192]
        basis = _kernel(np.sum((chunk[:, None] - target[None, :]) ** 2, axis=2))
        mapped[start:start + len(chunk)] = basis @ weights[:-3] + np.column_stack((np.ones(len(chunk)), chunk)) @ weights[-3:]
    map_x = mapped[:, 0].reshape(gh, gw).astype(np.float32)
    map_y = mapped[:, 1].reshape(gh, gw).astype(np.float32)
    if (gh, gw) != (height, width):
        # remap uses endpoint coordinates, unlike resize's half-pixel grid.
        mx, my = np.meshgrid(np.linspace(0, gw - 1, width, dtype=np.float32),
                             np.linspace(0, gh - 1, height, dtype=np.float32))
        map_x = cv2.remap(map_x, mx, my, cv2.INTER_LINEAR)
        map_y = cv2.remap(map_y, mx, my, cv2.INTER_LINEAR)
    return map_x, map_y


def warp_tps(image: np.ndarray, source_points: np.ndarray, target_points: np.ndarray,
             output_shape: tuple[int, int], grid_size: int | None = 128) -> np.ndarray:
    maps = tps_maps(source_points, target_points, output_shape, grid_size=grid_size)
    return cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_region(image: np.ndarray, geometry: CanonicalGeometry, region: str,
                size: int = 256) -> np.ndarray:
    x0, y0, x1, y1 = geometry.boxes[region]
    result = cv2.resize(image[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_LINEAR)
    return result[..., None] if image.ndim == 3 and result.ndim == 2 else result


def paste_region(crop: np.ndarray, geometry: CanonicalGeometry, region: str) -> np.ndarray:
    shape = (geometry.canvas_size, geometry.canvas_size) + tuple(crop.shape[2:])
    result = np.zeros(shape, dtype=crop.dtype)
    x0, y0, x1, y1 = geometry.boxes[region]
    resized = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    if result.ndim == 3 and resized.ndim == 2:
        resized = resized[..., None]
    result[y0:y1, x0:x1] = resized
    return result


def _polygon(shape, points, indices, value=1.0, mask=None):
    mask = np.zeros(shape, np.float32) if mask is None else mask
    cv2.fillPoly(mask, [np.rint(points[list(indices)]).astype(np.int32)], value)
    return mask


def face_region_mask(landmarks: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Geometry approximation; this does not detect occluding hands or hair."""
    points = _points(landmarks)
    mask = _polygon(tuple(image_shape[:2]), points, FACE_OVAL)
    for indices in (EYE_LEFT, EYE_RIGHT, LIP_INNER, BROW_LEFT, BROW_RIGHT):
        _polygon(mask.shape, points, indices, value=0.0, mask=mask)
    return mask


def region_mask(geometry: CanonicalGeometry, region: str) -> np.ndarray:
    points, size = geometry.landmarks, geometry.canvas_size
    mask = np.zeros((size, size), np.float32)
    if region == "lip":
        _polygon(mask.shape, points, LIP_OUTER, mask=mask)
        _polygon(mask.shape, points, LIP_INNER, value=0.0, mask=mask)
    elif region == "cheek":
        for indices in (CHEEK_LEFT, CHEEK_RIGHT):
            _polygon(mask.shape, points, indices, mask=mask)
    elif region == "eye":
        for eye, brow in ((EYE_LEFT, BROW_LEFT), (EYE_RIGHT, BROW_RIGHT)):
            hull = cv2.convexHull(np.rint(points[list(eye + brow)]).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1.0)
        mask = cv2.dilate(mask, np.ones((max(3, size // 40),) * 2, np.uint8))
    else:
        raise ValueError(f"Unknown region: {region}")
    return mask * face_region_mask(points, mask.shape)
