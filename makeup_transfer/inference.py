"""Extract one reference style, then render it on images or video frames.

The learned models are only called during :func:`extract_style`. Rendering uses
TPS and alpha compositing at the target's original resolution. Landmark masks
are a documented fallback for the paper's face parser; a caller can additionally
supply a semantic visibility mask to exclude hair, hands, and other occluders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .geometry import (
    REGIONS,
    CanonicalGeometry,
    FaceDetector,
    affine_align,
    crop_region,
    face_region_mask,
    region_landmarks,
    region_mask,
    warp_tps,
)


@dataclass(frozen=True)
class MakeupStyle:
    """Canonical, straight-alpha RGBA patches; RGB and alpha are in [0, 1]."""

    geometry: CanonicalGeometry
    patches: Mapping[str, np.ndarray]
    metadata: Mapping[str, object] = field(default_factory=dict)


class TemporalLandmarkSmoother:
    """EMA with a reset after detection failure, avoiding stale-face overlays.

    ``smoothing`` is the contribution of the previous detection, in [0, 1).
    """

    def __init__(self, smoothing: float = 0.6):
        if not np.isfinite(smoothing) or not 0 <= smoothing < 1:
            raise ValueError("smoothing must be finite and in [0, 1).")
        self.smoothing = float(smoothing)
        self._previous: np.ndarray | None = None

    def update(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        if landmarks is None:
            self._previous = None
            return None
        current = np.asarray(landmarks, dtype=np.float32)
        if current.ndim != 2 or current.shape[1] != 2 or not np.isfinite(current).all():
            raise ValueError("Landmarks must be a finite N x 2 array.")
        if self._previous is not None and self._previous.shape == current.shape:
            current = self.smoothing * self._previous + (1 - self.smoothing) * current
        self._previous = current.copy()
        return current


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Images must be H x W x 3 RGB uint8 arrays.")
    if min(image.shape[:2]) == 0:
        raise ValueError("An image cannot have an empty dimension.")
    return image


def _strength(value: float) -> float:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("strength must be finite and in [0, 1].")
    return float(value)


def _visibility_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=np.float32)
    values = np.asarray(mask)
    if values.shape == (*shape, 1):
        values = values[..., 0]
    if values.shape != shape:
        raise ValueError(f"A visibility mask must have target shape {shape}.")
    if values.dtype == np.uint8:
        # Accept both conventional binary masks and 8-bit grayscale masks.
        values = values.astype(np.float32) / (255.0 if values.max(initial=0) > 1 else 1.0)
    else:
        values = values.astype(np.float32)
    if not np.isfinite(values).all() or values.min(initial=0) < 0 or values.max(initial=0) > 1:
        raise ValueError("Visibility masks must be binary, uint8 grayscale, or floats in [0, 1].")
    return values


def alpha_composite(
    background: np.ndarray,
    foreground_rgba: np.ndarray,
    *,
    strength: float = 1.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blend straight-alpha RGBA over RGB, preserving zero-alpha pixels exactly."""
    background = _validate_rgb(background)
    rgba = np.asarray(foreground_rgba, dtype=np.float32)
    if rgba.shape != (*background.shape[:2], 4) or not np.isfinite(rgba).all():
        raise ValueError("foreground_rgba must be finite H x W x 4 matching the background.")
    rgba = np.clip(rgba, 0, 1)
    alpha = rgba[..., 3:4] * _strength(strength)
    alpha *= _visibility_mask(mask, background.shape[:2])[..., None]
    result = np.rint(background.astype(np.float32) * (1 - alpha) + 255 * rgba[..., :3] * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def _regions(regions: Sequence[str]) -> tuple[str, ...]:
    values = tuple(regions)
    if not values or len(set(values)) != len(values) or any(p not in REGIONS for p in values):
        raise ValueError(f"regions must be a nonempty, unique subset of {REGIONS}.")
    return values


def _checkpoint_paths(checkpoints: str | Path, regions: Sequence[str]) -> dict[str, Path]:
    root = Path(checkpoints)
    paths = {region: root / f"{region}.pt" for region in _regions(regions)}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Trained generator checkpoint(s) missing: " + ", ".join(missing)
            + ". Prepare the dataset and train the requested regions first. "
            "Inference does not substitute untrained models or color-transfer heuristics."
        )
    return paths


def _load_checkpoint(path: Path, device: object = "cpu") -> dict:
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"generator", "region", "base_channels", "average_alpha"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"{path} is not a makeup-transfer training checkpoint; required keys: {required}.")
    return checkpoint


def _resolve_geometry(
    checkpoints: str | Path, geometry_path: str | Path | None, regions: Sequence[str]
) -> CanonicalGeometry:
    paths = _checkpoint_paths(checkpoints, regions)
    if geometry_path is not None:
        path = Path(geometry_path)
    elif (Path(checkpoints) / "geometry.npz").is_file():
        path = Path(checkpoints) / "geometry.npz"
    else:
        checkpoint = _load_checkpoint(next(iter(paths.values())))
        stored = checkpoint.get("geometry_path")
        if not stored:
            raise ValueError("No canonical geometry found. Supply geometry_path from dataset preparation.")
        path = Path(stored)
        if not path.is_absolute() and not path.is_file():
            path = Path(checkpoints) / path
    if not path.is_file():
        raise FileNotFoundError(f"Canonical training geometry does not exist: {path}")
    return CanonicalGeometry.load(path)


def extract_style(
    reference_rgb: np.ndarray,
    checkpoints: str | Path,
    geometry: CanonicalGeometry,
    detector: FaceDetector,
    *,
    regions: Sequence[str] = REGIONS,
    device: str = "auto",
) -> MakeupStyle:
    """Run each trained region generator once on an affine-aligned reference."""
    reference_rgb = _validate_rgb(reference_rgb)
    paths = _checkpoint_paths(checkpoints, regions)
    import torch
    from .models import MakeupGenerator

    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    landmarks = detector.detect(reference_rgb)
    if landmarks is None:
        raise ValueError("No usable single face was detected in the makeup reference.")
    aligned, _, _ = affine_align(reference_rgb, landmarks, geometry)
    patches: dict[str, np.ndarray] = {}
    details: dict[str, dict] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for region, path in paths.items():
            # Training checkpoints also contain discriminator/optimizer states;
            # keep those on CPU instead of allocating them on the inference GPU.
            checkpoint = _load_checkpoint(path)
            if checkpoint["region"] != region:
                raise ValueError(f"Checkpoint {path} is for region {checkpoint['region']!r}, expected {region!r}.")
            model = MakeupGenerator(base_channels=int(checkpoint["base_channels"])).to(selected_device)
            model.load_state_dict(checkpoint["generator"], strict=True)
            model.eval()
            crop = crop_region(aligned, geometry, region, size=256)
            rgb_tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float()[None].to(selected_device) / 255
            prior = torch.as_tensor(checkpoint["average_alpha"], dtype=torch.float32, device=selected_device)
            if prior.shape == (256, 256):
                prior = prior[None]
            if prior.shape == (256, 256, 1):
                prior = prior.permute(2, 0, 1)
            if prior.shape == (1, 256, 256):
                prior = prior[None]
            if prior.shape != (1, 1, 256, 256) or not torch.isfinite(prior).all() or torch.any((prior < 0) | (prior > 1)):
                raise ValueError(f"Checkpoint {path} average_alpha must be a 256 x 256 probability mask.")
            rgba = model(rgb_tensor, prior)[0].detach().cpu().numpy().transpose(1, 2, 0)
            if rgba.shape != (256, 256, 4) or not np.isfinite(rgba).all():
                raise ValueError(f"Generator for {region} returned an invalid RGBA mask.")
            rgba = np.clip(rgba, 0, 1)
            region_support = crop_region(region_mask(geometry, region), geometry, region, size=256)
            rgba[..., 3] *= np.clip(np.squeeze(region_support), 0, 1)
            patches[region] = rgba.astype(np.float32)
            details[region] = {
                "checkpoint": str(path.resolve()),
                "epoch": int(checkpoint.get("epoch", 0)),
                "steps": int(checkpoint.get("steps", 0)),
                "training_complete": checkpoint.get("training_complete"),
                "limited_training": checkpoint.get("limited_training"),
                "paper_training_schedule_complete": checkpoint.get("paper_training_schedule_complete"),
            }
            del model, checkpoint, rgb_tensor, prior
    return MakeupStyle(geometry, patches, {
        "checkpoints": details,
        "device": str(selected_device),
        "generator_calls": len(patches),
        "extraction_seconds": time.perf_counter() - started,
    })


def render_style(
    target_rgb: np.ndarray,
    style: MakeupStyle,
    landmarks: np.ndarray,
    *,
    strength: float = 1.0,
    semantic_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """TPS-render cached RGBA patches without invoking any learned generators.

    Returns the original-resolution RGB result and cumulative applied alpha.
    Semantic masks are visibility masks, not parser class-index maps. Supplying
    one can additionally exclude occlusion but cannot re-enable protected eyes,
    eyebrows, or mouth interiors.
    """
    target_rgb = _validate_rgb(target_rgb)
    strength = _strength(strength)
    height, width = target_rgb.shape[:2]
    visibility = face_region_mask(landmarks, (height, width)).astype(np.float32)
    visibility *= _visibility_mask(semantic_mask, (height, width))
    output = target_rgb.copy()
    combined_alpha = np.zeros((height, width), dtype=np.float32)
    for region in ("cheek", "eye", "lip"):
        if region not in style.patches:
            continue
        patch = np.asarray(style.patches[region], dtype=np.float32)
        if patch.ndim != 3 or patch.shape[2] != 4 or not np.isfinite(patch).all():
            raise ValueError(f"Invalid canonical RGBA patch for {region}.")
        patch = np.clip(patch, 0, 1)
        x0, y0, x1, y1 = style.geometry.boxes[region]
        source_points = region_landmarks(style.geometry.landmarks, region).astype(np.float32).copy()
        # Match cv2.resize's pixel-center convention used by crop_region.
        source_points += np.array([0.5 - x0, 0.5 - y0], dtype=np.float32)
        source_points *= np.array([patch.shape[1] / (x1 - x0), patch.shape[0] / (y1 - y0)], dtype=np.float32)
        source_points -= 0.5
        target_points = region_landmarks(landmarks, region)
        # Interpolating premultiplied color avoids dark fringes at transparent borders.
        premultiplied = patch.copy()
        premultiplied[..., :3] *= patch[..., 3:4]
        warped = warp_tps(premultiplied, source_points, target_points, (height, width))
        alpha = np.clip(warped[..., 3:4], 0, 1)
        rgba = np.concatenate((np.divide(warped[..., :3], alpha, out=np.zeros_like(warped[..., :3]), where=alpha > 1e-7), alpha), axis=2)
        output = alpha_composite(output, rgba, strength=strength, mask=visibility)
        applied_alpha = alpha[..., 0] * strength * visibility
        combined_alpha = applied_alpha + combined_alpha * (1 - applied_alpha)
    return output, combined_alpha


def save_style(style: MakeupStyle, path: str | Path) -> Path:
    """Save extracted patches for inspection and reuse; contains no model weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{region}_rgba": patch for region, patch in style.patches.items()})
    return path


def _rgba_preview(rgba: np.ndarray, tile: int = 16) -> np.ndarray:
    """Composite straight-alpha RGBA on a checkerboard for human review."""
    rgba = np.asarray(rgba, dtype=np.float32)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or not np.isfinite(rgba).all():
        raise ValueError("RGBA preview expects a finite H x W x 4 array.")
    rgba = np.clip(rgba, 0, 1)
    height, width = rgba.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    checker = np.where(((xx // tile) + (yy // tile)) % 2 == 0, 0.88, 0.70).astype(np.float32)
    background = np.repeat(checker[..., None], 3, axis=2)
    alpha = rgba[..., 3:4]
    composite = background * (1 - alpha) + rgba[..., :3] * alpha
    return np.rint(np.clip(composite, 0, 1) * 255).astype(np.uint8)


def _read_rgb(path: str | Path) -> np.ndarray:
    data = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot decode visibility mask: {path}")
    return mask


def _write_image(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Could not encode output image: {path}")
    encoded.tofile(path)


def _output_path(output: str | Path, *inputs: str | Path) -> Path:
    path = Path(output)
    if any(path.resolve() == Path(value).resolve() for value in inputs):
        raise ValueError("Output must differ from every input path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_metadata(output: Path, metadata: dict) -> dict:
    path = output.with_suffix(output.suffix + ".json")
    metadata["metadata_path"] = str(path.resolve())
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return metadata


def run_extract(
    reference: str | Path,
    output: str | Path,
    checkpoints: str | Path,
    *,
    geometry_path: str | Path | None = None,
    landmark_model: str | Path = "models/face_landmarker.task",
    device: str = "auto",
    regions: Sequence[str] = REGIONS,
) -> dict:
    """Extract canonical RGBA makeup patches from one real reference image."""
    output = Path(output)
    if output.suffix.lower() != ".npz":
        raise ValueError("Extract output must use a .npz suffix.")
    preview_paths = [output.with_name(f"{output.stem}.{region}.png") for region in _regions(regions)]
    metadata_path = output.with_suffix(output.suffix + ".json")
    existing = [path for path in (output, metadata_path, *preview_paths) if path.exists()]
    if existing:
        raise FileExistsError("Extract output artifacts already exist: " + ", ".join(map(str, existing)))
    output = _output_path(output, reference)
    geometry = _resolve_geometry(checkpoints, geometry_path, regions)
    reference_rgb = _read_rgb(reference)
    with FaceDetector(landmark_model) as detector:
        style = extract_style(reference_rgb, checkpoints, geometry, detector, regions=regions, device=device)
    style_path = save_style(style, output)
    previews = {}
    for region, patch in style.patches.items():
        preview_path = output.with_name(f"{output.stem}.{region}.png")
        _write_image(preview_path, _rgba_preview(patch))
        previews[region] = str(preview_path.resolve())
    return _write_metadata(output, {
        "mode": "extract", "reference": str(Path(reference).resolve()),
        "output": str(style_path.resolve()), "regions": list(style.patches),
        "style": dict(style.metadata), "style_extractions": 1,
        "preview_paths": previews,
        "note": "Preview PNGs show extracted RGBA on a checkerboard; source-face pixels are not copied to the target until rendering.",
    })


def run_image(
    reference: str | Path,
    target: str | Path,
    output: str | Path,
    checkpoints: str | Path,
    *,
    geometry_path: str | Path | None = None,
    landmark_model: str | Path = "models/face_landmarker.task",
    device: str = "auto",
    regions: Sequence[str] = REGIONS,
    strength: float = 1.0,
    semantic_mask: str | Path | np.ndarray | None = None,
) -> dict:
    """Transfer makeup between files, saving result, style patches, and metadata."""
    strength = _strength(strength)
    output = _output_path(output, reference, target)
    geometry = _resolve_geometry(checkpoints, geometry_path, regions)
    reference_rgb, target_rgb = _read_rgb(reference), _read_rgb(target)
    if isinstance(semantic_mask, (str, Path)):
        semantic_mask = _read_mask(semantic_mask)
    with FaceDetector(landmark_model) as detector:
        style = extract_style(reference_rgb, checkpoints, geometry, detector, regions=regions, device=device)
        points = detector.detect(target_rgb)
        if points is None:
            raise ValueError("No usable single face was detected in the target image.")
        started = time.perf_counter()
        result, alpha = render_style(target_rgb, style, points, strength=strength, semantic_mask=semantic_mask)
        render_seconds = time.perf_counter() - started
    _write_image(output, result)
    style_path = save_style(style, output.with_suffix(".style.npz"))
    alpha_path = output.with_suffix(".alpha.png")
    _write_image(alpha_path, np.repeat(np.rint(alpha[..., None] * 255).astype(np.uint8), 3, axis=2))
    return _write_metadata(output, {
        "mode": "image", "reference": str(Path(reference).resolve()),
        "target": str(Path(target).resolve()), "output": str(output.resolve()),
        "width": result.shape[1], "height": result.shape[0], "regions": list(style.patches),
        "strength": strength, "style": dict(style.metadata), "style_extractions": 1,
        "style_path": str(style_path.resolve()), "alpha_path": str(alpha_path.resolve()),
        "face_mask": "landmark_geometry_plus_supplied_visibility" if semantic_mask is not None else "landmark_geometry_fallback",
        "occlusion_aware": semantic_mask is not None,
        "render_seconds": render_seconds,
    })


def run_video(
    reference: str | Path,
    target: str | Path,
    output: str | Path,
    checkpoints: str | Path,
    *,
    geometry_path: str | Path | None = None,
    landmark_model: str | Path = "models/face_landmarker.task",
    device: str = "auto",
    regions: Sequence[str] = REGIONS,
    strength: float = 1.0,
    smoothing: float = 0.6,
    max_frames: int | None = None,
    semantic_mask_provider: Callable[[np.ndarray, int], np.ndarray | None] | None = None,
) -> dict:
    """Extract once and render a video; no-face frames pass through unchanged.

    ``semantic_mask_provider`` receives RGB and a zero-based frame index. Its
    result must be a visibility mask at that frame's resolution. A missing mask
    falls back to landmark geometry. Output uses constant source FPS (30 when
    unavailable) and contains no audio. Measured processing FPS is reported;
    no real-time speed is assumed.
    """
    strength = _strength(strength)
    smoother = TemporalLandmarkSmoother(smoothing)
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer or None.")
    output = _output_path(output, reference, target)
    geometry = _resolve_geometry(checkpoints, geometry_path, regions)
    reference_rgb = _read_rgb(reference)
    capture = cv2.VideoCapture(str(target))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {target}")
    writer = None
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    output_fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 30.0
    frame_count = face_count = semantic_frames = 0
    frame_size = None
    try:
        with FaceDetector(landmark_model) as detector:
            style = extract_style(reference_rgb, checkpoints, geometry, detector, regions=regions, device=device)
            started = time.perf_counter()
            while max_frames is None or frame_count < max_frames:
                ok, bgr = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if writer is None:
                    frame_size = (rgb.shape[1], rgb.shape[0])
                    codec = "MJPG" if output.suffix.lower() == ".avi" else "mp4v"
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), output_fps, frame_size)
                    if not writer.isOpened():
                        raise OSError(f"Cannot open video writer: {output}")
                elif frame_size != (rgb.shape[1], rgb.shape[0]):
                    raise ValueError("Video frame dimensions changed during processing.")
                points = smoother.update(detector.detect(rgb))
                if points is not None:
                    semantic_mask = semantic_mask_provider(rgb, frame_count) if semantic_mask_provider else None
                    semantic_frames += int(semantic_mask is not None)
                    rgb, _ = render_style(rgb, style, points, strength=strength, semantic_mask=semantic_mask)
                    face_count += 1
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                frame_count += 1
            processing_seconds = time.perf_counter() - started
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_count == 0:
        raise ValueError(f"The video contained no decodable frames: {target}")
    style_path = save_style(style, output.with_suffix(".style.npz"))
    return _write_metadata(output, {
        "mode": "video", "reference": str(Path(reference).resolve()),
        "target": str(Path(target).resolve()), "output": str(output.resolve()),
        "width": frame_size[0], "height": frame_size[1], "regions": list(style.patches),
        "strength": strength, "smoothing_previous_weight": smoothing,
        "style": dict(style.metadata), "style_extractions": 1, "style_path": str(style_path.resolve()),
        "frames": frame_count, "frames_with_face": face_count,
        "frames_passthrough_no_face": frame_count - face_count,
        "frames_with_semantic_visibility": semantic_frames,
        "face_mask": "landmark_geometry_plus_supplied_visibility" if semantic_frames else "landmark_geometry_fallback",
        "audio_preserved": False,
        "source_fps": source_fps if np.isfinite(source_fps) and source_fps > 0 else None,
        "output_fps": output_fps, "processing_seconds": processing_seconds,
        "processing_fps": frame_count / processing_seconds if processing_seconds > 0 else None,
        "fps_scope": "decode, detect, smooth, TPS render, and encode; excludes reference extraction",
    })
