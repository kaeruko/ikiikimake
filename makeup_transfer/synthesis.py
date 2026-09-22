"""Procedural RGBA styles and Eq. (1) eye pseudo labels.

These public procedural templates replace the paper's unavailable style
library. They provide controllable paired supervision, not its exact renderer.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import (CanonicalGeometry, REGIONS, EYE_LEFT, EYE_RIGHT,
                       CHEEK_LEFT, CHEEK_RIGHT, region_mask)


def alpha_blend(background: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    background = np.asarray(background, dtype=np.float32)
    if background.max(initial=0) > 1:
        background = background / 255.0
    alpha = np.clip(rgba[..., 3:4], 0, 1)
    return np.clip(background * (1 - alpha) + rgba[..., :3] * alpha, 0, 1)


def _color(rng: np.random.Generator, region: str) -> np.ndarray:
    hue_ranges = {"lip": ((0, 18), (155, 179)), "cheek": ((0, 18), (157, 179)),
                  "eye": ((0, 179),)}
    choices = hue_ranges[region]
    lower, upper = choices[int(rng.integers(len(choices)))]
    saturation = rng.integers(65, 180) if region == "cheek" else rng.integers(85, 240)
    hsv = np.array([[[rng.integers(lower, upper + 1), saturation,
                       rng.integers(65 if region == "eye" else 140, 241)]]], np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32) / 255


def _feather_inside(binary_mask: np.ndarray, width: float) -> np.ndarray:
    """Smoothly approach zero INSIDE the support, keeping protected pixels zero.

    Blurring an alpha mask and then clipping it back to a polygon leaves a
    visible discontinuity at the clip boundary. An inward distance transform
    produces a continuous falloff without putting pigment in eyes or mouth.
    """
    distance = cv2.distanceTransform((binary_mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(distance / max(width, 1), 0, 1)
    return t * t * (3 - 2 * t)


def synthetic_templates(geometry: CanonicalGeometry, rng: np.random.Generator) -> dict[str, np.ndarray]:
    size = geometry.canvas_size
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    templates = {}
    for region in REGIONS:
        allowed = region_mask(geometry, region)
        color = _color(rng, region)
        second = _color(rng, region)
        opacity_ranges = {"lip": (0.3, 0.9), "eye": (0.18, 0.68), "cheek": (0.12, 0.42)}
        opacity = float(rng.uniform(*opacity_ranges[region]))
        feather_ranges = {"lip": (0.0025, 0.006), "eye": (0.009, 0.02), "cheek": (0.025, 0.045)}
        support = _feather_inside(allowed, max(1.5, size * rng.uniform(*feather_ranges[region])))
        if region == "lip":
            alpha = support
            blend = np.clip((yy - geometry.landmarks[13, 1]) / (size * 0.15) + 0.5, 0, 1)[..., None]
            rgb = color[None, None] * (1 - 0.25 * blend) + second[None, None] * 0.25 * blend
        else:
            alpha = np.zeros((size, size), np.float32)
            groups = (EYE_LEFT, EYE_RIGHT) if region == "eye" else (CHEEK_LEFT, CHEEK_RIGHT)
            for indices in groups:
                points = geometry.landmarks[list(indices)]
                center = points.mean(0)
                width = max(float(np.ptp(points[:, 0])), size * 0.025)
                if region == "eye":
                    center[1] -= width * float(rng.uniform(0.15, 0.35))
                    sx, sy = width * rng.uniform(0.45, 0.85), width * rng.uniform(0.23, 0.4)
                else:
                    sx, sy = width * rng.uniform(0.25, 0.4), width * rng.uniform(0.25, 0.38)
                    center += rng.uniform(-0.08, 0.08, 2) * width
                tilt = float(rng.uniform(-0.45, 0.45))
                dx, dy = xx - center[0], yy - center[1]
                gaussian = np.exp(-0.5 * (((dx + tilt * dy) / sx) ** 2 + (dy / sy) ** 2))
                alpha = np.maximum(alpha, gaussian)
            alpha *= support
            blend = (xx / size)[..., None]
            rgb = np.broadcast_to(color, (size, size, 3)).copy() * (1 - blend) + second * blend
            if region == "eye" and rng.random() < 0.65:
                liner = np.zeros((size, size), np.float32)
                for indices in (EYE_LEFT[:4], EYE_RIGHT[:4]):
                    cv2.polylines(liner, [np.rint(geometry.landmarks[list(indices)]).astype(np.int32)],
                                  False, 1.0, max(1, int(size * rng.uniform(0.001, 0.004))))
                liner *= support
                rgb = rgb * (1 - liner[..., None]) + np.array((0.035, 0.02, 0.025)) * liner[..., None]
                alpha = np.maximum(alpha, liner)
        # Modest shimmer/gloss variation, kept independent of the source face.
        if rng.random() < 0.45:
            grain = rng.normal(0, 0.015, (size, size, 1)).astype(np.float32)
            rgb = rgb + grain
        templates[region] = np.concatenate((np.clip(rgb, 0, 1),
                                           (alpha * opacity)[..., None]), axis=2).astype(np.float32)
    return templates


def composite_templates(templates: dict[str, np.ndarray]) -> np.ndarray:
    first = next(iter(templates.values()))
    alpha = np.zeros(first.shape[:2] + (1,), np.float32)
    premultiplied = np.zeros(first.shape[:2] + (3,), np.float32)
    for rgba in templates.values():
        a = rgba[..., 3:4]
        premultiplied = rgba[..., :3] * a + premultiplied * (1 - a)
        alpha = a + alpha * (1 - a)
    rgb = np.divide(premultiplied, alpha, out=np.zeros_like(premultiplied), where=alpha > 1e-7)
    return np.concatenate((rgb, alpha), axis=2)


# Public evaluation API: use the same sampled style on two different faces.
generate_style = synthetic_templates


def eye_kmeans_pseudo(canonical_rgb: np.ndarray, eye_mask: np.ndarray,
                      seed: int = 42, k: int = 6, s: int = 2) -> np.ndarray:
    """LAB cosine alpha; pseudo alpha weights RGB loss, never alpha supervision."""
    image = np.asarray(canonical_rgb, np.float32)
    if image.max(initial=0) > 1:
        image = image / 255
    lab = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2LAB)
    selected = np.asarray(eye_mask) > 0.5
    pixels = lab[selected].reshape(-1, 3)
    if len(pixels) < k or not 1 <= s <= k:
        raise ValueError("Not enough valid eye pixels for k-means, or invalid s/k")
    # Avoid OpenCV's process-global RNG so independent jobs remain reproducible.
    rng = np.random.default_rng(seed)
    fitted = pixels[rng.choice(len(pixels), min(len(pixels), 20000), replace=False)]
    centers = fitted[rng.choice(len(fitted), k, replace=False)].copy()
    for _ in range(40):
        labels = np.argmin(((fitted[:, None] - centers[None]) ** 2).sum(2), axis=1)
        updated = np.array([fitted[labels == i].mean(0) if np.any(labels == i) else centers[i]
                            for i in range(k)])
        if np.max(np.abs(updated - centers)) < 1e-4:
            centers = updated
            break
        centers = updated
    counts = np.bincount(labels, minlength=k)
    top = np.argsort(counts)[-s:]
    skin = np.average(centers[top], axis=0, weights=counts[top])
    denominator = np.linalg.norm(lab, axis=2) * np.linalg.norm(skin)
    similarity = np.divide(np.sum(lab * skin, axis=2), denominator,
                           out=np.ones(lab.shape[:2], np.float32), where=denominator > 1e-8)
    alpha = np.clip(1 - similarity, 0, 1) * selected
    return np.concatenate((image, alpha[..., None]), axis=2).astype(np.float32)
