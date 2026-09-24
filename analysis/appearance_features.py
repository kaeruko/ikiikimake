"""Fixed, descriptive color features on original-resolution facial regions.

This experimental geometry is not skin/hair/occlusion segmentation. Review every
returned mask. There is no beauty score or preferred direction of change. Indices
follow the installed MediaPipe face landmark lip/eye/eyebrow contour topology.
The public API neither reads files nor resizes, smooths, or registers images.
"""

from __future__ import annotations

import cv2
import numpy as np


BASE_NAMES = ("left_cheek", "right_cheek", "forehead")
SIDE_NAMES = ("screen_left", "screen_right")
BROW_INDICES = (
    (70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    (300, 293, 334, 296, 336, 285, 295, 282, 283, 276),
)
UPPER_EYE_INDICES = (
    (33, 246, 161, 160, 159, 158, 157, 173, 133),
    (263, 466, 388, 387, 386, 385, 384, 398, 362),
)
LOWER_EYE_INDICES = (
    (33, 7, 163, 144, 145, 153, 154, 155, 133),
    (263, 249, 390, 373, 374, 380, 381, 382, 362),
)
NOSE_WING_INDICES = (98, 327)
MOUTH_CORNER_INDICES = (61, 291)
OUTER_LIP_INDICES = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                     409, 270, 269, 267, 0, 37, 39, 40, 185)
INNER_LIP_INDICES = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                     415, 310, 311, 312, 13, 82, 81, 80, 191)
FACE_OVAL_INDICES = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
MIN_TARGET_PIXELS = {"brow": 32, "upper_lid": 24, "lips": 60, "skin": 100}
MIN_REFERENCE_PIXELS = 64
HIGHLIGHT_L_OFFSET = 8.0
TEXTURE_BLUR_SIGMA = 1.2
SESC_INSPIRED_THRESHOLD_MULTIPLIER = 19.0 / 13.0
SESC_INSPIRED_MAX_GRAY = 240.0
GVR_INSPIRED_CLAHE_CLIP_LIMIT = 2.0
GVR_INSPIRED_CLAHE_TILE_GRID = (8, 8)
SKIN_UNEVENNESS_SIGMA_FRACTION = 0.08
SKIN_FINE_LINE_KERNEL_FRACTION = 0.08
SKIN_FINE_LINE_KERNEL_MIN = 5
SKIN_FINE_LINE_KERNEL_MAX = 15
SKIN_FINE_LINE_PERCENTILE = 95
NASOLABIAL_MIN_CANDIDATE_PIXELS = 100
NASOLABIAL_MIN_CONTROL_PIXELS = 50
GVR_INSPIRED_REGIONS = (
    ("screen_left_upper_lid_skin", "画面左眉下の皮膚"),
    ("screen_right_upper_lid_skin", "画面右眉下の皮膚"),
    ("left_cheek", "画面左頬・対照"),
    ("right_cheek", "画面右頬・対照"),
    ("forehead", "額・対照"),
)

SKIN_APPEARANCE_REGIONS = (
    ("screen_left_upper_lid_skin", "画面左眉下の皮膚"),
    ("screen_right_upper_lid_skin", "画面右眉下の皮膚"),
    ("left_cheek", "画面左頬"),
    ("right_cheek", "画面右頬"),
    ("forehead", "額"),
)


def _binary_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != shape or mask.ndim != 2:
        raise ValueError(f"{name}: mask must match image height and width")
    if mask.dtype != np.bool_ and mask.dtype != np.uint8:
        raise ValueError(f"{name}: mask must be bool or binary uint8")
    values = set(np.unique(mask).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f"{name}: mask is not binary")
    return mask.astype(bool, copy=True)


def _polygon_mask(points: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    height, width = shape
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all()):
        raise ValueError(f"{name}: invalid polygon")
    if (np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height)):
        raise ValueError(f"{name}: polygon would be clipped by the image boundary")
    vertices = np.rint(points).astype(np.int32)
    if np.any(vertices[:, 0] >= width) or np.any(vertices[:, 1] >= height):
        raise ValueError(f"{name}: rounded polygon would leave the image")
    mask = np.zeros(shape, dtype=np.uint8)
    if abs(cv2.contourArea(vertices)) >= 0.5:
        cv2.fillPoly(mask, [vertices], 1)
    return mask.astype(bool)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _strip_mask(line: np.ndarray, direction: np.ndarray, near: float, far: float,
                shape: tuple[int, int], name: str) -> np.ndarray:
    polygon = np.vstack((line + direction * near, (line + direction * far)[::-1]))
    return _polygon_mask(polygon, shape, name)


def _quadratic_bezier_band_mask(
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
    half_width: float,
    shape: tuple[int, int],
    name: str,
    outward_hint: np.ndarray,
    normal_offset: float = 0.0,
    samples: int = 32,
) -> np.ndarray:
    if samples < 8:
        raise ValueError(f"{name}: samples must be >= 8")
    start = np.asarray(start, dtype=float)
    control = np.asarray(control, dtype=float)
    end = np.asarray(end, dtype=float)
    hint = np.asarray(outward_hint, dtype=float)
    if any(point.shape != (2,) or not np.isfinite(point).all()
           for point in (start, control, end, hint)):
        raise ValueError(f"{name}: invalid bezier geometry")
    if float(np.linalg.norm(end - start)) < 1.0:
        raise ValueError(f"{name}: bezier endpoints are degenerate")
    if float(np.linalg.norm(hint)) < 1.0:
        raise ValueError(f"{name}: outward hint is invalid")
    if half_width <= 0.0 or normal_offset < 0.0:
        raise ValueError(f"{name}: band widths must be positive")

    t = np.linspace(0.0, 1.0, samples)
    omt = 1.0 - t
    curve = (
        (omt * omt)[:, None] * start
        + (2.0 * omt * t)[:, None] * control
        + (t * t)[:, None] * end
    )
    tangent = (
        (2.0 * omt)[:, None] * (control - start)
        + (2.0 * t)[:, None] * (end - control)
    )
    tangent_length = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_length < 1e-6):
        raise ValueError(f"{name}: bezier tangent is degenerate")
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0])) / tangent_length[:, None]
    mid = len(normals) // 2
    if float(np.dot(normals[mid], hint)) < 0.0:
        normals = -normals

    outer = curve + normals * (normal_offset + half_width)
    inner = curve + normals * (normal_offset - half_width)
    polygon = np.vstack((outer, inner[::-1]))
    return _polygon_mask(polygon, shape, name)


def _odd_kernel_size(value: float) -> int:
    size = int(round(float(value)))
    size = max(SKIN_FINE_LINE_KERNEL_MIN, min(SKIN_FINE_LINE_KERNEL_MAX, size))
    if size % 2 == 0:
        size += 1 if size < SKIN_FINE_LINE_KERNEL_MAX else -1
    return size


def measure_skin_appearance_features(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    regions: tuple[tuple[str, str], ...] = SKIN_APPEARANCE_REGIONS,
) -> list[dict]:
    """Measure visible unevenness and fine dark-line appearance separately.

    These are image-space appearance features for makeup before/after review.
    They are not measurements of skin hydration, clinical dryness, or physical
    wrinkle depth.

    Low-frequency unevenness is measured after a mask-normalized Gaussian blur,
    so broad L* and a*/b* variation is separated from fine texture. Fine dark
    lines are measured with an L* black-hat transform inside an eroded ROI and
    normalized by the local median L*. The black-hat tail may still respond to
    pores, makeup boundaries, focus, lighting, and compression.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError("Expected a nonempty uint8 BGR image")
    if not isinstance(regions, tuple) or not regions:
        raise ValueError("regions must be a nonempty tuple of (mask_name, label) pairs")

    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB).astype(np.float64)
    rows = []
    for item in regions:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
            or not item[1]
        ):
            raise ValueError(f"invalid skin appearance region: {item!r}")
        name, label = item
        if name not in masks:
            raise ValueError(f"Missing skin appearance mask: {name}")
        mask = _binary_mask(masks[name], image.shape[:2], name)
        count = int(mask.sum())
        minimum = MIN_TARGET_PIXELS["skin"]

        if count < minimum:
            status = "insufficient_pixels"
            lowfreq_l_mad = None
            lowfreq_ab_mad = None
            fine_line_p95 = None
            sigma = None
            kernel_size = None
            valid_line_pixels = 0
        else:
            status = "ok"
            roi_scale = float(np.sqrt(count))
            sigma = max(1.0, SKIN_UNEVENNESS_SIGMA_FRACTION * roi_scale)
            mask_float = mask.astype(np.float64)
            weight = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
            if not np.isfinite(weight).all() or np.any(weight[mask] <= 1e-9):
                raise RuntimeError(f"{name}: invalid normalized-blur weights")

            smoothed = []
            for channel in range(3):
                numerator = cv2.GaussianBlur(
                    lab[:, :, channel] * mask_float,
                    (0, 0),
                    sigmaX=sigma,
                    sigmaY=sigma,
                )
                channel_smooth = numerator / np.maximum(weight, 1e-12)
                smoothed.append(channel_smooth)
            smooth_l, smooth_a, smooth_b = smoothed

            l_values = smooth_l[mask]
            a_values = smooth_a[mask]
            b_values = smooth_b[mask]
            med_l = float(np.median(l_values))
            med_a = float(np.median(a_values))
            med_b = float(np.median(b_values))
            lowfreq_l_mad = float(np.median(np.abs(l_values - med_l)))
            lowfreq_ab_mad = float(np.median(np.hypot(a_values - med_a, b_values - med_b)))

            local_median_l = float(np.median(lab[:, :, 0][mask]))
            if not np.isfinite(local_median_l) or local_median_l <= 1e-6:
                raise ValueError(f"{name}: local median L* is too small")
            kernel_size = _odd_kernel_size(SKIN_FINE_LINE_KERNEL_FRACTION * roi_scale)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            inner = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
            valid_line_pixels = int(inner.sum())
            if valid_line_pixels < minimum:
                fine_line_p95 = None
                status = "insufficient_line_pixels"
            else:
                filled_l = lab[:, :, 0].copy()
                filled_l[~mask] = local_median_l
                blackhat = cv2.morphologyEx(
                    filled_l.astype(np.float32), cv2.MORPH_BLACKHAT, kernel
                ).astype(np.float64)
                fine_line_p95 = float(
                    100.0
                    * np.percentile(blackhat[inner], SKIN_FINE_LINE_PERCENTILE)
                    / local_median_l
                )

        common_unevenness_note = (
            "ROI内部だけを使う正規化Gaussian blurで低周波成分を作り、"
            "そのL*またはa*/b*の中央値からのMADを計算。"
            "色・明るさの広いムラを見る画像指標で、乾燥そのものではない。"
            + (
                f" sigma={sigma:.2f}px（ROI面積由来）。"
                if sigma is not None else ""
            )
            + f" 対象{count}画素（必要{minimum}以上）。"
        )
        fine_line_note = (
            "ROI外を局所L*中央値で埋め、ROI面積に応じた黒帽変換で細い暗線候補を抽出し、"
            f"内部画素のp{SKIN_FINE_LINE_PERCENTILE}を局所L*中央値で正規化。"
            "乾燥で目立つ細線の候補を見るための画像指標だが、毛穴・メイク境界・"
            "ピント・照明・圧縮にも反応し得る。物理的なシワ深さではない。"
            + (
                f" kernel={kernel_size}px、内部{valid_line_pixels}画素。"
                if kernel_size is not None else ""
            )
        )
        rows.extend((
            {
                "id": f"{name}_lowfreq_L_mad",
                "region": name,
                "label": f"{label}の低周波明度ムラ（L* MAD）",
                "unit": "L*",
                "value": lowfreq_l_mad,
                "pixels": count,
                "status": status if status != "insufficient_line_pixels" else "ok",
                "note": common_unevenness_note,
            },
            {
                "id": f"{name}_lowfreq_ab_mad",
                "region": name,
                "label": f"{label}の低周波色ムラ（ab MAD）",
                "unit": "Lab",
                "value": lowfreq_ab_mad,
                "pixels": count,
                "status": status if status != "insufficient_line_pixels" else "ok",
                "note": common_unevenness_note,
            },
            {
                "id": f"{name}_fine_dark_line_p95_pct",
                "region": name,
                "label": f"{label}の細線候補・暗さコントラスト（p95）",
                "unit": "局所L*比 %",
                "value": fine_line_p95,
                "pixels": valid_line_pixels,
                "status": status,
                "note": fine_line_note,
            },
        ))
    return rows


def measure_gvr_inspired_features(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[dict]:
    """Measure a visible-light hydration-related GVR approximation.

    Wu et al. (2024) used HSI intensity I=(R+G+B)/3, CLAHE, a reflectance
    image obtained by subtracting the CLAHE image from the monochrome image,
    and GVR = mean grayscale value of ROI / sum grayscale value of whole image.

    The paper does not report CLAHE parameters. This implementation fixes
    OpenCV CLAHE to clipLimit=2.0 and tileGridSize=(8, 8), and clips negative
    I-CLAHE(I) residuals to zero. Therefore these values are named
    GVR-inspired and are not interchangeable with the paper's calibrated GVR.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError("Expected a nonempty uint8 BGR image")

    selected = {}
    for name, _ in GVR_INSPIRED_REGIONS:
        if name not in masks:
            raise ValueError(f"Missing GVR-inspired mask: {name}")
        selected[name] = _binary_mask(masks[name], image.shape[:2], name)

    # HSI intensity from the paper: I = (R + G + B) / 3.
    intensity = np.rint(image.astype(np.float32).mean(axis=2)).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=GVR_INSPIRED_CLAHE_CLIP_LIMIT,
        tileGridSize=GVR_INSPIRED_CLAHE_TILE_GRID,
    )
    enhanced = clahe.apply(intensity)
    reflectance = np.clip(
        intensity.astype(np.int16) - enhanced.astype(np.int16),
        0,
        None,
    ).astype(np.float64)
    whole_sum = float(reflectance.sum())
    if not np.isfinite(whole_sum) or whole_sum <= 0:
        raise ValueError(
            "GVR-inspired reflectance image has zero total intensity; metric is undefined"
        )

    rows = []
    for name, label in GVR_INSPIRED_REGIONS:
        mask = selected[name]
        count = int(mask.sum())
        minimum = MIN_TARGET_PIXELS["skin"]
        if count >= minimum:
            roi_mean = float(np.mean(reflectance[mask]))
            value = roi_mean / whole_sum
            status = "ok"
        else:
            value = None
            status = "insufficient_pixels"
        rows.append({
            "id": f"{name}_gvr_inspired_ratio",
            "region": name,
            "label": f"{label}のGVR-inspired反射プロキシ",
            "unit": "ratio",
            "value": value,
            "pixels": count,
            "status": status,
            "note": (
                "Wu et al. (2024) の可視光GVRを参考に、HSI intensity I=(R+G+B)/3へCLAHEを適用し、"
                "I-CLAHE(I)の負値を0へクリップしたreflectance-like画像について、"
                "ROI平均 / 画像全体の画素和を計算。論文ではCLAHEパラメータが未記載のため、"
                f"OpenCV clipLimit={GVR_INSPIRED_CLAHE_CLIP_LIMIT:.1f}, "
                f"tileGridSize={GVR_INSPIRED_CLAHE_TILE_GRID}を固定した研究用近似。"
                "EPISCANの標準化撮影を再現しておらず、皮膚水分量や乾燥の診断値ではない。 "
                f"元画像の対象{count}画素（必要{minimum}画素以上）。"
            ),
        })
    return rows



def measure_nasolabial_crease_features(
    image_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[dict]:
    """Measure image-space nasolabial crease conspicuity against nearby skin.

    This is not physical wrinkle depth. For each screen side, the median L* of
    the cheekward control band is the local skin baseline. Candidate-band pixels
    darker than that baseline are converted to positive relative darkness:
    100 * max(control_median_L - candidate_L, 0) / control_median_L.
    Median describes typical darkening; p90 describes the stronger dark tail.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError("Expected a nonempty uint8 BGR image")

    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float64)
    rows = []

    for side, label in (("screen_left", "画面左"), ("screen_right", "画面右")):
        candidate_name = f"{side}_nasolabial_candidate"
        control_name = f"{side}_nasolabial_outer_control"
        if candidate_name not in masks:
            raise ValueError(f"Missing nasolabial mask: {candidate_name}")
        if control_name not in masks:
            raise ValueError(f"Missing nasolabial mask: {control_name}")
        candidate = _binary_mask(masks[candidate_name], image.shape[:2], candidate_name)
        control = _binary_mask(masks[control_name], image.shape[:2], control_name)
        if np.any(candidate & control):
            raise ValueError(f"{side}: nasolabial candidate overlaps outer control")

        candidate_count = int(candidate.sum())
        control_count = int(control.sum())
        status = (
            "ok"
            if candidate_count >= NASOLABIAL_MIN_CANDIDATE_PIXELS
            and control_count >= NASOLABIAL_MIN_CONTROL_PIXELS
            else "insufficient_pixels"
        )

        if status == "ok":
            control_median_l = float(np.median(lightness[control]))
            if not np.isfinite(control_median_l) or control_median_l <= 1e-6:
                raise ValueError(f"{side}: nasolabial control median L* is too small")
            darkness = (
                100.0
                * np.maximum(control_median_l - lightness[candidate], 0.0)
                / control_median_l
            )
            median_value = float(np.median(darkness))
            p90_value = float(np.percentile(darkness, 90))
        else:
            median_value = None
            p90_value = None

        common_note = (
            "頬側の周囲皮膚対照帯のL*中央値を局所基準にし、候補帯の各画素について "
            "100×max(対照L*中央値−候補L*, 0)/対照L*中央値 を計算。"
            "物理的なシワ深さではなく、画像上で溝・陰影が周囲皮膚より暗く見える度合い。"
            f" 候補{candidate_count}画素（必要{NASOLABIAL_MIN_CANDIDATE_PIXELS}以上）、"
            f"対照{control_count}画素（必要{NASOLABIAL_MIN_CONTROL_PIXELS}以上）。"
        )
        rows.extend((
            {
                "id": f"{side}_nasolabial_crease_darkness_median_pct",
                "region": candidate_name,
                "label": f"{label}ほうれい線候補の周囲皮膚比・暗さコントラスト（中央値）",
                "unit": "局所L*比 %",
                "value": median_value,
                "pixels": candidate_count,
                "status": status,
                "note": common_note,
            },
            {
                "id": f"{side}_nasolabial_crease_darkness_p90_pct",
                "region": candidate_name,
                "label": f"{label}ほうれい線候補の周囲皮膚比・暗さコントラスト（p90）",
                "unit": "局所L*比 %",
                "value": p90_value,
                "pixels": candidate_count,
                "status": status,
                "note": common_note,
            },
        ))
    return rows


def build_feature_masks(
    image_shape: tuple,
    points_px: np.ndarray,
    base_masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return all target/reference masks, with screen-side names and no clipping.

    Brow targets follow the brow contour. Upper-lid bands use the middle five
    upper-eye contour points shifted toward the brow by .008–.025 face widths;
    the eye aperture and an extra pixel margin are excluded. Their skin reference
    uses .040–.068 widths above that contour, excluding eyebrows and all targets.
    Brow skin uses a strip .010–.040 widths above the top brow contour. Lower-eye
    skin uses a .012–.055-width strip below the lower-eye contour, outside the eye
    aperture. Nasolabial review ROIs use a quadratic Bezier curve from the
    nose-wing landmark toward the mouth corner, bowed toward the cheek. A second,
    non-overlapping band farther toward the cheek is retained as surrounding-skin
    control. These are review ROIs, not detected wrinkles or physical depth.
    Lip skin is an external .012–.035-width ring. Mouth interior has an additional exclusion
    margin; the outer lip loses one boundary pixel. All measurements keep native
    pixel counts, so empty/thin masks remain unavailable rather than being enlarged.
    """
    if len(image_shape) < 2:
        raise ValueError("image_shape must provide height and width")
    height, width = image_shape[:2]
    if (isinstance(height, bool) or isinstance(width, bool) or int(height) != height
            or int(width) != width or height < 1 or width < 1):
        raise ValueError("image dimensions must be positive integers")
    shape = (int(height), int(width))
    points = np.asarray(points_px, dtype=float)
    if (points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2
            or not np.isfinite(points).all()):
        raise ValueError("Expected at least 468 finite xy face landmarks")
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        raise ValueError("Face landmarks leave the image; clipping is not permitted")
    face_width = float(np.linalg.norm(points[454] - points[234]))
    if face_width < 1:
        raise ValueError("Face width is degenerate")
    masks = {}
    occupied = np.zeros(shape, bool)
    for name in BASE_NAMES:
        if name not in base_masks:
            raise ValueError(f"Missing base mask: {name}")
        mask = _binary_mask(base_masks[name], shape, name)
        if np.any(mask & occupied):
            raise ValueError("Base cheek/forehead masks must not overlap")
        masks[name] = mask
        occupied |= mask
    face_mask = _polygon_mask(points[list(FACE_OVAL_INDICES)], shape, "face oval")
    if not face_mask.any():
        raise ValueError("Face oval is degenerate")
    # Eye+brow groups stay together when the image is mirrored.
    groups = sorted(range(2), key=lambda i: points[list(UPPER_EYE_INDICES[i]), 0].mean())
    raw_references = {}
    eye_guard = np.zeros(shape, bool)
    brow_guard = np.zeros(shape, bool)
    margin = max(1, round(face_width * 0.004))
    for side, group in zip(SIDE_NAMES, groups):
        brow = points[list(BROW_INDICES[group])]
        upper_eye = points[list(UPPER_EYE_INDICES[group])]
        lower_eye = points[list(LOWER_EYE_INDICES[group])]
        eye_polygon = np.vstack((upper_eye, lower_eye[-2:0:-1]))
        eye = _polygon_mask(eye_polygon, shape, f"{side} eye aperture")
        group_eye_guard = _dilate(eye, margin)
        eye_guard |= group_eye_guard
        brow_target = _polygon_mask(brow, shape, f"{side} brow") & ~group_eye_guard & face_mask
        masks[f"{side}_brow"] = brow_target
        brow_guard |= _dilate(brow_target, margin)
        away_from_eye = brow.mean(axis=0) - eye_polygon.mean(axis=0)
        magnitude = float(np.linalg.norm(away_from_eye))
        if magnitude < 1:
            masks[f"{side}_upper_lid"] = np.zeros(shape, bool)
            raw_references[f"{side}_brow_skin"] = np.zeros(shape, bool)
            raw_references[f"{side}_upper_lid_skin"] = np.zeros(shape, bool)
            masks[f"{side}_lower_eye_skin"] = np.zeros(shape, bool)
            continue
        direction = away_from_eye / magnitude
        masks[f"{side}_upper_lid"] = _strip_mask(
            upper_eye[2:-2], direction, 0.008 * face_width, 0.025 * face_width,
            shape, f"{side} upper lid",
        ) & ~group_eye_guard & face_mask
        raw_references[f"{side}_brow_skin"] = _strip_mask(
            brow[:5], direction, 0.010 * face_width, 0.040 * face_width,
            shape, f"{side} brow skin",
        )
        raw_references[f"{side}_upper_lid_skin"] = _strip_mask(
            upper_eye[2:-2], direction, 0.040 * face_width, 0.068 * face_width,
            shape, f"{side} upper-lid skin",
        )
        masks[f"{side}_lower_eye_skin"] = _strip_mask(
            lower_eye[2:-2], -direction, 0.012 * face_width, 0.055 * face_width,
            shape, f"{side} lower-eye skin",
        ) & ~group_eye_guard & face_mask
    for side in SIDE_NAMES:
        masks[f"{side}_upper_lid"] &= ~brow_guard
    outer_lips = _polygon_mask(points[list(OUTER_LIP_INDICES)], shape, "outer lips")
    mouth = _polygon_mask(points[list(INNER_LIP_INDICES)], shape, "mouth interior")
    # A closed mouth can make the interior contour zero-area. Rasterize its
    # contour as a line as well, so that the mouth seam is always excluded.
    seam = np.zeros(shape, np.uint8)
    cv2.polylines(seam, [np.rint(points[list(INNER_LIP_INDICES)]).astype(np.int32)], True, 1, 1)
    mouth_guard = _dilate(mouth | seam.astype(bool), margin)

    nose_order = sorted(
        range(2),
        key=lambda i: points[NOSE_WING_INDICES[i], 0],
    )
    mouth_order = sorted(
        range(2),
        key=lambda i: points[MOUTH_CORNER_INDICES[i], 0],
    )
    for side, nose_i, mouth_i, cheek_name in (
        ("screen_left", nose_order[0], mouth_order[0], "left_cheek"),
        ("screen_right", nose_order[1], mouth_order[1], "right_cheek"),
    ):
        nose_point = points[NOSE_WING_INDICES[nose_i]]
        mouth_point = points[MOUTH_CORNER_INDICES[mouth_i]]
        cheek_y, cheek_x = np.nonzero(masks[cheek_name])
        if len(cheek_x) == 0:
            raise ValueError(f"{cheek_name}: base cheek mask is empty")
        cheek_center = np.array([cheek_x.mean(), cheek_y.mean()], dtype=float)

        end_point = nose_point + (mouth_point - nose_point) * 0.70
        line_midpoint = 0.5 * (nose_point + end_point)
        outward_hint = cheek_center - line_midpoint
        outward_length = float(np.linalg.norm(outward_hint))
        if outward_length < 1.0:
            raise ValueError(f"{side}: nasolabial outward direction is degenerate")
        outward_unit = outward_hint / outward_length
        control_point = line_midpoint + outward_unit * (0.040 * face_width)

        # The previous review showed that the outer cyan band, not the inner
        # yellow band, followed the visible nasolabial crease. Promote that
        # reviewed geometry to the candidate and move the control farther cheekward.
        candidate = _quadratic_bezier_band_mask(
            nose_point,
            control_point,
            end_point,
            0.014 * face_width,
            shape,
            f"{side} nasolabial candidate",
            outward_hint=outward_hint,
            normal_offset=0.052 * face_width,
        )
        candidate &= face_mask & ~mouth_guard

        outer_control = _quadratic_bezier_band_mask(
            nose_point,
            control_point,
            end_point,
            0.014 * face_width,
            shape,
            f"{side} nasolabial outer control",
            outward_hint=outward_hint,
            normal_offset=0.090 * face_width,
        )
        outer_control &= face_mask & ~mouth_guard & ~candidate

        if not candidate.any():
            raise ValueError(f"{side}: nasolabial candidate is empty")
        if not outer_control.any():
            raise ValueError(f"{side}: nasolabial outer control is empty")
        masks[f"{side}_nasolabial_candidate"] = candidate
        masks[f"{side}_nasolabial_outer_control"] = outer_control

    eroded_lips = cv2.erode(outer_lips.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    masks["lips"] = eroded_lips & ~mouth_guard & face_mask
    inner_radius = max(1, round(face_width * 0.012))
    outer_radius = max(inner_radius + 1, round(face_width * 0.035))
    if outer_lips.any():
        ys, xs = np.nonzero(outer_lips)
        if (xs.min() < outer_radius or ys.min() < outer_radius
                or xs.max() + outer_radius >= width or ys.max() + outer_radius >= height):
            raise ValueError("Lip skin ring would be clipped by the image boundary")
    raw_references["lip_skin"] = _dilate(outer_lips, outer_radius) & ~_dilate(outer_lips, inner_radius)
    target_union = outer_lips.copy()
    for side in SIDE_NAMES:
        target_union |= masks[f"{side}_brow"] | masks[f"{side}_upper_lid"]
    excluded_from_skin = _dilate(target_union, margin) | eye_guard | mouth_guard
    for name, reference in raw_references.items():
        masks[name] = reference & face_mask & ~excluded_from_skin
    return masks


def measure_features(image_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict]:
    """Measure fixed float-Lab statistics; small targets/references yield None.

    BGR uint8 is converted using ``cvtColor(image.astype(float32)/255, BGR2LAB)``.
    ``pixels`` counts the native target pixels; notes additionally record reference
    counts/requirements. No interpolation or resizing manufactures extra samples.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError("Expected a nonempty uint8 BGR image")
    targets = [f"{side}_{feature}" for side in SIDE_NAMES for feature in ("brow", "upper_lid")]
    lower_eye_regions = tuple(f"{side}_lower_eye_skin" for side in SIDE_NAMES)
    required = (*BASE_NAMES, *targets, *lower_eye_regions,
                *(f"{name}_skin" for name in targets), "lips", "lip_skin")
    selected = {}
    for name in required:
        if name not in masks:
            raise ValueError(f"Missing feature mask: {name}")
        selected[name] = _binary_mask(masks[name], image.shape[:2], name)
    all_targets = selected["lips"].copy()
    for name in targets:
        all_targets |= selected[name]
    for name in (*(f"{target}_skin" for target in targets), "lip_skin"):
        if np.any(selected[name] & all_targets):
            raise ValueError(f"{name}: reference skin overlaps a brow/eyelid/lip target")
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    values = {name: lab[mask] for name, mask in selected.items()}
    lightness = lab[:, :, 0]
    low_frequency = cv2.GaussianBlur(lightness, (0, 0), TEXTURE_BLUR_SIGMA)
    highpass_abs = np.abs(lightness - low_frequency)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    rows = []

    def add(identifier, region, label, unit, value_fn, note, minimum, reference=None):
        count = len(values[region])
        reference_count = len(values[reference]) if reference is not None else None
        ok = count >= minimum and (reference_count is None or reference_count >= MIN_REFERENCE_PIXELS)
        count_note = f" 元画像の対象{count}画素（必要{minimum}画素以上）。"
        if reference is not None:
            count_note += f"参照皮膚{reference_count}画素（必要{MIN_REFERENCE_PIXELS}画素以上）。"
        rows.append({"id": identifier, "region": region, "label": label, "unit": unit,
                     "value": float(value_fn()) if ok else None, "pixels": count,
                     "status": "ok" if ok else "insufficient_pixels", "note": note + count_note})

    for side, side_label in (("screen_left", "画面左"), ("screen_right", "画面右")):
        for feature, feature_label in (("brow", "眉"), ("upper_lid", "上まぶた外側帯")):
            name = f"{side}_{feature}"
            reference = f"{name}_skin"
            add(f"{name}_relative_L_contrast", name, f"{side_label}{feature_label}の相対明度コントラスト", "L*",
                lambda n=name, r=reference: np.median(values[r][:, 0]) - np.percentile(values[n][:, 0], 25),
                "参照皮膚L*中央値 − 対象L*第25百分位数（p25）。正値は対象が相対的に暗いことを示すだけで、"
                "改善判定ではない。毛・影・照明・位置ずれも混入する。", MIN_TARGET_PIXELS[feature], reference)
    for side, side_label in (("screen_left", "画面左"), ("screen_right", "画面右")):
        name = f"{side}_lower_eye_skin"
        count = int(selected[name].sum())
        minimum = MIN_TARGET_PIXELS["skin"]
        median_l = float(np.median(lightness[selected[name]])) if count else 0.0
        def texture_stat(percentile, n=name, base_l=median_l):
            if base_l <= 1e-6:
                raise ValueError(f"{n}: median L* is too small for normalized texture measurement")
            vals = highpass_abs[selected[n]]
            return 100.0 * np.percentile(vals, percentile) / base_l
        add(f"{name}_highpass_median_pct", name, f"{side_label}目の下の細かな質感コントラスト（中央値）", "局所L*比 %",
            lambda n=name, base_l=median_l: texture_stat(50, n, base_l),
            f"Gaussian blur σ={TEXTURE_BLUR_SIGMA:.1f}px を引いた |L*残差| の中央値を領域L*中央値で正規化。"
            "細かな凹凸・線・メイク境界・ピント・圧縮ノイズをまとめて拾う画像指標で、乾燥・シワの診断ではない。",
            minimum)
        add(f"{name}_highpass_p90_pct", name, f"{side_label}目の下の細かな質感コントラスト（p90）", "局所L*比 %",
            lambda n=name, base_l=median_l: texture_stat(90, n, base_l),
            f"Gaussian blur σ={TEXTURE_BLUR_SIGMA:.1f}px を引いた |L*残差| の90百分位を領域L*中央値で正規化。"
            "局所的に強く見える細線や粒状感の候補で、乾燥・シワの診断ではない。",
            minimum)

    for side, side_label in (("screen_left", "画面左"), ("screen_right", "画面右")):
        name = f"{side}_upper_lid_skin"
        count = int(selected[name].sum())
        minimum = MIN_TARGET_PIXELS["skin"]
        median_l = float(np.median(lightness[selected[name]])) if count else 0.0
        def texture_stat(percentile, n=name, base_l=median_l):
            if base_l <= 1e-6:
                raise ValueError(f"{n}: median L* is too small for normalized texture measurement")
            vals = highpass_abs[selected[n]]
            return 100.0 * np.percentile(vals, percentile) / base_l
        add(f"{name}_highpass_median_pct", name, f"{side_label}眉下の皮膚の細かな質感コントラスト（中央値）", "局所L*比 %",
            lambda n=name, base_l=median_l: texture_stat(50, n, base_l),
            f"Gaussian blur σ={TEXTURE_BLUR_SIGMA:.1f}px を引いた |L*残差| の中央値を領域L*中央値で正規化。"
            "細かな凹凸・線・メイク境界・ピント・圧縮ノイズをまとめて拾う画像指標で、乾燥・シワの診断ではない。",
            minimum)
        add(f"{name}_highpass_p90_pct", name, f"{side_label}眉下の皮膚の細かな質感コントラスト（p90）", "局所L*比 %",
            lambda n=name, base_l=median_l: texture_stat(90, n, base_l),
            f"Gaussian blur σ={TEXTURE_BLUR_SIGMA:.1f}px を引いた |L*残差| の90百分位を領域L*中央値で正規化。"
            "局所的に強く見える細線や粒状感の候補で、乾燥・シワの診断ではない。",
            minimum)

    for name, label in (("left_cheek", "画面左頬"), ("right_cheek", "画面右頬"), ("forehead", "額")):
        count = int(selected[name].sum())
        minimum = MIN_TARGET_PIXELS["skin"]
        median_l = float(np.median(lightness[selected[name]])) if count else 0.0
        def texture_stat(percentile, n=name, base_l=median_l):
            if base_l <= 1e-6:
                raise ValueError(f"{n}: median L* is too small for normalized texture measurement")
            vals = highpass_abs[selected[n]]
            return 100.0 * np.percentile(vals, percentile) / base_l
        add(f"{name}_highpass_median_pct", name, f"{label}の細かな質感コントラスト（中央値・対照）", "局所L*比 %",
            lambda n=name, base_l=median_l: texture_stat(50, n, base_l),
            f"Gaussian blur σ={TEXTURE_BLUR_SIGMA:.1f}px を引いた |L*残差| の中央値を領域L*中央値で正規化。"
            "眉下ROIと同じ式で撮影条件由来の高周波変化を確認する対照指標。乾燥・シワの診断ではない。",
            minimum)
        add(f"{name}_highpass_p90_pct", name, f"{label}の細かな質感コントラスト（p90・対照）", "局所L*比 %",
            lambda n=name, base_l=median_l: texture_stat(90, n, base_l),
            f"Gaussian blur σ={TEXTURE_BLUR_SIGMA:.1f}px を引いた |L*残差| の90百分位を領域L*中央値で正規化。"
            "眉下ROIと同じ式で撮影条件由来の局所的な細線・粒状感を確認する対照指標。乾燥・シワの診断ではない。",
            minimum)

    for name, label in (
        ("screen_left_upper_lid_skin", "画面左眉下の皮膚"),
        ("screen_right_upper_lid_skin", "画面右眉下の皮膚"),
        ("left_cheek", "画面左頬・対照"),
        ("right_cheek", "画面右頬・対照"),
        ("forehead", "額・対照"),
    ):
        def sesc_inspired(n=name):
            vals = gray[selected[n]]
            mean_gray = float(np.mean(vals))
            threshold = SESC_INSPIRED_THRESHOLD_MULTIPLIER * mean_gray
            bright_scale = (vals > threshold) & (vals <= SESC_INSPIRED_MAX_GRAY)
            return 100.0 * np.mean(bright_scale)
        add(
            f"{name}_sesc_inspired_scaliness_pct",
            name,
            f"{label}のSEsc-inspired bright-scaliness率",
            "%",
            sesc_inspired,
            "Visioscan SEscで公開されている閾値定義を参考に、ROI平均grayの19/13倍より明るく、"
            "gray 240以下の画素割合を計算する。通常のBGR動画をOpenCV grayへ変換した研究用の近似指標で、"
            "Visioscan専用UVA撮影によるSEscそのものではなく、乾燥・鱗屑の診断値でもない。",
            MIN_TARGET_PIXELS["skin"],
        )

    add("lips_relative_a", "lips", "唇と周囲皮膚のa*差", "a*",
        lambda: np.median(values["lips"][:, 1]) - np.median(values["lip_skin"][:, 1]),
        "唇a*中央値 − 周囲皮膚a*中央値。口腔内・口の境界線は除外。正負に良し悪しを割り当てない。",
        MIN_TARGET_PIXELS["lips"], "lip_skin")
    add("lips_skin_delta_e76", "lips", "唇と周囲皮膚のLab色差", "ΔE*ab（CIE76）",
        lambda: np.linalg.norm(np.median(values["lips"], axis=0) - np.median(values["lip_skin"], axis=0)),
        "唇と周囲皮膚のL*・a*・b*各中央値のユークリッド距離。口腔内を除外した領域間の色差で、"
        "塗布効果や知覚的な改善量を保証しない。", MIN_TARGET_PIXELS["lips"], "lip_skin")
    for name, label in (("left_cheek", "画面左頬"), ("right_cheek", "画面右頬"), ("forehead", "額")):
        for index, channel in enumerate(("L", "a", "b")):
            add(f"{name}_{channel}_median", name, f"{label}の{channel}*中央値", f"{channel}*",
                lambda n=name, c=index: np.median(values[n][:, c]),
                "固定ROI内の中央値。部位の色と撮影条件の確認用で、照明・影・化粧の影響を分離しない。",
                MIN_TARGET_PIXELS["skin"])
        if name == "forehead":
            continue
        add(f"{name}_ab_mad", name, f"{label}の色ばらつき", "Lab a*b*距離",
            lambda n=name: np.median(np.linalg.norm(values[n][:, 1:] - np.median(values[n][:, 1:], axis=0), axis=1)),
            "a*b*各中央値からの色平面距離の中央値（robust分散の代理量）。明度L*は含めない。"
            "値の大小を肌の良し悪しとみなさない。", MIN_TARGET_PIXELS["skin"])
        add(f"{name}_highlight_percent", name, f"{label}の明部率", "%",
            lambda n=name: 100 * np.mean(values[n][:, 0] >= np.median(values[n][:, 0]) + HIGHLIGHT_L_OFFSET),
            "ROI内L*中央値 + 8以上の画素割合。光沢の候補量であり、照明・影・形状・化粧の影響が"
            "混入する。0%は閾値を超える画素がないことを示し、光沢がないという意味ではない。"
            "光沢量の測定や改善判定ではない。", MIN_TARGET_PIXELS["skin"])
    return rows
