from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cheek color distributions in Lab color space."
    )
    parser.add_argument(
        "--before",
        type=Path,
        required=True,
        help="Before image path.",
    )
    parser.add_argument(
        "--after",
        type=Path,
        required=True,
        help="After image path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory.",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Failed to decode image: {path}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel BGR image, got shape={image.shape}: {path}"
        )

    return image


def select_roi(
    image: np.ndarray,
    window_name: str,
) -> tuple[int, int, int, int]:
    roi = cv2.selectROI(
        window_name,
        image,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(window_name)

    x, y, width, height = map(int, roi)

    if width <= 0 or height <= 0:
        raise ValueError(
            f"ROI selection was cancelled or invalid: "
            f"x={x}, y={y}, width={width}, height={height}"
        )

    image_height, image_width = image.shape[:2]

    if x < 0 or y < 0:
        raise ValueError(f"ROI origin is outside image: {roi}")

    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI exceeds image bounds: roi={roi}, "
            f"image_size=({image_width}, {image_height})"
        )

    return x, y, width, height


def crop_roi(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    x, y, width, height = roi
    crop = image[y : y + height, x : x + width]

    if crop.size == 0:
        raise ValueError(f"ROI produced empty image: {roi}")

    return crop


def to_lab_channels(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    l_cv, a_cv, b_cv = cv2.split(lab)

    # OpenCV:
    # L: 0..255 -> CIE L*: 0..100
    # a,b: 0..255, with 128 corresponding approximately to zero.
    l_star = l_cv.astype(np.float32) * (100.0 / 255.0)
    a_star = a_cv.astype(np.float32) - 128.0
    b_star = b_cv.astype(np.float32) - 128.0

    return l_star, a_star, b_star


def summarize(
    phase: str,
    side: str,
    crop: np.ndarray,
) -> dict[str, float | int | str]:
    l_star, a_star, b_star = to_lab_channels(crop)

    return {
        "phase": phase,
        "side": side,
        "pixels": int(a_star.size),
        "L_mean": float(np.mean(l_star)),
        "L_median": float(np.median(l_star)),
        "a_mean": float(np.mean(a_star)),
        "a_median": float(np.median(a_star)),
        "a_std": float(np.std(a_star)),
        "b_mean": float(np.mean(b_star)),
        "b_median": float(np.median(b_star)),
        "b_std": float(np.std(b_star)),
    }


def save_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)

    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def save_histogram(
    output_path: Path,
    before_crop: np.ndarray,
    after_crop: np.ndarray,
    title: str,
) -> None:
    _, before_a, _ = to_lab_channels(before_crop)
    _, after_a, _ = to_lab_channels(after_crop)

    before_values = before_a.ravel()
    after_values = after_a.ravel()

    value_min = float(min(before_values.min(), after_values.min()))
    value_max = float(max(before_values.max(), after_values.max()))

    if value_min == value_max:
        raise ValueError(
            f"Cannot create histogram because all a* values are identical: "
            f"{value_min}"
        )

    bins = np.linspace(value_min, value_max, 51)

    plt.figure(figsize=(8, 5))
    plt.hist(
        before_values,
        bins=bins,
        density=True,
        alpha=0.5,
        label="before",
    )
    plt.hist(
        after_values,
        bins=bins,
        density=True,
        alpha=0.5,
        label="after",
    )
    plt.xlabel("Lab a* (positive = redder)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    before = load_image(args.before)
    after = load_image(args.after)

    args.output.mkdir(parents=True, exist_ok=True)

    print("Select LEFT cheek in BEFORE image, then press Enter.")
    before_left_roi = select_roi(before, "BEFORE - LEFT cheek")

    print("Select LEFT cheek in AFTER image, then press Enter.")
    after_left_roi = select_roi(after, "AFTER - LEFT cheek")

    print("Select RIGHT cheek in BEFORE image, then press Enter.")
    before_right_roi = select_roi(before, "BEFORE - RIGHT cheek")

    print("Select RIGHT cheek in AFTER image, then press Enter.")
    after_right_roi = select_roi(after, "AFTER - RIGHT cheek")

    roi_data = {
        "before_left": before_left_roi,
        "after_left": after_left_roi,
        "before_right": before_right_roi,
        "after_right": after_right_roi,
    }

    with (args.output / "rois.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(roi_data, f, indent=2)

    before_left = crop_roi(before, before_left_roi)
    after_left = crop_roi(after, after_left_roi)
    before_right = crop_roi(before, before_right_roi)
    after_right = crop_roi(after, after_right_roi)

    save_image(args.output / "before_left.png", before_left)
    save_image(args.output / "after_left.png", after_left)
    save_image(args.output / "before_right.png", before_right)
    save_image(args.output / "after_right.png", after_right)

    rows = [
        summarize("before", "left", before_left),
        summarize("after", "left", after_left),
        summarize("before", "right", before_right),
        summarize("after", "right", after_right),
    ]

    csv_path = args.output / "lab_stats.csv"

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    save_histogram(
        args.output / "left_a_hist.png",
        before_left,
        after_left,
        "Left cheek: Lab a*",
    )

    save_histogram(
        args.output / "right_a_hist.png",
        before_right,
        after_right,
        "Right cheek: Lab a*",
    )

    for side in ("left", "right"):
        before_row = next(
            row
            for row in rows
            if row["phase"] == "before" and row["side"] == side
        )
        after_row = next(
            row
            for row in rows
            if row["phase"] == "after" and row["side"] == side
        )

        delta_a = float(after_row["a_median"]) - float(
            before_row["a_median"]
        )
        delta_l = float(after_row["L_median"]) - float(
            before_row["L_median"]
        )
        delta_b = float(after_row["b_median"]) - float(
            before_row["b_median"]
        )

        print()
        print(side.upper())
        print(f"  Δ median L*: {delta_l:+.3f}")
        print(f"  Δ median a*: {delta_a:+.3f}")
        print(f"  Δ median b*: {delta_b:+.3f}")

    print()
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()