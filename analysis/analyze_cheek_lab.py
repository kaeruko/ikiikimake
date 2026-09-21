"""Internal, mask-only Lab analysis API used by the reviewed-ROI runner.

There is deliberately no command-line entry point: use ``run_cheek01.py`` to
generate overlays and record human approval before calling this analysis.
``left_cheek`` and ``right_cheek`` always refer to sides of the displayed image.
Forehead subtraction is an experimental control comparison, not a validated
exposure correction, a cosmetic-effect estimate, or a score.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


ROI_NAMES = ("left_cheek", "right_cheek", "forehead")
A_THRESHOLDS = (20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
DELTA_METRICS = ("L_mean", "L_median", "a_mean", "a_median", "b_mean", "b_median")
CONTROL_NOTICE = (
    "Forehead subtraction is unvalidated: (cheek after - cheek before) - "
    "(forehead after - forehead before). Forehead treatment, shadow, pose, and "
    "lighting changes can invalidate the control; these are descriptive color "
    "differences, not scores or evidence of cosmetic efficacy."
)


def load_image(path: Path) -> np.ndarray:
    """Read BGR pixels, including paths containing non-ASCII characters."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def load_masks(path: Path, image_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Load disjoint nonempty HxW masks; uint8 may use 0/1 or 0/255."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask archive does not exist: {path}")
    masks: dict[str, np.ndarray] = {}
    occupied = np.zeros(image_shape[:2], dtype=bool)
    with np.load(path, allow_pickle=False) as archive:
        missing = set(ROI_NAMES) - set(archive.files)
        if missing:
            raise ValueError(f"Missing ROI masks: {', '.join(sorted(missing))}")
        for name in ROI_NAMES:
            mask = archive[name]
            if mask.ndim != 2 or mask.shape != image_shape[:2]:
                raise ValueError(
                    f"{name}: mask shape {mask.shape} does not match image "
                    f"shape {image_shape[:2]}"
                )
            if mask.dtype != np.bool_ and mask.dtype != np.uint8:
                raise ValueError(f"{name}: expected bool or uint8 binary mask")
            values = set(np.unique(mask).tolist())
            if not (values <= {0, 1} or values <= {0, 255}):
                raise ValueError(f"{name}: mask contains non-binary values")
            selected = mask.astype(bool)
            if not selected.any():
                raise ValueError(f"{name}: mask is empty")
            if np.any(occupied & selected):
                raise ValueError(f"{name}: ROI masks overlap")
            occupied |= selected
            masks[name] = selected
    return masks


def to_lab_channels(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the original script's OpenCV 8-bit Lab conversion exactly."""
    l_cv, a_cv, b_cv = cv2.split(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB))
    return (
        l_cv.astype(np.float32) * (100.0 / 255.0),
        a_cv.astype(np.float32) - 128.0,
        b_cv.astype(np.float32) - 128.0,
    )


def summarize(
    phase: str,
    side: str,
    channels: tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    """Calculate statistics on selected pixels only, never on padded crops."""
    l_star, a_star, b_star = (channel[mask] for channel in channels)
    row: dict[str, float | int | str] = {
        "phase": phase,
        "side": side,
        "pixels": int(a_star.size),
        "L_mean": float(np.mean(l_star)),
        "L_median": float(np.median(l_star)),
        "a_mean": float(np.mean(a_star)),
        "a_median": float(np.median(a_star)),
        "a_std": float(np.std(a_star)),
        "a_p75": float(np.percentile(a_star, 75)),
        "a_p90": float(np.percentile(a_star, 90)),
        "a_p95": float(np.percentile(a_star, 95)),
        "a_p99": float(np.percentile(a_star, 99)),
        "b_mean": float(np.mean(b_star)),
        "b_median": float(np.median(b_star)),
        "b_std": float(np.std(b_star)),
    }
    for threshold in A_THRESHOLDS:
        row[f"a_ratio_ge_{int(threshold)}"] = float(np.mean(a_star >= threshold))
    return row


def _save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_histogram(path: Path, before_values: tuple, after_values: tuple, name: str) -> None:
    figure = Figure(figsize=(12, 3.8))
    FigureCanvasAgg(figure)
    for index, label in enumerate(("L*", "a*", "b*")):
        before, after = before_values[index], after_values[index]
        lower = float(min(before.min(), after.min()))
        upper = float(max(before.max(), after.max()))
        # Constant-color ROIs are valid and common in small synthetic checks.
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        bins = np.linspace(lower, upper, 41)
        axes = figure.add_subplot(1, 3, index + 1)
        axes.hist(before, bins=bins, density=True, alpha=0.5, label="before")
        axes.hist(after, bins=bins, density=True, alpha=0.5, label="after")
        axes.set_xlabel(f"Lab {label}")
        axes.set_ylabel("Density of selected pixels")
        axes.legend()
    figure.suptitle(f"{name.replace('_', ' ')}: color distributions")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    figure.clear()


def _save_samples(path: Path, images: dict, masks: dict) -> None:
    """Display independent masked crops; do not imply pixel correspondence."""
    figure = Figure(figsize=(11, 7))
    FigureCanvasAgg(figure)
    for phase_index, phase in enumerate(("before", "after")):
        rgb = cv2.cvtColor(images[phase], cv2.COLOR_BGR2RGB)
        for roi_index, name in enumerate(ROI_NAMES):
            mask = masks[phase][name]
            ys, xs = np.nonzero(mask)
            y0, y1 = max(0, int(ys.min()) - 3), min(mask.shape[0], int(ys.max()) + 4)
            x0, x1 = max(0, int(xs.min()) - 3), min(mask.shape[1], int(xs.max()) + 4)
            crop_mask = mask[y0:y1, x0:x1]
            visible = np.full_like(rgb[y0:y1, x0:x1], 48)
            visible[crop_mask] = rgb[y0:y1, x0:x1][crop_mask]
            axes = figure.add_subplot(2, 3, phase_index * 3 + roi_index + 1)
            axes.imshow(visible, interpolation="nearest")
            axes.set_title(f"{phase}: {name.replace('_', ' ')}\n{int(mask.sum())} pixels")
            axes.axis("off")
    figure.suptitle("Selected ROI pixels (independent crops; no pixel alignment)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    figure.clear()



def write_summary_html(output_dir: Path, summary: dict) -> Path:
    """Write a one-page human review summary from an analysis summary."""
    output_dir = Path(output_dir)
    required = {
        "roi_samples.png",
        "left_cheek_lab_hist.png",
        "right_cheek_lab_hist.png",
        "forehead_lab_hist.png",
        "lab_stats.csv",
        "lab_deltas.csv",
        "analysis_summary.json",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise FileNotFoundError(
            "Cannot write summary.html; missing Lab artifacts: " + ", ".join(missing)
        )

    statistics = {
        (row["phase"], row["side"]): row
        for row in summary.get("statistics", [])
    }
    deltas = {
        (row["side"], row["metric"]): row
        for row in summary.get("deltas", [])
    }
    expected_stats = {
        (phase, side)
        for phase in ("before", "after")
        for side in ROI_NAMES
    }
    missing_stats = expected_stats - set(statistics)
    if missing_stats:
        raise ValueError(f"summary is missing statistics rows: {sorted(missing_stats)}")

    def number(value: float, digits: int = 2) -> str:
        return f"{float(value):.{digits}f}"

    labels = {
        "left_cheek": "画面左の頬",
        "right_cheek": "画面右の頬",
    }
    cheek_rows = []
    for side in ROI_NAMES[:2]:
        before = statistics["before", side]
        after = statistics["after", side]
        mean_delta = deltas[side, "a_mean"]
        median_delta = deltas[side, "a_median"]
        ratio_key = "a_ratio_ge_20"
        cheek_rows.append(
            "<tr>"
            f"<th>{html.escape(labels[side])}</th>"
            f"<td>{number(before['a_mean'])}</td>"
            f"<td>{number(after['a_mean'])}</td>"
            f"<td>{number(mean_delta['delta'])}</td>"
            f"<td>{number(mean_delta['delta_minus_forehead'])}</td>"
            f"<td>{number(before['a_median'])} → {number(after['a_median'])}"
            f" ({number(median_delta['delta'], 1)})</td>"
            f"<td>{number(before['a_p95'])} → {number(after['a_p95'])}</td>"
            f"<td>{number(100 * before[ratio_key], 1)}% → "
            f"{number(100 * after[ratio_key], 1)}%</td>"
            "</tr>"
        )

    forehead_before = statistics["before", "forehead"]
    forehead_after = statistics["after", "forehead"]
    forehead_rows = []
    for metric, label in (("L_mean", "L* 明るさ"), ("a_mean", "a* 赤み"), ("b_mean", "b* 黄み")):
        before = float(forehead_before[metric])
        after = float(forehead_after[metric])
        forehead_rows.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{number(before)}</td>"
            f"<td>{number(after)}</td>"
            f"<td>{number(after - before)}</td>"
            "</tr>"
        )

    control_notice = summary.get("control_correction", {}).get("notice", CONTROL_NOTICE)
    output = output_dir / "summary.html"
    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lab before / after summary</title>
<style>
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 1180px;
       margin: 32px auto; padding: 0 20px 60px; line-height: 1.6; color: #222; }}
h1 {{ margin-bottom: 0.25rem; }}
.note {{ background: #f5f5f5; border-left: 4px solid #777; padding: 12px 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; }}
img {{ max-width: 100%; height: auto; display: block; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 9px 10px; text-align: right; }}
th:first-child {{ text-align: left; }}
thead th {{ background: #f7f7f7; }}
small {{ color: #666; }}
a {{ color: inherit; }}
</style>
</head>
<body>
<h1>before / after Lab 解析</h1>
<p>まず ROI が正しいか確認し、その後に a*（赤み）の変化を見ます。
これは美しさの点数でも、化粧効果を証明する指標でもありません。</p>

<h2>1. ROI確認</h2>
<div class="card">
<img src="roi_samples.png" alt="before/after ROI samples">
</div>

<h2>2. 頬の a*（赤み）</h2>
<table>
<thead><tr>
<th>ROI</th><th>before平均</th><th>after平均</th><th>Δ平均</th>
<th>Δ平均−額Δ</th><th>中央値 (Δ)</th><th>p95</th><th>a*≥20画素率</th>
</tr></thead>
<tbody>{''.join(cheek_rows)}</tbody>
</table>
<p><small>Δ = after − before。p95 と a*≥20画素率は、ROI全体の平均では見えにくい
局所的な高a*画素の変化を見るための記述統計です。固定閾値20は妥当性を検証済みの判定基準ではありません。</small></p>

<h2>3. 額の変化（control確認）</h2>
<table>
<thead><tr><th>指標</th><th>before</th><th>after</th><th>Δ</th></tr></thead>
<tbody>{''.join(forehead_rows)}</tbody>
</table>
<p class="note">{html.escape(control_notice)}</p>

<h2>4. 分布を見る</h2>
<div class="grid">
<div class="card"><h3>画面左の頬</h3><img src="left_cheek_lab_hist.png" alt="left cheek Lab histogram"></div>
<div class="card"><h3>画面右の頬</h3><img src="right_cheek_lab_hist.png" alt="right cheek Lab histogram"></div>
<div class="card"><h3>額</h3><img src="forehead_lab_hist.png" alt="forehead Lab histogram"></div>
</div>

<h2>5. 元データ</h2>
<p>
<a href="lab_deltas.csv">lab_deltas.csv</a> /
<a href="lab_stats.csv">lab_stats.csv</a> /
<a href="analysis_summary.json">analysis_summary.json</a>
</p>
<p><small>左右は解剖学的左右ではなく、画像・画面上の左右です。
ROI間のピクセル対応付けは行っていません。</small></p>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return output


def analyze_pair(
    before_path: Path,
    after_path: Path,
    before_masks_path: Path,
    after_masks_path: Path,
    output_dir: Path,
) -> dict:
    """Analyze masks already approved through the runner's review workflow.

    Each mask archive is validated against its own image. Image sizes may differ;
    only region distributions and scalar statistics are compared. All validation
    occurs before output is created. Returns the JSON-serializable summary saved
    as ``analysis_summary.json``. This internal API does not record approval.
    """
    images = {"before": load_image(before_path), "after": load_image(after_path)}
    masks = {
        "before": load_masks(before_masks_path, images["before"].shape),
        "after": load_masks(after_masks_path, images["after"].shape),
    }
    channels = {phase: to_lab_channels(image) for phase, image in images.items()}
    rows = [
        summarize(phase, name, channels[phase], masks[phase][name])
        for name in ROI_NAMES
        for phase in ("before", "after")
    ]
    by_phase_side = {(row["phase"], row["side"]): row for row in rows}
    deltas = []
    for name in ROI_NAMES[:2]:
        for metric in DELTA_METRICS:
            before = float(by_phase_side["before", name][metric])
            after = float(by_phase_side["after", name][metric])
            forehead_before = float(by_phase_side["before", "forehead"][metric])
            forehead_after = float(by_phase_side["after", "forehead"][metric])
            delta = after - before
            forehead_delta = forehead_after - forehead_before
            deltas.append({
                "side": name,
                "metric": metric,
                "before": before,
                "after": after,
                "delta": delta,
                "forehead_before": forehead_before,
                "forehead_after": forehead_after,
                "forehead_delta": forehead_delta,
                "delta_minus_forehead": delta - forehead_delta,
                "correction_status": "unvalidated_forehead_control",
            })
    output_dir = Path(output_dir)
    summary = {
        "schema_version": 1,
        "side_convention": "image/screen sides, not anatomical sides",
        "color_conversion": "OpenCV uint8 BGR2LAB; L*=L_cv*100/255, a*=a_cv-128, b*=b_cv-128",
        "difference_convention": "after minus before",
        "control_correction": {"status": "unvalidated_forehead_control", "notice": CONTROL_NOTICE},
        "pixel_correspondence": False,
        "inputs": {
            "before": str(Path(before_path).resolve()),
            "after": str(Path(after_path).resolve()),
            "before_masks": str(Path(before_masks_path).resolve()),
            "after_masks": str(Path(after_masks_path).resolve()),
        },
        "statistics": rows,
        "deltas": deltas,
        "artifacts": {
            "statistics": "lab_stats.csv",
            "deltas": "lab_deltas.csv",
            "samples": "roi_samples.png",
            "summary_html": "summary.html",
            "histograms": [f"{name}_lab_hist.png" for name in ROI_NAMES],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(output_dir / "lab_stats.csv", rows)
    _save_csv(output_dir / "lab_deltas.csv", deltas)
    _save_samples(output_dir / "roi_samples.png", images, masks)
    for name in ROI_NAMES:
        values = {
            phase: tuple(channel[masks[phase][name]] for channel in channels[phase])
            for phase in ("before", "after")
        }
        _save_histogram(output_dir / f"{name}_lab_hist.png", values["before"], values["after"], name)
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    write_summary_html(output_dir, summary)
    return summary
