"""Generate face ROIs, stop for human review, then optionally analyze Lab.

Preparation never runs the analysis. Approval reuses the exact reviewed masks
and checks the input and artifact hashes before any analysis starts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


PHASES = ("before", "after")
ROI_ARTIFACTS = ("roi_masks.npz", "roi_points.json", "roi_overlay.png")
PAIR_LIMITS = {"yaw": 10.0, "pitch": 10.0, "roll": 12.0, "forehead_L_median": 12.0}
REVIEW_NOTICE = (
    "Check both overlays: cheek/forehead placement, hands, hair, shadow, and "
    "visible skin. Automated checks cannot prove that the regions are unobstructed. "
    "Forehead L* is an ambiguous control signal, not a measured exposure difference."
)


class ReviewError(ValueError):
    """A preparation or approval condition prevented analysis."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ReviewError(f"Required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def check_pair_quality(before: dict, after: dict) -> dict:
    """Reject missing quality signals and large pose/control differences."""
    result = {"limits": dict(PAIR_LIMITS), "deltas": {}, "errors": []}
    for field in PAIR_LIMITS:
        try:
            if field == "forehead_L_median":
                values = [float(item["quality"][field]) for item in (before, after)]
            else:
                values = [float(item["quality"]["pose_degrees"][field]) for item in (before, after)]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite quality value")
        except (KeyError, TypeError, ValueError):
            result["errors"].append(f"Missing or invalid pair quality signal: {field}")
            continue
        delta = values[1] - values[0]
        result["deltas"][field] = delta
        if abs(delta) > PAIR_LIMITS[field]:
            if field == "forehead_L_median":
                result["errors"].append(
                    f"Forehead control L* difference is too large ({delta:+.2f}; "
                    f"limit {PAIR_LIMITS[field]:g}). This may reflect treatment, "
                    "shadow, pose, or lighting; it is not an exposure measurement."
                )
            else:
                result["errors"].append(
                    f"Pair face {field} difference is too large "
                    f"({delta:+.2f} degrees; limit {PAIR_LIMITS[field]:g})."
                )
    return result


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ReviewError(f"Cannot decode review overlay: {path}")
    return image


def _save_review_image(output: Path) -> None:
    panels = []
    for phase in PHASES:
        image = _read_image(output / phase / "roi_overlay.png")
        scale = min(1.0, 900 / image.shape[1])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        panel = np.full((image.shape[0] + 44, image.shape[1], 3), 28, dtype=np.uint8)
        panel[44:] = image
        cv2.putText(panel, phase.upper(), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (240, 240, 240), 2, cv2.LINE_AA)
        panels.append(panel)
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels) + 16
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    offset = 0
    for panel in panels:
        canvas[:panel.shape[0], offset:offset + panel.shape[1]] = panel
        offset += panel.shape[1] + 16
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ReviewError("Could not encode review image")
    encoded.tofile(output / "review.png")


def _approval_command(output: Path, review_id: str) -> str:
    arguments = [sys.executable, str(Path(__file__).resolve()), "--output", str(output.resolve()),
                 "--approve-review", review_id]
    if os.name == "nt":
        # Single-quoted PowerShell literals also handle spaces, $, and apostrophes.
        return "& " + " ".join("'" + item.replace("'", "''") + "'" for item in arguments)
    return shlex.join(arguments)


def _quality_table(report: dict) -> str:
    def number(value, decimals=2):
        try:
            return f"{float(value):.{decimals}f}" if math.isfinite(float(value)) else "—"
        except (TypeError, ValueError):
            return "—"

    images = report.get("images", {})
    rows = []
    for axis, label in (("yaw", "顔の左右向き yaw（°）"), ("pitch", "顔の上下向き pitch（°）"),
                        ("roll", "顔の傾き roll（°）")):
        values = [images.get(phase, {}).get("quality", {}).get("pose_degrees", {}).get(axis) for phase in PHASES]
        rows.append((label, *(number(value) for value in values)))
    for name, label in (("left_cheek", "画面左頬の画素数"), ("right_cheek", "画面右頬の画素数"),
                        ("forehead", "額の画素数")):
        values = [images.get(phase, {}).get("rois", {}).get(name, {}).get("pixel_count") for phase in PHASES]
        rows.append((label, *(number(value, 0) for value in values)))
    rows.append(("額 L* 中央値（参考値）", *(number(images.get(phase, {}).get("quality", {}).get("forehead_L_median"))
                                        for phase in PHASES)))
    body = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    delta = report.get("pair_quality", {}).get("deltas", {}).get("forehead_L_median")
    return ("<table><thead><tr><th>項目</th><th>before</th><th>after</th></tr></thead>"
            f"<tbody>{body}</tbody></table><p>額 L* の前後差（after − before）：{number(delta)}。"
            "塗布・影・顔向き・照明が混ざる参考値で、露出差の測定値ではありません。</p>")


def _save_review_html(output: Path, report: dict) -> None:
    warnings = [str(item) for phase in PHASES for item in report["images"].get(phase, {}).get("warnings", [])]
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
    errors_html = "".join(f"<li>{html.escape(item)}</li>" for item in report["errors"])
    failed = bool(report["errors"]) or report["status"] == "failed"
    status = "生成時点：確認待ち（解析は未実行）" if not failed else "生成時点：条件を満たさないため停止"
    links = ['<a href="review.json">確認記録・最新の解析状態（JSON）</a>']
    individual_overlays = []
    for phase in PHASES:
        for filename, label in (("roi_overlay.png", "重ね合わせ画像・原寸"), ("roi_points.json", "領域情報・診断 JSON"),
                                ("roi_masks.npz", "領域マスク NPZ")):
            relative = f"{phase}/{filename}"
            if (output / relative).is_file():
                links.append(f'<a href="{relative}">{phase}：{label}</a>')
                if filename == "roi_overlay.png":
                    individual_overlays.append(f'<p>{phase}</p><img src="{relative}" alt="{phase} の ROI 診断画像">')
    if (output / "review.png").is_file():
        image_html = '<img src="review.png" alt="before と after に自動生成した ROI を重ねた画像">'
    else:
        image_html = "".join(individual_overlays) or "<p>表示できる重ね合わせ画像はありません。停止理由をご確認ください。</p>"
    if failed:
        action_html = "<p>この結果は承認できません。停止理由を確認し、別の画像や条件で新しい保存先に生成し直してください。</p>"
    else:
        command = html.escape(_approval_command(output, report["review_id"]))
        shell = "PowerShell" if os.name == "nt" else "端末"
        action_html = ("<p>両方の画像を確認し、問題がなければ次のコマンドを " + shell
                       + " に貼り付けて実行してください。画像やマスクを変更すると承認は拒否されます。</p>"
                       + f"<pre><code>{command}</code></pre>"
                       + "<p>問題があれば別の画像で生成し直してください。このページは生成時点の確認用スナップショットです。"
                       + "承認後の状態は上の「確認記録・最新の解析状態」で確認できます。</p>")
    content = f"""<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>頬・額 ROI 確認</title><style>
body{{font:16px/1.65 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:1400px;background:#fafafa;color:#20252a}}
img{{max-width:100%;height:auto;border:1px solid #bbb}}code{{overflow-wrap:anywhere}}
.status{{padding:1rem;background:#fff0bb;border-radius:.4rem}}li{{margin:.4rem 0}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#e9edf0;padding:1rem}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccd1d5;padding:.5rem 1rem;text-align:left}}
</style><h1>頬・額 ROI 確認</h1><p class="status">{status}</p>
<p>生成日時（UTC）：{html.escape(report['created_at'])}</p>
<p>画面左頬＝緑、画面右頬＝青、額＝黄。左右は画像上の位置です。</p>
{image_html}<p>{'<br>'.join(links)}</p>{_quality_table(report)}
<ul><li>頬の領域が目・鼻・口・輪郭を避けて、肌の上にありますか。</li>
<li>額の領域が髪や眉にかかっていませんか。</li>
<li>手や手の影、強い反射、顔の変形が領域に入り込んでいませんか。</li>
<li>前後で対応する位置が取れていて、比較できる見え方ですか。</li></ul>
<p>自動チェックだけで遮蔽の有無は保証できません。額の L* 差は、塗布・影・顔向き・照明の影響が混ざる参考値です。</p>
<ul>{errors_html}{warning_html}</ul><p>確認 ID：<code>{html.escape(report['review_id'])}</code></p>
{action_html}
<p>承認後の出力は Lab 色差の実験結果であり、メイク効果の点数ではありません。</p></html>
"""
    (output / "review.html").write_text(content, encoding="utf-8")


def prepare_review(
    before_path: Path,
    after_path: Path,
    output_dir: Path,
    model_dir: Path = ROOT / "models",
    *,
    extractor_factory: Callable | None = None,
) -> dict:
    """Save candidate ROIs and a review manifest; never run Lab analysis."""
    paths = {"before": Path(before_path).resolve(), "after": Path(after_path).resolve()}
    # Validate inputs before creating an output folder.
    sources = {phase: {"path": str(path), "sha256": _sha256(path)} for phase, path in paths.items()}
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ReviewError(f"Output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "review_id": secrets.token_hex(16),
        "status": "preparing",
        "created_at": _timestamp(),
        "sources": sources,
        "images": {},
        "artifacts": {},
        "errors": [],
        "notice": REVIEW_NOTICE,
    }
    _save_json(output / "review.json", report)
    active_phase = None
    try:
        if extractor_factory is None:
            from analysis.extract_face_rois import RoiConfig, RoiExtractor

            extractor_factory = lambda directory: RoiExtractor(directory, RoiConfig())
        with extractor_factory(Path(model_dir).resolve()) as extractor:
            for phase in PHASES:
                active_phase = phase
                metadata = extractor.extract(paths[phase], output / phase)
                report["images"][phase] = metadata
                if metadata.get("status") != "needs_review" or metadata.get("errors"):
                    raise ReviewError(f"ROI extraction did not pass for {phase}")
                source = metadata.get("source", {})
                if (Path(source.get("path", "")).resolve() != paths[phase]
                        or source.get("sha256") != sources[phase]["sha256"]
                        or _sha256(paths[phase]) != sources[phase]["sha256"]):
                    raise ReviewError(f"Source changed during ROI extraction: {phase}")
                for name in ROI_ARTIFACTS:
                    relative = f"{phase}/{name}"
                    report["artifacts"][relative] = _sha256(output / relative)
        report["pair_quality"] = check_pair_quality(report["images"]["before"], report["images"]["after"])
        report["errors"].extend(report["pair_quality"]["errors"])
        _save_review_image(output)
        _save_review_html(output, report)
        for relative in ("review.png", "review.html"):
            report["artifacts"][relative] = _sha256(output / relative)
        if report["errors"]:
            raise ReviewError("Pair quality checks failed")
        report["status"] = "needs_review"
        _save_json(output / "review.json", report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        if str(exc) not in report["errors"]:
            report["errors"].append(str(exc))
        diagnostic = getattr(exc, "report", None)
        if diagnostic is not None:
            report["extraction_failure"] = diagnostic
            if active_phase is not None:
                report["images"][active_phase] = diagnostic
        try:
            if not (output / "review.png").exists() and all((output / phase / "roi_overlay.png").is_file() for phase in PHASES):
                _save_review_image(output)
            _save_review_html(output, report)
            for phase in PHASES:
                for name in ROI_ARTIFACTS:
                    relative = f"{phase}/{name}"
                    if (output / relative).is_file():
                        report["artifacts"][relative] = _sha256(output / relative)
            for relative in ("review.html", "review.png"):
                if (output / relative).is_file():
                    report["artifacts"][relative] = _sha256(output / relative)
        except (OSError, ReviewError) as diagnostic_error:
            report["errors"].append(f"Could not save review diagnostic: {diagnostic_error}")
        _save_json(output / "review.json", report)
        raise ReviewError("ROI preparation failed: " + "; ".join(report["errors"]), report) from exc


def _load_review(output: Path) -> dict:
    try:
        report = json.loads((output / "review.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Cannot read review manifest: {output / 'review.json'}") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ReviewError("Unsupported or invalid review manifest")
    return report


def _validate_review(output: Path, report: dict, review_id: str,
                     before_path: Path | None, after_path: Path | None) -> dict[str, Path]:
    if report.get("status") != "needs_review" or report.get("errors"):
        raise ReviewError(f"Review cannot be approved in state: {report.get('status')}")
    if not review_id or not secrets.compare_digest(str(report.get("review_id", "")), review_id):
        raise ReviewError("Review ID is missing or incorrect")
    paths = {}
    for phase, supplied in (("before", before_path), ("after", after_path)):
        try:
            source = report["sources"][phase]
            path = Path(source["path"])
            metadata = report["images"][phase]
            if not path.is_absolute() or not isinstance(source["sha256"], str):
                raise ValueError("invalid source")
            if not isinstance(metadata, dict):
                raise ValueError("invalid ROI metadata")
            if metadata.get("status") != "needs_review" or metadata.get("errors"):
                raise ValueError("failed ROI metadata")
            if (Path(metadata["source"]["path"]).resolve() != path.resolve()
                    or metadata["source"]["sha256"] != source["sha256"]):
                raise ValueError("inconsistent source")
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewError(f"Invalid saved source or ROI metadata: {phase}") from exc
        if supplied is not None and Path(supplied).resolve() != path.resolve():
            raise ReviewError(f"Supplied {phase} path differs from the reviewed source")
        if _sha256(path) != source["sha256"]:
            raise ReviewError(f"Source changed since review: {phase}")
        paths[phase] = path
    expected = {f"{phase}/{name}" for phase in PHASES for name in ROI_ARTIFACTS} | {"review.png", "review.html"}
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ReviewError("Review artifact manifest is incomplete or invalid")
    for relative in sorted(expected):
        if _sha256(output / relative) != artifacts[relative]:
            raise ReviewError(f"Artifact changed since review: {relative}")
    # Recompute the pair guard from recorded metadata as well as checking status.
    pair = check_pair_quality(report["images"]["before"], report["images"]["after"])
    if pair["errors"]:
        raise ReviewError("Pair quality checks failed: " + "; ".join(pair["errors"]))
    if (output / "lab").exists():
        raise ReviewError("Analysis output already exists; generate a fresh review in a new folder")
    return paths


def approve_review(
    output_dir: Path,
    review_id: str,
    *,
    before_path: Path | None = None,
    after_path: Path | None = None,
    analyzer: Callable | None = None,
) -> dict:
    """Approve an unchanged review once and analyze its existing masks."""
    output = Path(output_dir).resolve()
    report = _load_review(output)
    paths = _validate_review(output, report, review_id, before_path, after_path)
    lock_path = output / ".approval.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ReviewError("Approval or analysis is already running for this review") from exc
    try:
        with lock:
            lock.write(review_id)
        # A second process may have completed between the first read and lock.
        report = _load_review(output)
        paths = _validate_review(output, report, review_id, before_path, after_path)
        if analyzer is None:
            from analysis.analyze_cheek_lab import analyze_pair

            analyzer = analyze_pair
        report["approved_at"] = _timestamp()
        report["status"] = "analyzing"
        _save_json(output / "review.json", report)
        try:
            analyzer(paths["before"], paths["after"], output / "before/roi_masks.npz",
                     output / "after/roi_masks.npz", output / "lab")
        except Exception as exc:
            report["status"] = "analysis_failed"
            report["errors"].append(f"Analysis failed: {exc}")
            _save_json(output / "review.json", report)
            raise ReviewError(f"Analysis failed; prepare a fresh review before retrying: {exc}", report) from exc
        report["status"] = "analyzed"
        report["analysis_completed_at"] = _timestamp()
        report["analysis_output"] = "lab"
        _save_json(output / "review.json", report)
        return report
    finally:
        lock_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, help="Before image (required for preparation)")
    parser.add_argument("--after", type=Path, help="After image (required for preparation)")
    parser.add_argument("--output", type=Path, required=True, help="New/empty directory, or saved review for approval")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--approve-review", metavar="REVIEW_ID", help="Confirm human overlay review and run Lab once")
    args = parser.parse_args(argv)
    if args.approve_review is None and (args.before is None or args.after is None):
        parser.error("--before and --after are required when preparing a review")
    try:
        if args.approve_review is not None:
            approve_review(args.output, args.approve_review, before_path=args.before, after_path=args.after)
            print(f"Lab analysis complete: {args.output.resolve() / 'lab'}")
        else:
            report = prepare_review(args.before, args.after, args.output, args.model_dir)
            print(f"ROI review required; no Lab analysis has run: {args.output.resolve() / 'review.html'}")
            print("Review both overlays. If acceptable, approve with:")
            print(_approval_command(args.output, report["review_id"]))
        return 0
    except (ReviewError, OSError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
