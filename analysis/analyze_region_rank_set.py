"""Analyze appearance metrics across multiple candidate before/after ranks.

This runner is for repeatability review. It never modifies
selected_region_pairs.json or promotes a candidate to an approved pair.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Iterable

from analysis.analyze_selected_regions import (
    REGION_LABELS,
    REGION_METRIC_IDS,
    _load_phase,
    _overlay,
    _save_png,
    sha256_file,
)
from analysis.match_video_regions import REGION_RULES, rank_region_pairs


RANK_SET_VERSION = "candidate-rank-set-v2"

EYE_TEXTURE_FOCUS_IDS = (
    "screen_left_upper_lid_skin_highpass_median_pct",
    "screen_left_upper_lid_skin_highpass_p90_pct",
    "screen_right_upper_lid_skin_highpass_median_pct",
    "screen_right_upper_lid_skin_highpass_p90_pct",
    "screen_left_nasolabial_crease_darkness_p90_pct",
    "screen_right_nasolabial_crease_darkness_p90_pct",
)


def _normalize_ranks(ranks: Iterable[int]) -> tuple[int, ...]:
    if isinstance(ranks, (str, bytes)):
        raise TypeError("ranks must be an iterable of positive integers")
    try:
        values = tuple(ranks)
    except TypeError as exc:
        raise TypeError("ranks must be an iterable of positive integers") from exc
    if not values:
        raise ValueError("ranks must not be empty")
    for rank in values:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"rank must be an integer, got {type(rank).__name__}: {rank!r}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
    if len(set(values)) != len(values):
        raise ValueError(f"ranks must not contain duplicates: {values}")
    return values


def _selected_frame(record: dict) -> dict:
    if not isinstance(record, dict):
        raise TypeError("scan record must be an object")
    required_keys = ("frame_id", "timestamp_seconds", "image_path", "roi_dir")
    missing = [key for key in required_keys if key not in record]
    if missing:
        raise ValueError(f"scan record is missing keys: {missing}")

    image_path = Path(record["image_path"])
    roi_dir = Path(record["roi_dir"])
    required_files = {
        "roi_masks_sha256": roi_dir / "roi_masks.npz",
        "roi_points_sha256": roi_dir / "roi_points.json",
        "roi_overlay_sha256": roi_dir / "roi_overlay.png",
    }
    for path in (image_path, *required_files.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    return {
        "frame_id": record["frame_id"],
        "timestamp_seconds": record["timestamp_seconds"],
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "roi_dir": str(roi_dir),
        **{name: sha256_file(path) for name, path in required_files.items()},
    }


def _delta_rows(
    region: str,
    rank: int,
    before_rows: dict[str, dict],
    after_rows: dict[str, dict],
) -> list[dict]:
    rows = []
    for identifier in REGION_METRIC_IDS[region]:
        before = before_rows[identifier]
        after = after_rows[identifier]
        status = (
            before["status"]
            if before["status"] == after["status"]
            else f'before={before["status"]}; after={after["status"]}'
        )
        before_value = before["value"]
        after_value = after["value"]
        delta = (
            float(after_value - before_value)
            if before_value is not None and after_value is not None
            else None
        )
        rows.append(
            {
                "rank": rank,
                "region": region,
                "id": identifier,
                "label": before["label"],
                "unit": before["unit"],
                "before": before_value,
                "after": after_value,
                "delta": delta,
                "before_pixels": before["pixels"],
                "after_pixels": after["pixels"],
                "status": status,
                "note": before["note"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "rank", "region", "id", "label", "unit", "before", "after", "delta",
        "before_pixels", "after_pixels", "status", "note",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value, signed: bool = False) -> str:
    if value is None:
        return "—"
    value = float(value)
    if value != 0 and abs(value) < 0.001:
        return f"{value:+.3e}" if signed else f"{value:.3e}"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _report_html(summary: dict) -> str:
    region = summary["region"]
    focus_ids = EYE_TEXTURE_FOCUS_IDS if region == "eye_texture" else ()
    focus_section = ""
    if focus_ids:
        labels = {
            row["id"]: row["label"]
            for row in summary["pairs"][0]["deltas"]
        }
        head = "".join(f"<th>{html.escape(labels[mid])}</th>" for mid in focus_ids)
        body = []
        for pair in summary["pairs"]:
            by_id = {row["id"]: row for row in pair["deltas"]}
            cells = "".join(
                f"<td>{_format_value(by_id[mid]['delta'], signed=True)}</td>"
                for mid in focus_ids
            )
            body.append(
                f"<tr><td>{pair['rank']}</td>"
                f"<td>{pair['selection']['before']['timestamp_seconds']:.2f}s → "
                f"{pair['selection']['after']['timestamp_seconds']:.2f}s</td>{cells}</tr>"
            )
        focus_section = (
            "<section><h2>rank横断の主要差分</h2>"
            "<p>値は after - before。正負の方向がrank間で再現するかを確認します。</p>"
            "<div class='scroll'><table><thead><tr><th>rank</th><th>before → after</th>"
            + head + "</tr></thead><tbody>" + "".join(body)
            + "</tbody></table></div></section>"
        )

    sections = []
    for pair in summary["pairs"]:
        rows = []
        for row in pair["deltas"]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(row['label'])}</td>"
                f"<td>{html.escape(row['unit'])}</td>"
                f"<td>{_format_value(row['before'])}</td>"
                f"<td>{_format_value(row['after'])}</td>"
                f"<td>{_format_value(row['delta'], signed=True)}</td>"
                f"<td>{html.escape(row['status'])}</td>"
                "</tr>"
            )
        sel = pair["selection"]
        gate = sel["region_gate"]
        sections.append(
            f"<section><h2>{html.escape(REGION_LABELS[region])} / rank {pair['rank']}</h2>"
            f"<p>score {sel['score']:.4f} ／ before {sel['before']['timestamp_seconds']:.2f}s "
            f"→ after {sel['after']['timestamp_seconds']:.2f}s</p>"
            f"<p>顔サイズ比 {sel['face_scale_ratio']:.4f} ／ "
            f"手の重なり before {gate['before_hand_overlap_ratio']:.2%} / "
            f"after {gate['after_hand_overlap_ratio']:.2%}</p>"
            f"<div class='pair'><img src='{html.escape(pair['images']['before'])}'>"
            f"<img src='{html.escape(pair['images']['after'])}'></div>"
            "<table><thead><tr><th>指標</th><th>単位</th><th>before</th>"
            "<th>after</th><th>差</th><th>状態</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )

    return f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>複数 candidate rank 再現性確認</title>
<style>
html,body{{background:#ffffff;color:#25322d}}body{{font:16px/1.7 system-ui,sans-serif;max-width:1400px;margin:28px auto;padding:0 22px}}
.report-root{{background:#ffffff !important;color:#25322d !important;color-scheme:light !important;padding:18px 20px 28px !important}}
.report-root,.report-root section,.report-root table,.report-root tbody,.report-root tr,.report-root td{{background-color:#ffffff !important;color:#25322d !important}}
.report-root h1,.report-root h2,.report-root p,.report-root td,.report-root th{{color:#25322d !important;opacity:1 !important}}
.report-root section{{border:1px solid #cfd8d3 !important;border-radius:10px;padding:18px;margin:22px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{max-width:100%;height:auto}}
.report-root table{{border-collapse:collapse;width:100%;margin-top:16px;background:#ffffff !important;color:#25322d !important}}
.report-root th,.report-root td{{padding:9px 10px;border-bottom:1px solid #aebbb4 !important;text-align:left;vertical-align:top;color:#25322d !important}}
.report-root th{{background:#eef3f0 !important;color:#17221d !important;font-weight:700;position:sticky;top:0}}
.scroll{{overflow-x:auto}}.report-root .notice{{background:#fff2c8 !important;color:#3c3214 !important;padding:14px;border-left:5px solid #d79b20 !important}}
@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}
</style><body><div class="report-root"><h1>複数 before / after candidate rank の再現性確認</h1>
<p class="notice">これは候補rankの探索的比較です。rankを承認済みペアへ昇格する処理ではありません。
画像上の記述指標であり、乾燥・シワの診断、物理的なシワ深さ、メイク効果の因果推定ではありません。
各候補は必ず画像を目視し、表情・照明・ピント・圧縮・手や道具の影響を確認してください。</p>
<p>候補再生成: before / after の同一フレーム再利用は禁止 ／
ペア全体が近い場合のみ diversity {summary["candidate_generation"]["diversity_seconds"]:.1f}秒で抑制 ／
最大 {summary["candidate_generation"]["top_k"]}件 ／
品質条件通過 {summary["candidate_generation"]["eligible_before_diversity"]}組 ／
unique before {summary["candidate_generation"]["eligible_unique_before_frames"]}枚 ／
unique after {summary["candidate_generation"]["eligible_unique_after_frames"]}枚 ／
endpointのみの最大マッチング {summary["candidate_generation"]["maximum_unique_endpoint_pairs"]}組 ／
貪欲法で選択 {summary["candidate_generation"]["selected_unique_endpoint_pairs"]}組。
別フレームであれば片側の時刻が近い候補は許可します。顔サイズ・手重なりの閾値は変更していません。</p>
{focus_section}
{''.join(sections)}
</div></body></html>"""


def analyze_region_rank_set(
    selected_path: Path,
    region: str,
    ranks: Iterable[int],
    output_root: Path,
    candidate_top_k: int = 20,
    diversity_seconds: float = 5.0,
) -> dict:
    selected_path = Path(selected_path)
    output_root = Path(output_root)
    ranks = _normalize_ranks(ranks)
    if isinstance(candidate_top_k, bool) or not isinstance(candidate_top_k, int) or candidate_top_k < 1:
        raise ValueError("candidate_top_k must be a positive integer")
    if candidate_top_k < max(ranks):
        raise ValueError(
            f"candidate_top_k={candidate_top_k} is smaller than requested max rank {max(ranks)}"
        )
    if isinstance(diversity_seconds, bool) or not isinstance(diversity_seconds, (int, float)):
        raise TypeError("diversity_seconds must be a finite nonnegative number")
    diversity_seconds = float(diversity_seconds)
    if not math.isfinite(diversity_seconds) or diversity_seconds < 0:
        raise ValueError("diversity_seconds must be a finite nonnegative number")

    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported selected_region_pairs schema_version: {selected.get('schema_version')!r}"
        )
    if region not in REGION_METRIC_IDS:
        raise ValueError(f"Unknown region: {region}")

    candidate_run_raw = selected.get("region_candidate_run")
    if not isinstance(candidate_run_raw, str) or not candidate_run_raw:
        raise ValueError("selected_region_pairs.json has no region_candidate_run")
    candidate_run = Path(candidate_run_raw)
    matching_path = candidate_run / "region_matching.json"
    occlusion_path = candidate_run / "region_occlusion.json"
    scan_manifest_path = selected_path.parent / "scan_manifest.json"
    for path in (matching_path, occlusion_path, scan_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(scan_manifest_path.read_text(encoding="utf-8"))
    original_matching = json.loads(matching_path.read_text(encoding="utf-8"))
    occlusion = json.loads(occlusion_path.read_text(encoding="utf-8"))
    video_info = manifest.get("video")
    if not isinstance(video_info, dict):
        raise ValueError("scan_manifest.json has no video object")
    if video_info.get("sha256") != selected.get("video_sha256"):
        raise RuntimeError("scan_manifest video SHA-256 differs from selected_region_pairs.json")
    selected_video_path = selected.get("video_path")
    manifest_video_path = video_info.get("path")
    if not isinstance(selected_video_path, str) or not isinstance(manifest_video_path, str):
        raise ValueError("video path is missing")
    if Path(selected_video_path).resolve() != Path(manifest_video_path).resolve():
        raise RuntimeError("scan_manifest video path differs from selected_region_pairs.json")

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("scan_manifest.json has no records")
    by_id = {}
    for record in records:
        if not isinstance(record, dict) or "frame_id" not in record:
            raise ValueError("scan_manifest record is invalid")
        frame_id = record["frame_id"]
        if frame_id in by_id:
            raise ValueError(f"scan_manifest has duplicate frame_id: {frame_id}")
        by_id[frame_id] = record

    original_regions = original_matching.get("regions")
    if not isinstance(original_regions, dict) or region not in original_regions:
        raise ValueError(f"region_matching.json has no region: {region}")

    current_rule = REGION_RULES[region]
    expected_rule = {
        "mask_names": list(current_rule.mask_names),
        "max_face_scale_ratio": current_rule.max_face_scale_ratio,
        "max_hand_overlap_ratio": current_rule.max_hand_overlap_ratio,
    }
    saved_rule = occlusion.get("rules", {}).get(region) if isinstance(occlusion, dict) else None
    if saved_rule != expected_rule:
        raise RuntimeError(
            f"{region}: saved region_occlusion rule differs from current rule; "
            f"saved={saved_rule!r} current={expected_rule!r}"
        )

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("scan_manifest.json has no config object")
    try:
        split_seconds = float(config["split_seconds"])
        min_gap_seconds = float(config["min_gap_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scan_manifest config lacks valid split/min-gap seconds") from exc

    regenerated_matching = rank_region_pairs(
        records,
        split_seconds,
        min_gap_seconds,
        occlusion,
        top_k=candidate_top_k,
        diversity_seconds=diversity_seconds,
    )
    if regenerated_matching.get("diversity_mode") != "unique_endpoints_joint_pair_time":
        raise RuntimeError(
            "regenerated matching does not use unique-endpoint pair diversity: "
            f"{regenerated_matching.get('diversity_mode')!r}"
        )
    regenerated_regions = regenerated_matching.get("regions")
    if not isinstance(regenerated_regions, dict) or region not in regenerated_regions:
        raise RuntimeError(f"regenerated matching has no region: {region}")
    ranked_pairs = regenerated_regions[region].get("ranked_pairs")
    if not isinstance(ranked_pairs, list):
        raise RuntimeError(f"{region}: regenerated ranked_pairs is not a list")
    if max(ranks) > len(ranked_pairs):
        region_stats = regenerated_regions[region]
        after_diag = region_stats.get("after_gate_diagnostics")
        if not isinstance(after_diag, dict):
            raise RuntimeError(f"{region}: after_gate_diagnostics is missing")
        after_counts = after_diag.get("counts")
        after_frames = after_diag.get("frames")
        if not isinstance(after_counts, dict) or not isinstance(after_frames, list):
            raise RuntimeError(f"{region}: after_gate_diagnostics is invalid")
        after_frame_summary = ", ".join(
            f"{float(row['after_time']):.2f}s:{row['status']}"
            for row in after_frames
        )
        raise ValueError(
            f"{region}: requested rank {max(ranks)}, but regenerated settings "
            f"top_k={candidate_top_k}, diversity_seconds={diversity_seconds:g} "
            f"produced only {len(ranked_pairs)} candidates. "
            f"Diagnostics: geometry_pairs={regenerated_matching.get('geometry_eligible_pairs')}, "
            f"geometry_unique_before={regenerated_matching.get('geometry_unique_before_frames')}, "
            f"geometry_unique_after={regenerated_matching.get('geometry_unique_after_frames')}, "
            f"region_eligible_pairs={region_stats.get('eligible_before_diversity')}, "
            f"region_unique_before={region_stats.get('eligible_unique_before_frames')}, "
            f"region_unique_after={region_stats.get('eligible_unique_after_frames')}, "
            f"maximum_unique_endpoint_pairs={region_stats.get('maximum_unique_endpoint_pairs')}, "
            f"greedy_selected={region_stats.get('selected_unique_endpoint_pairs')}, "
            f"after_gate_counts={after_counts}. "
            f"After frames: {after_frame_summary}"
        )

    implementation_path = Path(__file__)
    fingerprint_spec = {
        "version": RANK_SET_VERSION,
        "region": region,
        "ranks": list(ranks),
        "selected_region_pairs_sha256": sha256_file(selected_path),
        "scan_manifest_sha256": sha256_file(scan_manifest_path),
        "region_matching_sha256": sha256_file(matching_path),
        "region_occlusion_sha256": sha256_file(occlusion_path),
        "candidate_top_k": candidate_top_k,
        "diversity_seconds": diversity_seconds,
        "implementation_sha256": sha256_file(implementation_path),
        "selected_region_analysis_sha256": sha256_file(
            implementation_path.with_name("analyze_selected_regions.py")
        ),
        "appearance_features_sha256": sha256_file(
            implementation_path.with_name("appearance_features.py")
        ),
        "region_matching_implementation_sha256": sha256_file(
            implementation_path.with_name("match_video_regions.py")
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    report_path = output_root / "report.html"
    csv_path = output_root / "feature_deltas.csv"

    if summary_path.exists():
        if not summary_path.is_file():
            raise FileExistsError(f"summary.json is not a file: {summary_path}")
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            saved.get("fingerprint") == fingerprint
            and saved.get("fingerprint_spec") == fingerprint_spec
        ):
            expected = [report_path, csv_path]
            for rank in ranks:
                expected.extend(
                    (
                        output_root / f"{region}_rank{rank}_before.png",
                        output_root / f"{region}_rank{rank}_after.png",
                    )
                )
            missing = [str(path) for path in expected if not path.is_file()]
            if missing:
                raise FileExistsError(f"Cached rank-set output is incomplete: {missing}")
            return saved

    for pattern in (f"{region}_rank*_before.png", f"{region}_rank*_after.png"):
        for path in output_root.glob(pattern):
            if not path.is_file():
                raise FileExistsError(f"Expected generated file path: {path}")
            path.unlink()

    summary = {
        "schema_version": 1,
        "version": RANK_SET_VERSION,
        "region": region,
        "ranks": list(ranks),
        "fingerprint": fingerprint,
        "fingerprint_spec": fingerprint_spec,
        "selected_region_pairs_path": str(selected_path.resolve()),
        "scan_manifest_path": str(scan_manifest_path.resolve()),
        "region_matching_path": str(matching_path.resolve()),
        "region_occlusion_path": str(occlusion_path.resolve()),
        "candidate_generation": {
            "top_k": candidate_top_k,
            "diversity_seconds": diversity_seconds,
            "diversity_mode": regenerated_matching.get("diversity_mode"),
            "geometry_eligible_pairs": regenerated_matching.get("geometry_eligible_pairs"),
            "geometry_unique_before_frames": regenerated_matching.get("geometry_unique_before_frames"),
            "geometry_unique_after_frames": regenerated_matching.get("geometry_unique_after_frames"),
            "eligible_before_diversity": regenerated_regions[region].get("eligible_before_diversity"),
            "eligible_unique_before_frames": regenerated_regions[region].get("eligible_unique_before_frames"),
            "eligible_unique_after_frames": regenerated_regions[region].get("eligible_unique_after_frames"),
            "maximum_unique_endpoint_pairs": regenerated_regions[region].get("maximum_unique_endpoint_pairs"),
            "selected_unique_endpoint_pairs": regenerated_regions[region].get("selected_unique_endpoint_pairs"),
            "after_gate_diagnostics": regenerated_regions[region].get("after_gate_diagnostics"),
            "diversity_skipped": regenerated_regions[region].get("diversity_skipped"),
            "generated_candidates": len(ranked_pairs),
        },
        "output_dir": str(output_root.resolve()),
        "pairs": [],
    }
    all_rows = []

    for rank in ranks:
        candidate = ranked_pairs[rank - 1]
        for key in ("before_id", "after_id", "score", "terms", "region_gate"):
            if key not in candidate:
                raise ValueError(f"{region} rank {rank}: candidate is missing {key}")
        before_id = candidate["before_id"]
        after_id = candidate["after_id"]
        if before_id not in by_id or after_id not in by_id:
            raise ValueError(
                f"{region} rank {rank}: candidate frame is missing from scan_manifest"
            )
        terms = candidate["terms"]
        if not isinstance(terms, dict) or "face_scale_ratio" not in terms:
            raise ValueError(f"{region} rank {rank}: candidate terms lack face_scale_ratio")

        selection = {
            "rank": rank,
            "score": float(candidate["score"]),
            "face_scale_ratio": float(terms["face_scale_ratio"]),
            "region_gate": candidate["region_gate"],
            "before": _selected_frame(by_id[before_id]),
            "after": _selected_frame(by_id[after_id]),
        }
        before_image, before_masks, before_rows = _load_phase(selection["before"], region)
        after_image, after_masks, after_rows = _load_phase(selection["after"], region)
        if before_image.shape != after_image.shape:
            raise RuntimeError(
                f"{region} rank {rank}: source image dimensions differ: "
                f"{before_image.shape} vs {after_image.shape}"
            )

        deltas = _delta_rows(region, rank, before_rows, after_rows)
        all_rows.extend(deltas)
        before_name = f"{region}_rank{rank}_before.png"
        after_name = f"{region}_rank{rank}_after.png"
        _save_png(output_root / before_name, _overlay(before_image, before_masks, region))
        _save_png(output_root / after_name, _overlay(after_image, after_masks, region))
        summary["pairs"].append(
            {
                "rank": rank,
                "selection": selection,
                "images": {"before": before_name, "after": after_name},
                "deltas": deltas,
            }
        )

    _write_csv(csv_path, all_rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_report_html(summary), encoding="utf-8")
    return summary
