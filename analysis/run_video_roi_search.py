"""Sample a local video, extract each ROI, and search geometry-matched pairs.

Frames stay at source resolution. Failed frames are recorded and excluded;
the batch continues. No Lab analysis or human approval is performed here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from analysis.extract_face_rois import ROI_VERSION, RoiConfig, RoiExtractor, RoiGenerationError, save_image


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def probe_video(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f'Video does not exist: {path}')
    if not shutil.which('ffprobe'):
        raise ValueError('ffprobe is required to read video duration reliably')
    process = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate:format=duration', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True)
    data = json.loads(process.stdout)
    if not data.get('streams'):
        raise ValueError('No video stream')
    stream = data['streams'][0]
    numerator, denominator = map(float, stream['avg_frame_rate'].split('/'))
    duration = float(data['format']['duration'])
    fps = numerator / denominator
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid video duration/frame rate')
    return dict(path=str(path.resolve()), sha256=file_hash(path), duration_seconds=duration,
                fps=fps, width=int(stream['width']), height=int(stream['height']))


def sample_times(start: float, end: float, interval: float) -> list[float]:
    if not all(math.isfinite(v) for v in (start, end, interval)) or start < 0 or end <= start or interval <= 0:
        raise ValueError('Sampling needs 0 <= start < end and a positive interval')
    return [start + i * interval for i in range(math.ceil((end - start) / interval))
            if start + i * interval < end]


def validate_range(value, lower: float, upper: float, label: str) -> tuple[float, float]:
    start, end = (lower, upper) if value is None else map(float, value)
    if not all(math.isfinite(v) for v in (start, end)) or not lower <= start < end <= upper:
        raise ValueError(f'{label} range must be inside [{lower:.3f}, {upper:.3f}] with start < end')
    return start, end


def filter_records(records: list[dict], before_range: tuple, after_range: tuple) -> list[dict]:
    return [record for record in records if any(start <= record['timestamp_seconds'] < end
                                               for start, end in (before_range, after_range))]


def extract_samples(video: dict, output: Path, timestamps: list[float], records: list[dict],
                    extractor: RoiExtractor, stage: str) -> None:
    known = {item['frame_id'] for item in records}
    requested_done = {round(item['requested_timestamp_seconds'], 6) for item in records}
    cap = cv2.VideoCapture(video['path'])
    if not cap.isOpened():
        raise ValueError('Could not open video')
    (output / 'frames').mkdir(exist_ok=True)
    (output / 'rois').mkdir(exist_ok=True)
    added = 0
    try:
        with (output / 'frames.jsonl').open('a', encoding='utf-8') as log:
            for requested in timestamps:
                if round(requested, 6) in requested_done:
                    continue
                if not cap.set(cv2.CAP_PROP_POS_MSEC, requested * 1000):
                    raise ValueError(f'Video seeking failed at {requested:.3f}s')
                ok, image = cap.read()
                if not ok:
                    # Container duration can extend beyond the final video frame.
                    if video['duration_seconds'] - requested <= 1 / video['fps'] + 0.1:
                        continue
                    raise ValueError(f'Video decoding failed at {requested:.3f}s')
                actual = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                if not math.isfinite(actual) or actual < 0 or abs(actual - requested) > 1.0:
                    raise ValueError(f'Unreliable decoded timestamp: requested={requested}, decoded={actual}')
                frame_id = f'frame_{round(actual * 1000):010d}ms'
                if frame_id in known:
                    continue
                image_path = output / 'frames' / (frame_id + '.png')
                roi_dir = output / 'rois' / frame_id
                if image_path.exists() or roi_dir.exists():
                    raise ValueError(f'Incomplete prior frame output exists: {frame_id}. Use a new output directory.')
                save_image(image_path, image)
                try:
                    report = extractor.extract(image_path, roi_dir)
                except RoiGenerationError as exc:
                    report = exc.report
                record = dict(frame_id=frame_id, requested_timestamp_seconds=requested,
                              timestamp_seconds=actual, frame_index=int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
                              stage=stage, image_path=str(image_path), roi_dir=str(roi_dir), report=report)
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush()
                records.append(record)
                known.add(frame_id)
                requested_done.add(round(requested, 6))
                added += 1
                if added % 10 == 0:
                    passed = sum(r['report']['status'] == 'needs_review' for r in records)
                    print(f'{stage}: {len(records)} frames, {passed} ROI candidates; t={actual:.1f}s', flush=True)
    finally:
        cap.release()
    print(f'{stage}: added {added} frames', flush=True)


def extraction_signature(model_dir: Path) -> dict:
    return dict(roi_version=ROI_VERSION, config=asdict(RoiConfig()), mediapipe_version=version('mediapipe'),
                models={name: {'sha256': file_hash(model_dir / name)}
                        for name in ('face_landmarker.task', 'hand_landmarker.task')})


def verify_records(records: list[dict], expected_signature: dict | None = None) -> None:
    for record in records:
        if expected_signature is not None:
            saved_signature = {key: record['report'].get(key) for key in expected_signature}
            if saved_signature != expected_signature:
                raise ValueError(f"Extraction models/settings changed: {record['frame_id']}. Use a new scan.")
        source = record['report']['source']
        if source.get('sha256') != file_hash(Path(record['image_path'])):
            raise ValueError(f"Saved frame changed: {record['frame_id']}")
        actual_report = json.loads((Path(record['roi_dir']) / 'roi_points.json').read_text(encoding='utf-8'))
        if actual_report != record['report']:
            raise ValueError(f"ROI metadata changed: {record['frame_id']}")


def save_frame_index(output: Path, records: list[dict]) -> None:
    with (output / 'frames.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['frame_id', 'timestamp_seconds', 'stage', 'status',
            'yaw', 'pitch', 'roll', 'face_width_px', 'detected_hands', 'errors', 'image_path', 'roi_dir'])
        writer.writeheader()
        for record in sorted(records, key=lambda r: r['timestamp_seconds']):
            report, quality = record['report'], record['report'].get('quality', {})
            writer.writerow({**{k: record[k] for k in ('frame_id', 'timestamp_seconds', 'stage', 'image_path', 'roi_dir')},
                'status': report['status'], **{axis: quality.get('pose_degrees', {}).get(axis) for axis in ('yaw', 'pitch', 'roll')},
                'face_width_px': quality.get('face_width_px'), 'detected_hands': quality.get('detected_hands'),
                'errors': '; '.join(report['errors'])})


def run(args) -> dict:
    from analysis.match_video_rois import rank_pairs
    from analysis.video_roi_report import write_video_report

    video = probe_video(args.video.resolve())
    output = args.output.resolve()
    duration = video['duration_seconds']
    split = duration / 2 if args.split is None else args.split
    if not 0 < split < duration or not math.isfinite(split):
        raise ValueError('Split must lie inside the video')
    before_range = validate_range(args.before_range, 0, split, 'Before')
    after_range = validate_range(args.after_range, split, duration, 'After')
    interval = args.interval
    sample_times(0, duration, interval)  # Validate before creating output.
    if not math.isfinite(args.refine_interval) or not 0 < args.refine_interval <= interval:
        raise ValueError('Refinement interval must be >0 and <= coarse interval')
    gap = duration * 0.25 if args.min_gap is None else args.min_gap
    if not math.isfinite(gap) or gap < 0:
        raise ValueError('Minimum pair separation must be finite and nonnegative')
    if args.top < 1:
        raise ValueError('top must be positive')
    records = []
    signature = extraction_signature(args.model_dir)
    if args.resume:
        saved = json.loads((output / 'scan_config.json').read_text(encoding='utf-8'))
        if saved['video'] != video or saved['sample_interval_seconds'] != interval:
            raise ValueError('Resume requires the unchanged video and the same coarse interval')
        if saved.get('extraction_signature', signature) != signature:
            raise ValueError('Resume requires unchanged extraction models and settings')
        log_path = output / 'frames.jsonl'
        records = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line] if log_path.exists() else []
        verify_records(records, signature)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output must be new/empty; use --resume for an unchanged scan')
        output.mkdir(parents=True, exist_ok=True)
    (output / 'frames.jsonl').touch(exist_ok=True)
    write_json(output / 'scan_config.json', dict(video=video, sample_interval_seconds=interval, extraction_signature=signature))
    with RoiExtractor(args.model_dir, RoiConfig()) as extractor:
        extract_samples(video, output, sample_times(0, duration, interval), records, extractor, 'coarse')
        considered = filter_records(records, before_range, after_range)
        matching = rank_pairs(considered, split, gap, top_k=args.top)
        if not args.no_refine and matching['ranked_pairs']:
            times = set()
            for pair in matching['ranked_pairs'][:5]:
                for time_key, bounds in (('before_time', before_range), ('after_time', after_range)):
                    center = pair[time_key]
                    start, end = max(bounds[0], center - interval), min(bounds[1], center + interval)
                    if end > start:
                        times.update(sample_times(start, end, args.refine_interval))
            extract_samples(video, output, sorted(times), records, extractor, 'refined')
            matching = rank_pairs(filter_records(records, before_range, after_range), split, gap, top_k=args.top)
    config = dict(sample_interval_seconds=interval, refine_interval_seconds=None if args.no_refine else args.refine_interval,
                  split_seconds=split, min_gap_seconds=gap, before_range=list(before_range), after_range=list(after_range))
    manifest = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(), video=video, config=config,
        selected_ranges={'before': list(before_range), 'after': list(after_range)},
        notes=args.note or [], records=records,
        counts=dict(total=len(records), roi_candidates=sum(r['report']['status'] == 'needs_review' for r in records),
                    considered_for_matching=len(filter_records(records, before_range, after_range))),
        status='needs_review' if matching['ranked_pairs'] else 'no_eligible_pairs')
    save_frame_index(output, records)
    write_json(output / 'scan_manifest.json', manifest)
    write_json(output / 'matching.json', matching)
    artifacts = write_video_report(output, manifest, matching)
    print(json.dumps(dict(status=manifest['status'], counts=manifest['counts'], artifacts=artifacts), ensure_ascii=False), flush=True)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5.0)
    parser.add_argument('--refine-interval', type=float, default=1.0)
    parser.add_argument('--no-refine', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--split', type=float, help='Split timestamp in seconds; default midpoint')
    parser.add_argument('--before-range', type=float, nargs=2, metavar=('START', 'END'))
    parser.add_argument('--after-range', type=float, nargs=2, metavar=('START', 'END'))
    parser.add_argument('--min-gap', type=float, help='Minimum time separation, default 25%% of duration')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--note', action='append', help='Human-supplied scene observations shown in report')
    return parser.parse_args(argv)


if __name__ == '__main__':
    try:
        result = run(parse_args())
        raise SystemExit(0 if result['status'] == 'needs_review' else 2)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'Stopped: {error}', file=sys.stderr)
        raise SystemExit(1)
