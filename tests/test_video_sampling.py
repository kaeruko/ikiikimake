from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from analysis.extract_face_rois import RoiGenerationError
from analysis import run_video_roi_search as sampling


class FakeCapture:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.index = -1
        self.requested_ms = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, key, value):
        assert key == cv2.CAP_PROP_POS_MSEC
        self.requested_ms.append(value)
        self.index += 1
        return self.outcomes[self.index].get("seek_ok", True)

    def read(self):
        outcome = self.outcomes[self.index]
        return outcome.get("read_ok", True), np.full((32, 48, 3), outcome.get("pixel", 128), np.uint8)

    def get(self, key):
        outcome = self.outcomes[self.index]
        if key == cv2.CAP_PROP_POS_MSEC:
            return outcome["actual"] * 1000
        if key == cv2.CAP_PROP_POS_FRAMES:
            return outcome.get("next_frame", self.index + 1)
        raise AssertionError(f"Unexpected capture property {key}")

    def release(self):
        self.released = True


class FakeExtractor:
    def __init__(self, failed_calls=()):
        self.failed_calls = set(failed_calls)
        self.calls = []

    def extract(self, image_path, roi_dir):
        self.calls.append(image_path)
        failed = len(self.calls) in self.failed_calls
        report = {
            "status": "failed" if failed else "needs_review",
            "errors": ["synthetic hand occlusion"] if failed else [],
            "source": {"path": str(image_path.resolve()), "sha256": sampling.file_hash(image_path)},
        }
        roi_dir.mkdir()
        (roi_dir / "roi_points.json").write_text(json.dumps(report), encoding="utf-8")
        if failed:
            raise RoiGenerationError(report)
        return report


class SamplingBoundariesTests(unittest.TestCase):
    def test_sample_times_include_start_and_exclude_end(self):
        self.assertEqual(sampling.sample_times(0, 10, 5), [0, 5])
        self.assertEqual(sampling.sample_times(2, 10, 5), [2, 7])
        self.assertEqual(sampling.sample_times(1, 1.1, 5), [1])
        self.assertEqual(sampling.sample_times(0, 0.3, 0.1), [0, 0.1, 0.2])

    def test_rounding_does_not_create_out_of_range_or_duplicate_samples(self):
        for start, end, interval in ((0.9999998, 1.0, 1.0), (0, 0.0000005, 0.0000001)):
            with self.subTest(start=start, end=end, interval=interval):
                times = sampling.sample_times(start, end, interval)
                self.assertTrue(all(start <= value < end for value in times))
                self.assertEqual(len(times), len(set(times)))

    def test_bad_sampling_inputs_fail(self):
        for values in ((-1, 10, 1), (0, 0, 1), (1, 0, 1), (0, 10, 0), (0, 10, -1),
                       (0, float("inf"), 1), (float("nan"), 10, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                sampling.sample_times(*values)

    def test_ranges_are_within_phase_and_nonempty(self):
        self.assertEqual(sampling.validate_range(None, 0, 100, "before"), (0, 100))
        self.assertEqual(sampling.validate_range([65, 100], 0, 100, "before"), (65, 100))
        for value in ((-1, 99), (0, 101), (99, 99), (100, 90), (0, float("nan"))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sampling.validate_range(value, 0, 100, "before")

    def test_filter_uses_decoded_time_and_exclusive_end(self):
        times = (64.99, 65, 195.99, 196, 2021.99, 2022, 2123.99, 2124)
        records = [{"timestamp_seconds": time, "requested_timestamp_seconds": 0} for time in times]
        original = deepcopy(records)
        result = sampling.filter_records(records, (65, 196), (2022, 2124))
        self.assertEqual([r["timestamp_seconds"] for r in result], [65, 195.99, 2022, 2123.99])
        self.assertEqual(records, original)


class ExtractionSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = {"path": str(self.root / "fake.mp4"), "duration_seconds": 20, "fps": 10}

    def extract_fake(self, requested, outcomes, *, failed_calls=(), records=None):
        capture = FakeCapture(outcomes)
        extractor = FakeExtractor(failed_calls)
        records = [] if records is None else records
        with patch.object(sampling.cv2, "VideoCapture", return_value=capture):
            sampling.extract_samples(self.video, self.root, requested, records, extractor, "test")
        return records, capture, extractor

    def test_actual_timestamp_and_frame_index_are_recorded(self):
        records, capture, _ = self.extract_fake([1.02], [{"actual": 1.1, "next_frame": 12}])
        record = records[0]
        self.assertEqual(record["requested_timestamp_seconds"], 1.02)
        self.assertEqual(record["timestamp_seconds"], 1.1)
        self.assertEqual(record["frame_index"], 11)
        self.assertEqual(record["frame_id"], "frame_0000001100ms")
        self.assertEqual(capture.requested_ms, [1020])
        self.assertTrue(capture.released)

    def test_failed_roi_is_persisted_and_next_sample_is_processed(self):
        records, _, extractor = self.extract_fake([1, 2], [{"actual": 1}, {"actual": 2}], failed_calls=(1,))
        self.assertEqual(len(extractor.calls), 2)
        self.assertEqual([r["report"]["status"] for r in records], ["failed", "needs_review"])
        lines = (self.root / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line) for line in lines], records)
        sampling.verify_records(records)

    def test_resume_skips_completed_requests_including_failed_rois(self):
        records, _, _ = self.extract_fake([1, 2], [{"actual": 1}, {"actual": 2}], failed_calls=(1,))
        resumed, capture, extractor = self.extract_fake([1, 2], [], records=records)
        self.assertEqual(len(resumed), 2)
        self.assertEqual(capture.requested_ms, [])
        self.assertEqual(extractor.calls, [])

    def test_two_seek_requests_for_same_decoded_frame_do_not_duplicate_it(self):
        records, _, extractor = self.extract_fake([1, 1.02], [{"actual": 1}, {"actual": 1}])
        self.assertEqual(len(records), 1)
        self.assertEqual(len(extractor.calls), 1)

    def test_unreliable_timestamps_fail_before_extraction(self):
        for actual in (-1, float("nan"), 3):
            with self.subTest(actual=actual):
                capture = FakeCapture([{"actual": actual}])
                extractor = FakeExtractor()
                with patch.object(sampling.cv2, "VideoCapture", return_value=capture), self.assertRaisesRegex(ValueError, "timestamp"):
                    sampling.extract_samples(self.video, self.root, [1], [], extractor, "test")
                self.assertFalse(extractor.calls)
                self.assertTrue(capture.released)

    def test_failed_decode_near_video_tail_is_skipped_but_middle_is_error(self):
        records, capture, _ = self.extract_fake([19.95], [{"read_ok": False}])
        self.assertFalse(records)
        self.assertTrue(capture.released)
        with patch.object(sampling.cv2, "VideoCapture", return_value=FakeCapture([{"read_ok": False}])):
            with self.assertRaisesRegex(ValueError, "decoding failed"):
                sampling.extract_samples(self.video, self.root, [10], [], FakeExtractor(), "test")

    def test_resume_detects_changed_saved_frame(self):
        records, _, _ = self.extract_fake([1], [{"actual": 1}])
        frame = Path(records[0]["image_path"])
        frame.write_bytes(frame.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "Saved frame changed"):
            sampling.verify_records(records)

    def test_resume_detects_changed_roi_metadata(self):
        records, _, _ = self.extract_fake([1], [{"actual": 1}])
        roi_path = Path(records[0]["roi_dir"]) / "roi_points.json"
        modified = deepcopy(records[0]["report"])
        modified["errors"] = ["changed"]
        roi_path.write_text(json.dumps(modified), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ROI metadata changed"):
            sampling.verify_records(records)

    def test_resume_rejects_changed_source_video_before_extracting(self):
        old_video = dict(self.video, sha256="old", width=48, height=32)
        sampling.write_json(self.root / "scan_config.json", {"video": old_video, "sample_interval_seconds": 5})
        args = sampling.parse_args(["--video", self.video["path"], "--output", str(self.root), "--resume", "--no-refine"])
        with patch.object(sampling, "probe_video", return_value=dict(old_video, sha256="changed")), \
                patch.object(sampling, "extraction_signature", return_value={"models": {}}), \
                patch.object(sampling, "RoiExtractor") as extractor:
            with self.assertRaisesRegex(ValueError, "unchanged video"):
                sampling.run(args)
            extractor.assert_not_called()

    def test_resume_rejects_changed_model_signature(self):
        video = dict(self.video, sha256="unchanged", width=48, height=32)
        old_signature = {"models": {"face_landmarker.task": {"sha256": "old"}}}
        new_signature = {"models": {"face_landmarker.task": {"sha256": "new"}}}
        sampling.write_json(self.root / "scan_config.json", {
            "video": video, "sample_interval_seconds": 5, "extraction_signature": old_signature,
        })
        args = sampling.parse_args(["--video", self.video["path"], "--output", str(self.root), "--resume", "--no-refine"])
        with patch.object(sampling, "probe_video", return_value=video), \
                patch.object(sampling, "extraction_signature", return_value=new_signature), \
                patch.object(sampling, "RoiExtractor") as extractor:
            with self.assertRaisesRegex(ValueError, "unchanged extraction"):
                sampling.run(args)
            extractor.assert_not_called()

    def test_per_frame_signature_guard_also_protects_legacy_scan_config(self):
        records, _, _ = self.extract_fake([1], [{"actual": 1}])
        report = records[0]["report"]
        signature = {"roi_version": "test-v1", "config": {"polygon_scale": 0.88}, "models": {"model": "same"}}
        report.update(signature)
        (Path(records[0]["roi_dir"]) / "roi_points.json").write_text(json.dumps(report), encoding="utf-8")
        sampling.verify_records(records, signature)
        with self.assertRaisesRegex(ValueError, "models/settings changed"):
            sampling.verify_records(records, dict(signature, roi_version="test-v2"))

    def test_resume_recovers_when_startup_failed_before_log_creation(self):
        video = dict(self.video, sha256="unchanged", width=48, height=32)
        sampling.write_json(self.root / "scan_config.json", {"video": video, "sample_interval_seconds": 5})
        self.assertFalse((self.root / "frames.jsonl").exists())
        args = sampling.parse_args(["--video", self.video["path"], "--output", str(self.root), "--resume", "--no-refine"])
        with patch.object(sampling, "probe_video", return_value=video), \
                patch.object(sampling, "extraction_signature", return_value={"models": {}}), \
                patch.object(sampling, "RoiExtractor"), \
                patch.object(sampling, "extract_samples") as extract, \
                patch("analysis.video_roi_report.write_video_report", return_value={}):
            manifest = sampling.run(args)
        self.assertTrue((self.root / "frames.jsonl").is_file())
        self.assertEqual(manifest["counts"]["total"], 0)
        self.assertEqual(extract.call_count, 1)

    def test_real_decoder_timestamp_matches_encoded_frame_content(self):
        path = self.root / "timestamp_test.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (48, 32))
        if not writer.isOpened():
            self.skipTest("MJPG encoder is unavailable in this OpenCV build")
        try:
            for index in range(15):
                writer.write(np.full((32, 48, 3), index * 12, np.uint8))
        finally:
            writer.release()
        video = {"path": str(path), "duration_seconds": 1.5, "fps": 10}
        records = []
        sampling.extract_samples(video, self.root, [0, 0.22, 0.78, 1.2], records, FakeExtractor(), "real")
        self.assertEqual(len(records), 4)
        for record in records:
            pixels = cv2.imread(record["image_path"])
            content_index = round(float(pixels.mean()) / 12)
            self.assertEqual(record["frame_index"], content_index)
            self.assertAlmostEqual(record["timestamp_seconds"], content_index / 10, places=5)


if __name__ == "__main__":
    unittest.main()
