from __future__ import annotations

import copy
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from analysis.run_cheek01 import (
    PAIR_LIMITS,
    ReviewError,
    _approval_command,
    approve_review,
    check_pair_quality,
    main,
    prepare_review,
)


class FakeExtractionError(ValueError):
    def __init__(self):
        super().__init__("left cheek is occluded")
        self.report = {"status": "failed", "errors": [str(self)]}


class FakeExtractor:
    def __init__(self, quality: dict | None = None, fail_phase: str | None = None):
        self.quality = quality or {}
        self.fail_phase = fail_phase
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def extract(self, image_path: Path, output_dir: Path) -> dict:
        phase = output_dir.name
        self.calls.append(phase)
        output_dir.mkdir()
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imencode(".png", image)[1].tofile(output_dir / "roi_overlay.png")
        if phase == self.fail_phase:
            error = FakeExtractionError()
            (output_dir / "roi_points.json").write_text(json.dumps(error.report), encoding="utf-8")
            raise error
        masks = {name: np.zeros(image.shape[:2], dtype=np.uint8)
                 for name in ("left_cheek", "right_cheek", "forehead")}
        masks["left_cheek"][12:18, 3:10] = 1
        masks["right_cheek"][12:18, 22:29] = 1
        masks["forehead"][3:8, 12:20] = 1
        np.savez_compressed(output_dir / "roi_masks.npz", **masks)
        quality = {
            "pose_degrees": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "face_width_px": 24,
            "forehead_L_median": 60.0,
        }
        quality.update(copy.deepcopy(self.quality.get(phase, {})))
        metadata = {
            "schema_version": 1,
            "status": "needs_review",
            "source": {
                "path": str(image_path.resolve()),
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "width": image.shape[1],
                "height": image.shape[0],
            },
            "quality": quality,
            "rois": {name: {"pixel_count": int(np.count_nonzero(mask))} for name, mask in masks.items()},
            "warnings": [],
            "errors": [],
        }
        (output_dir / "roi_points.json").write_text(json.dumps(metadata), encoding="utf-8")
        return metadata


class RoiRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.before = self.root / "塗布前.png"
        self.after = self.root / "塗布後.png"
        for path, value in ((self.before, 110), (self.after, 120)):
            image = np.full((24, 32, 3), value, dtype=np.uint8)
            cv2.imencode(".png", image)[1].tofile(path)
        self.output = self.root / "output"

    def prepare(self, extractor=None, output=None):
        self.extractor = extractor or FakeExtractor()
        return prepare_review(self.before, self.after, output or self.output,
                              self.root / "models", extractor_factory=lambda _: self.extractor)

    def manifest(self):
        return json.loads((self.output / "review.json").read_text(encoding="utf-8"))

    def test_preparation_stops_for_review_and_hashes_every_artifact(self):
        with patch("analysis.analyze_cheek_lab.analyze_pair") as analyzer:
            result = self.prepare()
        analyzer.assert_not_called()
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(self.extractor.calls, ["before", "after"])
        self.assertEqual(len(result["artifacts"]), 8)
        self.assertEqual(len(result["review_id"]), 32)
        self.assertFalse((self.output / "lab").exists())
        for name, digest in result["artifacts"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(self.manifest(), result)
        self.assertIsNotNone(cv2.imdecode(np.fromfile(self.output / "review.png", dtype=np.uint8), cv2.IMREAD_COLOR))

    def test_review_html_contains_exact_runtime_command_quality_and_file_links(self):
        report = self.prepare()
        page = (self.output / "review.html").read_text(encoding="utf-8")
        command = _approval_command(self.output, report["review_id"])
        self.assertIn(command, html.unescape(page))
        self.assertIn(sys.executable, html.unescape(page))
        self.assertIn("生成日時（UTC）", page)
        self.assertIn("生成時点の確認用スナップショット", page)
        self.assertIn("画面左頬の画素数</td><td>42</td><td>42", page)
        self.assertIn("額 L* の前後差（after − before）：0.00", page)
        self.assertIn("露出差の測定値ではありません", page)
        for phase in ("before", "after"):
            for filename in ("roi_overlay.png", "roi_points.json", "roi_masks.npz"):
                self.assertIn(f'href="{phase}/{filename}"', page)

    def test_approval_uses_reviewed_masks_once_without_redetection(self):
        result = self.prepare()
        analyzer = Mock(return_value={"status": "complete"})
        approved = approve_review(self.output, result["review_id"], analyzer=analyzer)
        analyzer.assert_called_once_with(
            self.before, self.after, self.output / "before/roi_masks.npz",
            self.output / "after/roi_masks.npz", self.output / "lab",
        )
        self.assertEqual(approved["status"], "analyzed")
        self.assertIn("approved_at", approved)
        self.assertEqual(self.extractor.calls, ["before", "after"])
        with self.assertRaisesRegex(ReviewError, "state: analyzed"):
            approve_review(self.output, result["review_id"], analyzer=analyzer)
        analyzer.assert_called_once()

    def test_synthetic_review_approval_runs_real_lab_analysis(self):
        # Uniform grayscale inputs have the same Lab change in all three ROIs.
        report = self.prepare()
        approved = approve_review(self.output, report["review_id"])
        self.assertEqual(approved["status"], "analyzed")
        self.assertEqual(self.manifest()["status"], "analyzed")
        self.assertEqual(self.extractor.calls, ["before", "after"])
        lab = self.output / "lab"
        self.assertTrue((lab / "lab_stats.csv").is_file())
        self.assertTrue((lab / "analysis_summary.json").is_file())
        with (lab / "lab_deltas.csv").open(encoding="utf-8", newline="") as handle:
            deltas = list(csv.DictReader(handle))
        self.assertEqual(len(deltas), 12)
        for row in deltas:
            self.assertAlmostEqual(float(row["delta_minus_forehead"]), 0.0, places=5)
            if row["metric"] in ("L_mean", "L_median"):
                self.assertGreater(float(row["delta"]), 0.0)
        self.assertFalse((self.output / ".approval.lock").exists())

    def test_wrong_or_missing_id_cannot_trigger_analysis(self):
        self.prepare()
        analyzer = Mock()
        for review_id in ("", "wrong-review-id"):
            with self.subTest(review_id=review_id):
                with self.assertRaisesRegex(ReviewError, "ID is missing or incorrect"):
                    approve_review(self.output, review_id, analyzer=analyzer)
        analyzer.assert_not_called()
        self.assertEqual(self.manifest()["status"], "needs_review")

    def test_failure_html_displays_available_overlay_without_broken_after_links(self):
        with self.assertRaises(ReviewError):
            self.prepare(FakeExtractor(fail_phase="before"))
        report = self.manifest()
        page = (self.output / "review.html").read_text(encoding="utf-8")
        self.assertIn("left cheek is occluded", page)
        self.assertIn('src="before/roi_overlay.png"', page)
        self.assertIn('href="before/roi_points.json"', page)
        self.assertNotIn('href="after/', page)
        self.assertNotIn('src="review.png"', page)
        self.assertNotIn('href="before/roi_masks.npz"', page)
        self.assertNotIn("--approve-review", page)
        self.assertIn("この結果は承認できません", page)
        self.assertEqual(report["artifacts"]["review.html"],
                         hashlib.sha256((self.output / "review.html").read_bytes()).hexdigest())

    def test_model_setup_failure_has_html_diagnostic_without_image_links(self):
        with self.assertRaisesRegex(ReviewError, "model unavailable"):
            prepare_review(self.before, self.after, self.output,
                           extractor_factory=Mock(side_effect=RuntimeError("model unavailable")))
        page = (self.output / "review.html").read_text(encoding="utf-8")
        self.assertIn("model unavailable", page)
        self.assertNotIn("<img ", page)
        self.assertNotIn('href="before/', page)
        self.assertNotIn('href="after/', page)
        self.assertEqual(self.manifest()["status"], "failed")

    def test_changed_and_missing_artifacts_cannot_trigger_analysis(self):
        result = self.prepare()
        analyzer = Mock()
        for name in result["artifacts"]:
            path = self.output / name
            original = path.read_bytes()
            for missing in (False, True):
                with self.subTest(name=name, missing=missing):
                    if missing:
                        path.unlink()
                    else:
                        path.write_bytes(original + b"modified")
                    with self.assertRaises(ReviewError):
                        approve_review(self.output, result["review_id"], analyzer=analyzer)
                    path.write_bytes(original)
        analyzer.assert_not_called()

    def test_changed_missing_or_different_source_cannot_trigger_analysis(self):
        result = self.prepare()
        analyzer = Mock()
        original = self.before.read_bytes()
        self.before.write_bytes(original + b"change")
        with self.assertRaisesRegex(ReviewError, "Source changed"):
            approve_review(self.output, result["review_id"], analyzer=analyzer)
        self.before.unlink()
        with self.assertRaisesRegex(ReviewError, "file is missing"):
            approve_review(self.output, result["review_id"], analyzer=analyzer)
        self.before.write_bytes(original)
        replacement = self.root / "other-before.png"
        replacement.write_bytes(original)
        with self.assertRaisesRegex(ReviewError, "differs from the reviewed source"):
            approve_review(self.output, result["review_id"], before_path=replacement, analyzer=analyzer)
        analyzer.assert_not_called()

    def test_extraction_failure_stops_immediately_and_cannot_be_approved(self):
        extractor = FakeExtractor(fail_phase="before")
        analyzer = Mock()
        with self.assertRaisesRegex(ReviewError, "occluded"):
            self.prepare(extractor)
        report = self.manifest()
        self.assertEqual(extractor.calls, ["before"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["extraction_failure"]["status"], "failed")
        self.assertFalse((self.output / "before/roi_masks.npz").exists())
        with self.assertRaisesRegex(ReviewError, "state: failed"):
            approve_review(self.output, report["review_id"], analyzer=analyzer)
        analyzer.assert_not_called()

    def test_pair_guard_rejects_large_pose_or_control_differences(self):
        cases = {
            "yaw": {"pose_degrees": {"yaw": 10.01, "pitch": 0, "roll": 0}},
            "pitch": {"pose_degrees": {"yaw": 0, "pitch": -10.01, "roll": 0}},
            "roll": {"pose_degrees": {"yaw": 0, "pitch": 0, "roll": 12.01}},
            "control": {"forehead_L_median": 72.01},
        }
        analyzer = Mock()
        for name, quality in cases.items():
            with self.subTest(name=name):
                output = self.root / name
                with self.assertRaisesRegex(ReviewError, "too large"):
                    self.prepare(FakeExtractor(quality={"after": quality}), output)
                report = json.loads((output / "review.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                with self.assertRaisesRegex(ReviewError, "state: failed"):
                    approve_review(output, report["review_id"], analyzer=analyzer)
                self.assertTrue((output / "review.png").is_file())
                if name == "control":
                    self.assertIn("not an exposure measurement", " ".join(report["errors"]))
        analyzer.assert_not_called()

    def test_pair_limits_include_boundary_and_fail_on_missing_or_nonfinite_values(self):
        before = {"quality": {"pose_degrees": {"yaw": 0, "pitch": 0, "roll": 0}, "forehead_L_median": 50}}
        after = {"quality": {"pose_degrees": {"yaw": 10, "pitch": -10, "roll": 12}, "forehead_L_median": 62}}
        self.assertEqual(check_pair_quality(before, after)["errors"], [])
        for field in PAIR_LIMITS:
            for value in (None, float("nan"), float("inf")):
                broken = copy.deepcopy(after)
                target = broken["quality"] if field == "forehead_L_median" else broken["quality"]["pose_degrees"]
                target[field] = value
                with self.subTest(field=field, value=value):
                    self.assertTrue(check_pair_quality(before, broken)["errors"])
        self.assertEqual(len(check_pair_quality({}, {})["errors"]), 4)

    def test_existing_output_is_not_overwritten_and_empty_output_is_accepted(self):
        self.output.mkdir()
        report = self.prepare()
        before = (self.output / "review.json").read_bytes()
        with self.assertRaisesRegex(ReviewError, "new or empty"):
            self.prepare()
        self.assertEqual((self.output / "review.json").read_bytes(), before)
        self.assertEqual(self.manifest()["review_id"], report["review_id"])

    def test_existing_analysis_output_and_lock_reject_approval(self):
        report = self.prepare()
        analyzer = Mock()
        (self.output / "lab").mkdir()
        with self.assertRaisesRegex(ReviewError, "already exists"):
            approve_review(self.output, report["review_id"], analyzer=analyzer)
        (self.output / "lab").rmdir()
        (self.output / ".approval.lock").write_text("other-process", encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "already running"):
            approve_review(self.output, report["review_id"], analyzer=analyzer)
        analyzer.assert_not_called()
        self.assertTrue((self.output / ".approval.lock").exists())

    def test_analysis_failure_is_recorded_and_cannot_be_retried_as_approved(self):
        report = self.prepare()
        analyzer = Mock(side_effect=RuntimeError("disk full"))
        with self.assertRaisesRegex(ReviewError, "disk full"):
            approve_review(self.output, report["review_id"], analyzer=analyzer)
        self.assertEqual(self.manifest()["status"], "analysis_failed")
        self.assertFalse((self.output / ".approval.lock").exists())
        with self.assertRaisesRegex(ReviewError, "state: analysis_failed"):
            approve_review(self.output, report["review_id"], analyzer=analyzer)
        analyzer.assert_called_once()

    def test_incomplete_hash_manifest_rejects_approval(self):
        report = self.prepare()
        del report["artifacts"]["before/roi_overlay.png"]
        (self.output / "review.json").write_text(json.dumps(report), encoding="utf-8")
        analyzer = Mock()
        with self.assertRaisesRegex(ReviewError, "manifest is incomplete"):
            approve_review(self.output, report["review_id"], analyzer=analyzer)
        analyzer.assert_not_called()

    def test_cli_failure_is_nonzero_and_missing_inputs_do_not_create_output(self):
        with patch("sys.stderr"):
            code = main(["--output", str(self.output), "--approve-review", "unknown"])
        self.assertEqual(code, 1)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ReviewError, "file is missing"):
            prepare_review(self.root / "missing.png", self.after, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
