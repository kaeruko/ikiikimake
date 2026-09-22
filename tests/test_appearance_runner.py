from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import analysis.analyze_appearance as appearance


def metric(value=10.0, *, status="ok", ident="left_cheek_L_mean") -> dict:
    return {
        "id": ident, "region": "left_cheek", "label": "Synthetic luminance",
        "unit": "L*", "value": value, "status": status, "pixels": 30,
        "note": "Synthetic fixture for runner I/O checks",
    }


class CompareMeasurementsTests(unittest.TestCase):
    def test_identical_values_are_zero_and_swap_reverses_sign(self):
        before = [metric(12.25), metric(-3.0, ident="left_cheek_a_mean")]
        after = [metric(19.5), metric(-5.0, ident="left_cheek_a_mean")]
        self.assertEqual([row["delta"] for row in appearance.compare_measurements(before, before)], [0.0, 0.0])
        forward = appearance.compare_measurements(before, after)
        reverse = appearance.compare_measurements(after, before)
        for first, second in zip(forward, reverse):
            self.assertEqual(first["delta"], -second["delta"])
            self.assertEqual(first["before"], second["after"])
            self.assertEqual(first["after"], second["before"])

    def test_unavailable_values_remain_none(self):
        missing = metric(None, status="insufficient_pixels")
        for before, after in ((missing, metric(12)), (metric(12), missing), (missing, missing)):
            with self.subTest(before=before["value"], after=after["value"]):
                row = appearance.compare_measurements([before], [after])[0]
                self.assertIsNone(row["delta"])
                self.assertEqual(row["before"], before["value"])
                self.assertEqual(row["after"], after["value"])
                self.assertEqual(row["status"], "insufficient_pixels")

    def test_nonfinite_measurements_are_rejected_in_any_status(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for status in ("ok", "insufficient_pixels"):
                for nonfinite_first in (True, False):
                    with self.subTest(value=value, status=status, nonfinite_first=nonfinite_first):
                        values = ([metric(value, status=status)], [metric(12)])
                        if not nonfinite_first:
                            values = values[::-1]
                        with self.assertRaisesRegex(ValueError, "non-finite"):
                            appearance.compare_measurements(*values)

    def test_duplicate_missing_or_changed_definitions_are_rejected(self):
        for before, after in (([metric(), metric()], [metric()]),
                              ([metric()], [metric(ident="different")]),
                              ([metric()], [dict(metric(), unit="percent")])):
            with self.subTest(before=before, after=after):
                with self.assertRaises(ValueError):
                    appearance.compare_measurements(before, after)


class AppearanceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selected_path = self.root / "selected_pair.json"
        self.output = self.root / "appearance"
        self.shape = (32, 48, 3)
        self.selected = {"video_path": str(self.root / "fixture.mp4")}
        for phase, level in (("before", 80), ("after", 100)):
            image_path = self.root / f"{phase}.png"
            self.save_image(image_path, level)
            roi_dir = self.root / phase
            roi_dir.mkdir()
            masks = {name: np.zeros(self.shape[:2], dtype=np.uint8)
                     for name in ("left_cheek", "right_cheek", "forehead")}
            masks["left_cheek"][18:24, 4:10] = 1
            masks["right_cheek"][18:24, 38:44] = 1
            masks["forehead"][4:9, 20:28] = 1
            np.savez_compressed(roi_dir / "roi_masks.npz", **masks)
            landmarks = np.tile([0.5, 0.5, 0.0], (478, 1))
            landmarks[234, 0] = 0.2
            landmarks[454, 0] = 0.8
            report = {
                "status": "needs_review", "errors": [],
                "source": {"path": str(image_path), "sha256": self.digest(image_path),
                           "width": self.shape[1], "height": self.shape[0]},
                "face_landmarks_normalized": landmarks.tolist(),
                "quality": {"pose_degrees": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}},
            }
            (roi_dir / "roi_points.json").write_text(json.dumps(report), encoding="utf-8")
            (roi_dir / "roi_overlay.png").write_bytes(image_path.read_bytes())
            self.selected[phase] = {"image_path": str(image_path), "image_sha256": self.digest(image_path),
                                    "roi_dir": str(roi_dir), "timestamp_seconds": 1 if phase == "before" else 12}
            self.update_roi_hashes(phase)
        self.save_selected()
        self.build_patch = patch.object(appearance, "build_feature_masks", side_effect=lambda shape, points, masks: masks)
        self.measure_patch = patch.object(appearance, "measure_features", side_effect=lambda image, masks: [metric(float(image.mean()))])
        self.visual_patch = patch.object(appearance, "_save_visuals", side_effect=self.fake_visuals)
        self.build = self.build_patch.start()
        self.measure = self.measure_patch.start()
        self.visuals = self.visual_patch.start()
        self.addCleanup(self.build_patch.stop)
        self.addCleanup(self.measure_patch.stop)
        self.addCleanup(self.visual_patch.stop)

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_image(self, path: Path, level: int):
        cv2.imencode(".png", np.full(self.shape, level, dtype=np.uint8))[1].tofile(path)

    def update_roi_hashes(self, phase: str):
        item = self.selected[phase]
        for name, key in (("roi_masks.npz", "roi_masks_sha256"), ("roi_points.json", "roi_points_sha256"),
                          ("roi_overlay.png", "roi_overlay_sha256")):
            item[key] = self.digest(Path(item["roi_dir"]) / name)

    def save_selected(self):
        self.selected_path.write_text(json.dumps(self.selected), encoding="utf-8")

    @staticmethod
    def fake_visuals(output: Path, frames: dict, deltas: list):
        encoded = cv2.imencode(".png", np.full((8, 8, 3), 128, dtype=np.uint8))[1]
        for name in ("appearance_rois.png", "region_samples.png", "feature_changes.png"):
            encoded.tofile(output / name)

    def run_analysis(self):
        return appearance.analyze_selected_appearance(self.selected_path, self.output)

    def test_saved_run_reuses_only_verified_artifacts_without_remeasuring(self):
        first = self.run_analysis()
        run = Path(first["output_dir"])
        self.assertEqual(run.parent, self.output)
        self.assertEqual(run.name, first["fingerprint"][:16])
        self.assertEqual(first["deltas"][0]["delta"], 20.0)
        self.assertIsNone(first["overall_impression"])
        self.assertTrue(first["native_pixels_only"])
        manifest = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["files"]), set(appearance.ARTIFACTS))
        for name, digest in manifest["files"].items():
            self.assertEqual(self.digest(run / name), digest)
        snapshot = {path.name: path.read_bytes() for path in run.iterdir()}
        second = self.run_analysis()
        self.assertEqual(second, first)
        self.assertEqual(self.measure.call_count, 2)
        self.visuals.assert_called_once()
        self.assertEqual(snapshot, {path.name: path.read_bytes() for path in run.iterdir()})

    def test_changed_or_missing_input_fails_before_creating_output(self):
        for phase in ("before", "after"):
            item = self.selected[phase]
            paths = [Path(item["image_path"])] + [Path(item["roi_dir"]) / name
                                                      for name in ("roi_masks.npz", "roi_points.json", "roi_overlay.png")]
            for path in paths:
                original = path.read_bytes()
                for missing in (False, True):
                    with self.subTest(phase=phase, file=path.name, missing=missing):
                        if missing:
                            path.unlink()
                        else:
                            path.write_bytes(original + b"changed")
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            self.run_analysis()
                        self.assertFalse(self.output.exists())
                        path.write_bytes(original)
        self.measure.assert_not_called()
        self.visuals.assert_not_called()

    def test_source_dimensions_are_checked_even_when_all_hashes_match(self):
        item = self.selected["before"]
        path = Path(item["roi_dir"]) / "roi_points.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["source"]["width"] += 1
        path.write_text(json.dumps(report), encoding="utf-8")
        self.update_roi_hashes("before")
        self.save_selected()
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            self.run_analysis()
        self.assertFalse(self.output.exists())
        self.measure.assert_not_called()

    def test_tampered_or_missing_saved_artifact_is_rejected(self):
        result = self.run_analysis()
        run = Path(result["output_dir"])
        for name in appearance.ARTIFACTS:
            path = run / name
            original = path.read_bytes()
            for missing in (False, True):
                with self.subTest(name=name, missing=missing):
                    if missing:
                        path.unlink()
                    else:
                        path.write_bytes(original + b"tampered")
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        self.run_analysis()
                    path.write_bytes(original)
        self.assertEqual(self.measure.call_count, 2)
        self.visuals.assert_called_once()
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_incomplete_or_wrong_saved_manifest_is_rejected(self):
        result = self.run_analysis()
        path = Path(result["output_dir"]) / "artifact_manifest.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for mode in ("missing_file", "wrong_fingerprint"):
            altered = copy.deepcopy(original)
            if mode == "missing_file":
                del altered["files"]["report.html"]
            else:
                altered["fingerprint"] = "0" * 64
            with self.subTest(mode=mode):
                path.write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid artifact manifest"):
                    self.run_analysis()
        self.assertEqual(self.measure.call_count, 2)

    def test_new_content_at_same_paths_creates_separate_run_and_preserves_old_run(self):
        first = self.run_analysis()
        old_run = Path(first["output_dir"])
        old_files = {path.name: path.read_bytes() for path in old_run.iterdir()}
        item = self.selected["before"]
        image_path, roi_dir = Path(item["image_path"]), Path(item["roi_dir"])
        self.save_image(image_path, 90)
        item["image_sha256"] = self.digest(image_path)
        points_path = roi_dir / "roi_points.json"
        points = json.loads(points_path.read_text(encoding="utf-8"))
        points["source"]["sha256"] = item["image_sha256"]
        points_path.write_text(json.dumps(points), encoding="utf-8")
        (roi_dir / "roi_overlay.png").write_bytes(image_path.read_bytes())
        self.update_roi_hashes("before")
        self.save_selected()
        second = self.run_analysis()
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        self.assertEqual(second["deltas"][0]["delta"], 10.0)
        self.assertEqual(len(list(self.output.iterdir())), 2)
        self.assertEqual(old_files, {path.name: path.read_bytes() for path in old_run.iterdir()})
        self.assertEqual(first["input_specification"]["inputs"]["before"]["image_path"],
                         second["input_specification"]["inputs"]["before"]["image_path"])

    def test_version_implementation_and_runtime_changes_have_separate_runs(self):
        baseline = self.run_analysis()
        real_hash = appearance.sha256_file
        def changed_implementation(path):
            return "f" * 64 if Path(path).name == "appearance_features.py" else real_hash(path)
        contexts = (
            patch.object(appearance, "APPEARANCE_VERSION", "test-new-version"),
            patch.object(appearance, "sha256_file", side_effect=changed_implementation),
            patch.object(appearance.cv2, "__version__", "test-new-runtime"),
        )
        fingerprints = {baseline["fingerprint"]}
        for context in contexts:
            with context:
                result = self.run_analysis()
                self.assertNotIn(result["fingerprint"], fingerprints)
                fingerprints.add(result["fingerprint"])
        self.assertEqual(len(list(self.output.iterdir())), 4)

    def test_nonfinite_metric_does_not_create_output(self):
        self.measure.side_effect = lambda image, masks: [metric(float("nan"))]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.run_analysis()
        self.assertFalse(self.output.exists())
        self.visuals.assert_not_called()


if __name__ == "__main__":
    unittest.main()
