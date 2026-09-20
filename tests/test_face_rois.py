from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from analysis.extract_face_rois import (
    ROI_INDICES,
    RoiConfig,
    RoiExtractor,
    RoiGenerationError,
    assess_geometry,
    hand_coverage,
    polygons_from_landmarks,
    pose_from_matrix,
    rasterize_polygons,
)


def synthetic_face() -> np.ndarray:
    points = np.full((478, 2), 200.0)
    points[234] = (70, 220)
    points[454] = (330, 220)
    for name, center, radii in (
        ("cheek_a", (125, 240), (36, 28)),
        ("cheek_b", (275, 240), (36, 28)),
        ("forehead", (200, 100), (55, 25)),
    ):
        indices = ROI_INDICES[name]
        angles = np.arange(len(indices)) * (2 * math.pi / len(indices))
        polygon = np.column_stack((np.cos(angles), np.sin(angles))) * radii + center
        points[list(indices)] = polygon
    return points


def rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    cosine, sine = math.cos(theta), math.sin(theta)
    rotations = {
        "pitch": [[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]],
        "yaw": [[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]],
        "roll": [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]],
    }
    matrix = np.eye(4)
    matrix[:3, :3] = rotations[axis]
    return matrix


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((400, 400, 3), 128, dtype=np.uint8)
        self.points = synthetic_face()
        self.config = RoiConfig()

    def test_good_geometry_still_requires_review_and_does_not_invent_confidence(self) -> None:
        polygons, masks, quality, errors, warnings, _ = assess_geometry(
            self.image, self.points, np.eye(4), [], self.config
        )
        self.assertFalse(errors)
        self.assertEqual(set(masks), {"left_cheek", "right_cheek", "forehead"})
        self.assertLess(polygons["left_cheek"][:, 0].mean(), polygons["right_cheek"][:, 0].mean())
        self.assertIsNone(quality["landmark_confidence"])
        self.assertIn("no per-landmark confidence", quality["confidence_note"])
        self.assertTrue(any("Human review required" in item for item in warnings))

    def test_mirror_preserves_screen_side_convention(self) -> None:
        original = polygons_from_landmarks(self.points)
        mirrored = self.points.copy()
        mirrored[:, 0] = 400 - mirrored[:, 0]
        result = polygons_from_landmarks(mirrored)
        self.assertLess(result["left_cheek"][:, 0].mean(), result["right_cheek"][:, 0].mean())
        np.testing.assert_allclose(result["left_cheek"][:, 0], 400 - original["right_cheek"][:, 0])

    def test_outside_and_rounding_past_edge_are_rejected_without_clipping(self) -> None:
        for polygon in (
            np.array([[-0.1, 10], [10, 10], [10, 20], [0, 20]]),
            np.array([[390, 10], [399.8, 10], [399.8, 20], [390, 20]]),
            np.array([[390, 10], [400, 10], [400, 20], [390, 20]]),
        ):
            with self.subTest(polygon=polygon.tolist()):
                masks, errors = rasterize_polygons({"left_cheek": polygon}, (400, 400), 1)
                self.assertNotIn("left_cheek", masks)
                self.assertTrue(any("outside image" in item for item in errors))

    def test_overlapping_masks_are_rejected(self) -> None:
        square = np.array([[20, 20], [40, 20], [40, 40], [20, 40]])
        _, errors = rasterize_polygons({"left_cheek": square, "right_cheek": square + 10}, (80, 80), 10)
        self.assertTrue(any("overlap" in item for item in errors))

    def test_collinear_or_tiny_polygon_is_rejected(self) -> None:
        for polygon in (
            np.array([[10, 10], [11, 10], [12, 10]]),
            np.array([[10, 10], [11, 10], [11, 11], [10, 11]]),
        ):
            _, errors = rasterize_polygons({"forehead": polygon}, (80, 80), 100)
            self.assertTrue(any("too few pixels" in item for item in errors))

    def test_hand_hull_fills_between_landmarks_and_dilates_margin(self) -> None:
        corners = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], dtype=float)
        hand = np.vstack((np.tile(corners, (5, 1)), corners[:1]))
        undilated, hulls = hand_coverage([hand], (80, 80), 0)
        dilated, _ = hand_coverage([hand], (80, 80), 3)
        self.assertEqual(len(hulls), 1)
        self.assertEqual(undilated[30, 30], 1)
        self.assertEqual(undilated[30, 18], 0)
        self.assertEqual(dilated[30, 18], 1)
        self.assertEqual(dilated[30, 15], 0)
        self.assertGreater(np.count_nonzero(dilated), np.count_nonzero(undilated))
        empty, hulls = hand_coverage([], (80, 80), 3)
        self.assertFalse(empty.any())
        self.assertEqual(hulls, [])

    def test_detected_hand_covering_cheek_is_rejected(self) -> None:
        corners = np.array([[85, 205], [165, 205], [165, 275], [85, 275]], dtype=float)
        hand = np.vstack((np.tile(corners, (5, 1)), corners[:1]))
        _, _, quality, errors, _, _ = assess_geometry(self.image, self.points, np.eye(4), [hand], self.config)
        self.assertGreater(quality["hand_overlap_ratios"]["left_cheek"], 0.9)
        self.assertTrue(any("left_cheek: suspected hand occlusion" in item for item in errors))

    def test_pose_limits_reject_each_axis_and_scale_does_not_hide_pose(self) -> None:
        for axis in ("pitch", "yaw", "roll"):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    degrees = sign * (getattr(self.config, f"max_{axis}_degrees") + 1)
                    matrix = rotation_matrix(axis, degrees)
                    matrix[:3, :3] *= 2.5
                    self.assertAlmostEqual(pose_from_matrix(matrix)[axis], degrees)
                    _, _, _, errors, _, _ = assess_geometry(self.image, self.points, matrix, [], self.config)
                    self.assertTrue(any(f"face {axis} too large" in item for item in errors))

    def test_invalid_pose_matrices_are_rejected(self) -> None:
        reflected = np.eye(4)
        reflected[0, 0] = -1
        nonfinite = np.eye(4)
        nonfinite[0, 0] = float("nan")
        for matrix in (np.eye(3), np.zeros((4, 4)), reflected, nonfinite):
            with self.subTest(matrix=matrix.tolist()):
                with self.assertRaises(ValueError):
                    pose_from_matrix(matrix)

    def test_small_face_and_channel_clipping_are_rejected(self) -> None:
        small = self.points.copy()
        small[454] = small[234] + (10, 0)
        _, _, _, errors, _, _ = assess_geometry(self.image, small, np.eye(4), [], self.config)
        self.assertTrue(any("face too small" in item for item in errors))
        clipped = self.image.copy()
        clipped[:, :, 2] = 255
        _, _, quality, errors, _, _ = assess_geometry(clipped, self.points, np.eye(4), [], self.config)
        self.assertTrue(all(ratio == 1 for ratio in quality["clipped_pixel_ratios"].values()))
        self.assertEqual(sum("excessive clipped pixels" in item for item in errors), 3)

    def test_invalid_landmark_arrays_fail_with_value_error(self) -> None:
        nonfinite = self.points.copy()
        nonfinite[100, 0] = float("nan")
        for points in (self.points[:10], np.ones((478, 1)), nonfinite):
            with self.subTest(shape=points.shape):
                with self.assertRaises(ValueError):
                    assess_geometry(self.image, points, np.eye(4), [], self.config)


class ExtractionFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image_path = self.root / "input.png"
        self.assertTrue(cv2.imwrite(str(self.image_path), np.full((400, 400, 3), 128, np.uint8)))
        models = self.root / "models"
        models.mkdir()
        for name in ("face_landmarker.task", "hand_landmarker.task"):
            (models / name).write_bytes(b"fake model only for testing report flow")
        self.extractor = RoiExtractor(models)
        self.extractor.mp = SimpleNamespace(
            __version__="test-fake",
            ImageFormat=SimpleNamespace(SRGB="SRGB"),
            Image=lambda **kwargs: kwargs,
        )
        self.extractor.hand = SimpleNamespace(detect=lambda image: SimpleNamespace(hand_landmarks=[]))

    def fake_faces(self, count: int, matrix: bool = True, malformed: bool = False) -> None:
        points = synthetic_face()
        landmarks = [SimpleNamespace(x=x / 400, y=y / 400, z=0.0) for x, y in points]
        if malformed:
            landmarks = landmarks[:10]
        result = SimpleNamespace(
            face_landmarks=[landmarks] * count,
            facial_transformation_matrixes=[np.eye(4)] if matrix else [],
        )
        self.extractor.face = SimpleNamespace(detect=lambda image: result)

    def assert_failed_without_masks(self, output: Path) -> dict:
        with self.assertRaises(RoiGenerationError):
            self.extractor.extract(self.image_path, output)
        self.assertTrue((output / "roi_overlay.png").is_file())
        self.assertFalse((output / "roi_masks.npz").exists())
        self.assertFalse(list(output.glob("*_mask.png")))
        report = json.loads((output / "roi_points.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["errors"])
        return report

    def test_no_face_and_multiple_faces_fail_without_exporting_masks(self) -> None:
        for count in (0, 2):
            with self.subTest(count=count):
                self.fake_faces(count)
                report = self.assert_failed_without_masks(self.root / f"out_{count}")
                self.assertEqual(report["quality"]["detected_faces"], count)
                self.assertIn("Expected exactly one face", report["errors"][0])

    def test_missing_pose_fails_without_exporting_masks(self) -> None:
        self.fake_faces(1, matrix=False)
        report = self.assert_failed_without_masks(self.root / "out_pose")
        self.assertIn("pose is unavailable", report["errors"][0])

    def test_malformed_landmarks_fail_with_persisted_report(self) -> None:
        self.fake_faces(1, malformed=True)
        self.assert_failed_without_masks(self.root / "out_landmarks")

    def test_empty_or_undecodable_image_fails_with_report_and_no_masks(self) -> None:
        self.fake_faces(1)
        for index, encoded in enumerate((b"", b"not an image")):
            with self.subTest(encoded=encoded):
                self.image_path.write_bytes(encoded)
                output = self.root / f"bad_image_{index}"
                with self.assertRaises(RoiGenerationError):
                    self.extractor.extract(self.image_path, output)
                self.assertFalse((output / "roi_masks.npz").exists())
                report = json.loads((output / "roi_points.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["errors"])

    def test_accepted_geometry_is_never_marked_approved(self) -> None:
        self.fake_faces(1)
        output = self.root / "review"
        report = self.extractor.extract(self.image_path, output)
        self.assertEqual(report["status"], "needs_review")
        self.assertTrue((output / "roi_masks.npz").is_file())
        self.assertIsNone(report["quality"]["landmark_confidence"])


if __name__ == "__main__":
    unittest.main()
