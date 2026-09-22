from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from makeup_transfer.geometry import (
    BROW_LEFT, BROW_RIGHT, CHEEK_LEFT, CHEEK_RIGHT, EYE_LEFT, EYE_RIGHT,
    FACE_OVAL, LIP_INNER, LIP_OUTER, CanonicalGeometry, face_region_mask,
)
from makeup_transfer.inference import (
    MakeupStyle, TemporalLandmarkSmoother, alpha_composite, extract_style,
    render_style, run_video,
)


def face_points():
    points = np.full((478, 2), 128.0, dtype=np.float32)
    for indices, cx, cy, rx, ry in (
        (FACE_OVAL, 128, 125, 103, 119),
        (EYE_LEFT, 88, 90, 20, 8), (EYE_RIGHT, 168, 90, 20, 8),
        (BROW_LEFT, 88, 66, 23, 4), (BROW_RIGHT, 168, 66, 23, 4),
        (LIP_OUTER, 128, 181, 27, 14), (LIP_INNER, 128, 181, 16, 5),
        (CHEEK_LEFT, 73, 140, 22, 17), (CHEEK_RIGHT, 183, 140, 22, 17),
    ):
        angles = np.linspace(0, 2 * np.pi, len(indices), endpoint=False)
        points[list(indices)] = np.column_stack((cx + rx * np.cos(angles), cy + ry * np.sin(angles)))
    points[1], points[6], points[168] = (128, 127), (128, 109), (128, 99)
    return points


class AlphaCompositionTests(unittest.TestCase):
    def test_known_half_alpha_and_exact_zero_alpha(self):
        background = np.array([[[0, 0, 0], [37, 83, 151]]], dtype=np.uint8)
        rgba = np.array([[[1, 0, 0, 0.5], [0, 1, 1, 0]]], dtype=np.float32)
        result = alpha_composite(background, rgba)
        np.testing.assert_array_equal(result[0, 0], [128, 0, 0])
        np.testing.assert_array_equal(result[0, 1], background[0, 1])

    def test_mask_and_strength_are_multiplicative(self):
        background = np.zeros((2, 2, 3), dtype=np.uint8)
        rgba = np.ones((2, 2, 4), dtype=np.float32)
        visibility = np.array([[1, 0], [0.5, 1]], dtype=np.float32)
        result = alpha_composite(background, rgba, strength=0.5, mask=visibility)
        np.testing.assert_array_equal(result[..., 0], [[128, 0], [64, 128]])
        np.testing.assert_array_equal(alpha_composite(background, rgba, strength=0), background)

    def test_invalid_strength_or_mask_rejected(self):
        background = np.zeros((2, 2, 3), dtype=np.uint8)
        rgba = np.ones((2, 2, 4), dtype=np.float32)
        for strength in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                alpha_composite(background, rgba, strength=strength)
        with self.assertRaises(ValueError):
            alpha_composite(background, rgba, mask=np.ones((3, 3)))


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.points = face_points()
        self.geometry = CanonicalGeometry(self.points, 256, {p: (0, 0, 256, 256) for p in ("eye", "lip", "cheek")})
        rgba = np.zeros((256, 256, 4), dtype=np.float32)
        rgba[..., 0] = 1
        rgba[..., 3] = 0.5
        self.style = MakeupStyle(self.geometry, {"cheek": rgba})
        self.background = np.full((256, 256, 3), 70, dtype=np.uint8)

    def test_protected_features_and_nonface_pixels_unchanged(self):
        output, alpha = render_style(self.background, self.style, self.points)
        protected = face_region_mask(self.points, self.background.shape[:2]) == 0
        np.testing.assert_array_equal(output[protected], self.background[protected])
        self.assertTrue(np.all(alpha[protected] == 0))
        for y, x in ((90, 88), (90, 168), (66, 88), (66, 168), (181, 128), (0, 0)):
            np.testing.assert_array_equal(output[y, x], self.background[y, x])
        self.assertGreater(int(output[140, 73, 0]), int(self.background[140, 73, 0]))

    def test_original_resolution_and_semantic_visibility(self):
        background = np.full((128, 128, 3), 70, dtype=np.uint8)
        visibility = np.ones((128, 128), dtype=np.float32)
        visibility[:, :64] = 0
        output, alpha = render_style(background, self.style, self.points / 2, semantic_mask=visibility)
        self.assertEqual(output.shape, background.shape)
        self.assertEqual(alpha.shape, background.shape[:2])
        np.testing.assert_array_equal(output[:, :64], background[:, :64])
        self.assertTrue(np.all(alpha[:, :64] == 0))
        self.assertTrue(np.any(output[:, 64:] != background[:, 64:]))

    def test_pixels_outside_style_alpha_unchanged(self):
        patch_rgba = self.style.patches["cheek"].copy()
        patch_rgba[..., 3] = 0
        patch_rgba[127:151, 61:85, 3] = 0.5
        style = MakeupStyle(self.geometry, {"cheek": patch_rgba})
        output, alpha = render_style(self.background, style, self.points)
        np.testing.assert_array_equal(output[alpha == 0], self.background[alpha == 0])
        self.assertGreater(np.count_nonzero(alpha), 0)
        self.assertLess(np.count_nonzero(alpha), 1000)

    def test_resized_crop_reconstructs_original_pixel_coordinates(self):
        # A coordinate ramp makes a crop/resize half-pixel shift observable in
        # RGB values after rendering it back on the identical face geometry.
        yy, xx = np.mgrid[:256, :256]
        rgba = np.stack((xx / 255, yy / 255, np.zeros_like(xx), np.ones_like(xx)), axis=2).astype(np.float32)
        geometry = CanonicalGeometry(self.points, 256, {"cheek": (40, 100, 120, 180)})
        crop = cv2.resize(rgba[100:180, 40:120], (10, 10), interpolation=cv2.INTER_LINEAR)
        output, _ = render_style(self.background, MakeupStyle(geometry, {"cheek": crop}), self.points)
        np.testing.assert_allclose(output[140, 70], [70, 140, 0], atol=1)

    def test_missing_weights_raise_before_detection_or_torch_import(self):
        detector = MagicMock()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "Trained generator checkpoint"):
                extract_style(self.background, folder, self.geometry, detector)
        detector.detect.assert_not_called()


class TemporalTests(unittest.TestCase):
    def test_smoothing_and_reset_on_lost_face(self):
        smoother = TemporalLandmarkSmoother(0.75)
        first = np.array([[0, 0], [2, 2]], dtype=np.float32)
        second = first + 8
        np.testing.assert_allclose(smoother.update(first), first)
        np.testing.assert_allclose(smoother.update(second), first + 2)
        self.assertIsNone(smoother.update(None))
        np.testing.assert_array_equal(smoother.update(second), second)

    def test_video_reuses_style_and_passes_failed_frames_through(self):
        frames = [np.full((8, 10, 3), value, dtype=np.uint8) for value in (20, 40, 60)]
        points = np.array([[1, 1], [2, 2], [3, 1]], dtype=np.float32)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25.0
        capture.read.side_effect = [(True, frame.copy()) for frame in frames] + [(False, None)]
        writer = MagicMock()
        writer.isOpened.return_value = True
        detector = MagicMock()
        detector.detect.side_effect = [points, None, points + 10]
        detector.__enter__.return_value = detector
        geometry = MagicMock()
        style = MakeupStyle(geometry, {}, {"generator_calls": 3})
        rendered_points = []

        def render(rgb, cached_style, landmarks, **kwargs):
            self.assertIs(cached_style, style)
            rendered_points.append(landmarks.copy())
            return rgb + 1, np.zeros(rgb.shape[:2], dtype=np.float32)

        with tempfile.TemporaryDirectory() as folder, \
                patch("makeup_transfer.inference._resolve_geometry", return_value=geometry), \
                patch("makeup_transfer.inference._read_rgb", return_value=frames[0]), \
                patch("makeup_transfer.inference.cv2.VideoCapture", return_value=capture), \
                patch("makeup_transfer.inference.cv2.VideoWriter", return_value=writer), \
                patch("makeup_transfer.inference.FaceDetector", return_value=detector), \
                patch("makeup_transfer.inference.extract_style", return_value=style) as extract, \
                patch("makeup_transfer.inference.render_style", side_effect=render) as render_call:
            output = Path(folder) / "result.mp4"
            metadata = run_video("reference.png", "target.mp4", output, "checkpoints")
            extract.assert_called_once()
            self.assertEqual(render_call.call_count, 2)
            self.assertEqual(writer.write.call_count, 3)
            np.testing.assert_array_equal(writer.write.call_args_list[1].args[0], frames[1])
            np.testing.assert_array_equal(rendered_points[1], points + 10)
            self.assertEqual(metadata["style_extractions"], 1)
            self.assertEqual(metadata["frames_passthrough_no_face"], 1)
            self.assertEqual(metadata["frames_with_face"], 2)
            self.assertFalse(metadata["audio_preserved"])
            self.assertGreater(metadata["processing_fps"], 0)
            saved = json.loads(Path(metadata["metadata_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["frames"], 3)
        capture.release.assert_called_once()
        writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
