from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from makeup_transfer.benchmark import _held_out_sources, benchmark, psnr
from makeup_transfer.inference import MakeupStyle, _write_image


class MetricsTests(unittest.TestCase):
    def test_psnr_known_error_and_exact_match(self):
        target = np.zeros((2, 2, 3), dtype=np.uint8)
        prediction = np.full_like(target, 10)
        self.assertAlmostEqual(psnr(prediction, target), 20 * np.log10(255 / 10))
        self.assertEqual(psnr(target, target), float("inf"))

    def test_face_roi_excludes_background_error(self):
        target = np.zeros((2, 2, 3), dtype=np.uint8)
        prediction = target.copy()
        prediction[0, 0] = 100
        mask = np.array([[False, True], [True, True]])
        self.assertTrue(np.isfinite(psnr(prediction, target)))
        self.assertEqual(psnr(prediction, target, mask), float("inf"))
        with self.assertRaises(ValueError):
            psnr(prediction, target, np.zeros((2, 2), dtype=bool))


class HoldoutTests(unittest.TestCase):
    def manifest(self):
        return {"sources": [
            {"id": "a", "source": "a.png", "split": "test", "kind": "synthetic"},
            {"id": "b", "source": "b.png", "split": "test", "kind": "synthetic"},
            {"id": "c", "source": "c.png", "split": "train", "kind": "synthetic"},
            {"id": "d", "source": "d.png", "split": "test", "kind": "real_eye"},
        ], "canonical_training_sources": ["c"], "records": []}

    def test_selects_only_distinct_held_out_synthetic_sources(self):
        manifest = self.manifest()
        manifest["sources"].append(manifest["sources"][0].copy())
        self.assertEqual([source["id"] for source in _held_out_sources(manifest, "test")], ["a", "b"])

    def test_train_split_and_overlap_are_rejected(self):
        with self.assertRaises(ValueError):
            _held_out_sources(self.manifest(), "train")
        manifest = self.manifest()
        manifest["canonical_training_sources"].append("a")
        with self.assertRaisesRegex(ValueError, "overlap"):
            _held_out_sources(manifest, "test")
        manifest = self.manifest()
        manifest["records"] = [{"source_id": "b", "split": "train"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            _held_out_sources(manifest, "test")

    def test_single_held_out_source_cannot_be_paired_with_itself(self):
        manifest = self.manifest()
        manifest["sources"] = manifest["sources"][:1]
        with self.assertRaisesRegex(ValueError, "At least two"):
            _held_out_sources(manifest, "test")


class ProtocolTests(unittest.TestCase):
    def test_cross_face_protocol_writes_groundtruth_predictions_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared, output = root / "prepared", root / "results"
            prepared.mkdir()
            sources = []
            for index, level in enumerate((30, 80)):
                path = root / f"source{index}.png"
                _write_image(path, np.full((8, 10, 3), level, dtype=np.uint8))
                sources.append({"source": str(path), "id": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "split": "test", "kind": "synthetic"})
            manifest_path = prepared / "manifest.json"
            manifest_path.write_text(json.dumps({"sources": sources, "records": []}), encoding="utf-8")
            geometry = SimpleNamespace(canvas_size=8, boxes={}, landmarks=np.zeros((3, 2)))
            templates = {region: np.zeros((8, 8, 4), dtype=np.float32) for region in ("eye", "lip", "cheek")}
            learned = MakeupStyle(geometry, templates, {"generator_calls": 3})
            detector = MagicMock()
            detector.__enter__.return_value = detector
            detector.detect.return_value = np.zeros((3, 2))
            render_backgrounds = []

            def render(rgb, style, points):
                render_backgrounds.append(rgb.copy())
                return rgb + 10, np.ones(rgb.shape[:2], dtype=np.float32)

            with patch("makeup_transfer.benchmark._checkpoint_paths", return_value={"eye": root / "eye.pt"}), \
                    patch("makeup_transfer.benchmark._load_checkpoint", return_value={"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}), \
                    patch("makeup_transfer.benchmark._resolve_geometry", return_value=geometry), \
                    patch("makeup_transfer.benchmark.CanonicalGeometry.load", return_value=geometry), \
                    patch("makeup_transfer.benchmark.FaceDetector", return_value=detector), \
                    patch("makeup_transfer.benchmark.generate_style", return_value=templates), \
                    patch("makeup_transfer.benchmark.crop_region", side_effect=lambda image, *_: image), \
                    patch("makeup_transfer.benchmark.render_style", side_effect=render) as renderer, \
                    patch("makeup_transfer.benchmark.extract_style", return_value=learned) as extract, \
                    patch("makeup_transfer.benchmark.face_region_mask", return_value=np.ones((8, 10))), \
                    patch("makeup_transfer.benchmark._perceptual_metrics") as perceptual:
                report = benchmark(prepared, root / "checkpoints", output)
            self.assertEqual(report["evaluated_pairs"], 1)
            self.assertEqual(renderer.call_count, 3)
            extract.assert_called_once()
            perceptual.assert_not_called()
            # Known makeup is rendered on two different originals; learned
            # transfer and ground truth both use the same original target B.
            self.assertFalse(np.array_equal(render_backgrounds[0], render_backgrounds[1]))
            np.testing.assert_array_equal(render_backgrounds[1], render_backgrounds[2])
            self.assertEqual(report["summary"]["psnr_full_db_mean"], "Infinity")
            self.assertNotEqual(report["pairs"][0]["source_a_id"], report["pairs"][0]["source_b_id"])
            for key in ("reference", "target", "ground_truth", "prediction", "review"):
                self.assertTrue(Path(report["pairs"][0][key]).is_file())
            persisted = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(persisted["evaluated_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
