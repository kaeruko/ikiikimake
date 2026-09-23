"""Geometry, source isolation, and synthetic-supervision invariants."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from makeup_transfer.geometry import (REGIONS, EYE_LEFT, EYE_RIGHT, BROW_LEFT, BROW_RIGHT,
    LIP_OUTER, LIP_INNER, CHEEK_LEFT, CHEEK_RIGHT, FACE_OVAL, CanonicalGeometry,
    affine_align, alignment_anchors, build_canonical, face_region_mask, region_mask,
    tps_maps, warp_tps)
from makeup_transfer.prepare import (inventory_images, prepare_dataset, source_split,
                                     verify_artifact_hash, verify_prepared_integrity)
from makeup_transfer.synthesis import alpha_blend, eye_kmeans_pseudo, generate_style


def face_points(size=128):
    points = np.full((478, 2), size * 0.5, np.float32)
    for indices, center, radius, start in (
        (FACE_OVAL, (0.5, 0.5), (0.4, 0.47), -np.pi / 2),
        (EYE_LEFT, (0.34, 0.37), (0.08, 0.025), np.pi),
        (EYE_RIGHT, (0.66, 0.37), (0.08, 0.025), np.pi),
        (BROW_LEFT, (0.34, 0.29), (0.09, 0.016), np.pi),
        (BROW_RIGHT, (0.66, 0.29), (0.09, 0.016), np.pi),
        (LIP_OUTER, (0.5, 0.68), (0.13, 0.055), np.pi),
        (LIP_INNER, (0.5, 0.68), (0.08, 0.019), np.pi),
        (CHEEK_LEFT, (0.26, 0.57), (0.075, 0.075), np.pi),
        (CHEEK_RIGHT, (0.74, 0.57), (0.075, 0.075), np.pi),
    ):
        angles = np.arange(len(indices)) * 2 * np.pi / len(indices) + start
        points[list(indices)] = (np.column_stack((np.cos(angles), np.sin(angles))) * radius + center) * size
    points[1], points[6], points[168] = (size * 0.5, size * 0.54), (size * 0.5, size * 0.45), (size * 0.5, size * 0.38)
    return points


class MakeupGeometryTests(unittest.TestCase):
    def test_tps_identity_translation_and_duplicate_controls(self):
        points = np.array([[0, 0], [63, 0], [0, 63], [63, 63], [31, 31]], np.float32)
        image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        np.testing.assert_allclose(warp_tps(image, points, points, (64, 64)), image, atol=0.01)
        duplicate = np.vstack((points, points[-1]))
        mx, my = tps_maps(duplicate, duplicate + (2, 3), (64, 64), grid_size=None)
        np.testing.assert_allclose(mx[10, 20], 18, atol=1e-4)
        np.testing.assert_allclose(my[10, 20], 7, atol=1e-4)
        gx, gy = tps_maps(points, points, (128, 128), grid_size=32)
        np.testing.assert_allclose(gx[64, 70], 70, atol=0.07)
        np.testing.assert_allclose(gy[64, 70], 64, atol=0.07)

    def test_alignment_is_invariant_to_source_translation_scale(self):
        source = face_points()
        geometry = build_canonical([source], canvas_size=128)
        transformed = source * 1.5 + (11, -8)
        _, aligned, _ = affine_align(np.zeros((210, 210, 3), np.uint8), transformed, geometry)
        np.testing.assert_allclose(aligned[:468], geometry.landmarks, atol=1e-4)
        np.testing.assert_allclose(alignment_anchors(aligned), alignment_anchors(geometry.landmarks), atol=1e-4)

    def test_style_reproducible_bounded_and_excludes_anatomy(self):
        geometry = build_canonical([face_points()], canvas_size=128)
        one = generate_style(geometry, np.random.default_rng(12))
        two = generate_style(geometry, np.random.default_rng(12))
        application_mask = face_region_mask(geometry.landmarks, (128, 128))
        for region in REGIONS:
            np.testing.assert_array_equal(one[region], two[region])
            self.assertEqual(one[region].shape, (128, 128, 4))
            self.assertTrue(np.isfinite(one[region]).all())
            self.assertGreaterEqual(one[region].min(), 0)
            self.assertLessEqual(one[region].max(), 1)
            self.assertGreater(float(one[region][..., 3].sum()), 1)
            self.assertEqual(float(one[region][..., 3][application_mask == 0].sum()), 0)
        blank = np.ones((5, 5, 3), np.float32) * 0.7
        rgba = np.zeros((5, 5, 4), np.float32)
        np.testing.assert_array_equal(alpha_blend(blank, rgba), blank)

    def test_kmeans_uses_lab_cosine_alpha(self):
        rgb = np.empty((32, 32, 3), np.float32)
        rgb[:] = (0.7, 0.5, 0.4)
        rgb[:4] = (0.12, 0.12, 0.5)
        rgb[4:6] = (0.72, 0.52, 0.42)
        mask = np.ones((32, 32), np.float32)
        mask[:, :2] = 0
        label = eye_kmeans_pseudo(rgb, mask, seed=1)
        self.assertEqual(label.shape, (32, 32, 4))
        self.assertEqual(float(label[:, :2, 3].sum()), 0)
        self.assertGreater(float(label[:4, 2:, 3].mean()), float(label[8:, 2:, 3].mean()))
        np.testing.assert_array_equal(label[..., :3], rgb)

    def test_cheek_alpha_fades_inside_polygon_boundary(self):
        geometry = build_canonical([face_points()], canvas_size=256)
        mask = region_mask(geometry, "cheek")
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        boundary = (distance > 0) & (distance <= 1.1)
        interior = distance > 6
        for seed in (3, 12, 42):
            alpha = generate_style(geometry, np.random.default_rng(seed))["cheek"][..., 3]
            self.assertTrue(boundary.any() and interior.any())
            self.assertLess(float(alpha[boundary].max()), 0.2 * float(alpha.max()))
            self.assertGreater(float(alpha[interior].mean()), 4 * float(alpha[boundary].mean()))
            self.assertEqual(float(alpha[mask == 0].sum()), 0.0)


class MakeupDataTests(unittest.TestCase):
    def test_changed_synthesis_changes_artifact_fingerprint_and_detects_tampering(self):
        class Detector:
            def __init__(self, *_):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def detect(self, rgb):
                return face_points(64)

        def faded_style(geometry, rng):
            styles = generate_style(geometry, rng)
            for rgba in styles.values():
                rgba[..., 3] *= 0.5
            return styles

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            cv2.imencode(".png", np.full((64, 64, 3), 128, np.uint8))[1].tofile(source / "face.png")
            arguments = dict(variants=1, image_size=32, canvas_size=64, model_path=root / "mock_missing.task")
            with patch("makeup_transfer.prepare.FaceDetector", Detector):
                original = prepare_dataset(source, root / "original", **arguments)
                with patch("makeup_transfer.prepare.generate_style", faded_style):
                    faded = prepare_dataset(source, root / "faded", **arguments)
            self.assertEqual(original["provenance"], faded["provenance"])
            self.assertIsNone(original["provenance"]["landmark_model_sha256"])
            self.assertEqual(set(original["provenance"]["code_sha256"]), {"geometry.py", "synthesis.py", "prepare.py"})
            self.assertEqual(original["provenance"]["versions"]["numpy"], np.__version__)
            self.assertNotEqual(original["dataset_fingerprint"], faded["dataset_fingerprint"])
            self.assertNotEqual(original["records"][0]["sha256"], faded["records"][0]["sha256"])
            original_bytes = (root / "original" / "manifest.json").read_bytes()
            faded_bytes = (root / "faded" / "manifest.json").read_bytes()
            self.assertNotEqual(original_bytes, faded_bytes)
            self.assertEqual(verify_prepared_integrity(root / "original")["checked"], 7)
            self.assertEqual(verify_prepared_integrity(root / "faded")["without_hash"], 0)
            try:
                from makeup_transfer.data import MakeupDataset
            except ImportError:
                return
            with patch("makeup_transfer.data.verify_artifact_hash", wraps=verify_artifact_hash) as check:
                dataset = MakeupDataset(root / "original", "eye")
                dataset[0]
                dataset[0]
                # Geometry/prior verified once at load; sample once on first use.
                self.assertEqual(check.call_count, 3)
            sample_path = root / "original" / dataset.records[0]["path"]
            with sample_path.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                dataset[0]
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_prepared_integrity(root / "original")
            # An old manifest without any hash fields remains readable.
            legacy = dict(faded)
            legacy.pop("geometry_sha256")
            legacy.pop("average_alpha_sha256")
            legacy.pop("dataset_fingerprint")
            for record in legacy["records"]:
                record.pop("sha256")
            (root / "faded" / "manifest.json").write_text(json.dumps(legacy), encoding="utf-8")
            self.assertEqual(verify_prepared_integrity(root / "faded")["without_hash"], 7)
            self.assertEqual(tuple(MakeupDataset(root / "faded", "eye")[0]["input"].shape), (3, 32, 32))

    def test_inventory_excludes_documentation_and_split_order_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("ffhq-dataset/docs", "ffhq-dataset/senior_female_50plus", "fairface/50-59"):
                (root / directory).mkdir(parents=True)
            for name in ("ffhq-dataset/docs/example.png", "ffhq-dataset/ffhq-teaser.png",
                         "ffhq-dataset/senior_female_50plus/00001.png", "fairface/50-59/person.jpg"):
                (root / name).write_bytes(b"placeholder")
            images = inventory_images(root)
            self.assertEqual(len(images), 2)
            self.assertEqual(len(inventory_images(root, sources=["ffhq"])), 1)
        ids = [str(i) for i in range(20)]
        self.assertEqual(source_split(ids, seed=8), source_split(ids[::-1], seed=8))
        self.assertEqual(set(source_split(ids, seed=8).values()), {"train", "val", "test"})

    def test_prepare_has_no_source_or_prior_leakage(self):
        class Detector:
            def __init__(self, *_):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def detect(self, rgb):
                points = face_points(64)
                # Only cheek geometry differs, leaving affine anchors stable.
                points[list(CHEEK_LEFT), 0] += float(rgb[0, 0, 0]) / 30
                return points

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output, real = root / "source", root / "prepared", root / "real"
            source.mkdir()
            real.mkdir()
            expected_points = {}
            for index in range(5):
                image = np.full((64, 64, 3), (70 + index * 25, 100, 130), np.uint8)
                path = source / f"face{index}.png"
                cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tofile(path)
                identity = hashlib.sha256(path.read_bytes()).hexdigest()
                expected_points[identity] = Detector().detect(image)
            (source / "duplicate.png").write_bytes((source / "face0.png").read_bytes())
            real_image = np.full((64, 64, 3), (180, 120, 70), np.uint8)
            cv2.imencode(".png", real_image)[1].tofile(real / "real_makeup.png")
            real_image_two = np.full((64, 64, 3), (90, 180, 120), np.uint8)
            cv2.imencode(".png", real_image_two)[1].tofile(real / "real_makeup_two.png")
            with patch("makeup_transfer.prepare.FaceDetector", Detector):
                manifest = prepare_dataset(source, output, variants=2, image_size=32, canvas_size=64,
                                           real_makeup_dir=real, max_real_makeup=1, real_preview_count=2)
            self.assertEqual(len(manifest["sources"]), 6)
            self.assertEqual(len(manifest["duplicates"]), 1)
            self.assertEqual(len(manifest["records"]), 5 * 2 * 3 + 1)
            real_records = [r for r in manifest["records"] if r["kind"] == "real_eye"]
            self.assertEqual(len(real_records), 1)
            real_record = real_records[0]
            self.assertEqual(real_record["region"], "eye")
            with np.load(output / real_record["path"]) as real_sample:
                self.assertEqual(float(real_sample["has_alpha"]), 0)
            self.assertEqual(manifest["summary"]["detected_real_makeup_sources"], 1)
            self.assertEqual(manifest["summary"]["records_by_kind"]["real_eye"], 1)
            self.assertEqual(manifest["summary"]["real_eye_previews"], 1)
            previews = list((output / "real_eye_previews").glob("*.png"))
            self.assertEqual(len(previews), 1)
            preview = cv2.imdecode(np.fromfile(previews[0], dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(preview.shape[:2], (64, 192))
            for identity in expected_points:
                splits = {r["split"] for r in manifest["records"] if r["source_id"] == identity}
                self.assertEqual(len(splits), 1)
            training_ids = [s["id"] for s in manifest["sources"] if s["split"] == "train" and s["kind"] == "synthetic"]
            expected = build_canonical([expected_points[key] for key in training_ids], canvas_size=64)
            actual = CanonicalGeometry.load(output / "geometry.npz")
            np.testing.assert_allclose(actual.landmarks, expected.landmarks, atol=1e-4)
            for region in REGIONS:
                train_records = [r for r in manifest["records"] if r["region"] == region and r["split"] == "train" and r["kind"] == "synthetic"]
                alphas = []
                for record in train_records:
                    with np.load(output / record["path"]) as sample:
                        self.assertEqual(sample["input"].shape, (32, 32, 3))
                        self.assertEqual(sample["mask"].shape, (32, 32, 1))
                        self.assertEqual(float(sample["has_alpha"]), 1)
                        alphas.append(sample["target"][..., 3:4].astype(np.float32))
                np.testing.assert_allclose(np.load(output / manifest["average_alpha"][region]), np.mean(alphas, axis=0), atol=0.0003)
            saved_manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(saved_manifest["schema_version"], 1)
            self.assertEqual(saved_manifest["summary"]["records"], len(manifest["records"]))
            try:
                from makeup_transfer.data import MakeupDataset
            except ImportError:
                return
            dataset = MakeupDataset(output, "lip", split="train")
            sample = dataset[0]
            self.assertEqual(tuple(sample["input"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["target"].shape), (4, 32, 32))
            self.assertEqual(tuple(sample["average_alpha"].shape), (1, 32, 32))


if __name__ == "__main__":
    unittest.main()
