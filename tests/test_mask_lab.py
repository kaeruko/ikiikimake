from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from analysis.analyze_cheek_lab import (
    A_THRESHOLDS,
    ROI_NAMES,
    analyze_pair,
    load_masks,
    summarize,
    to_lab_channels,
)


class MaskLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shape = (24, 32, 3)
        self.masks = {name: np.zeros(self.shape[:2], dtype=np.uint8) for name in ROI_NAMES}
        self.masks["left_cheek"][12:19, 3:10] = 1
        self.masks["right_cheek"][12:19, 22:29] = 1
        self.masks["forehead"][3:8, 12:20] = 1

    def save_masks(self, masks: dict | None = None, name: str = "masks.npz") -> Path:
        path = self.root / name
        np.savez_compressed(path, **(self.masks if masks is None else masks))
        return path

    def save_image(self, name: str, image: np.ndarray) -> Path:
        path = self.root / name
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        encoded.tofile(path)
        return path

    def test_outside_mask_pixels_cannot_change_statistics(self) -> None:
        rng = np.random.default_rng(482)
        original = rng.integers(0, 256, size=self.shape, dtype=np.uint8)
        union = np.logical_or.reduce([mask.astype(bool) for mask in self.masks.values()])
        changed = original.copy()
        changed[~union] = 255 - changed[~union]
        for name, mask in self.masks.items():
            selected = mask.astype(bool)
            original_row = summarize("before", name, to_lab_channels(original), selected)
            changed_row = summarize("before", name, to_lab_channels(changed), selected)
            self.assertEqual(original_row, changed_row)
            self.assertEqual(original_row["pixels"], int(mask.sum()))

    def test_statistics_match_existing_lab_conversion_and_thresholds(self) -> None:
        image = np.zeros(self.shape, dtype=np.uint8)
        image[:, :16] = (20, 60, 240)
        image[:, 16:] = (130, 160, 190)
        mask = np.ones(self.shape[:2], dtype=bool)
        row = summarize("before", "left_cheek", to_lab_channels(image), mask)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_star = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
        a_star = lab[:, :, 1].astype(np.float32) - 128.0
        b_star = lab[:, :, 2].astype(np.float32) - 128.0
        self.assertEqual(row["L_mean"], float(l_star.mean()))
        self.assertEqual(row["a_median"], float(np.median(a_star)))
        self.assertEqual(row["b_std"], float(b_star.std()))
        self.assertEqual(row["a_p95"], float(np.percentile(a_star, 95)))
        for threshold in A_THRESHOLDS:
            self.assertEqual(row[f"a_ratio_ge_{int(threshold)}"], float(np.mean(a_star >= threshold)))

    def test_accept_bool_and_both_uint8_binary_encodings(self) -> None:
        for dtype, multiplier in ((bool, 1), (np.uint8, 1), (np.uint8, 255)):
            with self.subTest(dtype=dtype, multiplier=multiplier):
                encoded = {name: (mask * multiplier).astype(dtype) for name, mask in self.masks.items()}
                loaded = load_masks(self.save_masks(encoded), self.shape)
                for name in ROI_NAMES:
                    np.testing.assert_array_equal(loaded[name], self.masks[name].astype(bool))

    def test_reject_invalid_masks(self) -> None:
        cases = []
        missing = {name: mask.copy() for name, mask in self.masks.items() if name != "forehead"}
        cases.append(("missing", missing, "Missing ROI"))
        for case, replacement, message in (
            ("empty", np.zeros(self.shape[:2], dtype=np.uint8), "empty"),
            ("wrong_shape", np.ones((3, 4), dtype=np.uint8), "shape"),
            ("three_dimensional", np.ones((*self.shape[:2], 1), dtype=np.uint8), "shape"),
            ("non_binary", self.masks["left_cheek"] * 2, "non-binary"),
            ("float", self.masks["left_cheek"].astype(float), "bool or uint8"),
            ("overlap", self.masks["right_cheek"].copy(), "overlap"),
        ):
            masks = {name: mask.copy() for name, mask in self.masks.items()}
            masks["left_cheek"] = replacement
            cases.append((case, masks, message))
        mixed = {name: mask.copy() for name, mask in self.masks.items()}
        mixed["left_cheek"][12, 3] = 255
        cases.append(("mixed_encoding", mixed, "non-binary"))
        for case, masks, message in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, message):
                    load_masks(self.save_masks(masks), self.shape)

    def test_invalid_pair_does_not_create_output(self) -> None:
        image = np.full(self.shape, 128, dtype=np.uint8)
        before = self.save_image("before.png", image)
        after = self.save_image("after.png", image)
        valid = self.save_masks(name="valid.npz")
        invalid = dict(self.masks)
        invalid["forehead"] = np.zeros(self.shape[:2], dtype=np.uint8)
        invalid_path = self.save_masks(invalid, "invalid.npz")
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "empty"):
            analyze_pair(before, after, valid, invalid_path, output)
        self.assertFalse(output.exists())

    def test_pair_outputs_and_control_subtraction_with_constant_regions(self) -> None:
        before = np.full(self.shape, (100, 140, 180), dtype=np.uint8)
        # Different image dimensions must not require pixel alignment.
        after = np.full((28, 36, 3), (110, 150, 190), dtype=np.uint8)
        after_masks = {}
        for name, mask in self.masks.items():
            bigger = np.zeros(after.shape[:2], dtype=np.uint8)
            bigger[2:26, 2:34] = mask
            after_masks[name] = bigger
        after[after_masks["left_cheek"].astype(bool)] = (100, 140, 210)
        before_path = self.save_image("前.png", before)
        after_path = self.save_image("後.png", after)
        before_masks_path = self.save_masks(name="before.npz")
        after_masks_path = self.save_masks(after_masks, "after.npz")
        output = self.root / "output"
        result = analyze_pair(before_path, after_path, before_masks_path, after_masks_path, output)
        self.assertEqual(len(result["statistics"]), 6)
        self.assertEqual(len(result["deltas"]), 12)
        self.assertFalse(result["pixel_correspondence"])
        self.assertEqual(result["control_correction"]["status"], "unvalidated_forehead_control")
        rows = {(row["phase"], row["side"]): row for row in result["statistics"]}
        for delta in result["deltas"]:
            metric, side = delta["metric"], delta["side"]
            expected_raw = rows["after", side][metric] - rows["before", side][metric]
            expected_forehead = rows["after", "forehead"][metric] - rows["before", "forehead"][metric]
            self.assertEqual(delta["delta"], expected_raw)
            self.assertEqual(delta["delta_minus_forehead"], expected_raw - expected_forehead)
        for filename in ("lab_stats.csv", "lab_deltas.csv", "analysis_summary.json", "roi_samples.png"):
            self.assertGreater((output / filename).stat().st_size, 0)
        for name in ROI_NAMES:
            self.assertIsNotNone(cv2.imread(str(output / f"{name}_lab_hist.png")))
        with (output / "analysis_summary.json").open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), result)
        self.assertFalse(list(output.glob("*delta*heatmap*")))


if __name__ == "__main__":
    unittest.main()
