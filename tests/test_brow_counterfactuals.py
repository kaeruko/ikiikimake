from __future__ import annotations

import unittest

import numpy as np

from analysis.generate_brow_counterfactuals import (
    composite_brow_layer, composite_residual, remove_original_brow, translate_array,
)


class BrowCounterfactualTests(unittest.TestCase):
    def test_vertical_translation_does_not_wrap(self):
        values = np.arange(20).reshape(4, 5)
        down = translate_array(values, 1)
        np.testing.assert_array_equal(down[0], np.zeros(5, dtype=values.dtype))
        np.testing.assert_array_equal(down[1:], values[:-1])
        up = translate_array(values, -1)
        np.testing.assert_array_equal(up[:-1], values[1:])
        np.testing.assert_array_equal(up[-1], np.zeros(5, dtype=values.dtype))

    def test_composite_is_bit_identical_outside_shifted_support(self):
        before = np.full((9, 11, 3), 100, np.uint8)
        residual = np.zeros_like(before, dtype=np.float32)
        support = np.zeros(before.shape[:2], bool)
        support[3:5, 4:7] = True
        residual[support] = (-20, 5, 12)
        result, shifted_support, outside_delta = composite_residual(before, residual, support, 2)
        self.assertEqual(outside_delta, 0)
        np.testing.assert_array_equal(result[~shifted_support], before[~shifted_support])
        expected = np.broadcast_to(np.array([80, 105, 112], np.uint8), (2, 3, 3))
        np.testing.assert_array_equal(result[5:7, 4:7], expected)

    def test_composite_clips_without_touching_other_pixels(self):
        before = np.full((5, 5, 3), 250, np.uint8)
        residual = np.zeros_like(before, dtype=np.float32)
        support = np.zeros(before.shape[:2], bool)
        support[2, 2] = True
        residual[2, 2] = (20, -300, 1)
        result, shifted_support, outside_delta = composite_residual(before, residual, support, 0)
        self.assertEqual(result[2, 2].tolist(), [255, 0, 251])
        self.assertEqual(outside_delta, 0)
        np.testing.assert_array_equal(result[~shifted_support], before[~shifted_support])

    def test_layer_composite_erases_old_brow_before_shifting_new_one(self):
        before = np.full((9, 11, 3), 180, np.uint8)
        before[3, 4:7] = 80
        erase = np.zeros(before.shape[:2], bool)
        erase[3, 4:7] = True
        clean = before.copy()
        clean[erase] = 180
        alpha = np.zeros(before.shape[:2], np.float32)
        alpha[3, 4:7] = 1.0
        layer = np.zeros_like(before, np.float32)
        layer[3, 4:7] = 60
        result, support, outside_delta = composite_brow_layer(
            before, clean, erase, layer, alpha, -2)
        self.assertEqual(outside_delta, 0)
        np.testing.assert_array_equal(result[1, 4:7], np.full((3, 3), 60, np.uint8))
        np.testing.assert_array_equal(result[3, 4:7], np.full((3, 3), 180, np.uint8))
        np.testing.assert_array_equal(result[~support], before[~support])

    def test_original_brow_fill_is_scoped_to_mask(self):
        before = np.full((8, 7, 3), 75, np.uint8)
        before[4, 2:5] = 20
        mask = np.zeros(before.shape[:2], bool)
        mask[4, 2:5] = True
        clean, support = remove_original_brow(before, mask, 3)
        self.assertGreater(int(clean[4, 3, 0]), 20)
        np.testing.assert_array_equal(clean[~support], before[~support])


if __name__ == "__main__":
    unittest.main()
