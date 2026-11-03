from __future__ import annotations

import unittest

import numpy as np

from analysis.generate_brow_curve_counterfactuals import (
    evaluate_tps,
    fit_tps_displacement,
    five_point_centerline,
    variant_destination,
)


class BrowCurveCounterfactualTests(unittest.TestCase):
    def test_five_point_centerline_orders_nearest_face_centre_as_head(self):
        points = np.zeros((20, 2), np.float64)
        indices = tuple(range(10))
        xs = np.repeat([10, 20, 30, 40, 50], 2)
        points[:10] = np.column_stack((xs, np.tile([8, 12], 5)))
        centres = five_point_centerline(points, indices, face_center_x=80)
        np.testing.assert_allclose(centres[:, 0], [50, 40, 30, 20, 10])
        np.testing.assert_allclose(centres[:, 1], 10)

    def test_tps_reproduces_control_displacements(self):
        controls = np.array([
            [0, 0], [10, 0], [0, 10], [10, 10], [5, 4], [3, 7],
        ], np.float64)
        displacement = np.array([
            [0, 0], [0, 0], [0, 0], [0, 0], [1.5, -2], [-1, 0.5],
        ], np.float64)
        model = fit_tps_displacement(controls, displacement, smoothing=1e-9)
        np.testing.assert_allclose(evaluate_tps(model, controls), displacement, atol=1e-5)

    def test_variants_keep_head_fixed_and_move_tail_in_expected_direction(self):
        source = np.column_stack((np.arange(5) * 10.0, [10, 7, 5, 6, 9]))
        for variant in ("soft_arch", "flat", "tail_up", "tail_down"):
            destination = variant_destination(source, variant, amplitude=8)
            np.testing.assert_allclose(destination[0], source[0])
        self.assertLess(variant_destination(source, "tail_up", 8)[-1, 1], source[-1, 1])
        self.assertGreater(variant_destination(source, "tail_down", 8)[-1, 1], source[-1, 1])
        self.assertLess(variant_destination(source, "soft_arch", 8)[3, 1], source[3, 1])


if __name__ == "__main__":
    unittest.main()
