from __future__ import annotations

import unittest

import numpy as np

from analysis.brow_shape_metrics import centerline_metrics, mask_metrics


class BrowShapeMetricTests(unittest.TestCase):
    def test_centerline_reports_apex_fraction_curvature_and_semantic_slope(self):
        curve = np.array([[20, 40], [30, 34], [40, 28], [50, 30], [60, 36]], np.float64)
        values = centerline_metrics(
            curve, eye_inner=np.array([20, 60]), eye_outer=np.array([60, 60]),
            face_up=np.array([0, -1]),
        )
        self.assertEqual(values["apex_control_point"], 3)
        self.assertGreater(values["max_chord_distance_px"], 5)
        self.assertGreater(values["apex_arc_fraction"], 0.4)
        self.assertLess(values["apex_arc_fraction"], 0.6)
        self.assertGreater(values["head_to_tail_slope_degrees"], 0)

    def test_mask_area_and_width_are_alpha_weighted(self):
        alpha = np.zeros((80, 90), np.float64)
        alpha[35:45, 20:61] = 0.5
        curve = np.array([[20, 40], [30, 40], [40, 40], [50, 40], [60, 40]], np.float64)
        values = mask_metrics(
            alpha, curve, eye_inner=np.array([20, 60]), eye_outer=np.array([60, 60]),
            face_width=100,
        )
        self.assertAlmostEqual(values["effective_area_px2"], 205)
        self.assertAlmostEqual(values["average_width_px"], 5.125)
        self.assertEqual(values["binary_area_px2_at_alpha_0_5"], 410)
        self.assertAlmostEqual(values["tail_to_head_thickness_ratio"], 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
