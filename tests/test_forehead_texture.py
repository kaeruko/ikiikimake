"""Behavioral checks of the exploratory contrast measurement, not validation on people."""
import unittest

import numpy as np

from analysis.analyze_forehead_texture import line_response


class ForeheadTextureTests(unittest.TestCase):
    def test_uniform_skin_has_no_lines(self):
        _, response = line_response(np.full((80, 120), 75., np.float32))
        self.assertLess(float(response.max()), 1e-4)

    def test_horizontal_line_depth_increases_response(self):
        weak = np.full((80, 120), 75., np.float32)
        strong = weak.copy()
        weak[39:41, 10:110] -= 2
        strong[39:41, 10:110] -= 6
        _, a = line_response(weak)
        _, b = line_response(strong)
        self.assertGreater(float(b[39:41,20:100].mean()), float(a[39:41,20:100].mean()) * 2)

    def test_vertical_hair_is_suppressed_relative_to_horizontal_line(self):
        horizontal = np.full((100,100),75.,np.float32)
        horizontal[49:51,:] -= 6
        _, a = line_response(horizontal)
        _, b = line_response(horizontal.T.copy())
        self.assertGreater(float(a[10:-10,10:-10].mean()), .01)
        self.assertLess(float(b[10:-10,10:-10].max()), 1e-4)

    def test_relative_response_invariant_to_multiplicative_L_change(self):
        # This checks the formula, NOT invariance to illumination in real photos.
        im = np.full((80,120),60.,np.float32)
        im[39:41,:] -= 4
        _, a = line_response(im)
        _, b = line_response(im*1.2)
        np.testing.assert_allclose(a,b,atol=1e-4)


if __name__ == '__main__':
    unittest.main()
