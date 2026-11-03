from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.brow_shape_report import build_report


def side(apex=3, fraction=0.62):
    return {
        "length_eye_widths": 1.5, "effective_area_eye_width2": 0.4,
        "average_width_eye_widths": 0.25, "apex_arc_fraction": fraction,
        "max_chord_distance_eye_widths": 0.15, "head_to_tail_slope_degrees": 5.0,
        "tail_to_head_thickness_ratio": 0.7, "apex_control_point": apex,
        "length_px": 100.0, "effective_area_px2": 1200.0, "average_width_px": 12.0,
        "max_chord_distance_px": 10.0,
    }


class BrowShapeReportTests(unittest.TestCase):
    def test_report_contains_sections_and_relative_artifact_links(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            conditions = []
            for variant in ("baseline", "soft_arch", "flat", "tail_up", "tail_down"):
                image = directory / f"curve_{variant}.png"
                image.write_bytes(b"png")
                left = side(1, 0.0) if variant == "flat" else side()
                right = side(1, 0.0) if variant == "flat" else side()
                differences = {key: {"absolute_difference": 0.0}
                               for key in ("length_eye_widths", "effective_area_eye_width2",
                                           "average_width_eye_widths", "apex_arc_fraction",
                                           "max_chord_distance_eye_widths",
                                           "head_to_tail_slope_degrees",
                                           "tail_to_head_thickness_ratio")}
                conditions.append({
                    "variant": variant, "file": image.name, "changed_pixels": 10,
                    "outside_support_max_abs_delta": 0,
                    "shape_metrics": {"sides": {"screen_left": left, "screen_right": right},
                                      "left_right_differences": differences},
                })
            for name in ("contact_sheet.png", "five_control_points.png", "shape_metrics.csv"):
                (directory / name).write_bytes(b"x")
            (directory / "report.json").write_text(json.dumps({
                "registration": {"median_control_error_px": 1.5}, "conditions": conditions,
            }), encoding="utf-8")
            output = directory / "shape_report.html"
            build_report(directory, output)
            body = output.read_text(encoding="utf-8")
            self.assertIn("眉5点曲線・形状指標レポート", body)
            self.assertIn("指標の定義", body)
            self.assertIn("curve_flat.png", body)
            self.assertIn("shape_metrics.csv", body)


if __name__ == "__main__":
    unittest.main()
