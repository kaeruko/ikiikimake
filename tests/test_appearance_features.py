from __future__ import annotations

from copy import deepcopy
import json
import unittest

import cv2
import numpy as np

from analysis.appearance_features import (
    BASE_NAMES, BROW_INDICES, FACE_OVAL_INDICES, INNER_LIP_INDICES,
    LOWER_EYE_INDICES, OUTER_LIP_INDICES, UPPER_EYE_INDICES,
    build_feature_masks, measure_features,
)


def synthetic_landmarks(scale=1.0):
    points = np.full((478, 2), (200.0, 200.0))
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, len(FACE_OVAL_INDICES), endpoint=False)
    points[list(FACE_OVAL_INDICES)] = np.column_stack((200 + 150 * np.cos(angles), 200 + 185 * np.sin(angles)))
    # The face-oval topology places these width anchors on opposite sides.
    points[234], points[454] = (50, 200), (350, 200)
    for i, center_x in enumerate((130, 270)):
        upper_x = np.linspace(center_x - 32, center_x + 32, 5)
        upper_y = np.array([111, 106, 105, 107, 112])
        points[list(BROW_INDICES[i])] = np.vstack((np.column_stack((upper_x, upper_y)),
                                                 np.column_stack((upper_x[::-1], upper_y[::-1] + 9))))
        eye_x = np.linspace(center_x - 30, center_x + 30, 9)
        curve = np.sin(np.linspace(0, np.pi, 9))
        points[list(UPPER_EYE_INDICES[i])] = np.column_stack((eye_x, 158 - 9 * curve))
        points[list(LOWER_EYE_INDICES[i])] = np.column_stack((eye_x, 158 + 7 * curve))
    # Lip topology traverses the bottom from left to right, then the top back.
    for indices, rx, ry in ((OUTER_LIP_INDICES, 52, 22), (INNER_LIP_INDICES, 32, 8)):
        angles = np.linspace(np.pi, -np.pi, len(indices), endpoint=False)
        points[list(indices)] = np.column_stack((200 + rx * np.cos(angles), 288 + ry * np.sin(angles)))
    return points * scale


def fixture(scale=1.0):
    size = round(400 * scale)
    shape = (size, size, 3)
    base = {name: np.zeros(shape[:2], bool) for name in BASE_NAMES}
    for name, (x0, y0, x1, y1) in {
        "left_cheek": (90, 200, 155, 250), "right_cheek": (245, 200, 310, 250),
        "forehead": (160, 48, 240, 80),
    }.items():
        base[name][round(y0 * scale):round(y1 * scale), round(x0 * scale):round(x1 * scale)] = True
    points = synthetic_landmarks(scale)
    masks = build_feature_masks(shape, points, base)
    image = np.full(shape, (120, 150, 180), np.uint8)
    return image, points, masks, base


def rows_by_id(image, masks):
    return {row["id"]: row for row in measure_features(image, masks)}


class AppearanceFeatureTests(unittest.TestCase):
    def test_constant_color_produces_zero_contrasts_dispersion_and_highlights(self):
        image, _, masks, _ = fixture()
        rows = measure_features(image, masks)
        self.assertEqual(len(rows), 23)
        self.assertEqual(len({row["id"] for row in rows}), 23)
        for row in rows:
            with self.subTest(feature=row["id"]):
                self.assertEqual(row["status"], "ok")
                if not row["id"].endswith("_median"):
                    self.assertAlmostEqual(row["value"], 0)
        json.dumps(rows, ensure_ascii=False, allow_nan=False)

    def test_float_lab_conversion_matches_documented_quality_medians(self):
        image, _, masks, _ = fixture()
        expected = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[0, 0]
        rows = rows_by_id(image, masks)
        for index, channel in enumerate(("L", "a", "b")):
            self.assertAlmostEqual(rows[f"left_cheek_{channel}_median"]["value"], float(expected[index]))

    def test_mouth_teeth_and_oral_cavity_cannot_change_lip_features(self):
        image, points, masks, _ = fixture()
        mouth = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mouth, [np.rint(points[list(INNER_LIP_INDICES)]).astype(np.int32)], 1)
        self.assertFalse(np.any(masks["lips"] & mouth.astype(bool)))
        baseline = rows_by_id(image, masks)
        for color in ((255, 255, 255), (0, 0, 0), (0, 0, 255)):
            changed = image.copy()
            changed[mouth.astype(bool)] = color
            rows = rows_by_id(changed, masks)
            self.assertEqual(rows["lips_relative_a"], baseline["lips_relative_a"])
            self.assertEqual(rows["lips_skin_delta_e76"], baseline["lips_skin_delta_e76"])

    def test_reference_color_changes_relative_value_but_outside_pixels_do_not(self):
        image, _, masks, _ = fixture()
        baseline = rows_by_id(image, masks)
        reference_changed = image.copy()
        reference_changed[masks["screen_left_brow_skin"]] = (220, 220, 220)
        changed_rows = rows_by_id(reference_changed, masks)
        key = "screen_left_brow_relative_L_contrast"
        self.assertGreater(changed_rows[key]["value"], baseline[key]["value"])
        union = np.logical_or.reduce(list(masks.values()))
        outside_changed = image.copy()
        outside_changed[~union] = (255, 0, 255)
        self.assertEqual(measure_features(image, masks), measure_features(outside_changed, masks))

    def test_small_target_or_reference_returns_null_without_upsampling(self):
        image, _, masks, _ = fixture()
        tiny = deepcopy(masks)
        tiny["screen_left_brow"][:] = False
        tiny["screen_left_brow"][110, 130] = True
        tiny["screen_right_upper_lid_skin"][:] = False
        tiny["screen_right_upper_lid_skin"][125, 270] = True
        tiny["lips"][:] = False
        tiny["left_cheek"][:] = False
        tiny["left_cheek"][210, 110] = True
        rows = rows_by_id(image, tiny)
        for name in ("screen_left_brow_relative_L_contrast", "screen_right_upper_lid_relative_L_contrast",
                     "lips_relative_a", "lips_skin_delta_e76", "left_cheek_L_median", "left_cheek_ab_mad"):
            with self.subTest(name=name):
                self.assertIsNone(rows[name]["value"])
                self.assertEqual(rows[name]["status"], "insufficient_pixels")
        self.assertEqual(rows["screen_left_brow_relative_L_contrast"]["pixels"], 1)

    def test_lid_and_reference_masks_exclude_eye_and_other_targets(self):
        image, points, masks, _ = fixture()
        targets = masks["lips"].copy()
        for side in ("screen_left", "screen_right"):
            targets |= masks[f"{side}_brow"] | masks[f"{side}_upper_lid"]
        for name, mask in masks.items():
            if name.endswith("_skin"):
                self.assertFalse(np.any(mask & targets), name)
        for i, side in enumerate(("screen_left", "screen_right")):
            eye = np.zeros(image.shape[:2], np.uint8)
            polygon = np.vstack((points[list(UPPER_EYE_INDICES[i])], points[list(LOWER_EYE_INDICES[i])][-2:0:-1]))
            cv2.fillPoly(eye, [np.rint(polygon).astype(np.int32)], 1)
            self.assertFalse(np.any(masks[f"{side}_upper_lid"] & eye.astype(bool)))

    def test_reference_target_overlap_is_rejected_at_measurement(self):
        image, _, masks, _ = fixture()
        masks["screen_left_upper_lid_skin"] |= masks["screen_left_brow"]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            measure_features(image, masks)

    def test_highlight_percentage_and_chromatic_dispersion_are_descriptive(self):
        image, _, masks, _ = fixture()
        pixels = np.argwhere(masks["left_cheek"])
        for y, x in pixels[:len(pixels) // 4]:
            image[y, x] = (250, 250, 250)
        rows = rows_by_id(image, masks)
        expected = (len(pixels) // 4) / len(pixels) * 100
        self.assertAlmostEqual(rows["left_cheek_highlight_percent"]["value"], expected)
        self.assertIn("照明", rows["left_cheek_highlight_percent"]["note"])
        self.assertAlmostEqual(rows["left_cheek_ab_mad"]["value"], 0)

    def test_outside_nonfinite_bad_shape_and_overlapping_base_are_rejected(self):
        image, points, _, base = fixture()
        for bad in (points[:10], np.full(points.shape, np.nan), points - (500, 0)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                build_feature_masks(image.shape, bad, base)
        overlapped = deepcopy(base)
        overlapped["forehead"] |= base["left_cheek"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_feature_masks(image.shape, points, overlapped)

    def test_native_low_resolution_masks_are_not_enlarged_to_meet_guard(self):
        image, _, masks, _ = fixture(scale=0.20)
        rows = rows_by_id(image, masks)
        self.assertEqual(image.shape, (80, 80, 3))
        unavailable = [row for row in rows.values() if row["status"] == "insufficient_pixels"]
        self.assertGreater(len(unavailable), 0)
        self.assertTrue(all(row["value"] is None for row in unavailable))

    def test_mirroring_swaps_screen_side_targets(self):
        image, points, masks, base = fixture()
        mirrored_points = points.copy()
        mirrored_points[:, 0] = image.shape[1] - 1 - points[:, 0]
        mirrored_base = {
            "left_cheek": base["right_cheek"][:, ::-1],
            "right_cheek": base["left_cheek"][:, ::-1],
            "forehead": base["forehead"][:, ::-1],
        }
        mirrored = build_feature_masks(image.shape, mirrored_points, mirrored_base)
        for screen_side, original_side in (("screen_left", "screen_right"), ("screen_right", "screen_left")):
            actual = mirrored[f"{screen_side}_brow"]
            expected = masks[f"{original_side}_brow"][:, ::-1]
            # OpenCV polygon edge rasterization can differ at one-pixel ties.
            self.assertGreater(np.count_nonzero(actual & expected) / np.count_nonzero(actual | expected), 0.98)
            self.assertLess(abs(np.nonzero(actual)[1].mean() - np.nonzero(expected)[1].mean()), 0.2)


    def test_lower_eye_texture_masks_avoid_eye_aperture(self):
        image, points, masks, _ = fixture()
        for i, side in enumerate(("screen_left", "screen_right")):
            eye = np.zeros(image.shape[:2], np.uint8)
            polygon = np.vstack((points[list(UPPER_EYE_INDICES[i])],
                                 points[list(LOWER_EYE_INDICES[i])][-2:0:-1]))
            cv2.fillPoly(eye, [np.rint(polygon).astype(np.int32)], 1)
            region = masks[f"{side}_lower_eye_skin"]
            self.assertGreater(np.count_nonzero(region), 100)
            self.assertFalse(np.any(region & eye.astype(bool)))

    def test_lower_eye_highpass_metric_increases_for_added_fine_texture(self):
        image, _, masks, _ = fixture()
        baseline = rows_by_id(image, masks)
        textured = image.copy()
        region = masks["screen_left_lower_eye_skin"]
        ys, xs = np.where(region)
        for y, x in zip(ys, xs):
            if (x + y) % 2:
                textured[y, x] = np.clip(textured[y, x].astype(np.int16) - 35, 0, 255).astype(np.uint8)
            else:
                textured[y, x] = np.clip(textured[y, x].astype(np.int16) + 35, 0, 255).astype(np.uint8)
        changed = rows_by_id(textured, masks)
        for suffix in ("highpass_median_pct", "highpass_p90_pct"):
            key = f"screen_left_lower_eye_skin_{suffix}"
            self.assertEqual(baseline[key]["status"], "ok")
            self.assertEqual(changed[key]["status"], "ok")
            self.assertGreater(changed[key]["value"], baseline[key]["value"])
            self.assertIn("乾燥", changed[key]["note"])


if __name__ == "__main__":
    unittest.main()
