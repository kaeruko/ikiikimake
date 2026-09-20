from __future__ import annotations

from copy import deepcopy
import json
import math
import unittest

import numpy as np

from analysis.extract_face_rois import polygons_from_landmarks, ROI_INDICES
from analysis.match_video_rois import rank_pairs, similarity_procrustes_rms


def make_record(frame_id: str, timestamp: float, *, size=(800, 400), scale=1.0,
                shift=(0, 0), roll=0.0, yaw=0.0, pitch=0.0,
                eye_aperture=10.0, mouth_opening=4.0, mirrored=False) -> dict:
    width, height = size
    points = np.full((478, 2), (200.0, 180.0))
    points[234], points[454] = (70, 220), (330, 220)
    for name, center, radii in (
        ("cheek_a", (125, 240), (36, 28)),
        ("cheek_b", (275, 240), (36, 28)),
        ("forehead", (200, 100), (55, 25)),
    ):
        indices = ROI_INDICES[name]
        angles = np.arange(len(indices)) * 2 * math.pi / len(indices)
        points[list(indices)] = np.column_stack((np.cos(angles), np.sin(angles))) * radii + center
    for corners, lid_pairs, center_x in (
        ((33, 133), ((159, 145), (158, 153)), 140),
        ((362, 263), ((386, 374), (385, 380)), 260),
    ):
        points[list(corners)] = ((center_x - 20, 175), (center_x + 20, 175))
        for i, (top, bottom) in enumerate(lid_pairs):
            points[top] = (center_x + i * 5, 175 - eye_aperture / 2)
            points[bottom] = (center_x + i * 5, 175 + eye_aperture / 2)
    points[13], points[14] = (200, 285 - mouth_opening / 2), (200, 285 + mouth_opening / 2)
    if mirrored:
        points[:, 0] = 400 - points[:, 0]
    angle = math.radians(roll)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = ((points - (200, 200)) @ rotation.T) * scale + (200, 200) + shift
    polygons = polygons_from_landmarks(points)
    report = {
        "status": "needs_review", "errors": [], "warnings": [],
        "source": {"width": width, "height": height},
        "config": {"polygon_scale": 0.88},
        "quality": {"face_width_px": float(np.linalg.norm(points[454] - points[234])),
                    "pose_degrees": {"yaw": yaw, "pitch": pitch, "roll": roll},
                    "detected_faces": 1, "forehead_L_median": 50.0},
        "face_landmarks_normalized": np.column_stack((points / (width, height), np.zeros(len(points)))).tolist(),
        "rois": {name: {"polygon_px": polygon.tolist()} for name, polygon in polygons.items()},
    }
    return {"frame_id": frame_id, "timestamp_seconds": timestamp, "image_path": "not-read.png",
            "roi_dir": "not-read", "report": report}


class GeometryMatchingTests(unittest.TestCase):
    def test_identical_geometry_scores_zero_without_identity_claim(self) -> None:
        records = [make_record("before", 10), make_record("after", 100)]
        original = deepcopy(records)
        result = rank_pairs(records, 50, 30)
        self.assertEqual(records, original)
        pair = result["ranked_pairs"][0]
        self.assertEqual(pair["score"], 0)
        self.assertFalse(pair["diagnostics"]["identity_verified"])
        self.assertTrue(pair["diagnostics"]["needs_human_review"])
        self.assertFalse(result["settings"]["thresholds_validated"])
        json.dumps(result, allow_nan=False)

    def test_lab_and_rgb_changes_never_change_eligibility_or_ranking(self) -> None:
        records = [make_record("b", 10), make_record("a1", 100, yaw=2), make_record("a2", 130)]
        baseline = rank_pairs(records, 50, 30)
        for index, record in enumerate(records):
            quality = record["report"]["quality"]
            quality["forehead_L_median"] = (float("nan"), -1e30, 1e30)[index]
            quality["Lab"] = [99, -55, 78]
            quality["rgb"] = [255, 0, 255]
            quality["clipped_pixel_ratios"] = {"forehead": 1}
            quality["hand_overlap_ratios"] = {"left_cheek": 1}
        changed = rank_pairs(records, 50, 30)
        self.assertEqual(baseline, changed)
        self.assertFalse(changed["settings"]["forehead_L_used_for_pair_gate"])

    def test_similarity_alignment_removes_translation_rotation_scale_not_reflection(self) -> None:
        source = np.array([[0, 0], [2, 0], [3, 1], [0, 3]], dtype=float)
        theta = math.radians(17)
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        self.assertEqual(similarity_procrustes_rms(source, source @ rotation * 1.3 + (9, 7)), 0)
        reflected = source * (-1, 1)
        self.assertGreater(similarity_procrustes_rms(source, reflected), 0.1)

    def test_pair_shape_is_invariant_but_pose_and_scale_remain_separate_terms(self) -> None:
        result = rank_pairs([make_record("b", 10), make_record("a", 100, scale=1.2, shift=(40, 0), roll=5)], 50, 0)
        pair = result["ranked_pairs"][0]
        self.assertEqual(pair["terms"]["roi_procrustes_rms"], 0)
        self.assertAlmostEqual(pair["terms"]["face_scale_ratio"], 1.2)
        self.assertEqual(pair["terms"]["roll_gap_degrees"], 5)
        self.assertAlmostEqual(pair["terms"]["eye_aperture_gap"], 0)
        self.assertAlmostEqual(pair["score"], sum(pair["terms"]["contributions"].values()))

    def test_non_square_image_dimensions_do_not_distort_expression(self) -> None:
        result = rank_pairs([make_record("b", 10, size=(800, 400)), make_record("a", 100, size=(1600, 450))], 50, 0)
        pair = result["ranked_pairs"][0]
        self.assertEqual(pair["score"], 0)
        self.assertAlmostEqual(pair["diagnostics"]["before_expression"]["screen_eye_apertures"][0], 10 / 260)

    def test_pose_scale_expression_and_mirror_gates(self) -> None:
        for kwargs, reason in (
            ({"yaw": 10.1}, "yaw_gap"),
            ({"pitch": -10.1}, "pitch_gap"),
            ({"roll": 12.1}, "roll_gap"),
            ({"scale": 1.51}, "face_scale_ratio"),
            ({"eye_aperture": 21}, "eye_aperture_gap"),
            ({"mouth_opening": 25}, "mouth_opening_gap"),
            ({"mirrored": True}, "mirrored_landmark_order_changed"),
        ):
            with self.subTest(reason=reason):
                result = rank_pairs([make_record("b", 10), make_record("a", 100, **kwargs)], 50, 0)
                self.assertFalse(result["ranked_pairs"])
                self.assertEqual(result["stats"]["pair_rejection_counts"][reason], 1)

    def test_geometry_sort_and_deterministic_ties(self) -> None:
        records = [make_record("b", 10), make_record("worse", 100, yaw=5),
                   make_record("later", 140), make_record("earlier", 120)]
        result = rank_pairs(records[::-1], 50, 0, diversity_seconds=0)
        self.assertEqual([p["after_id"] for p in result["ranked_pairs"]], ["earlier", "later", "worse"])

    def test_local_roi_deformation_increases_shape_term(self) -> None:
        deformed = make_record("deformed", 100)
        report = deformed["report"]
        landmarks = np.array(report["face_landmarks_normalized"])
        landmarks[116, 0] += 0.015
        report["face_landmarks_normalized"] = landmarks.tolist()
        polygons = polygons_from_landmarks(landmarks[:, :2] * (800, 400))
        report["rois"] = {name: {"polygon_px": polygon.tolist()} for name, polygon in polygons.items()}
        result = rank_pairs([make_record("b", 10), deformed, make_record("same", 130)], 50, 0, diversity_seconds=0)
        self.assertEqual([pair["after_id"] for pair in result["ranked_pairs"]], ["same", "deformed"])
        self.assertGreater(result["ranked_pairs"][1]["terms"]["roi_procrustes_rms"], 0)

    def test_split_boundary_and_minimum_gap_are_enforced(self) -> None:
        records = [make_record("b", 49), make_record("boundary", 50), make_record("a", 80)]
        result = rank_pairs(records, 50, 30)
        self.assertEqual([(p["before_id"], p["after_id"]) for p in result["ranked_pairs"]], [("b", "a")])
        self.assertEqual(result["stats"]["before_frames"], 1)
        self.assertEqual(result["stats"]["after_frames"], 2)
        self.assertEqual(result["stats"]["pair_rejection_counts"]["minimum_time_gap"], 1)

    def test_diversity_rejects_only_when_both_times_are_near(self) -> None:
        records = [make_record("b1", 10), make_record("b2", 12), make_record("b3", 35),
                   make_record("a1", 100), make_record("a2", 102), make_record("a3", 130)]
        result = rank_pairs(records, 50, 0, top_k=20, diversity_seconds=15)
        self.assertEqual(len(result["ranked_pairs"]), 4)
        self.assertEqual(result["stats"]["diversity_skipped_pairs"], 5)
        unlimited = rank_pairs(records, 50, 0, top_k=20, diversity_seconds=0)
        self.assertEqual(len(unlimited["ranked_pairs"]), 9)
        limited = rank_pairs(records, 50, 0, top_k=2, diversity_seconds=0)
        self.assertEqual(len(limited["ranked_pairs"]), 2)
        self.assertEqual(limited["stats"]["top_k_omitted_pairs"], 7)

    def test_failed_nonfinite_malformed_and_duplicate_frames_are_rejected(self) -> None:
        bad_records = []
        for i, mutate in enumerate((
            lambda r: r["report"].update(status="failed"),
            lambda r: r["report"].update(errors=["hand occlusion"]),
            lambda r: r["report"].update(face_landmarks_normalized=[[0, 0, 0]]),
            lambda r: r["report"]["face_landmarks_normalized"][0].__setitem__(0, float("nan")),
            lambda r: r["report"]["quality"]["pose_degrees"].update(yaw=float("inf")),
            lambda r: r["report"]["quality"].update(face_width_px=1),
            lambda r: r["report"]["rois"]["left_cheek"].update(polygon_px=[]),
            lambda r: r.update(timestamp_seconds=-1),
            lambda r: r.update(frame_id=float("nan")),
        )):
            record = make_record(f"bad{i}", 20)
            mutate(record)
            bad_records.append(record)
        bad_records.extend([make_record("duplicate", 10), make_record("duplicate", 15)])
        result = rank_pairs(bad_records + [make_record("good", 100)], 50, 0)
        self.assertFalse(result["ranked_pairs"])
        self.assertEqual(result["stats"]["rejected_frame_count"], len(bad_records))
        self.assertEqual(result["stats"]["accepted_frames"], 1)
        json.dumps(result, allow_nan=False)

    def test_argument_validation_and_empty_result(self) -> None:
        for kwargs in ({"split_seconds": float("nan")}, {"min_gap_seconds": -1},
                       {"diversity_seconds": -1}, {"top_k": 0}, {"top_k": True}):
            options = {"split_seconds": 50, "min_gap_seconds": 10}
            options.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rank_pairs([], **options)
        empty = rank_pairs([], 50, 10)
        self.assertEqual(empty["stats"]["candidate_pairs"], 0)
        self.assertFalse(empty["ranked_pairs"])


if __name__ == "__main__":
    unittest.main()
