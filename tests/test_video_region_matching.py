from __future__ import annotations

import unittest

from analysis.match_video_regions import (
    REGION_RULES,
    RegionRule,
    filter_region_pairs,
)


def pair(before_id: str, after_id: str, score: float, scale: float,
         before_time: float, after_time: float) -> dict:
    return {
        "before_id": before_id,
        "after_id": after_id,
        "before_time": before_time,
        "after_time": after_time,
        "score": score,
        "terms": {"face_scale_ratio": scale},
    }


def occlusion_entry(**overlaps: float) -> dict:
    return {
        "regions": {
            region: {"hand_overlap_ratio": float(overlaps.get(region, 0.0)), "pixels": 1000}
            for region in REGION_RULES
        }
    }


class RegionPairFilteringTests(unittest.TestCase):
    def test_region_specific_hand_gate_keeps_eye_pair_but_rejects_same_pair_for_lips(self):
        pairs = [
            pair("b1", "a1", 0.10, 1.00, 10, 100),
            pair("b2", "a2", 0.20, 1.02, 30, 130),
            pair("b3", "a3", 0.30, 1.08, 50, 150),
        ]
        frames = {
            "b1": occlusion_entry(),
            "a1": occlusion_entry(lips=0.40),
            "b2": occlusion_entry(),
            "a2": occlusion_entry(),
            "b3": occlusion_entry(),
            "a3": occlusion_entry(),
        }
        result = filter_region_pairs(
            pairs, {"frames": frames}, top_k=10, diversity_seconds=0
        )
        self.assertEqual(
            result["regions"]["eye_texture"]["ranked_pairs"][0]["before_id"], "b1"
        )
        self.assertEqual(
            result["regions"]["lips"]["ranked_pairs"][0]["before_id"], "b2"
        )
        self.assertEqual(result["regions"]["lips"]["rejected"]["hand_overlap"], 1)
        self.assertEqual(
            result["regions"]["eye_texture"]["rejected"]["face_scale_ratio"], 1
        )

    def test_missing_occlusion_metadata_stops_instead_of_assuming_zero(self):
        pairs = [pair("b1", "a1", 0.1, 1.0, 10, 100)]
        with self.assertRaisesRegex(ValueError, "missing hand-occlusion"):
            filter_region_pairs(
                pairs, {"frames": {"b1": occlusion_entry()}},
                top_k=10, diversity_seconds=0,
            )

    def test_no_candidate_does_not_relax_scale_limit(self):
        strict = {
            "eye_texture": RegionRule(
                REGION_RULES["eye_texture"].mask_names, 1.001, 0.01
            )
        }
        pairs = [pair("b1", "a1", 0.1, 1.02, 10, 100)]
        frames = {"b1": occlusion_entry(), "a1": occlusion_entry()}
        result = filter_region_pairs(
            pairs, {"frames": frames}, rules=strict,
            top_k=10, diversity_seconds=0,
        )
        self.assertEqual(result["regions"]["eye_texture"]["ranked_pairs"], [])
        self.assertEqual(
            result["regions"]["eye_texture"]["rejected"]["face_scale_ratio"], 1
        )

    def test_diversity_rejects_reuse_or_nearby_before_and_after_endpoints(self):
        pairs = [
            pair("b1", "a1", 0.10, 1.00, 10, 100),
            # before is far away, but the same after endpoint must not be reused.
            pair("b2", "a1", 0.11, 1.00, 40, 100),
            # after is far away, but the same before endpoint must not be reused.
            pair("b1", "a2", 0.12, 1.00, 10, 150),
            # Both endpoints are independently far enough from the first pair.
            pair("b3", "a3", 0.13, 1.00, 40, 140),
        ]
        frames = {
            frame_id: occlusion_entry()
            for frame_id in ("b1", "a1", "b2", "a2", "b3", "a3")
        }
        result = filter_region_pairs(
            pairs, {"frames": frames}, top_k=10, diversity_seconds=15
        )
        selected = result["regions"]["eye_texture"]["ranked_pairs"]
        self.assertEqual(
            [(item["before_id"], item["after_id"]) for item in selected],
            [("b1", "a1"), ("b3", "a3")],
        )
        self.assertEqual(result["diversity_mode"], "independent_endpoints")
        self.assertEqual(
            result["regions"]["eye_texture"]["diversity_skipped"], 2
        )


if __name__ == "__main__":
    unittest.main()
