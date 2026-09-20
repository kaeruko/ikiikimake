from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import unquote

import cv2
import numpy as np

from analysis.video_roi_report import write_video_report


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src"):
                self.paths.append(value)


class VideoReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "report"
        self.output.mkdir()
        self.records = []
        for index, color in enumerate(((40, 60, 90), (50, 70, 100))):
            source = self.root / f"画像 {index}.png"
            image = np.full((80, 120, 3), color, dtype=np.uint8)
            cv2.imencode(".png", image)[1].tofile(source)
            roi_dir = self.root / f"roi {index}"
            roi_dir.mkdir()
            overlay = image.copy()
            overlay[30:40, 20:35] = (0, 255, 0)
            cv2.imencode(".png", overlay)[1].tofile(roi_dir / "roi_overlay.png")
            report = {
                "status": "needs_review",
                "errors": [],
                "quality": {"pose_degrees": {"yaw": 1, "pitch": 2, "roll": 3}},
                "face_landmarks_normalized": [[0.2, 0.1, 0], [0.8, 0.9, 0]],
                "rois": {},
            }
            (roi_dir / "roi_points.json").write_text(json.dumps(report), encoding="utf-8")
            self.records.append({"frame_id": f"frame_{index}", "timestamp_seconds": 1.5 + 20 * index,
                                 "image_path": str(source), "roi_dir": str(roi_dir), "report": report})
        self.manifest = {
            "video": {"path": "メイク.mp4", "sha256": "abc", "duration_seconds": 60, "width": 120, "height": 80, "fps": 30},
            "config": {"sample_interval_seconds": 1, "split_seconds": 20, "min_gap_seconds": 10},
            "notes": ["塗布前後の候補範囲を手動で選択。<script>危険なタグ</script>"],
            "selected_ranges": {"before": [0, 10], "after": [20, 30]},
            "records": self.records,
        }
        self.matching = {"ranked_pairs": [{"before_id": "frame_0", "after_id": "frame_1", "before_time": 1.5,
                                            "after_time": 21.5, "score": 0.25, "terms": {"contributions": {"pose": 0.25}},
                                            "diagnostics": {"notice": "geometry only"}}], "stats": {"valid_pairs": 1}}

    def test_ranked_report_preserves_originals_and_separates_overlays(self):
        before_hashes = {record["image_path"]: hashlib.sha256(Path(record["image_path"]).read_bytes()).hexdigest()
                         for record in self.records}
        artifacts = write_video_report(self.output, self.manifest, self.matching)
        self.assertEqual(set(artifacts), {"html", "best_pair", "best_pair_faces", "best_pair_roi_overlay"})
        for name in ("best_pair", "best_pair_faces", "best_pair_roi_overlay"):
            image = cv2.imdecode(np.fromfile(self.output / artifacts[name], dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            has_green = np.any(np.all(image == [0, 255, 0], axis=2))
            self.assertEqual(bool(has_green), name == "best_pair_roi_overlay")
        for path, digest in before_hashes.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        page = (self.output / artifacts["html"]).read_text(encoding="utf-8")
        self.assertIn("今回の選択範囲内の先頭候補", page)
        self.assertIn("00:01.50", page)
        self.assertIn("00:21.50", page)
        self.assertIn("幾何差 0.2500", page)
        self.assertIn("今回の選択範囲（以下の範囲内で候補を比較）", page)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("Lab 解析・メイクの採点は行っていません", page)
        links = Links()
        links.feed(page)
        self.assertTrue(links.paths)
        for relative in links.paths:
            self.assertTrue((self.output / unquote(relative)).is_file(), relative)

    def test_zero_pairs_reports_failure_counts_without_broken_images(self):
        self.records[0]["report"].update(status="failed", errors=["left_cheek: suspected hand occlusion 8.0%"])
        self.records[1]["report"].update(status="failed", errors=["face too small: 95 px", "right_cheek: suspected hand occlusion 9.0%"])
        artifacts = write_video_report(self.output, self.manifest, {"ranked_pairs": [], "stats": {"passed": 0}})
        self.assertEqual(artifacts, {"html": "report.html"})
        page = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn("比較候補は見つかりませんでした", page)
        self.assertIn("手が領域に重なる疑い</td><td>2", page)
        self.assertIn("顔が小さい</td><td>1", page)
        self.assertNotIn("<img", page)
        self.assertFalse((self.output / "best_pair.png").exists())

    def test_unknown_frame_reference_is_rejected(self):
        self.matching["ranked_pairs"][0]["after_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown frame ID"):
            write_video_report(self.output, self.manifest, self.matching)
        self.assertFalse((self.output / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
