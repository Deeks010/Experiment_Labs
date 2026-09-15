import unittest

from floor_map_analytics import Track, build_summary, observation_spans


class FloorMapAnalyticsTests(unittest.TestCase):
    def test_observation_spans_preserve_sample_count_across_small_gaps(self) -> None:
        self.assertEqual(
            observation_spans([1, 2, 5, 12]),
            [
                {"start_sec": 1, "end_sec": 6, "span_sec": 5, "sampled_inside_seconds": 3},
                {"start_sec": 12, "end_sec": 13, "span_sec": 1, "sampled_inside_seconds": 1},
            ],
        )

    def test_area_counts_require_footpoint_inside_box(self) -> None:
        track = Track(
            ref="subject_1",
            start=0.0,
            end=2.0,
            points=[(0.0, 9.0, 5.0), (1.0, 10.0, 5.0), (2.0, 11.0, 5.0)],
            confidence=0.9,
            quality="good",
            tracker_ids=["4", "9"],
            reconnect_count=1,
        )
        summary, _ = build_summary(
            "camera",
            "run",
            [track],
            [{"name": "box", "box": [10.0, 0.0, 20.0, 10.0], "metadata": {}}],
            100,
            50,
            0.0,
            2,
            10,
        )

        area = summary["mapped_area_observations"][0]
        self.assertEqual(area["footpoint_inside_person_seconds"], 2)
        self.assertEqual(summary["identity_observations"]["raw_tracker_id_count"], 2)
        self.assertNotIn("near", str(summary).lower())


if __name__ == "__main__":
    unittest.main()
