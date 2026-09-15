import unittest

from experiment.cctv_intelligence.floor_activity_tracker import (
    ActiveSubject,
    assign_reconnections,
    suppress_duplicate_detections,
)


def appearance(first: float, second: float) -> dict:
    histogram = [first, second]
    return {
        "full_hsv_hist": histogram,
        "upper_hsv_hist": histogram,
        "lower_hsv_hist": histogram,
        "upper_color": "red" if first > second else "blue",
        "lower_color": "dark",
    }


def subject(internal_id: str, point: tuple[float, float], signature: dict | None = None) -> ActiveSubject:
    value = ActiveSubject(
        internal_id=internal_id,
        subject_ref=f"subject_{internal_id}",
        first_time=0.0,
        last_time=0.0,
        last_seen_frame=0,
    )
    value.last_smoothed = point
    if signature:
        value.appearance_samples = [signature]
    return value


def detection(internal_id: str, point: tuple[float, float], signature: dict | None = None) -> dict:
    return {
        "internal_id": internal_id,
        "foot": point,
        "appearance": signature,
    }


class ReconnectionTests(unittest.TestCase):
    def match(self, detections, active=None, lost=None, t_sec=1.0):
        return assign_reconnections(
            detections=detections,
            active=active or {},
            lost=lost or {},
            seen_active_ids=set(),
            t_sec=t_sec,
            max_missing_sec=4.0,
            max_reconnect_px=250.0,
            reid_lost_sec=30.0,
            reid_max_distance_px=680.0,
            reid_min_appearance=0.68,
            reid_min_score=0.58,
        )

    def test_appearance_wins_when_positions_are_ambiguous(self):
        red = appearance(1.0, 0.0)
        blue = appearance(0.0, 1.0)
        active = {
            "old_red": subject("old_red", (0.0, 0.0), red),
            "old_blue": subject("old_blue", (10.0, 0.0), blue),
        }
        detections = [
            detection("new_red", (9.0, 0.0), red),
            detection("new_blue", (1.0, 0.0), blue),
        ]

        matches = self.match(detections, active=active)

        self.assertEqual(matches[0][1], "old_red")
        self.assertEqual(matches[1][1], "old_blue")

    def test_one_old_track_cannot_match_two_detections(self):
        red = appearance(1.0, 0.0)
        active = {"old_red": subject("old_red", (0.0, 0.0), red)}
        detections = [
            detection("new_1", (1.0, 0.0), red),
            detection("new_2", (200.0, 0.0), red),
        ]

        matches = self.match(detections, active=active)

        self.assertEqual(len(matches), 1)

    def test_missing_appearance_is_only_allowed_for_a_short_gap(self):
        active = {"old": subject("old", (0.0, 0.0))}
        detections = [detection("new", (5.0, 0.0))]

        self.assertEqual(len(self.match(detections, active=active, t_sec=1.0)), 1)
        self.assertEqual(len(self.match(detections, active=active, t_sec=2.0)), 0)

    def test_ambiguous_identity_is_not_merged(self):
        red = appearance(1.0, 0.0)
        active = {
            "old_1": subject("old_1", (0.0, 0.0), red),
            "old_2": subject("old_2", (0.0, 0.0), red),
        }

        matches = self.match([detection("new", (1.0, 0.0), red)], active=active)

        self.assertEqual(matches, {})

    def test_path_distance_is_not_net_displacement(self):
        value = subject("loop", (0.0, 0.0))
        value.points = [
            {"foot": [0.0, 0.0]},
            {"foot": [100.0, 0.0]},
            {"foot": [0.0, 0.0]},
        ]

        self.assertEqual(value.net_displacement_px(), 0.0)
        self.assertEqual(value.path_distance_px(), 200.0)

    def test_nested_boxes_with_matching_pose_are_suppressed(self):
        pose = [[100.0 + index, 100.0 + index, 0.9] for index in range(17)]
        detections = [
            {"internal_id": "old", "bbox": [50.0, 50.0, 180.0, 350.0], "conf": 0.40, "keypoints": pose},
            {"internal_id": "new", "bbox": [50.0, 50.0, 180.0, 240.0], "conf": 0.80, "keypoints": pose},
        ]

        kept = suppress_duplicate_detections(detections, {"old"})

        self.assertEqual([item["internal_id"] for item in kept], ["old"])

    def test_overlapping_people_with_different_poses_are_kept(self):
        pose_a = [[100.0 + index, 100.0 + index, 0.9] for index in range(17)]
        pose_b = [[150.0 + index, 200.0 + index, 0.9] for index in range(17)]
        detections = [
            {"internal_id": "a", "bbox": [50.0, 50.0, 180.0, 350.0], "conf": 0.80, "keypoints": pose_a},
            {"internal_id": "b", "bbox": [50.0, 50.0, 180.0, 240.0], "conf": 0.75, "keypoints": pose_b},
        ]

        kept = suppress_duplicate_detections(detections, set())

        self.assertEqual(len(kept), 2)

    def test_same_origin_nested_boxes_without_pose_are_suppressed(self):
        detections = [
            {"internal_id": "old", "bbox": [50.0, 50.0, 180.0, 350.0], "conf": 0.40, "keypoints": None},
            {"internal_id": "new", "bbox": [52.0, 51.0, 178.0, 240.0], "conf": 0.80, "keypoints": None},
        ]

        kept = suppress_duplicate_detections(detections, {"old"})

        self.assertEqual([item["internal_id"] for item in kept], ["old"])

    def test_alternating_nested_box_reconnects_despite_shifted_foot(self):
        active_subject = subject("old", (115.0, 340.0))
        active_subject.last_bbox = [50.0, 50.0, 180.0, 350.0]
        detections = [
            {
                "internal_id": "new",
                "bbox": [52.0, 51.0, 178.0, 240.0],
                "foot": (115.0, 240.0),
                "appearance": None,
            }
        ]

        matches = self.match(detections, active={"old": active_subject}, t_sec=0.5)

        self.assertEqual(matches[0][1], "old")
        self.assertEqual(matches[0][2]["nested_bbox_continuation"], 1.0)


if __name__ == "__main__":
    unittest.main()
