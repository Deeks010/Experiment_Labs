### q04
TOOLS: list_cameras(); get_scene_log(camera='cam07', start='2026-10-09 10:00', end='10:10')

**Answer:** Today's recording covers only 10:00–10:10 on cam07. In that window, people entered the view 5 times: 3 times from the East exit (right) and 2 times from the West exit (left). Re-entries of the same person cannot be told apart, so this counts entries, not people.
**Evidence:**
- list_cameras: today's only recording is cam07, 2026-10-09 10:00:00–10:10:00.
- get_scene_log: IN at East exit 10:02:11, 10:02:13, 10:02:19.
- get_scene_log: IN at West exit 10:06:44, 10:08:41.
- get_scene_log: 5 exits in the same window (East 10:02:24, 10:05:29, 10:07:20, 10:08:46; West 10:03:56).
- get_scene_log: camera OFFLINE 10:05:45–10:06:25.
**Confidence:** high — edge crossings are reliable data; only 40 s are unknown.
**Cannot know:** any entry during the 40 s offline gap, or outside 10:00–10:10 (not recorded).

### q11
TOOLS: list_cameras(); compare_periods(camera='cam07', a_start='2026-10-08 10:00', a_end='2026-10-08 10:10', b_start='2026-10-09 10:00', b_end='2026-10-09 10:10'); get_dwell_overview(camera='cam07', start='2026-10-08 10:00', end='2026-10-08 10:10'); get_dwell_overview(camera='cam07', start='2026-10-09 10:00', end='2026-10-09 10:10'); get_events(camera='cam07', start='2026-10-08 10:00', end='2026-10-08 10:10'); get_events(camera='cam07', start='2026-10-09 10:00', end='2026-10-09 10:10')

**Answer:** Today was only slightly busier: on average 5.1 people were in view vs 4.9 yesterday, with a peak of 7 vs 6. Entries and exits were the same (5 each). The bigger change is where people were. Today a group stood at the control panel and Op 15 was empty for a while. Yesterday Op 14 and Op 16 had empty spells, and one person stood still on open floor for a long time.
**Evidence:**
- compare_periods: avg people 4.9 → 5.1, peak 6 → 7, edge in/out 5/5 both days. Camera online 10.0 min yesterday vs 9.3 min today.
- compare_periods / get_events: Op 15 empty 3.7 min today (absence 10:03:20–10:07:40, including 40 s offline) vs 0 yesterday. Yesterday Op 14 was empty 10:01:42–10:04:24 and Op 16 10:08:04–10:09:18. Neither was empty today.
- get_dwell_overview: Control panel front had 7.7 person-min today (flagged UNUSUAL) and is not in yesterday's list. get_events: crowding there 10:02:16–10:05:45 (peak 4), plus 2 close_pair events (10:03:04–10:03:55, 10:03:55–10:05:29).
- get_events (yesterday): long_stationary 10:02:12–10:06:50 on open floor near the notice board and 10:06:39–10:10:00 in Op 17. Outside-work-area dwell 10:02:14–10:06:41. Today: no long_stationary.
- get_events (today): one fast_movement in the Yellow aisle 10:08:42–10:08:44 (2 s). None yesterday.
**Confidence:** medium — counts and station states are reliable, but the windows are only 10 min and today is missing 40 s.
**Cannot know:** why people gathered at the panel or why the stations were empty. Whether yesterday's still person was resting has not been checked with frames.
**Suggested check:** get_image_frames on Control panel front, 2026-10-09 10:02:40–10:05:10, to see what the group was doing.

### q14
TOOLS: get_camera_info(camera='cam07', place='Press 15'); get_scene_log(camera='cam07', start='2026-10-08 10:01', end='10:02', focus='Op 15', detail='positions')

**Answer:** At 10:01:30 yesterday (2026-10-08), one person stood inside the Op 15 standing area, between Press 15 and Bench L. They were about 0.5 body heights (~0.9 m) in front of the middle of Press 15 (camera side) and about 0.5 bh (~0.9 m) behind the right part of Bench L (far side). Their feet were visible.
**Evidence:**
- get_camera_info: "Press 15" maps to the machine Press 15 and its operator area, Op 15.
- get_scene_log (positions): track p8, 10:01:24–10:02:00, in Op 15, BETWEEN Press 15 and Bench L, feet visible, xy=(465,330)px.
- get_scene_log: Op 15 occupied for the whole 10:01–10:02 window. A short step-away was logged at 10:01:11: just before that, a person was on the Bench L footprint (10:01:11–10:01:24).
**Confidence:** high — feet were visible and the camera was online. The position is a median foot point over 10:01:24–10:02:00, not a single frame.
**Cannot know:** whether this person ran the press. Whether it is the same person as track p2, who was at Op 15 until 10:01:11, is also unknown (track numbers are not identities).

### q16
TOOLS: list_cameras(); get_scene_log(camera='cam07', start='2026-10-09 10:00', end='10:10')

**Answer:** Between 7 and 9 different people appeared on cam07 today, 10:00–10:10. At least 7 were in view at the same time. At most 9 is 4 at the start plus 5 entries, if no one came back in. The exact number cannot be known because people are not identified.
**Evidence:**
- get_scene_log: peak of 7 people in frame 10:02:16–10:03:56.
- get_scene_log: 4 people at 10:00:00 and 5 entries (East 10:02:11, 10:02:13, 10:02:19; West 10:06:44, 10:08:41).
- get_scene_log DERIVED: "at least 7, at most 9".
- get_scene_log: camera OFFLINE 10:05:45–10:06:25.
**Confidence:** medium — the lower bound is solid. The upper bound assumes no re-entries and no one appearing during the offline gap.
**Cannot know:** whether a person who left came back (e.g. the 10:03:56 West exit and the 10:06:44 West entry), or anyone who passed only during the 40 s offline gap.
**Suggested check:** get_image_frames around the West exit at 10:03:56 and 10:06:44 to see if the same-looking person re-entered.
