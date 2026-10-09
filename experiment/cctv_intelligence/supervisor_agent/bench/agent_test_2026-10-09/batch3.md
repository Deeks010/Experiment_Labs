### q09
TOOLS: list_cameras(); get_events(start='2026-10-08 00:00', end='2026-10-08 23:59', camera='cam07'); get_image_frames(camera='cam07', start='2026-10-08 10:02:30', end='2026-10-08 10:06:30', count=4); get_image_frames(camera='cam07', start='2026-10-08 10:07:00', end='2026-10-08 10:09:50', count=4, crop_zone='Op 17')
**Answer:** No sleeping is supported. Yesterday (2026-10-08) only cam07 has data, and only for 10:00–10:10. Two people stood still for a long time, but the frames show both standing upright, not lying down.
**Evidence:**
- list_cameras: the only recording yesterday is cam07, 2026-10-08 10:00:00–10:10:00.
- get_events: long_stationary 10:02:12–10:06:50 on open floor beyond the left end of the Notice board (not a work zone); outside_work_area_dwell 10:02:14–10:06:41 near the same spot.
- get_image_frames: 4 frames at 10:02:30, 10:03:50, 10:05:10 and 10:06:30 show one person standing upright between the right end of Bench R and the Notice board.
- get_events: long_stationary 10:06:39–10:10:00 in Op 17. get_image_frames: 4 frames from 10:07:00 to 10:09:50 show a person standing at Press 17.
**Confidence:** medium. Positions are reliable, but posture is judged from 8 frames only.
**Cannot know:** anything outside 10:00–10:10 yesterday, because there is no recording. The frames cannot show if eyes were closed.
**Suggested check:** If you need the whole day, get the rest of yesterday's footage processed for cam07.

### q10
TOOLS: list_cameras(); get_camera_info(camera='cam07', place='Op 14'); get_scene_log(camera='cam07', start='2026-10-08 10:00', end='10:10', focus='Op 14'); get_image_frames(camera='cam07', start='2026-10-08 10:01:30', end='2026-10-08 10:04:30', count=6)
**Answer:** Op 14 was empty from 10:01:42 to 10:04:24 yesterday. That is the only gap in the recorded window, 10:00–10:10. The person who left Op 14 walked to Bench L, through the Passage, along the Yellow aisle, and left the view at the East exit at 10:02:41. Where they went after that is not visible.
**Evidence:**
- get_scene_log: Op 14 EMPTY 10:01:42–10:04:24. The camera was online for the whole window.
- get_scene_log, movement m1: Op 14 until 10:01:42 → at Bench L 10:01:42–10:01:57 → Passage 10:01:57–10:02:15 → Yellow aisle 10:02:15–10:02:41 → left view at East exit 10:02:41.
- get_scene_log: a person entered at the East exit at 10:03:47. Op 14 was occupied again from 10:04:24.
- get_image_frames: a person in red was at Op 14 at 10:01:30 and in the Passage at 10:02:06. No one was in red at 10:02:42 or 10:03:18. A person in red was in the aisle near the east side at 10:03:54 and at Op 14 at 10:04:30.
**Confidence:** high for the empty times, because station states are reliable. Medium for the route out of view. Low that the same person came back, because that rests only on clothing colour.
**Cannot know:** where the person was from 10:02:41 to about 10:03:47 (out of view), or why they left. Only 10:00–10:10 was recorded yesterday.
**Suggested check:** Look at the camera covering the area past the East exit, if one exists, for 10:02:41–10:03:47.

### q11
TOOLS: compare_periods(camera='cam07', a_start='2026-10-09 10:00', a_end='2026-10-09 10:10', b_start='2026-10-08 10:00', b_end='2026-10-08 10:10'); get_events(start='2026-10-09 10:00', end='2026-10-09 10:10', camera='cam07'); get_dwell_overview(camera='cam07', start='2026-10-09 10:00', end='2026-10-09 10:10'); get_dwell_overview(camera='cam07', start='2026-10-08 10:00', end='2026-10-08 10:10')
**Answer:** Today was about as busy as yesterday. The average was 5.1 people in view today and 4.9 yesterday, and the peak was 7 vs 6. What changed was where people gathered. Today a group stood at the Control panel front for several minutes and Op 15 was empty. Yesterday Op 14 and Op 16 had gaps, and people stood still near Bench R and the Notice board.
**Evidence:**
- compare_periods: average 5.1 people today vs 4.9 yesterday. Peak 7 vs 6. Entries and exits were 5 and 5 on both days. Today the camera was online 9.3 of 10 minutes (offline 10:05:45–10:06:25, from get_dwell_overview).
- get_events (today): crowding at the Control panel front 10:02:16–10:05:45 (peak 4 people). Two close_pair events there, 10:03:04–10:03:55 and 10:03:55–10:05:29.
- get_dwell_overview: the Control panel front had 7.7 person-minutes today. It is not in yesterday's list.
- Op 15 was empty 3.7 min today and 0 yesterday. get_events: absence 10:03:20–10:07:40, including 40 s unknown. Op 14 was empty 2.7 min and Op 16 1.2 min yesterday, and 0 for both today (compare_periods).
- Yesterday had 2 long_stationary events (8.0 min) and today had none. Outside-work-area dwell was 4.5 min yesterday vs 2.6 min today (compare_periods).
**Confidence:** high for counts and station states, which are reliable. Medium on reasons, because the data shows positions only.
**Cannot know:** why people gathered at the control panel today, or whether the Op 15 person was among them. Identities are not tracked. Nothing is known for 10:05:45–10:06:25 today.
**Suggested check:** Use get_image_frames at the Control panel front around 10:03–10:05 today to see what the group was doing.

### q12
TOOLS: get_camera_info(camera='cam07', place='Press 16 and Press 17 operators'); get_events(start='2026-10-08 00:00', end='2026-10-08 23:59', camera='cam07', types=['close_pair']); get_scene_log(camera='cam07', start='2026-10-08 10:07', end='10:10', focus='Op 17', detail='positions'); get_image_frames(camera='cam07', start='2026-10-08 10:07:55', end='2026-10-08 10:09:25', count=4, crop_zone='Op 17')
**Answer:** Possibly. Yesterday from 10:08:04 to 10:09:18, two people stood next to each other in Op 17, and Op 16 was empty at that time. That fits a chat, but there is no audio, so a conversation cannot be confirmed.
**Evidence:**
- get_events: close_pair at Op 17, 10:08:05–10:09:17 (72 s). This is the only close_pair yesterday.
- get_scene_log: Op 16 EMPTY 10:08:04–10:09:18. Op 17 had 2 people 10:08:04–10:09:18. Both stations had one person again from 10:09:18.
- get_image_frames: at 10:07:55 there was one person at Press 16 (blue) and one at Press 17 (purple). At 10:08:25 and 10:08:55 both stood side by side in Op 17. At 10:09:25 they were back at Press 16 and Press 17.
**Confidence:** medium. Positions and station states are reliable. That these were "the Press 16 and Press 17 operators" rests on clothing colour in 4 frames. Being close does not prove they talked.
**Cannot know:** whether they talked or what about, since there is no audio. Who they are is also unknown. Only 10:00–10:10 was recorded yesterday.
