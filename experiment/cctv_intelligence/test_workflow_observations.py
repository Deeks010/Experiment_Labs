import json
import os
import argparse
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

import cctv_tools
import floor_activity_tracker as tracker
from workflow_observations import evidence_frame, get_observations, init_observations, inside, measure_frames, select_run
from workflow_knowledge import build_knowledge, knowledge_markdown, map_version


def detection(ref="subject_1", x=20, y=20):
    return {"subject_ref": ref, "tracker_id": "1", "foot": [x,y],
            "bbox": [x-10,0,x+10,100], "confidence": 0.9, "foot_source": "ankle_midpoint"}


def frame(t, detections=None):
    return {"t": t, "detections": [detection()] if detections is None else detections}


class MeasurementTests(unittest.TestCase):
    def measure(self, frames, end=3, zones=None):
        return measure_frames(frames, zones or [], 0, end, 1)

    def test_stationary_is_not_waiting_and_never_exceeds_window(self):
        result = self.measure([frame(0),frame(1),frame(2)])
        self.assertEqual(result["people"][0]["low_motion_sec"], 2)
        self.assertEqual(result["people"][0]["observed_sec"], 2)
        self.assertNotIn("waiting", result["people"][0])

    def test_missing_detection_splits_intervals(self):
        result = self.measure([frame(0),frame(1),frame(2,[]),frame(3),frame(4)], 5)
        self.assertEqual(result["people"][0]["low_motion_sec"], 2)
        self.assertEqual(result["people"][0]["low_motion_intervals"], [[0,1],[3,4]])
        self.assertEqual(result["frames_without_detections"], 1)
        self.assertEqual(result["people"][0]["unobserved_samples_within_span"], 1)

    def test_missing_processed_frames_never_filled(self):
        result = self.measure([frame(0),frame(4)], 5)
        self.assertEqual(result["people"][0]["observed_sec"], 0)
        self.assertEqual(result["adjacent_sample_coverage_sec"], 0)

    def test_one_sample_has_no_measured_duration(self):
        result = self.measure([frame(0)], 0.2)
        self.assertEqual(result["people"][0]["observed_sec"], 0)

    def test_no_detections_is_preserved(self):
        result = self.measure([frame(0,[]),frame(1,[])], 2)
        self.assertEqual(result["processed_frames"], 2)
        self.assertEqual(result["frames_without_detections"], 2)
        self.assertEqual(result["local_track_labels"], 0)

    def test_polygon_not_bounding_box_used(self):
        geometry = {"shape":"polygon", "points":[[0,0],[100,0],[0,100]], "box":[0,0,100,100]}
        self.assertFalse(inside([90,90], geometry))
        self.assertTrue(inside([20,20], geometry))

    def test_zone_dwell_requires_both_endpoints(self):
        zone = {"name":"LANE", "kind":"floor_area", "geometry":{"box":[0,0,50,50]}}
        result = self.measure([frame(0),frame(1),frame(2,[detection(x=90)])], zones=[zone])
        self.assertEqual(result["areas"][0]["occupied_sec"], 1)

    def test_duration_not_rounded_up_to_whole_seconds(self):
        result = measure_frames([frame(0),frame(.125),frame(.25)], [], 0, .3, .125)
        self.assertEqual(result["people"][0]["observed_sec"], .25)

    def test_out_of_window_data_excluded(self):
        result = self.measure([frame(0),frame(1),frame(2),frame(3)], 2)
        self.assertEqual(result["processed_frames"], 2)

    def test_invalid_window_rejected(self):
        for end in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                self.measure([], end)

    def test_uncertain_low_motion_track_stays_uncertain(self):
        person = tracker.ActiveSubject('1','s1',0,1,1)
        person.uncertain_notes.append('identity ambiguous')
        self.assertEqual(person.to_json()['quality'], 'uncertain')


class RunIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'observations.sqlite3'
        self.path_patch = patch.object(tracker, 'DB_PATH', self.db)
        self.path_patch.start()
        tracker.init_db()
        tracker.get_or_create_camera('Camera', 'missing.mp4')
        self.conn = sqlite3.connect(self.db)
        self.conn.row_factory = sqlite3.Row
        config = json.dumps({'start_sec':0,'end_sec':3,'fps':1,'frame_stride':1,'width':100,'height':100})
        for run in ['old','new']:
            self.conn.execute("INSERT INTO tracking_runs(run_id,camera_id,source_path,source_kind,config_json,status) VALUES (?,1,'missing.mp4','synthetic',?,'complete')", (run,config))
        self.conn.execute("INSERT INTO frame_observations VALUES ('old',0,0,?)", (json.dumps([detection(),detection('subject_2')]),))
        self.conn.execute("INSERT INTO frame_observations VALUES ('new',0,0,?)", (json.dumps([detection()]),))
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.path_patch.stop()
        self.temp.cleanup()

    def test_latest_run_not_combined(self):
        self.assertEqual(get_observations(self.db,'Camera')['peak_detected_people'], 1)
        self.assertEqual(get_observations(self.db,'Camera','old')['peak_detected_people'], 2)

    def test_knowledge_ids_do_not_cross_runs(self):
        old = build_knowledge(get_observations(self.db, 'Camera', 'old'))
        new = build_knowledge(get_observations(self.db, 'Camera', 'new'))
        self.assertNotEqual(old['people'][0]['scoped_person_ref'], new['people'][0]['scoped_person_ref'])
        self.assertEqual(new['people'][0]['person_ref'], 'subject_1')
        self.assertEqual(new['scope']['suggested_frame_times_sec'], [0])
        self.assertNotIn(3, new['scope']['suggested_frame_times_sec'])

    def test_markdown_brief_and_map_share_labels_and_scope(self):
        with patch.object(cctv_tools, 'DB_PATH', self.db):
            brief = cctv_tools.get_workflow_brief('Camera', 'old')
            self.assertIn('Camera/old', brief)
            self.assertIn('subject_2', brief)
            self.assertNotIn('row1_', brief)
            self.assertIn('Camera-view map', brief)
            self.assertNotIn('subject_2', cctv_tools.get_map('Camera', 0, 3, 'new'))

    def test_brief_distinguishes_missing_person_from_empty_frame(self):
        for index, detections in [(1,[detection('subject_2')]),(2,[detection(), detection('subject_2')])]:
            self.conn.execute("INSERT INTO frame_observations VALUES ('new',?,?,?)", (index,index,json.dumps(detections)))
        self.conn.commit()
        knowledge = build_knowledge(get_observations(self.db, 'Camera', 'new'))
        self.assertEqual(knowledge['measurements']['frames_with_no_people_detected'], 0)
        self.assertEqual(knowledge['measurements']['missing_person_samples_within_spans'], 1)
        self.assertIn('Missing person samples within tracked spans: 1', knowledge_markdown(knowledge))

    def test_intervals_and_evidence_references_are_deterministic(self):
        self.conn.execute("INSERT INTO frame_observations VALUES ('new',1,1,?)", (json.dumps([detection()]),))
        self.conn.commit()
        summary = get_observations(self.db, 'Camera', 'new')
        a, b = build_knowledge(summary), build_knowledge(summary)
        self.assertEqual(a, b)
        self.assertEqual(a['events'][0]['kind'], 'low_motion')
        self.assertTrue(a['events'][0]['evidence_ref'].startswith('Camera/new/event/'))
        self.assertNotIn('waiting', {event['kind'] for event in a['events']})

    def test_map_version_is_order_independent_but_geometry_sensitive(self):
        a = {'name':'A','geometry':{'box':[0,0,50,50]},'kind':'floor_area'}
        b = {'name':'B','geometry':{'box':[50,50,90,90]},'kind':'static_object'}
        self.assertEqual(map_version([a,b]), map_version([b,a]))
        version=map_version([a,b])
        a['geometry']['box'][0]=1
        self.assertNotEqual(version, map_version([a,b]))

    def test_visual_gets_labeled_frames_and_same_knowledge(self):
        source = Path(self.temp.name)/'visual.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 1, (100,100))
        self.assertTrue(writer.isOpened())
        for _ in range(3): writer.write(np.full((100,100,3), 90, np.uint8))
        writer.release()
        self.conn.execute("UPDATE tracking_runs SET source_path=? WHERE run_id='new'", (str(source),))
        for index in (1,2):
            self.conn.execute("INSERT INTO frame_observations VALUES ('new',?,?,?)", (index,index,json.dumps([detection()])))
        self.conn.commit()
        client=MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content='subject_1: unclear'
        with patch.object(cctv_tools, 'DB_PATH', self.db), patch.object(cctv_tools, 'BASE_DIR', Path(self.temp.name)):
            result=cctv_tools._inspect_workflow_frames('Camera','new',[0,1,2],'Describe subject_1',client)
        content=client.chat.completions.create.call_args.kwargs['messages'][0]['content']
        self.assertIn('Camera/new', content[0]['text'])
        self.assertIn('subject_1', content[0]['text'])
        self.assertEqual(len([p for p in content if p['type']=='image_url']), 3)
        metadata=[json.loads(p['text']) for p in content[1:] if p['type']=='text']
        self.assertEqual([p['actual_time_sec'] for p in metadata],[0,1,2])
        self.assertTrue(all(p['person_refs']==['subject_1'] for p in metadata))
        self.assertTrue(all(p['image_interpreted'] is False for p in metadata))
        self.assertTrue(all(p['map_version']==map_version([]) for p in metadata))
        self.assertIn('VISUAL INTERPRETATION (not a new measurement',result)

    def test_day_summary_refuses_repeated_processing_runs(self):
        with patch.object(cctv_tools, 'DB_PATH', self.db):
            self.assertIn('Do not sum', cctv_tools.get_day_summary('Camera'))

    def visual(self, run_id, timestamps):
        with patch.object(cctv_tools, 'DB_PATH', self.db), \
             patch.dict(os.environ, {'OPENAI_API_KEY':'test-only'}), patch('openai.OpenAI') as client:
            result = cctv_tools.get_visual_grid('Camera', timestamps, 'Describe visible evidence', run_id)
            client.return_value.chat.completions.create.assert_not_called()
            return result

    def test_visual_unknown_run_does_not_fall_back(self):
        self.assertIn('No matching tracking run', self.visual('missing', [0]))

    def test_visual_latest_empty_run_uses_its_own_source(self):
        self.conn.execute("UPDATE tracking_runs SET source_path='new-source.mp4' WHERE run_id='new'")
        self.conn.commit()
        self.assertIn('new-source.mp4', self.visual(None, [0]))

    def test_visual_timestamp_must_be_inside_run(self):
        self.assertIn('outside the selected run', self.visual('new', [3]))

    def test_no_baseline_means_unknown_not_normal(self):
        result = get_observations(self.db,'Camera')
        self.assertEqual(result['normality'], 'insufficient_evidence')
        self.assertIsNone(result['baseline'])

    def test_unknown_run_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            get_observations(self.db,'Camera','does-not-exist')

    def test_failed_latest_run_is_visible(self):
        self.conn.execute("UPDATE tracking_runs SET status='failed' WHERE run_id='new'")
        self.conn.commit()
        self.assertEqual(get_observations(self.db,'Camera')['status'], 'failed')

    def test_evidence_outside_window_rejected(self):
        with self.assertRaises(ValueError):
            evidence_frame(self.db,'Camera','new',3,Path(self.temp.name))

    def test_legacy_query_does_not_fall_back_to_older_run(self):
        subject = {'subject_ref':'s1','path_points':[{'t':2,'foot':[1,1]}], 'start_time_sec':2,'end_time_sec':2}
        tracker.insert_floor_segment(1,{'subjects':[subject]}, {}, {'run_id':'old'})
        self.assertEqual(cctv_tools._load_segments(1,0,3,self.conn.cursor()), [])

    def test_empty_video_run_keeps_zero_based_frame_observations(self):
        source = Path(self.temp.name) / 'source.mp4'
        source.touch()
        class Capture:
            position = 0
            def isOpened(self): return True
            def release(self): pass
            def get(self, prop):
                return {cv2.CAP_PROP_FPS:20, cv2.CAP_PROP_FRAME_COUNT:4,
                        cv2.CAP_PROP_FRAME_WIDTH:100, cv2.CAP_PROP_FRAME_HEIGHT:100,
                        cv2.CAP_PROP_POS_FRAMES:self.position}.get(prop, 0)
            def read(self):
                self.position += 1
                return (True,np.zeros((100,100,3),np.uint8)) if self.position <= 4 else (False,None)
        class Model:
            def track(self, *args, **kwargs): return [None]
        args = argparse.Namespace(video=str(source), camera='Empty', frame_stride=1, start_sec=0,
            end_sec=None, model='mock', device='test', preview=False, tracker='botsort.yaml',
            conf=.35, imgsz=640, max_missing_sec=4, max_reconnect_px=250, reid_lost_sec=30,
            reid_max_distance_px=680, reid_min_appearance=.68, reid_min_score=.58,
            discard_low_motion=False, min_segment_sec=1, min_movement_px=35,
            save_preview_video=False, source_kind='synthetic')
        with patch.object(tracker,'OUTPUT_DIR',Path(self.temp.name)/'runs'), \
             patch.object(tracker,'load_yolo',return_value=Model()), \
             patch.object(tracker.cv2,'VideoCapture',return_value=Capture()), \
             patch.object(tracker,'extract_detections',return_value=[]):
            result = tracker.process_video(args)
        observations = self.conn.execute("SELECT frame_index,time_sec,detections_json FROM frame_observations WHERE run_id=? ORDER BY frame_index", (result['run_id'],)).fetchall()
        self.assertEqual([(r[0],r[1]) for r in observations], [(0,0),(1,.05),(2,.1),(3,.15)])
        self.assertTrue(all(r[2] == '[]' for r in observations))
        self.assertEqual(get_observations(self.db,'Empty')['normality'],'insufficient_evidence')


if __name__ == '__main__':
    unittest.main()
