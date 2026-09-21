import copy
import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from contextlib import closing
from pathlib import Path
from unittest.mock import patch, MagicMock
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import cctv_agent
import cctv_tools
import map_editor_server as editor
import site_context as site
import site_tools
import floor_activity_tracker as tracker
import agent_web_server as web
from workflow_observations import get_observations
from workflow_knowledge import build_knowledge
from test_workflow_observations import detection


class SiteTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.db=self.root/'site.sqlite3'
        self.image=self.root/'source.jpg'
        cv2.imwrite(str(self.image),np.full((100,100,3),90,np.uint8))
        with patch.object(tracker,'DB_PATH',self.db):
            tracker.init_db()
            for name in ['Dock','Transfer']:
                tracker.get_or_create_camera(name,str(self.image))
        with closing(sqlite3.connect(self.db)) as conn, conn:
            config=json.dumps({'start_sec':0,'end_sec':4,'fps':1,'frame_stride':1,'width':100,'height':100})
            for i,name in enumerate(['Dock','Transfer'],1):
                conn.execute('INSERT INTO tracking_runs(run_id,camera_id,source_path,source_kind,config_json,status) '
                             'VALUES (?,?,?,\'synthetic\',?,\'complete\')',(name,i,str(self.image),config))
                for t in range(4):
                    detections=[detection()] if t<2 else [detection(),detection('subject_future')]
                    conn.execute('INSERT INTO frame_observations VALUES (?,?,?,?)',(name,t,t,json.dumps(detections)))
        self.doc={'schema_version':1,'name':'Test place','rooms':[{'id':'a'},{'id':'b'},{'id':'blind'}],
            'cameras':[{'name':'Dock','room_id':'a'},{'name':'Transfer','room_id':'b'}],
            'routes':[{'id':'direct','from_room':'a','to_room':'b','bidirectional':True,'min_seconds':20,'max_seconds':80},
                      {'id':'alt1','from_room':'a','to_room':'blind','min_seconds':10,'max_seconds':40},
                      {'id':'alt2','from_room':'blind','to_room':'b','min_seconds':10,'max_seconds':40}],
            'workflows':[{'id':'work','room_sequence':['a','b']}],'exceptions':[]}
        self.revision=site.save_site(self.db,self.doc)
        site.bind_clock(self.db,'Dock','Dock','2026-09-18T11:00:00+05:30','simulated')
        site.bind_clock(self.db,'Transfer','Transfer','2026-09-18T11:01:00+05:30','simulated')
        self.zone={'name':'Lane','shape':'polygon','points':[[0,0],[99,0],[0,99]],'box':[0,0,99,99],
                   'metadata':{'kind':'floor_area','description':'Main approach','category':'lane'}}
        self.patch=patch.object(cctv_tools,'DB_PATH',self.db)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def save_map(self,camera='Dock',reviewed=True):
        data=editor.load_saved_camera(camera,self.db)
        return editor.save_camera_zones({'camera_name':camera,'zones':[self.zone],
            'expected_revision':data['revision'],'reviewed':reviewed},self.db)

    def timeline(self,cutoff='2026-09-18T11:00:02+05:30'):
        return site.site_timeline(self.db,'2026-09-18T11:00:00+05:30','2026-09-18T11:02:00+05:30',cutoff)

    def test_naive_time_rejected(self):
        with self.assertRaises(ValueError):site.parse_time('2026-09-18T11:00:00')

    def test_invalid_site_rejected(self):
        for change in [{'rooms':['bad']},{'cameras':[{'name':'x','room_id':'missing'}]},
                       {'routes':[{'id':'x','from_room':'a','to_room':'b','min_seconds':80,'max_seconds':2}]}]:
            with self.assertRaises(ValueError):site.validate_site({**self.doc,**change})

    def test_site_stale_edit_rejected(self):
        with self.assertRaises(ValueError):site.save_site(self.db,self.doc,'old')

    def test_invalid_timezone_rejected(self):
        with self.assertRaises(ValueError):site.validate_site({**self.doc,'timezone':'bad-zone'})

    def test_background_is_not_observed_activity(self):
        result=json.loads(site_tools.get_site_context())
        self.assertEqual(result['evidence_class'],'supplied_background_NOT_observed_activity')
        result=json.loads(site_tools.get_site_timeline('2026-09-18T11:00:00+05:30','2026-09-18T11:00:02+05:30'))
        self.assertIn('Inspection or task completion',result['does_not_measure'])

    def test_duplicate_recording_cannot_double_count(self):
        with closing(sqlite3.connect(self.db)) as conn,conn:
            conn.execute("INSERT INTO tracking_runs(run_id,camera_id,source_path,source_kind,config_json,status) "
                         "SELECT 'repeat',camera_id,source_path,source_kind,config_json,status FROM tracking_runs WHERE run_id='Dock'")
        with self.assertRaises(ValueError):site.bind_clock(self.db,'Dock','repeat','2026-09-18T11:00:00+05:30','simulated')

    def test_simulated_clock_cannot_be_claimed_real(self):
        with self.assertRaises(ValueError):site.bind_clock(self.db,'Dock','Dock','2026-09-18T11:00:00+05:30','camera_clock')

    def test_source_zero_clock_handles_trimmed_run(self):
        with closing(sqlite3.connect(self.db)) as conn,conn:
            config=json.loads(conn.execute("SELECT config_json FROM tracking_runs WHERE run_id='Dock'").fetchone()[0])
            config.update(start_sec=2,end_sec=4)
            conn.execute("UPDATE tracking_runs SET config_json=? WHERE run_id='Dock'",(json.dumps(config),))
        self.assertEqual(site.recording_schedule(self.db)[0]['start_at'],'2026-09-18T11:00:02+05:30')

    def test_no_future_people_or_cameras(self):
        result=self.timeline()
        self.assertEqual([r['camera'] for r in result['recordings']],['Dock'])
        self.assertEqual(result['recordings'][0]['measurements']['processed_frames'],2)
        self.assertNotIn('subject_future',json.dumps(result))

    def test_before_recording_has_no_events(self):
        result=self.timeline('2026-09-18T10:59:59+05:30')
        self.assertEqual(result['events'],[])
        self.assertEqual(result['recordings'],[])

    def test_available_report_uses_latest_data_not_wall_clock(self):
        window=site.available_report_window(self.db)
        self.assertEqual(site.parse_time(window['as_of']),site.parse_time('2026-09-18T11:01:04+05:30'))
        self.assertEqual(window['time_basis'],'simulated')
        self.assertFalse(site.site_context(self.db)['ready_for_analysis'])

    def test_future_schedule_without_observations_does_not_extend_report(self):
        with closing(sqlite3.connect(self.db)) as conn,conn:
            conn.execute("DELETE FROM frame_observations WHERE run_id='Transfer'")
        window=site.available_report_window(self.db)
        self.assertEqual(site.parse_time(window['as_of']),site.parse_time('2026-09-18T11:00:04+05:30'))

    def test_available_report_rejects_empty_data(self):
        with closing(sqlite3.connect(self.db)) as conn,conn:
            conn.execute('DELETE FROM frame_observations')
        with self.assertRaises(ValueError):site.available_report_window(self.db)

    def test_unreviewed_visual_excludes_map(self):
        with patch('openai.OpenAI'),patch.object(cctv_tools,'_inspect_workflow_frames',return_value='observed') as inspect:
            site_tools.inspect_site_interval('Dock','2026-09-18T11:00:00+05:30',
                '2026-09-18T11:00:02+05:30','Check movement')
            self.assertFalse(inspect.call_args.kwargs['include_map'])

    def test_gaps_not_added_as_empty_observations(self):
        result=site.site_timeline(self.db,'2026-09-18T11:00:10+05:30','2026-09-18T11:00:50+05:30')
        self.assertEqual(result['recordings'],[])
        self.assertIn('unknown',result['meaning'])

    def test_alternate_routes_and_uncovered_rooms(self):
        result=site.site_routes(self.db,'a','b')
        self.assertEqual(len(result['routes']),2)
        self.assertTrue(any('blind' in r['unobserved_rooms'] for r in result['routes']))
        self.assertEqual(len(site.site_routes(self.db,'b','a')['routes']),1)

    def test_dated_closure_only_applies_when_active(self):
        self.doc['exceptions']=[{'id':'closure','start_at':'2026-09-18T11:00:00+05:30',
            'end_at':'2026-09-18T12:00:00+05:30','closed_routes':['direct']}]
        site.save_site(self.db,self.doc,self.revision)
        self.assertEqual(len(site.site_routes(self.db,'a','b','2026-09-18T11:30:00+05:30')['routes']),1)
        self.assertEqual(len(site.site_routes(self.db,'a','b','2026-09-18T12:00:00+05:30')['routes']),2)

    def test_comparison_is_not_identity(self):
        result=site.compare_events(self.db,'Dock','2026-09-18T11:00:00+05:30','Transfer','2026-09-18T11:01:00+05:30')
        self.assertIn('does not match',result['conclusion'])
        self.assertTrue(all(r['within_estimated_travel_range'] for r in result['routes']))

    def test_comparison_cannot_look_ahead(self):
        with self.assertRaises(ValueError):site.compare_events(self.db,'Dock','2026-09-18T11:00:00+05:30',
            'Transfer','2026-09-18T11:01:00+05:30','2026-09-18T11:00:30+05:30')

    def test_polygon_metadata_and_video_source_preserved(self):
        self.save_map()
        loaded=editor.load_saved_camera('Dock',self.db)
        self.assertEqual(loaded['zones'][0]['points'],self.zone['points'])
        self.assertEqual(loaded['zones'][0]['metadata']['description'],'Main approach')
        self.assertEqual(loaded['camera']['source_path'],str(self.image))
        self.assertEqual(loaded['revision'],build_knowledge(get_observations(self.db,'Dock','Dock'))['scope']['map_version'])
        self.assertTrue(site.map_review(self.db,'Dock')['reviewed'])

    def test_map_change_invalidates_review(self):
        self.save_map();self.save_map(reviewed=False)
        self.assertFalse(site.map_review(self.db,'Dock')['reviewed'])

    def test_map_stale_save_is_atomic(self):
        self.save_map()
        with self.assertRaises(ValueError):editor.save_camera_zones({'camera_name':'Dock','zones':[],
                                                                   'expected_revision':'old'},self.db)
        self.assertTrue(site.map_review(self.db,'Dock')['reviewed'])

    def test_invalid_polygon_rejected(self):
        for pts in [[[0,0],[99,99],[99,0],[0,99]], [[0,0],[101,0],[0,99]]]:
            with self.assertRaises(ValueError):editor.normalize_zones([{**self.zone,'points':pts}],100,100)

    def test_draft_area_events_withheld_then_recomputed(self):
        self.save_map(reviewed=False)
        self.assertFalse(any(e['kind']=='observed_in_area' for e in self.timeline()['events']))
        self.save_map()
        result=self.timeline()
        self.assertTrue(any(e['kind']=='observed_in_area' for e in result['events']))
        self.assertIn('floor_brief',result['recordings'][0])

    def test_replay_blocks_unbounded_tools(self):
        token=site.REPLAY_AS_OF.set('2026-09-18T11:00:02+05:30')
        try:
            self.assertIn('ERROR',cctv_agent.call_tool('get_workflow_brief',{'camera_name':'Dock','run_id':'Dock'}))
            result=json.loads(site_tools.get_site_timeline('2026-09-18T11:00:00+05:30','2026-09-18T11:02:00+05:30',
                                                         '2026-09-18T12:00:00+05:30'))
            self.assertNotIn('subject_future',json.dumps(result))
            self.assertIn('ERROR',cctv_agent.call_tool('get_site_context',{'at_time':'2026-09-18T12:00:00+05:30'}))
        finally:site.REPLAY_AS_OF.reset(token)

    def test_replay_discards_later_history_and_resets_context(self):
        with patch.object(cctv_agent,'run_agent_turn_openai',return_value=('test',[])) as run:
            cctv_agent.run_agent_turn([{'content':'future secret'}],'Assess',as_of='2026-09-18T11:00:02+05:30')
            self.assertEqual(run.call_args.args[0],[])
        self.assertIsNone(site.REPLAY_AS_OF.get())

    def test_visual_selection_and_context_bounded(self):
        self.save_map()
        with patch('openai.OpenAI'),patch.object(cctv_tools,'_inspect_workflow_frames',return_value='observed') as inspect:
            result=json.loads(site_tools.inspect_site_interval('Dock','2026-09-18T11:00:00+05:30',
                '2026-09-18T11:00:04+05:30','Check movement','2026-09-18T11:00:02+05:30'))
            self.assertEqual(inspect.call_args.args[2],[0,1])
            self.assertEqual(inspect.call_args.kwargs['context_end_sec'],2)
            self.assertEqual(len(result['evidence'][0]['event_times']),2)

    def test_site_tools_registered_both_providers(self):
        self.assertTrue(site.SITE_TOOLS<={t['function']['name'] for t in cctv_agent.OPENAI_TOOLS})
        self.assertTrue(site.SITE_TOOLS<={t.name for t in cctv_agent.GEMINI_TOOL_DECLARATIONS})

    def test_api_requires_review_and_rejects_unknown_camera(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),web.AgentWebHandler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with patch.object(cctv_agent,'run_agent_turn',return_value=('Recorded activity only.',[])) as agent:
                request=urllib.request.Request(base+'/api/chat',data=json.dumps({'scope':'available',
                    'message':'What happened as of now?'}).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request) as response:
                    events=[json.loads(line) for line in response.read().splitlines()]
                self.assertTrue(any(e['type']=='coverage' for e in events))
                self.assertEqual(site.parse_time(agent.call_args.kwargs['as_of']),site.parse_time('2026-09-18T11:01:04+05:30'))
                self.assertIn('across all cameras',agent.call_args.args[1])
            for path,payload in [('/api/chat',{'scope':'site','message':'Assess','as_of':'2026-09-18T11:00:02+05:30'}),
                                 ('/api/site/map',{'camera_name':'unknown','zones':[],'expected_revision':'x'})]:
                request=urllib.request.Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
                self.assertEqual(error.exception.code,400);error.exception.close()
            self.save_map();self.save_map('Transfer')
            with patch.object(cctv_agent,'run_agent_turn',return_value=('No confirmed issue.',[])) as agent:
                request=urllib.request.Request(base+'/api/chat',data=json.dumps({'scope':'site','message':'Assess',
                    'as_of':'2026-09-18T11:00:02+05:30','session_id':'old'}).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request) as response:self.assertIn(b'No confirmed',response.read())
                self.assertEqual(agent.call_args.kwargs['as_of'],'2026-09-18T11:00:02+05:30')
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
