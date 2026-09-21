"""Operator-supplied place knowledge and event-time queries, independent of a scenario."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from workflow_observations import get_observations, select_run
from workflow_knowledge import build_knowledge

REPLAY_AS_OF = ContextVar('site_replay_as_of', default=None)
SITE_TOOLS = {'get_site_context', 'get_site_routes', 'get_site_timeline',
              'compare_site_events', 'inspect_site_interval'}


def parse_time(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Event times need a timezone offset, for example +05:30.')
    return result


def init_site_tables(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS site_context (
            id INTEGER PRIMARY KEY CHECK(id=1), revision TEXT NOT NULL, document_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS recording_clocks (
            run_id TEXT PRIMARY KEY, source_zero_at TEXT NOT NULL, provenance TEXT NOT NULL,
            uncertainty_sec REAL NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS camera_map_reviews (
            camera_name TEXT PRIMARY KEY, map_version TEXT NOT NULL, reviewed_at TEXT NOT NULL);
    ''')


def digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:16]


def validate_site(document):
    if not isinstance(document, dict) or document.get('schema_version') != 1:
        raise ValueError('Unsupported site document.')
    if not str(document.get('name', '')).strip():
        raise ValueError('A site name is required.')
    document = copy.deepcopy(document)
    try:
        ZoneInfo(document.get('timezone', 'UTC'))
    except (ZoneInfoNotFoundError, TypeError) as exc:
        raise ValueError('Use a valid site timezone, such as Asia/Kolkata.') from exc
    collections = {}
    for field, key in [('rooms','id'), ('cameras','name'), ('routes','id'), ('workflows','id'), ('exceptions','id')]:
        items = document.get(field, [])
        if not isinstance(items, list) or len(items) > 500:
            raise ValueError(f'Invalid {field} list.')
        if any(not isinstance(item, dict) for item in items):
            raise ValueError(f'Invalid {field} entry.')
        ids = [item.get(key) for item in items]
        if any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{field} must have unique nonempty {key} values.')
        collections[field] = set(ids)
    rooms, cameras, routes = (collections[k] for k in ('rooms','cameras','routes'))
    if not rooms:
        raise ValueError('At least one room or area is required.')
    for room in document['rooms']:
        room.setdefault('name', room['id'])
        if not isinstance(room['name'], str) or not room['name'].strip():
            raise ValueError('Room names must be nonempty text.')
        position = room.get('position', [.5, .5])
        if (not isinstance(position, list) or len(position) != 2 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in position)):
            raise ValueError('Room positions need two numbers between zero and one.')
    for camera in document.get('cameras', []):
        if camera.get('room_id') not in rooms:
            raise ValueError('Camera references an unknown room.')
        if not isinstance(camera.get('coverage', ''), str):
            raise ValueError('Camera coverage must be text.')
    for route in document.get('routes', []):
        if route.get('from_room') not in rooms or route.get('to_room') not in rooms:
            raise ValueError('Route references an unknown room.')
        lower, upper = route.get('min_seconds'), route.get('max_seconds')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (lower, upper)) or not 0 <= lower <= upper:
            raise ValueError('Route travel-time estimates must be finite and ordered.')
        if not isinstance(route.get('bidirectional', False), bool):
            raise ValueError('bidirectional must be true or false.')
    for workflow in document.get('workflows', []):
        for field in ('room_sequence','steps','expected_checks'):
            if not isinstance(workflow.get(field, []), list) or any(not isinstance(v,str) for v in workflow.get(field, [])):
                raise ValueError(f'Workflow {field} must be a list of text values.')
        if any(room not in rooms for room in workflow.get('room_sequence', [])):
            raise ValueError('Workflow references an unknown room.')
    for exception in document.get('exceptions', []):
        for field in ('closed_routes','affected_workflows'):
            if not isinstance(exception.get(field, []), list):
                raise ValueError(f'Exception {field} must be a list.')
        if parse_time(exception.get('start_at')) >= parse_time(exception.get('end_at')):
            raise ValueError('Exception end must follow its start.')
        if any(route not in routes for route in exception.get('closed_routes', [])):
            raise ValueError('Exception references an unknown route.')
        if any(w not in collections['workflows'] for w in exception.get('affected_workflows', [])):
            raise ValueError('Exception references an unknown workflow.')
    return copy.deepcopy(document)


def save_site(db, document, expected_revision=None):
    document = validate_site(document)
    revision = digest(document)
    with closing(sqlite3.connect(db)) as conn:
        init_site_tables(conn)
        with conn:
            conn.execute('BEGIN IMMEDIATE')
            current = conn.execute('SELECT revision FROM site_context WHERE id=1').fetchone()
            if current and expected_revision != current[0]:
                raise ValueError('Site changed since loading. Reload before saving.')
            conn.execute('INSERT OR REPLACE INTO site_context VALUES (1,?,?)', (revision, json.dumps(document)))
    return revision


def load_site(db):
    with closing(sqlite3.connect(db)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='site_context'").fetchone():
            return None
        row = conn.execute('SELECT revision,document_json FROM site_context WHERE id=1').fetchone()
    return {'revision': row[0], 'document': json.loads(row[1])} if row else None


def bind_clock(db, camera, run_id, source_zero_at, provenance, uncertainty_sec=0):
    start = parse_time(source_zero_at)
    if provenance not in {'simulated','operator_declared','camera_clock'}:
        raise ValueError('Unknown clock provenance.')
    if not math.isfinite(uncertainty_sec) or uncertainty_sec < 0:
        raise ValueError('Clock uncertainty must be nonnegative.')
    with closing(sqlite3.connect(db)) as conn:
        conn.row_factory = sqlite3.Row
        init_site_tables(conn)
        conn.execute('BEGIN IMMEDIATE')
        run = select_run(conn, camera, run_id)
        if run['source_kind'] == 'synthetic' and provenance != 'simulated':
            raise ValueError('Generated footage must have a simulated clock.')
        config = json.loads(run['config_json'])
        begin, finish = (start+timedelta(seconds=config[key]) for key in ('start_sec','end_sec'))
        for other in recording_schedule(db):
            if other['camera'] == camera and other['run_id'] != run_id and (
                    parse_time(other['start_at']) < finish and parse_time(other['end_at']) > begin):
                raise ValueError('This camera already has a recording at that time. Choose one run, not duplicate observations.')
        with conn:
            conn.execute('INSERT OR REPLACE INTO recording_clocks VALUES (?,?,?,?)',
                         (run_id, start.isoformat(), provenance, uncertainty_sec))


def recording_schedule(db):
    with closing(sqlite3.connect(db)) as conn:
        conn.row_factory = sqlite3.Row
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='recording_clocks'").fetchone():
            return []
        rows = conn.execute('''SELECT k.*, r.config_json,r.status,r.source_kind,c.name AS camera
            FROM recording_clocks k JOIN tracking_runs r ON r.run_id=k.run_id
            JOIN cameras c ON c.id=r.camera_id''').fetchall()
    result = []
    for row in rows:
        item = dict(row)
        config = json.loads(item.pop('config_json'))
        zero = parse_time(item['source_zero_at'])
        item.update(start_at=(zero+timedelta(seconds=config['start_sec'])).isoformat(),
                    end_at=(zero+timedelta(seconds=config['end_sec'])).isoformat())
        result.append(item)
    return sorted(result, key=lambda r: parse_time(r['start_at']))


def available_report_window(db, now=None):
    """Latest observed site day, not processing time or future scheduled footage."""
    saved = load_site(db)
    if not saved:
        raise ValueError('Site knowledge is unavailable.')
    cameras = {c['name'] for c in saved['document']['cameras']}
    now = now or datetime.now(timezone.utc)
    ends, provenance = [], set()
    with closing(sqlite3.connect(db)) as conn:
        for recording in recording_schedule(db):
            if recording['camera'] not in cameras:
                continue
            zero = parse_time(recording['source_zero_at'])
            limit = parse_time(recording['end_at'])
            if recording['provenance'] != 'simulated':
                limit = min(limit, now)
            row = conn.execute('''SELECT MAX(f.time_sec), r.config_json
                FROM tracking_runs r JOIN frame_observations f ON f.run_id=r.run_id
                WHERE r.run_id=? AND f.time_sec<?''',
                (recording['run_id'], (limit-zero).total_seconds())).fetchone()
            if row[0] is None:
                continue
            config = json.loads(row[1])
            ends.append(min(limit, zero+timedelta(seconds=row[0]+config['frame_stride']/config['fps'])))
            provenance.add(recording['provenance'])
    if not ends:
        raise ValueError('No recorded observations are available yet.')
    if 'simulated' in provenance and len(provenance) > 1:
        raise ValueError('Simulated and real-clock recordings need separate report requests.')
    end = max(ends).astimezone(ZoneInfo(saved['document'].get('timezone','UTC')))
    start = (end-timedelta(microseconds=1)).replace(hour=0,minute=0,second=0,microsecond=0)
    return {'start_at':start.isoformat(), 'as_of':end.isoformat(),
            'time_basis':'simulated' if provenance == {'simulated'} else 'recorded',
            'meaning':'Latest day with existing observations. Missing time is unknown, not a complete shift.'}


def map_review(db, camera):
    from map_editor_server import load_saved_camera, map_revision
    # The editor functions take an explicit database for this shared, request-safe path.
    data = load_saved_camera(camera, db_path=db)
    if data is None:
        return {'reviewed': False, 'reason': 'Camera has no map.'}
    revision = map_revision(data['zones'])
    with closing(sqlite3.connect(db)) as conn:
        row = None
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='camera_map_reviews'").fetchone():
            row = conn.execute('SELECT map_version,reviewed_at FROM camera_map_reviews WHERE camera_name=?', (camera,)).fetchone()
    return {'reviewed': bool(row and row[0] == revision), 'map_version': revision,
            'reviewed_at': row[1] if row and row[0] == revision else None}


def active_exceptions(document, at_time):
    if at_time is None:
        return []
    now = parse_time(at_time)
    return [e for e in document.get('exceptions', []) if parse_time(e['start_at']) <= now < parse_time(e['end_at'])]


def site_context(db, at_time=None):
    saved = load_site(db)
    if not saved:
        return {'configured': False, 'next_step': 'Provide site rooms, cameras, routes and workflow expectations.'}
    doc = saved['document']
    camera_names = {c['name'] for c in doc.get('cameras', [])}
    reviews = {name: map_review(db, name) for name in camera_names}
    from map_editor_server import load_saved_camera
    maps = {}
    for name in camera_names:
        data = load_saved_camera(name, db_path=db)
        maps[name] = [{'name':z['name'], 'geometry':{k:z.get(k) for k in ('shape','points','box')},
                       **{k:z.get('metadata',{}).get(k,'') for k in ('kind','description','category')}}
                      for z in data['zones']] if data else []
    return {'configured': True, **saved, 'at_time': at_time, 'active_exceptions': active_exceptions(doc, at_time),
            'recording_schedule': [r for r in recording_schedule(db) if r['camera'] in camera_names],
            'map_reviews': reviews, 'ready_for_analysis': bool(reviews) and all(r['reviewed'] for r in reviews.values()),
            'camera_maps': maps,
            'unobserved_rooms': [r['id'] for r in doc['rooms'] if not any(c['room_id']==r['id'] for c in doc['cameras'])],
            'limits': ['Site connections and operating expectations are supplied knowledge, not inferred from video.',
                       'Scheduled recordings describe availability, not observed events or worker identity.',
                       'No cross-camera identity or causal link has been established.']}


def site_routes(db, from_room, to_room, at_time=None):
    saved = load_site(db)
    if not saved:
        raise ValueError('Site not configured.')
    doc = saved['document']
    rooms = {r['id'] for r in doc['rooms']}
    if from_room not in rooms or to_room not in rooms:
        raise ValueError('Unknown room.')
    exceptions = active_exceptions(doc, at_time)
    closed = {route for e in exceptions for route in e.get('closed_routes', [])}
    adjacency = {room: [] for room in rooms}
    for route in doc['routes']:
        if route['id'] in closed:
            continue
        adjacency[route['from_room']].append((route['to_room'], route))
        if route.get('bidirectional'):
            adjacency[route['to_room']].append((route['from_room'], route))
    results, stack, truncated = [], [(from_room, [from_room], [])], False
    while stack:
        current, visited, edges = stack.pop()
        if current == to_room:
            results.append({'rooms': visited, 'route_ids': [e['id'] for e in edges],
                            'min_seconds': sum(e['min_seconds'] for e in edges),
                            'max_seconds': sum(e['max_seconds'] for e in edges),
                            'unobserved_rooms': [r for r in visited if not any(c['room_id']==r for c in doc['cameras'])]})
            if len(results) >= 32:
                truncated = bool(stack)
                break
        elif len(visited) < 12:
            for destination, route in adjacency[current]:
                if destination not in visited:
                    stack.append((destination, visited+[destination], edges+[route]))
        elif adjacency[current]:
            truncated = True
    return {'from_room': from_room, 'to_room': to_room, 'at_time': at_time, 'routes': results,
            'closed_routes': sorted(closed), 'truncated': truncated, 'site_revision': saved['revision'],
            'provenance': doc.get('provenance', 'operator_supplied'),
            'meaning': 'Possible configured routes, not evidence anyone took one. Travel times are supplied estimates.'}


def site_timeline(db, start_time, end_time, as_of=None):
    start, end = parse_time(start_time), parse_time(end_time)
    if end <= start or (end-start).total_seconds() > 86400:
        raise ValueError('Choose a positive time window of at most 24 hours.')
    cutoff = min(end, parse_time(as_of)) if as_of else end
    saved = load_site(db)
    if not saved:
        raise ValueError('Site not configured.')
    camera_map = {c['name']: c for c in saved['document']['cameras']}
    slices, events = [], []
    for recording in recording_schedule(db):
        camera = recording['camera']
        if camera not in camera_map:
            continue
        begin, finish = max(start, parse_time(recording['start_at'])), min(cutoff, parse_time(recording['end_at']))
        if finish <= begin:
            continue
        zero = parse_time(recording['source_zero_at'])
        review = map_review(db, camera)
        summary = get_observations(db, camera, recording['run_id'], (begin-zero).total_seconds(), (finish-zero).total_seconds())
        summary['limitations'] = [v for v in summary['limitations'] if not v.startswith('Separate clips')]
        summary['limitations'].append(f"Site clock origin: {recording['provenance']}. Assigned time does not establish continuity or cross-camera identity.")
        knowledge = build_knowledge(summary)
        slices.append({'camera': camera, 'room_id': camera_map[camera]['room_id'], 'run_id': recording['run_id'],
            'start_at': begin.isoformat(), 'end_at': finish.isoformat(), 'status': recording['status'],
            'clock_provenance': recording['provenance'], 'clock_uncertainty_sec': recording['uncertainty_sec'],
            'map_review': review, 'measurements': knowledge['measurements'],
            'people': [{k:v for k,v in p.items() if k!='first_and_last_observation'} for p in knowledge['people']]})
        if review['reviewed']:
            from workflow_knowledge import knowledge_markdown
            from cctv_tools import render_workflow_map
            slices[-1]['floor_brief'] = knowledge_markdown(knowledge) + '\n\n' + render_workflow_map(summary)
        for event in knowledge['events']:
            if event['kind'] == 'observed_in_area' and not review['reviewed']:
                continue
            events.append({**event, 'camera': camera, 'room_id': camera_map[camera]['room_id'],
                'start_at': (zero+timedelta(seconds=event['start_sec'])).isoformat(),
                'end_at': (zero+timedelta(seconds=event['end_sec'])).isoformat(),
                'clock_provenance': recording['provenance'], 'map_version': knowledge['scope']['map_version']})
    events.sort(key=lambda e: (parse_time(e['start_at']),e['camera'],e['person_ref']))
    return {'start_at': start.isoformat(), 'end_at': end.isoformat(), 'visible_before': cutoff.isoformat(),
            'site_revision': saved['revision'], 'recordings': slices, 'events': events[:500],
            'truncated': len(events)>500,
            'meaning': 'Observed intervals only. Unrecorded time and rooms are unknown, not empty. No cross-camera identity.',
            'mapping_policy': 'Area-membership events are withheld until this camera map is reviewed.'}


def compare_events(db, from_camera, from_time, to_camera, to_time, as_of=None):
    start, end = parse_time(from_time), parse_time(to_time)
    if end < start or (as_of and end > parse_time(as_of)):
        raise ValueError('Comparison must be chronological and not beyond the replay cursor.')
    saved = load_site(db)
    if not saved:
        raise ValueError('Site not configured.')
    doc = saved['document']
    cameras = {c['name']: c for c in doc['cameras']}
    if from_camera not in cameras or to_camera not in cameras:
        raise ValueError('Unknown camera.')
    routes = site_routes(db, cameras[from_camera]['room_id'], cameras[to_camera]['room_id'], from_time)
    elapsed = (end-start).total_seconds()
    exceptions = [e for e in doc.get('exceptions', [])
        if parse_time(e['start_at']) <= end and parse_time(e['end_at']) > start]
    for route in routes['routes']:
        route['within_estimated_travel_range'] = route['min_seconds'] <= elapsed <= route['max_seconds']
        route['closure_during_window'] = any(set(route['route_ids']) & set(e.get('closed_routes', [])) for e in exceptions)
    covered = {}
    for camera, instant in ((from_camera,start),(to_camera,end)):
        covered[camera] = [r['run_id'] for r in recording_schedule(db) if r['camera']==camera
                           and parse_time(r['start_at']) <= instant < parse_time(r['end_at'])]
    return {**routes, 'elapsed_seconds': elapsed, 'recordings_at_requested_times': covered,
            'conclusion': 'Route/time compatibility only. It does not match a person, load, event or cause across cameras.',
            'exceptions_during_window': exceptions,
            'limits': 'Estimated ranges do not model stops or clock uncertainty. A closure during the window requires review.'}
