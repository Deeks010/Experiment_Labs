"""One evidence vocabulary for measurements, maps, frames and agent briefs."""
from __future__ import annotations

import hashlib
import json

from workflow_observations import inside


def scope_id(camera, run_id):
    return f"{camera}/{run_id}"


def map_version(zones):
    payload = sorted(zones, key=lambda z: z['name'])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def frame_ref(camera, run_id, frame_index):
    return f"{scope_id(camera, run_id)}/frame/{frame_index}"


def build_knowledge(summary):
    scope = scope_id(summary['camera'], summary['run_id'])
    zones = summary['zones']
    times = [row['time_sec'] for row in summary['count_timeline']]
    frame_times = list(dict.fromkeys([times[0], times[len(times)//2], times[-1]])) if times else []
    people = []
    events = []

    def event(person, kind, start, end, **details):
        content = [scope, map_version(zones), person, kind, start, end, details]
        suffix = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
        item = {'evidence_ref': f'{scope}/event/{suffix}', 'person_ref': person,
                'kind': kind, 'start_sec': start, 'end_sec': end, **details}
        events.append(item)

    for person in summary['people']:
        ref = person['subject_ref']
        path = person['path']
        endpoints = []
        for point in ([path[0], path[-1]] if path else []):
            endpoints.append({'time_sec': point['t'], 'image_xy': [round(v, 1) for v in point['foot']],
                'floor_areas': [z['name'] for z in zones if z['kind'] == 'floor_area'
                                and inside(point['foot'], z['geometry'])]})
        people.append({'person_ref': ref, 'scoped_person_ref': f'{scope}/person/{ref}',
            'observed_sec': person['observed_sec'], 'low_motion_sec': person['low_motion_sec'],
            'moving_sec': person['moving_sec'], 'samples': person['samples'],
            'missing_samples_within_span': person['unobserved_samples_within_span'],
            'first_and_last_observation': endpoints})
        for start, end in person['low_motion_intervals']:
            event(ref, 'low_motion', start, end)
        for name, intervals in person['zone_visits'].items():
            zone = next(z for z in zones if z['name'] == name)
            if zone['kind'] == 'floor_area':
                for start, end in intervals:
                    event(ref, 'observed_in_area', start, end, area_ref=name)
    return {'schema': 'floor-evidence/v1', 'scope': {'scope_id': scope,
        'camera': summary['camera'], 'run_id': summary['run_id'], 'period': summary['period'],
        'source_kind': summary['source_kind'], 'status': summary['status'],
        'map_version': map_version(zones), 'suggested_frame_times_sec': frame_times,
        'window_end_exclusive': True, 'coordinate_system': 'camera image: x right, y down; pixels, not metres',
        'identity_scope': 'camera and run only; same label in another scope is unrelated'},
        'measurements': {'processed_frames': summary['processed_frames'],
            'peak_detected_people': summary['peak_detected_people'],
            'local_track_labels': summary['local_track_labels'],
            'frames_with_no_people_detected': summary['frames_without_detections'],
            'missing_person_samples_within_spans': sum(p['missing_samples_within_span'] for p in people)},
        'areas': summary['areas'], 'people': people,
        'events': sorted(events, key=lambda e: (e['start_sec'], e['person_ref'], e['kind'])),
        'limits': summary['limitations']}


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def knowledge_markdown(knowledge, max_events=60):
    scope = knowledge['scope']
    counts = knowledge['measurements']
    p = scope['period']
    lines = [f"# Floor evidence: {scope['camera']}",
        f"Scope: {scope['scope_id']} | {p['start_sec']:g}-{p['end_sec']:g}s | {scope['source_kind']} | {scope['status']}",
        f"Map version: {scope['map_version']}",
        f"Available example frame times (seconds): {scope['suggested_frame_times_sec']}. Use these for visual requests. The reporting end is exclusive, not a frame timestamp.",
        "Evidence state: measurements only. Saving a proof frame does not inspect its contents; a successful visual tool result is needed for visual claims.",
        "Person labels below are identical to labels on proof frames. They are not employee identities.",
        f"Coordinates: {scope['coordinate_system']}. Map boundaries are draft camera-image geometry.",
        "", "## Measurement meanings",
        "- observed_in_area: detected foot point inside a named floor area, not proof of a task.",
        "- low_motion: little image movement, not waiting, idleness or absence.",
        "- missing person sample: this label was not detected between its first and last sightings; cause unknown.",
        "- no-people frame: zero detections, not proof the floor was empty.",
        "- visual interpretation: a model's reading of frames, not a replacement for measurements.",
        "", "## Counts and gaps",
        f"Processed frames: {counts['processed_frames']}; peak detected: {counts['peak_detected_people']}; local labels: {counts['local_track_labels']}.",
        f"Frames with no people detected: {counts['frames_with_no_people_detected']}.",
        f"Missing person samples within tracked spans: {counts['missing_person_samples_within_spans']} (a DIFFERENT measure).",
        "", "## People", "| Person reference | Observed s | Low motion s | Moving s | Missing samples |",
        "|---|---:|---:|---:|---:|"]
    for person in knowledge['people']:
        lines.append(f"| {_cell(person['person_ref'])} | {person['observed_sec']} | {person['low_motion_sec']} | {person['moving_sec']} | {person['missing_samples_within_span']} |")
    lines += ["", "## Start and end positions (not a continuous path)"]
    for person in knowledge['people']:
        ends = person['first_and_last_observation']
        for label, point in zip(('First', 'Last'), ends):
            areas = ', '.join(point['floor_areas']) or 'outside named floor areas'
            lines.append(f"- {person['person_ref']} {label}: {point['time_sec']:g}s; {areas}; image x,y={point['image_xy']}.")
    lines += ["", "## Floor map legend", "| Area reference | Kind | Peak detected | Occupied s |",
              "|---|---|---:|---:|"]
    for area in knowledge['areas']:
        lines.append(f"| {_cell(area['name'])} | {area['kind']} | {area['peak_detected']} | {area['occupied_sec']} |")
    lines += ["", "## Measured intervals", "| Evidence reference | Person | Observation | Area | Seconds |",
              "|---|---|---|---|---|"]
    for e in knowledge['events'][:max_events]:
        lines.append(f"| {_cell(e['evidence_ref'])} | {_cell(e['person_ref'])} | {e['kind']} | {_cell(e.get('area_ref', '-'))} | {e['start_sec']:g}-{e['end_sec']:g} |")
    if len(knowledge['events']) > max_events:
        lines.append(f"Showing {max_events} of {len(knowledge['events'])} intervals. Request a narrower window for the rest.")
    lines += ["", "## Unknown / not established"] + [f'- {limit}' for limit in knowledge['limits']]
    return '\n'.join(lines)
