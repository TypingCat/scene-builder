# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

DATASPEC = MappingProxyType({
    'num_frames': 100,
    'frame_unit': 0.1,
    'num_agents': 128,
    'num_links': 512,
    'num_link_points': 128,
    'num_link_connections': 4,
    'num_link_adjacents': 8,
    'num_link_boundaries': 8,
    'num_lanes': 128,
    'num_lane_points': 128,
    'num_markers': 64,
    'num_marker_points': 16,
    'num_traffics': 32,
    'num_paths': 10,
    'num_path_points': 300,
    'num_agent_types': 4,
    'num_traffic_types': 4,
    'num_traffic_states': 4,
    'num_link_types': 4,
    'num_link_sections': 6,
    'num_lane_types': 7,
    'num_marker_types': 9,
})

DS = SimpleNamespace(**DATASPEC)

LINK_TRUST_RADIUS_M = 5.0

SYMBOLS = MappingProxyType({
    'A': DS.num_agents, 'F': DS.num_frames,
    'I': DS.num_links, 'I_points': DS.num_link_points,
    'I_connections': DS.num_link_connections, 'I_adjacents': DS.num_link_adjacents,
    'I_boundaries': DS.num_link_boundaries,
    'L': DS.num_lanes, 'L_points': DS.num_lane_points,
    'M': DS.num_markers, 'M_points': DS.num_marker_points,
    'T': DS.num_traffics,
    'P': DS.num_paths, 'P_points': DS.num_path_points,
})


class AgentType:
    unknown = 0
    vehicle = 1
    pedestrian = 2
    cyclist = 3


class TrafficType:
    unknown = 0
    vehicle = 1
    pedestrian = 2
    cyclist = 3


class TrafficState:
    unknown = 0
    go = 1
    caution = 2
    stop = 3


class LinkType:
    unknown = 0
    vehicle = 1
    pedestrian = 2
    cyclist = 3


class LinkSection:
    unknown = 0
    intersection = 1
    school_zone = 2
    tunnel = 3
    hi_pass = 4
    freeway = 5


class LaneType:
    unknown = 0
    center = 1
    solid = 2
    dashed = 3
    edge = 4
    boundary = 5
    median = 6


class MarkerType:
    unknown = 0
    stop_sign = 1
    stop_line = 2
    crosswalk = 3
    speed_bump = 4
    driveway = 5
    bus_stop = 6
    safe_point = 7
    arrow = 8


GROUPS = ('agents', 'links', 'lanes', 'traffics', 'markers', 'paths', 'goal', 'meta')

FIELDS = MappingProxyType({
    'agents': {
        'id': (('A',), 'int32'),
        'type': (('A',), 'int32'),
        'is_ego': (('A',), 'bool'),
        'size_xyz': (('A', 3), 'float32'),
        'time_s': (('A', 'F', 1), 'float32'),
        'pos_xy': (('A', 'F', 2), 'float32'),
        'dir_xy': (('A', 'F', 2), 'float32'),
        'vel_xy': (('A', 'F', 2), 'float32'),
        'acc_xy': (('A', 'F', 2), 'float32'),
        'valid': (('A', 'F'), 'bool'),
    },
    'links': {
        'id': (('I',), 'int32'),
        'type': (('I',), 'int32'),
        'section': (('I',), 'int32'),
        'speed_limit': (('I',), 'float32'),
        'pos_xy': (('I', 'I_points', 2), 'float32'),
        'dir_xy': (('I', 'I_points', 2), 'float32'),
        'valid': (('I', 'I_points'), 'bool'),
        'entry_links': (('I', 'I_connections'), 'int32'),
        'exit_links': (('I', 'I_connections'), 'int32'),
        'left_links': (('I', 'I_adjacents'), 'int32'),
        'right_links': (('I', 'I_adjacents'), 'int32'),
        'left_lanes': (('I', 'I_boundaries'), 'int32'),
        'right_lanes': (('I', 'I_boundaries'), 'int32'),
    },
    'lanes': {
        'id': (('L',), 'int32'),
        'type': (('L',), 'int32'),
        'pos_xy': (('L', 'L_points', 2), 'float32'),
        'dir_xy': (('L', 'L_points', 2), 'float32'),
        'valid': (('L', 'L_points'), 'bool'),
    },
    'traffics': {
        'id': (('T',), 'int32'),
        'type': (('T',), 'int32'),
        'target_link': (('T',), 'int32'),
        'dir_xy': (('T', 2), 'float32'),
        'pos_xy': (('T', 2), 'float32'),
        'state': (('T', 'F'), 'int32'),
        'time_s': (('T', 'F', 1), 'float32'),
        'valid': (('T', 'F'), 'bool'),
    },
    'markers': {
        'id': (('M',), 'int32'),
        'type': (('M',), 'int32'),
        'pos_xy': (('M', 'M_points', 2), 'float32'),
        'dir_xy': (('M', 'M_points', 2), 'float32'),
        'valid': (('M', 'M_points'), 'bool'),
    },
    'paths': {
        'near_link': (('P', 'P_points'), 'int32'),
        'pos_xy': (('P', 'P_points', 2), 'float32'),
        'dir_xy': (('P', 'P_points', 2), 'float32'),
        'arc_length': (('P', 'P_points', 1), 'float32'),
        'is_route': (('P',), 'bool'),
        'cost': (('P',), 'float32'),
        'valid': (('P', 'P_points'), 'bool'),
    },
    'goal': {
        'pos_xy': ((2,), 'float32'),
        'dir_xy': ((2,), 'float32'),
        'valid': ((), 'bool'),
    },
})

META_KEYS = ('scene_id', 'map_id', 'anchor_xy')

ENUMS = MappingProxyType({
    ('agents', 'type'): DS.num_agent_types,
    ('links', 'type'): DS.num_link_types,
    ('links', 'section'): DS.num_link_sections,
    ('lanes', 'type'): DS.num_lane_types,
    ('traffics', 'type'): DS.num_traffic_types,
    ('traffics', 'state'): DS.num_traffic_states,
    ('markers', 'type'): DS.num_marker_types,
})

REFERENCES = MappingProxyType({
    ('links', 'entry_links'): 'links',
    ('links', 'exit_links'): 'links',
    ('links', 'left_links'): 'links',
    ('links', 'right_links'): 'links',
    ('links', 'left_lanes'): 'lanes',
    ('links', 'right_lanes'): 'lanes',
    ('traffics', 'target_link'): 'links',
    ('paths', 'near_link'): 'links',
})


def shape_of(group: str, field: str) -> tuple[int, ...]:
    shape, _ = FIELDS[group][field]
    return tuple(SYMBOLS[s] if isinstance(s, str) else s for s in shape)
