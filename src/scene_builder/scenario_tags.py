# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

"""Scenario measures derived from TFExample fields."""

from __future__ import annotations

import numpy as np

from .dataspec.spec import AgentType, LinkSection, LinkType
from .route_tags import ROUTE_TAGS, route_tags

_AGENT_TAGS = {
    'Vehicles': AgentType.vehicle,
    'Pedestrians': AgentType.pedestrian,
    'Cyclists': AgentType.cyclist,
}
_SECTION_TAGS = {'Intersections': LinkSection.intersection, 'Freeways': LinkSection.freeway}
_MOTION_TAGS = (
    'vel', 'acc', 'dir_rate', 'Δvel', 'Δacc', 'Δdir', 'Δpos', 'Δpos_x', 'Δpos_y',
)
_FIT_HALF_S = 1.0
SCENARIO_TAGS = {
    'scene': (*_AGENT_TAGS, 'Traffics', 'Links', 'Lanes', *_SECTION_TAGS),
    'route': ROUTE_TAGS,
    'ego': _MOTION_TAGS,
}


def _ego_track(agents):
    rows = np.flatnonzero(agents['is_ego'])
    if not rows.size:
        return None
    row = rows[0]
    valid = agents['valid'][row]
    if not valid.any():
        return None
    t = agents['time_s'][row, valid, 0].astype(np.float64)
    pos = agents['pos_xy'][row, valid].astype(np.float64)
    dirs = agents['dir_xy'][row, valid].astype(np.float64)
    heading = np.unwrap(np.arctan2(dirs[:, 1], dirs[:, 0]))
    return t, pos, dirs, heading


def _on_map(pos, links):
    valid = links['valid'] & (links['type'] == LinkType.vehicle)[:, None]
    segments = valid[:, :-1] & valid[:, 1:]
    start = links['pos_xy'][:, :-1][segments].astype(np.float64)
    end = links['pos_xy'][:, 1:][segments].astype(np.float64)
    on_map = np.zeros(len(pos), dtype=bool)

    for left in range(0, len(start), 2048):
        a, b = start[left:left + 2048], end[left:left + 2048]
        v = b - a
        delta = pos[:, None] - a
        fraction = np.clip((delta * v).sum(-1) / np.maximum((v * v).sum(-1), 1e-6), 0, 1)
        on_map |= ((np.square(delta - fraction[..., None] * v).sum(-1)) <= 3.0**2).any(axis=1)
    return on_map


def _lane_share(pos, lanes):
    points = lanes['pos_xy'][lanes['valid']]
    near = np.zeros(len(pos), dtype=bool)
    for left in range(0, len(points), 2048):
        delta = pos[:, None] - points[left:left + 2048]
        near |= (np.square(delta).sum(-1) <= 3.5**2).any(axis=1)
    return float(near.mean())


def _fit_state(t, pos, heading, frame):
    mask = np.abs(t - t[frame]) <= _FIT_HALF_S + 1e-6
    if mask.sum() < 3:
        return None
    times = t[mask] - t[frame]
    coefficients = np.polyfit(times, pos[mask] - pos[frame], 2)
    return (float(np.linalg.norm(coefficients[1])), 2 * coefficients[0],
            float(np.polyfit(times, heading[mask], 1)[0]))


def _ego_motion(ego):
    if ego is None or len(ego[0]) < 3:
        return {}
    t, pos, dirs, heading = ego
    mid = len(t) // 2
    state = _fit_state(t, pos, heading, mid)
    if state is None:
        return {}
    forward = dirs[mid]
    lateral = np.array([-forward[1], forward[0]])
    speed, acc, rate = state
    out = {'vel': speed, 'acc': float(acc @ forward), 'dir_rate': rate}
    first = int(np.searchsorted(t, t[0] + _FIT_HALF_S - 1e-6))
    last = int(np.searchsorted(t, t[-1] - _FIT_HALF_S + 1e-6, side='right')) - 1
    if first < last:
        delta = pos[last] - pos[first]
        out.update({'Δdir': float(heading[last] - heading[first]),
                    'Δpos': float(np.linalg.norm(delta)),
                    'Δpos_x': float(delta @ forward), 'Δpos_y': float(delta @ lateral)})
        a = _fit_state(t, pos, heading, first)
        b = _fit_state(t, pos, heading, last)
        if a is not None and b is not None:
            out.update({'Δvel': b[0] - a[0], 'Δacc': float((b[1] - a[1]) @ forward)})
    return out


def classify(scene: dict) -> dict:
    """Measure the scenario tags in a NumPy-decoded scene."""
    agents, links = scene['agents'], scene['links']
    valid = agents['valid'] & ~agents['is_ego'][:, None]
    measures = {tag: int(valid[agents['type'] == kind].sum(axis=0).max(initial=0))
                for tag, kind in _AGENT_TAGS.items()}
    measures['Traffics'] = int(scene['traffics']['valid'].sum(axis=0).max(initial=0))
    sections = links['section'][links['valid'].any(axis=1)]
    measures.update({tag: int((sections == kind).sum()) for tag, kind in _SECTION_TAGS.items()})
    ego = _ego_track(agents)
    on_map = _on_map(ego[1], links) if ego is not None else None
    measures['Links'] = float(on_map.mean()) if on_map is not None else 0.0
    measures['Lanes'] = _lane_share(ego[1], scene['lanes']) if ego is not None else 0.0
    motion = _ego_motion(ego)
    return {'scene': measures, 'route': route_tags(scene, ego, on_map),
            'ego': {tag: motion.get(tag) for tag in _MOTION_TAGS}}
