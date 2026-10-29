# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

from types import SimpleNamespace

import plotly.graph_objects as go
import torch

from ..dataspec.spec import LaneType
from . import theme
from .primitives import (
    Figure,
    add_time_slider,
    boxes_to_polygons,
    create_lines,
    create_markers,
    create_polygons,
    frame_indices,
    heading_segments,
    rail_segments,
)
from .theme import LANE_STYLE, PALETTE, TRAFFIC_COLOR, rgba

View = SimpleNamespace


def _last_valid_frame(agents: View, F: int) -> int:
    if agents.valid.numel() == 0:
        return F - 1
    fv = agents.valid.any(dim=0)
    return int(fv.nonzero()[-1]) if bool(fv.any()) else F - 1


def _ego_width(agents: View) -> float | None:
    if agents.valid.numel() == 0 or not bool(agents.is_ego.any()):
        return None
    idx = int(agents.is_ego.nonzero(as_tuple=True)[0][0])
    return float(agents.size_xyz[idx, 1])


def _markers_traces(markers: View) -> list[go.Scatter]:
    if markers.valid.numel() == 0:
        return []
    pos_xy = markers.pos_xy
    valid = markers.valid
    mtype = markers.type

    polygons: list[torch.Tensor] = []
    points: list[torch.Tensor] = []
    for i in range(valid.shape[0]):
        v = valid[i]
        if not bool(v.any()):
            continue
        pts = pos_xy[i, v]
        if int(mtype[i]) in theme.POLYGON_MARKER_TYPES and pts.shape[0] >= 3:
            polygons.append(torch.cat([pts, pts[:1]], dim=0))
        else:
            points.append(pts[:1])

    out: list[go.Scatter] = []
    if polygons:
        out.append(create_polygons(polygons, fillcolor=PALETTE.surface))
    if points:
        out.append(create_markers(torch.cat(points, dim=0),
                                  marker=dict(size=theme.MARKER.point_px, color=PALETTE.surface)))
    return out


def _links_base_traces(links: View, subsample: int = 1) -> list[go.Scatter]:
    if links.valid.numel() == 0:
        return []
    pos_xy = links.pos_xy
    valid = links.valid
    pts = pos_xy[valid]
    if pts.numel() == 0:
        return []
    pts = pts[::max(1, subsample)]
    return [create_markers(pts, marker=dict(
        size=theme.LINK.base_px, color=rgba(PALETTE.gray, theme.LINK.base_alpha)))]


def _lanes_traces(lanes: View, subsample: int = 1) -> list[go.Scatter]:
    if lanes.valid.numel() == 0:
        return []
    pos_xy = lanes.pos_xy
    valid = lanes.valid
    ltype = lanes.type
    step = max(1, subsample)

    by_type: dict[int, list[torch.Tensor]] = {}
    for i in range(valid.shape[0]):
        v = valid[i]
        if not bool(v.any()):
            continue
        p = pos_xy[i, v]
        if step > 1 and p.shape[0] > 2:
            p = torch.cat([p[::step], p[-1:]])
        by_type.setdefault(int(ltype[i]), []).append(p)

    out: list[go.Scatter] = []
    for lt, polylines in by_type.items():
        color, width, dash, alpha = LANE_STYLE.get(lt, LANE_STYLE[LaneType.unknown])
        line = dict(color=rgba(color, alpha) if alpha < 1.0 else color, width=width)
        if dash is not None:
            line['dash'] = dash
        out.append(create_lines(polylines, line=line))
    return out


def _route_traces(paths: View, ego_width: float | None, stride: int = 2) -> list[go.Scatter]:
    if paths.valid.numel() == 0:
        return []
    pos_xy = paths.pos_xy
    valid = paths.valid
    is_route = paths.is_route

    route_pts = {(round(float(p[0]), 2), round(float(p[1]), 2))
                 for i in range(valid.shape[0]) if bool(is_route[i])
                 for p in pos_xy[i, valid[i]]}

    half_w = (ego_width * theme.ROUTE.half_width_factor) if ego_width else theme.ROUTE.fallback_half_width_m
    route_segs: list[torch.Tensor] = []
    other_segs: list[torch.Tensor] = []
    for i in range(valid.shape[0]):
        route = bool(is_route[i])
        v = valid[i]
        if not bool(v.any()):
            continue
        pts = pos_xy[i, v][::stride]
        if not route:
            keep = [(round(float(p[0]), 2), round(float(p[1]), 2)) not in route_pts for p in pts]
            pts = pts[torch.tensor(keep)] if not all(keep) else pts
        (route_segs if route else other_segs).extend(rail_segments(pts, half_w))
    out: list[go.Scatter] = []
    if other_segs:
        out.append(create_lines(other_segs, line=dict(width=theme.ROUTE.width_px, color=theme.ROUTE.other_color)))
    if route_segs:
        out.append(create_lines(route_segs, line=dict(width=theme.ROUTE.width_px, color=theme.ROUTE.color)))
    return out


def _signal_static_traces(traffics: View, now: int) -> list[go.Scatter]:
    if traffics.valid.numel() == 0 or traffics.valid.shape[1] == 0:
        return []
    t = min(now, traffics.valid.shape[1] - 1)
    valid = traffics.valid[:, t]
    if not bool(valid.any()):
        return []
    pos_xy = traffics.pos_xy
    dir_xy = traffics.dir_xy

    ring_pos = pos_xy[valid]
    ticks = heading_segments(ring_pos, dir_xy[valid], theme.SIGNAL.tick_r0_m, theme.SIGNAL.tick_r1_m)
    return [
        create_markers(ring_pos, marker=dict(
            size=theme.SIGNAL.ring_px, color=theme.SIGNAL.ring_color, symbol='circle-open', line=dict(width=2))),
        create_lines(ticks, line=dict(width=2, color=theme.SIGNAL.ring_color)),
    ]


def _link_state_at(traffics: View, t: int) -> dict[int, int]:
    out: dict[int, int] = {}
    if traffics.valid.numel() == 0 or t >= traffics.valid.shape[1]:
        return out
    valid = traffics.valid[:, t]
    state = traffics.state[:, t]
    target = traffics.target_link
    for i in valid.nonzero(as_tuple=True)[0].tolist():
        s = int(state[i])
        if s == 0:
            continue
        tl = target[i]
        for fid in tl[tl > 0].reshape(-1).tolist():
            if s > out.get(int(fid), 0):
                out[int(fid)] = s
    return out


def _link_state_traces(links: View, traffics: View, t: int) -> list[go.Scatter]:
    state_map = _link_state_at(traffics, t)
    if not state_map or links.valid.numel() == 0:
        return []
    link_id = links.id
    pos_xy = links.pos_xy
    valid = links.valid
    by_state: dict[int, list[torch.Tensor]] = {}
    for i in range(valid.shape[0]):
        s = state_map.get(int(link_id[i]))
        if s is not None and bool(valid[i].any()):
            by_state.setdefault(s, []).append(pos_xy[i, valid[i]])
    return [create_markers(torch.cat(g, dim=0), marker=dict(size=theme.LINK.state_px, color=TRAFFIC_COLOR[s]))
            for s, g in by_state.items()]


def _signal_exit_traces(traffics: View, t: int) -> list[go.Scatter]:
    if traffics.valid.numel() == 0 or t >= traffics.valid.shape[1]:
        return []
    valid = traffics.valid[:, t]
    state = traffics.state[:, t]
    pos_xy = traffics.pos_xy
    dir_xy = traffics.dir_xy
    by_state: dict[int, list[torch.Tensor]] = {}
    for i in valid.nonzero(as_tuple=True)[0].tolist():
        s = int(state[i])
        if s != 0:
            by_state.setdefault(s, []).extend(heading_segments(pos_xy[i:i + 1], dir_xy[i:i + 1], 0.0, theme.SIGNAL.tick_r0_m))
    return [create_lines(segs, line=dict(width=2.5, color=TRAFFIC_COLOR[s])) for s, segs in by_state.items()]


def agent_frame_traces(agents: View, t: int) -> list[go.Scatter]:
    if agents.valid.numel() == 0 or t >= agents.valid.shape[1] or not bool(agents.valid[:, t].any()):
        return []
    valid = agents.valid[:, t]
    pos = agents.pos_xy[valid, t]
    dir_ = agents.dir_xy[valid, t]
    size = agents.size_xyz[valid]
    boxes = boxes_to_polygons(pos, dir_, size)
    out = [create_polygons(boxes, fillcolor=theme.AGENT.fill)]

    ego = agents.is_ego[valid]
    groups = [(ego, theme.AGENT.ego_tick_color, theme.AGENT.ego_center_border)]
    if bool((~ego).any()):
        groups.append((~ego, theme.AGENT.tick_color, theme.AGENT.center_border))
    for mask, tick_color, border_color in groups:
        if not bool(mask.any()):
            continue
        ticks = heading_segments(pos[mask], dir_[mask], theme.AGENT.tick_r0_m, theme.AGENT.tick_r1_m)
        out.append(create_lines(ticks, line=dict(width=theme.AGENT.tick_px, color=tick_color)))
        out.append(create_markers(pos[mask], marker=dict(
            size=theme.AGENT.center_px, color=theme.AGENT.center_fill,
            line=dict(width=theme.AGENT.center_border_px, color=border_color))))
    return out


def _scene_frame_traces(scene: View, t: int) -> list[go.Scatter]:
    return (_link_state_traces(scene.links, scene.traffics, t)
            + _signal_exit_traces(scene.traffics, t)
            + agent_frame_traces(scene.agents, t))


def _autocenter_ego(figure: Figure, agents: View, now: int, r: int) -> None:
    if agents.valid.numel() == 0 or agents.valid.shape[1] == 0:
        return
    t = min(now, agents.valid.shape[1] - 1)
    ego_mask = agents.is_ego & agents.valid[:, t]
    if not bool(ego_mask.any()):
        return
    c = agents.pos_xy[ego_mask, t][0]
    cx, cy = float(c[0]), float(c[1])
    figure.update_layout(
        xaxis=dict(range=[cx - r, cx + r]),
        yaxis=dict(range=[cy - r, cy + r], scaleanchor='x', scaleratio=1),
    )


def plot_scene(
    scene: View,
    *,
    now: int,
    frame_unit: float,
    slider: bool = True,
    frame_stride: int = 5,
    map_subsample: int = 2,
    path_stride: int = 2,
    viewport_radius: int = 50,
) -> Figure:
    figure = Figure(viewport_radius=viewport_radius)
    agents = scene.agents
    F = agents.valid.shape[1] if agents.valid.numel() else 0
    if not 0 <= now < F:
        now = F // 2

    for tr in _markers_traces(scene.markers):
        figure.add_trace(tr)
    for tr in _links_base_traces(scene.links, map_subsample):
        figure.add_trace(tr)
    for tr in _lanes_traces(scene.lanes, map_subsample):
        figure.add_trace(tr)
    for tr in _route_traces(scene.paths, _ego_width(agents), path_stride):
        figure.add_trace(tr)
    for tr in _signal_static_traces(scene.traffics, now):
        figure.add_trace(tr)

    if F > 0:
        if slider:
            frames = frame_indices(F - 1, frame_stride or 1)
            groups = [(t, _scene_frame_traces(scene, t)) for t in frames]
            add_time_slider(figure, groups, frame_unit=frame_unit,
                            last_valid=_last_valid_frame(agents, F))
        else:
            for tr in _scene_frame_traces(scene, now):
                figure.add_trace(tr)

    _autocenter_ego(figure, agents, now, viewport_radius)
    return figure
