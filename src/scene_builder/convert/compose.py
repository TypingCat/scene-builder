# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import torch

from ..dataspec.schema import empty_goal, empty_paths
from ..dataspec.shard import make_stamp
from ..dataspec.spec import DATASPEC, DS, TrafficType
from . import track
from .codes import traffic_state_from_waymo
from .markers import concat_marker_cats, fill_markers, stop_line_markers
from .paths import assign_near_link, finalize_path_slots, infer_route_from_ego, resample_paths
from .polyline import LANE_INTERVAL, distance_truncate, sort_by_distance, sort_by_encounter
from .roads import roads_to_schema
from .timeline import fit_frames

_STAMP = make_stamp(dict(DATASPEC))


def _agents(raw: dict, dt: float) -> dict:
    traj = raw['log_trajectory']
    meta = raw['object_metadata']
    yaw = traj['yaw']
    pos_xy = torch.stack([traj['x'], traj['y']], dim=-1)
    vel_xy, acc_xy = track.derive(pos_xy, traj['valid'], dt)
    size = torch.stack([traj['length'], traj['width'], traj['height']], dim=-1)
    A, F = traj['valid'].shape
    time_s = (torch.arange(F) * dt).unsqueeze(0).unsqueeze(-1).expand(A, F, 1)
    types = meta['object_types']
    return dict(
        id=meta['ids'],
        type=torch.where(types >= 4, torch.zeros_like(types), types),
        is_ego=meta['is_sdc'],
        size_xyz=size.amax(dim=-2),
        time_s=time_s,
        pos_xy=pos_xy,
        dir_xy=torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1),
        vel_xy=vel_xy,
        acc_xy=acc_xy,
        valid=traj['valid'],
    )


def _ego_refs(agents: dict) -> tuple[int | None, torch.Tensor | None]:
    if not agents['is_ego'].any():
        return None, None
    k = int(agents['is_ego'].nonzero(as_tuple=True)[0][0])
    pos, valid = agents['pos_xy'][k], agents['valid'][k]
    F = valid.shape[-1]
    refs = [pos[f] for f in (0, F // 4, F // 2, 3 * F // 4, F - 1) if valid[f]]
    if refs:
        return k, torch.stack([torch.tensor([float(p[0]), float(p[1])]) for p in refs])
    if valid.any():
        frames = valid.nonzero(as_tuple=True)[0]
        f = frames[(frames - F // 2).abs().argmin()]
        return k, torch.tensor([float(pos[f, 0]), float(pos[f, 1])])
    return k, None


def _traffics(raw: dict, dt: float, links: dict, resolve_link_fid, ego_pos):
    tl = raw['log_traffic_light']
    L_src, F = tl['valid'].shape
    tl_time = (torch.arange(F) * dt).unsqueeze(0).unsqueeze(-1).expand(L_src, F, 1)
    tl_pos = torch.stack([tl['x'], tl['y']], dim=-1)
    state_all = traffic_state_from_waymo(tl['state'])

    rows = []
    for i in range(L_src):
        v = tl['valid'][i]
        if not v.any():
            continue
        vf = v.nonzero(as_tuple=True)[0]
        lane_id = int(tl['lane_ids'][i, vf[0]])
        if lane_id == 0:
            continue
        stop_xy = tl_pos[i, vf[len(vf) // 2]]
        rows.append((resolve_link_fid(lane_id, stop_xy), i, stop_xy))

    T = DS.num_traffics
    if rows and ego_pos is not None:
        keep = sort_by_distance(torch.stack([r[2] for r in rows]), ego_pos, T).tolist()
    else:
        keep = list(range(min(len(rows), T)))

    out = dict(
        id=torch.zeros(T, dtype=torch.int32),
        type=torch.zeros(T, dtype=torch.int32),
        target_link=torch.zeros(T, dtype=torch.int32),
        dir_xy=torch.zeros((T, 2), dtype=torch.float32),
        state=torch.zeros((T, F), dtype=state_all.dtype),
        time_s=torch.zeros((T, F, 1), dtype=tl_time.dtype),
        pos_xy=torch.zeros((T, 2), dtype=tl_pos.dtype),
        valid=torch.zeros((T, F), dtype=tl['valid'].dtype),
    )
    link_ids = links['id'].to(torch.long)
    for slot, r in enumerate(keep):
        fid, i, stop_xy = rows[r]
        out['id'][slot] = slot + 1
        out['type'][slot] = TrafficType.vehicle
        out['target_link'][slot] = fid
        out['state'][slot] = state_all[i]
        out['time_s'][slot] = tl_time[i]
        out['pos_xy'][slot] = stop_xy
        out['valid'][slot] = tl['valid'][i]
        match = (link_ids == fid).nonzero(as_tuple=True)[0] if fid != 0 else []
        if len(match) and links['valid'][int(match[0])].any():
            li = int(match[0])
            out['dir_xy'][slot] = links['dir_xy'][li, int(links['valid'][li].nonzero(as_tuple=True)[0][0])]
    return out, len(keep)


def _paths(raw: dict, agents: dict, ego_idx: int | None, links: dict) -> dict:
    sp = raw['sdc_paths']
    if not sp['valid'].any():
        return empty_paths(P=DS.num_paths, P_points=DS.num_path_points)
    ego_xy = agents['pos_xy'][ego_idx] if ego_idx is not None else None
    ego_valid = agents['valid'][ego_idx] if ego_idx is not None else None
    paths = resample_paths(sp, interval=LANE_INTERVAL, P_points=DS.num_path_points,
                           ego_xy=ego_xy, ego_valid=ego_valid)
    route = paths['is_route']
    if route.sum() != 1 and ego_idx is not None:
        paths['is_route'] = infer_route_from_ego(
            ego_xy=ego_xy, ego_valid=ego_valid,
            path_pos=paths['pos_xy'], path_valid=paths['valid'],
            candidate_mask=route if route.any() else None)
    paths = finalize_path_slots(paths)
    paths['near_link'] = assign_near_link(paths, links)
    return paths


def _to_scene(raw: dict) -> dict:
    dt = 1.0 / raw['_source_hz']
    agents = _agents(raw, dt)
    ego_idx, ego_pos = _ego_refs(agents)

    A = agents['valid'].shape[0]
    if ego_pos is not None and A > 0:
        order = sort_by_encounter(agents['pos_xy'], agents['valid'],
                                  agents['pos_xy'][ego_idx], agents['valid'][ego_idx],
                                  DS.num_agents, always_first=agents['is_ego'])
    else:
        order = torch.arange(min(A, DS.num_agents))
    agents = {k: v[order] for k, v in agents.items()}
    ego = agents['is_ego'].nonzero(as_tuple=True)[0]
    ego_idx = int(ego[0]) if ego.numel() else None

    links, lanes, markers_cat, resolve_link_fid, id_allocator = roads_to_schema(
        raw['roads'], raw['lane_graph'], ego_pos)

    traffics, n_kept = _traffics(raw, dt, links, resolve_link_fid, ego_pos)

    stop_lines = stop_line_markers(traffics['pos_xy'], n_kept, id_allocator, DS.num_marker_points)
    joint = distance_truncate(concat_marker_cats(markers_cat, stop_lines), ego_pos, DS.num_markers)
    markers = fill_markers(joint)

    return dict(
        agents=agents, traffics=traffics, paths=_paths(raw, agents, ego_idx, links),
        links=links, lanes=lanes, markers=markers,
        goal=empty_goal(),
        meta=dict(scene_id=str(raw['_scene_id']), map_id=None, anchor_xy=None),
    )


def convert_raw(raw: dict) -> list[dict]:
    scene = _to_scene(fit_frames(raw, DS.num_frames))
    return [scene] if scene['paths']['valid'].any() else []


def stamp() -> dict:
    return dict(_STAMP)
