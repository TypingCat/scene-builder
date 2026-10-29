# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import numpy as np
import torch

from ..dataspec.spec import DS

_MIN_POLYLINE_POINTS = 2


def _group_polylines(points: dict) -> list[dict]:
    valid, types, ids = points['valid'], points['types'], points['ids']
    idx_by_fid: dict[tuple[int, int], list[int]] = {}
    for i in np.flatnonzero(valid & (types != 0)):
        idx_by_fid.setdefault((int(types[i]), int(ids[i])), []).append(int(i))

    polylines = []
    for (t, fid), idxs in idx_by_fid.items():
        if len(idxs) < _MIN_POLYLINE_POINTS:
            continue
        pos = np.stack([points['x'][idxs], points['y'][idxs]], axis=-1).astype(np.float32)
        dirs = np.stack([points['dir_x'][idxs], points['dir_y'][idxs]], axis=-1).astype(np.float32)
        polylines.append(dict(type=t, id=fid, pos=pos, dir=dirs))
    return polylines


def _nearest(polylines: list[dict], ego_xy: np.ndarray | None, max_roads: int) -> list[dict]:
    if len(polylines) <= max_roads:
        return polylines
    if ego_xy is None:
        return polylines[:max_roads]
    centroids = np.stack([p['pos'].mean(axis=0) for p in polylines])
    order = np.argsort(np.linalg.norm(centroids - ego_xy, axis=-1))[:max_roads]
    return [polylines[i] for i in order]


def _pack_roads(polylines: list[dict]) -> dict:
    I, W = DS.num_links, DS.num_link_points
    pos_xy = np.zeros((I, W, 2), dtype=np.float32)
    dir_xy = np.zeros((I, W, 2), dtype=np.float32)
    road_type = np.zeros(I, dtype=np.int32)
    road_id = np.zeros(I, dtype=np.int32)
    valid = np.zeros((I, W), dtype=bool)
    for r, p in enumerate(polylines[:I]):
        pos, dirs = p['pos'], p['dir']
        if len(pos) > W:
            idx = np.round(np.linspace(0, len(pos) - 1, W)).astype(int)
            pos, dirs = pos[idx], dirs[idx]
        n = len(pos)
        pos_xy[r, :n] = pos
        dir_xy[r, :n] = dirs
        road_type[r] = p['type']
        road_id[r] = p['id']
        valid[r, :n] = True
    return dict(
        pos_xy=torch.from_numpy(pos_xy),
        dir_xy=torch.from_numpy(dir_xy),
        type=torch.from_numpy(road_type),
        id=torch.from_numpy(road_id),
        speed_limit=torch.zeros(I, dtype=torch.float32),
        valid=torch.from_numpy(valid),
    )


def _sides(my_c, my_dir, cand_c, cand_dir, band_extra, k: int, exclude: int | None = None):
    rel = cand_c - my_c
    d = np.linalg.norm(rel, axis=-1)
    cos = (my_dir[None, :] * cand_dir).sum(axis=-1)
    band = band_extra(d, cos)
    if exclude is not None:
        band[exclude] = False
    cross = my_dir[0] * rel[:, 1] - my_dir[1] * rel[:, 0]
    left = np.where(band & (cross > 0))[0]
    right = np.where(band & (cross < 0))[0]
    return left[np.argsort(d[left])][:k], right[np.argsort(d[right])][:k]


def _lane_graph(polylines: list[dict], *, lane_endpoint_thresh: float, lane_dir_cos_min: float,
                lane_width_min: float, lane_width_max: float, neighbor_cos_min: float,
                boundary_dist_max: float, boundary_cos_min: float) -> dict:
    I = DS.num_links
    C, J, B = DS.num_link_connections, DS.num_link_adjacents, DS.num_link_boundaries
    lg = dict(
        entry=np.full((I, C), -1, dtype=np.int32),
        exit=np.full((I, C), -1, dtype=np.int32),
        left_links=np.full((I, J), -1, dtype=np.int32),
        right_links=np.full((I, J), -1, dtype=np.int32),
        left_lanes=np.full((I, B), -1, dtype=np.int32),
        right_lanes=np.full((I, B), -1, dtype=np.int32),
    )
    n = min(len(polylines), I)
    types = np.array([p['type'] for p in polylines[:n]], dtype=np.int32)
    lane_idx = np.where((types >= 1) & (types <= 3))[0]
    line_idx = np.where((types >= 4) & (types <= 13))[0]
    if len(lane_idx) == 0:
        return {k: torch.from_numpy(v) for k, v in lg.items()}

    start = np.array([polylines[r]['pos'][0] for r in lane_idx], dtype=np.float32)
    end = np.array([polylines[r]['pos'][-1] for r in lane_idx], dtype=np.float32)
    start_dir = np.array([polylines[r]['dir'][0] for r in lane_idx], dtype=np.float32)
    end_dir = np.array([polylines[r]['dir'][-1] for r in lane_idx], dtype=np.float32)
    center = np.array([polylines[r]['pos'].mean(axis=0) for r in lane_idx], dtype=np.float32)

    for li, r in enumerate(lane_idx):
        for key, mine, mine_dir, theirs, their_dir in (
                ('exit', end[li], end_dir[li], start, start_dir),
                ('entry', start[li], start_dir[li], end, end_dir)):
            d = np.linalg.norm(theirs - mine[None, :], axis=-1)
            d[li] = np.inf
            cos = (mine_dir * their_dir).sum(axis=-1)
            cand = np.where((d < lane_endpoint_thresh) & (cos > lane_dir_cos_min))[0]
            cand = cand[np.argsort(d[cand])][:C]
            lg[key][r, :len(cand)] = lane_idx[cand]

        left, right = _sides(
            center[li], end_dir[li], center, end_dir,
            lambda d, cos: (d >= lane_width_min) & (d <= lane_width_max) & (np.abs(cos) > neighbor_cos_min),
            J, exclude=li)
        lg['left_links'][r, :len(left)] = lane_idx[left]
        lg['right_links'][r, :len(right)] = lane_idx[right]

    if len(line_idx) > 0:
        line_center = np.array([polylines[r]['pos'].mean(axis=0) for r in line_idx], dtype=np.float32)
        line_dir = np.array([polylines[r]['dir'][-1] for r in line_idx], dtype=np.float32)
        for li, r in enumerate(lane_idx):
            left, right = _sides(
                center[li], end_dir[li], line_center, line_dir,
                lambda d, cos: (d < boundary_dist_max) & (np.abs(cos) > boundary_cos_min),
                B)
            lg['left_lanes'][r, :len(left)] = line_idx[left]
            lg['right_lanes'][r, :len(right)] = line_idx[right]

    return {k: torch.from_numpy(v) for k, v in lg.items()}


def assemble_roadgraph(
    points: dict,
    ego_xy: tuple[float, float] | None = None,
    *,
    lane_endpoint_thresh: float = 1.0,
    lane_dir_cos_min: float = 0.5,
    lane_width_min: float = 2.5,
    lane_width_max: float = 6.0,
    neighbor_cos_min: float = 0.7,
    boundary_dist_max: float = 3.0,
    boundary_cos_min: float = 0.5,
) -> tuple[dict, dict]:
    polylines = _group_polylines(points)
    ego_arr = None if ego_xy is None else np.asarray(ego_xy, dtype=np.float32).reshape(2)
    polylines = _nearest(polylines, ego_arr, DS.num_links)
    lane_graph = _lane_graph(
        polylines,
        lane_endpoint_thresh=lane_endpoint_thresh, lane_dir_cos_min=lane_dir_cos_min,
        lane_width_min=lane_width_min, lane_width_max=lane_width_max,
        neighbor_cos_min=neighbor_cos_min,
        boundary_dist_max=boundary_dist_max, boundary_cos_min=boundary_cos_min,
    )
    return _pack_roads(polylines), lane_graph
