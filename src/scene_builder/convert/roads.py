# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import torch

from ..dataspec.schema import empty_lanes, empty_links
from ..dataspec.spec import DS, LinkSection
from .codes import (
    RoadType,
    lane_type_from_road_type,
    link_section_from_road_type,
    link_type_from_road_type,
    marker_type_from_road_type,
)
from .polyline import (
    LANE_INTERVAL,
    LINE_INTERVAL,
    MARKER_INTERVAL,
    distance_truncate,
    piece_centroid,
    piece_endpoints,
    resample_split_batch,
)


def _split_category(src: torch.Tensor, roads: dict, W_target: int, interval: float,
                    id_allocator: list, *, mode: str = 'split') -> dict:
    pos, dirs, valid, parents, piece_idx = resample_split_batch(
        src, roads['pos_xy'], roads['dir_xy'], roads['valid'], W_target, interval, mode=mode)
    n = pos.shape[0]

    feat_id = torch.zeros(n, dtype=torch.int32)
    pieces_per_src = torch.zeros(n, dtype=torch.int32)
    count: dict[int, int] = {}
    for p in parents:
        count[p] = count.get(p, 0) + 1
    for i in range(n):
        feat_id[i] = id_allocator[0]
        id_allocator[0] += 1
        pieces_per_src[i] = count[parents[i]]

    if n > 0:
        centroid = torch.stack([piece_centroid(pos[i], valid[i]) for i in range(n)])
        ends = [piece_endpoints(pos[i], valid[i]) for i in range(n)]
        start_xy = torch.stack([e[0] for e in ends])
        end_xy = torch.stack([e[1] for e in ends])
        parent_t = torch.tensor(parents, dtype=torch.int64)
        road_type = roads['type'][parent_t]
        speed_limit = roads['speed_limit'][parent_t]
    else:
        centroid = torch.zeros((0, 2), dtype=torch.float32)
        start_xy = torch.zeros((0, 2), dtype=torch.float32)
        end_xy = torch.zeros((0, 2), dtype=torch.float32)
        parent_t = torch.zeros(0, dtype=torch.int64)
        road_type = torch.zeros(0, dtype=torch.int32)
        speed_limit = torch.zeros(0, dtype=torch.float32)

    return dict(
        pos=pos, dir=dirs, valid=valid,
        feat_id=feat_id,
        parent=parent_t,
        piece_idx=torch.tensor(piece_idx, dtype=torch.int64) if piece_idx else torch.zeros(0, dtype=torch.int64),
        pieces_per_src=pieces_per_src,
        centroid=centroid,
        start_xy=start_xy,
        end_xy=end_xy,
        road_type=road_type,
        speed_limit=speed_limit,
    )


def _pieces_by_src(cat: dict) -> dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]]:
    grouped: dict[int, list] = {}
    for i in range(cat['pos'].shape[0]):
        grouped.setdefault(int(cat['parent'][i]), []).append(
            (int(cat['piece_idx'][i]), int(cat['feat_id'][i]), cat['start_xy'][i], cat['end_xy'][i]))
    out = {}
    for si, lst in grouped.items():
        lst.sort(key=lambda t: t[0])
        out[si] = [(fid, s, e) for _pi, fid, s, e in lst]
    return out


def _nearest_piece(pieces, query_xy: torch.Tensor) -> tuple[int, float]:
    best_fid, best_d = pieces[0][0], float('inf')
    for fid, s_xy, e_xy in pieces:
        d = min(float((s_xy - query_xy).norm().item()), float((e_xy - query_xy).norm().item()))
        if d < best_d:
            best_fid, best_d = fid, d
    return best_fid, best_d


def _resolver(by_src: dict):
    def resolve(si: int, query_xy: torch.Tensor | None = None, mode: str = 'first') -> int:
        pieces = by_src.get(int(si))
        if not pieces:
            return 0
        if mode == 'first':
            return pieces[0][0]
        if mode == 'last':
            return pieces[-1][0]
        return _nearest_piece(pieces, query_xy)[0]
    return resolve


def roads_to_schema(roads: dict, lane_graph: dict, ego_pos: torch.Tensor | None):
    road_types = roads['type']
    roads_id = roads['id']

    link_src = torch.where(RoadType.is_lane_center(road_types))[0]
    lane_src = torch.where(RoadType.is_road_line(road_types) | RoadType.is_road_edge(road_types))[0]
    marker_src = torch.where(RoadType.is_marker(road_types))[0]

    max_existing = int(roads_id.max().item()) if roads_id.numel() > 0 else 0
    id_allocator = [max_existing + 1]

    links_cat = _split_category(link_src, roads, DS.num_link_points, LANE_INTERVAL, id_allocator)
    links_cat = distance_truncate(links_cat, ego_pos, DS.num_links)

    lanes_cat = _split_category(lane_src, roads, DS.num_lane_points, LINE_INTERVAL, id_allocator)
    lanes_cat = distance_truncate(lanes_cat, ego_pos, DS.num_lanes)

    markers_cat = _split_category(marker_src, roads, DS.num_marker_points, MARKER_INTERVAL,
                                  id_allocator, mode='fit_single')
    markers_cat['marker_type'] = marker_type_from_road_type(markers_cat['road_type']).to(torch.int32)
    markers_cat = distance_truncate(markers_cat, ego_pos, DS.num_markers)

    link_pieces = _pieces_by_src(links_cat)
    resolve_link = _resolver(link_pieces)
    resolve_lane = _resolver(_pieces_by_src(lanes_cat))

    fid_to_srcs: dict[int, list[int]] = {}
    for si in link_src.tolist():
        fid_to_srcs.setdefault(int(roads_id[si]), []).append(int(si))

    def resolve_link_fid(orig_fid: int, query_xy: torch.Tensor) -> int:
        candidates = fid_to_srcs.get(int(orig_fid))
        if not candidates:
            return 0
        if len(candidates) == 1:
            return resolve_link(candidates[0], query_xy, mode='nearest')
        best_si, best_d = None, float('inf')
        for si in candidates:
            pieces = link_pieces.get(si)
            if pieces:
                d = _nearest_piece(pieces, query_xy)[1]
                if d < best_d:
                    best_si, best_d = si, d
        if best_si is None:
            best_si = candidates[0]
        return resolve_link(best_si, query_xy, mode='nearest')

    n_links = links_cat['pos'].shape[0]
    links = empty_links(I=DS.num_links, I_points=DS.num_link_points,
                        I_connections=DS.num_link_connections,
                        I_adjacents=DS.num_link_adjacents,
                        I_boundaries=DS.num_link_boundaries)
    links['id'][:n_links] = links_cat['feat_id']
    links['pos_xy'][:n_links] = links_cat['pos']
    links['dir_xy'][:n_links] = links_cat['dir']
    links['valid'][:n_links] = links_cat['valid']
    links['type'][:n_links] = link_type_from_road_type(links_cat['road_type'])
    links['section'][:n_links] = link_section_from_road_type(links_cat['road_type'])
    links['speed_limit'][:n_links] = links_cat['speed_limit']

    lg = lane_graph
    intersection = ((lg['entry'] >= 0).sum(dim=-1) > 1) | ((lg['exit'] >= 0).sum(dim=-1) > 1)
    n_conn = DS.num_link_connections

    for i in range(n_links):
        si = int(links_cat['parent'][i])
        pi = int(links_cat['piece_idx'][i])
        n_pieces = int(links_cat['pieces_per_src'][i])
        siblings = link_pieces.get(si, [])

        if pi == 0:
            fids = [resolve_link(r, mode='last') for r in lg['entry'][si].tolist() if r >= 0][:n_conn]
            if fids:
                links['entry_links'][i, :len(fids)] = torch.tensor(fids, dtype=torch.int32)
        else:
            links['entry_links'][i, 0] = siblings[pi - 1][0] if pi - 1 < len(siblings) else 0
        if pi == n_pieces - 1:
            fids = [resolve_link(r, mode='first') for r in lg['exit'][si].tolist() if r >= 0][:n_conn]
            if fids:
                links['exit_links'][i, :len(fids)] = torch.tensor(fids, dtype=torch.int32)
        else:
            links['exit_links'][i, 0] = siblings[pi + 1][0] if pi + 1 < len(siblings) else 0

        query_xy = links_cat['centroid'][i]
        for key, resolve in (('left_links', resolve_link), ('right_links', resolve_link),
                             ('left_lanes', resolve_lane), ('right_lanes', resolve_lane)):
            for slot, r in enumerate(lg[key][si].tolist()):
                if r >= 0:
                    links[key][i, slot] = resolve(r, query_xy, mode='nearest')

        if bool(intersection[si]):
            links['section'][i] = LinkSection.intersection

    n_lanes = lanes_cat['pos'].shape[0]
    lanes = empty_lanes(L=DS.num_lanes, L_points=DS.num_lane_points)
    lanes['id'][:n_lanes] = lanes_cat['feat_id']
    lanes['pos_xy'][:n_lanes] = lanes_cat['pos']
    lanes['dir_xy'][:n_lanes] = lanes_cat['dir']
    lanes['valid'][:n_lanes] = lanes_cat['valid']
    lanes['type'][:n_lanes] = lane_type_from_road_type(lanes_cat['road_type'])

    return links, lanes, markers_cat, resolve_link_fid, id_allocator
