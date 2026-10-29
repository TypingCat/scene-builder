# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import math

import torch

from ..dataspec.spec import LINK_TRUST_RADIUS_M
from .polyline import arc_length, dedupe, normalize_arc, point_dirs, resample_split, sample_polyline

NEAR_LINK_COS_MIN = 0.0
PATH_SANITIZE_GAP_M = 3.0
PATH_SANITIZE_TURN_DEG = 60.0
PATH_SANITIZE_TURN_STEP_M = 1.5
PATH_OVERLAP_DIST_M = 1.5
PATH_OVERLAP_LEN_M = 10.0


def _segment_path_geometry(
    pos: torch.Tensor,
    arc: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if pos.shape[0] == 0:
        return []
    arc = normalize_arc(pos, arc)
    if pos.shape[0] == 1:
        return [(pos, arc)]

    step = pos[1:] - pos[:-1]
    step_dist = step.norm(dim=-1)
    step_heading = torch.atan2(step[:, 1], step[:, 0])
    break_mask = step_dist > PATH_SANITIZE_GAP_M

    if step_heading.shape[0] > 1:
        delta = torch.remainder(step_heading[1:] - step_heading[:-1] + math.pi, 2 * math.pi) - math.pi
        turn_break = torch.zeros_like(step_dist, dtype=torch.bool)
        turn_break[1:] = (delta.abs() > math.radians(PATH_SANITIZE_TURN_DEG)) & (
            step_dist[1:] > PATH_SANITIZE_TURN_STEP_M
        )
        break_mask = break_mask | turn_break

    split_idx = (break_mask.nonzero(as_tuple=True)[0] + 1).tolist()
    bounds = [0] + split_idx + [pos.shape[0]]
    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg_pos = pos[start:end]
        seg_arc = arc[start:end] - arc[start]
        if seg_pos.shape[0] == 0:
            continue
        segments.append((seg_pos, seg_arc))
    return segments

def _find_longest_true_run(mask: torch.Tensor) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    run_start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag:
            if run_start is None:
                run_start = idx
        elif run_start is not None:
            if best is None or idx - run_start > best[1] - best[0]:
                best = (run_start, idx)
            run_start = None
    if run_start is not None:
        end = mask.shape[0]
        if best is None or end - run_start > best[1] - best[0]:
            best = (run_start, end)
    return best

def _find_overlap_window(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    interval: float,
) -> tuple[int, int, int, int] | None:
    if pos_a.shape[0] == 0 or pos_b.shape[0] == 0:
        return None
    dists = torch.cdist(pos_a.unsqueeze(0), pos_b.unsqueeze(0)).squeeze(0)
    min_dist, nearest_b = dists.min(dim=1)
    mask = min_dist <= PATH_OVERLAP_DIST_M
    run = _find_longest_true_run(mask)
    if run is None:
        return None
    start_a, end_a = run
    min_points = max(2, int(math.ceil(PATH_OVERLAP_LEN_M / interval)) + 1)
    if end_a - start_a < min_points:
        return None
    overlap_b = nearest_b[start_a:end_a]
    start_b = int(overlap_b.min().item())
    end_b = int(overlap_b.max().item())
    overlap_a_len = (end_a - start_a - 1) * interval
    overlap_b_len = max(0, end_b - start_b) * interval
    if overlap_a_len < PATH_OVERLAP_LEN_M or overlap_b_len < PATH_OVERLAP_LEN_M:
        return None
    return start_a, end_a - 1, start_b, end_b

def _merge_overlapping_segments(
    seg_a: tuple[torch.Tensor, torch.Tensor],
    seg_b: tuple[torch.Tensor, torch.Tensor],
    interval: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    sample_a, _ = sample_polyline(seg_a[0], interval, seg_a[1])
    sample_b, _ = sample_polyline(seg_b[0], interval, seg_b[1])
    window = _find_overlap_window(sample_a, sample_b, interval)
    if window is None:
        return None
    entry_a, _exit_a, _start_b, exit_b = window
    prefix = sample_a[:entry_a + 1]
    suffix = sample_b[exit_b:]
    merged = torch.cat([prefix, suffix], dim=0)
    merged = dedupe(merged)
    return merged, arc_length(merged)

def _select_representative_segment(
    segments: list[tuple[torch.Tensor, torch.Tensor]],
    ego_xy: torch.Tensor | None,
    ego_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(segments) == 1:
        return segments[0]
    if ego_xy is None or ego_valid is None or not bool(ego_valid.any()):
        return max(segments, key=lambda seg: float(seg[1][-1].item()) if seg[1].numel() > 0 else 0.0)

    ego_pts = ego_xy[ego_valid]
    best_idx = 0
    best_dist = float('inf')
    for idx, (seg_pos, _seg_arc) in enumerate(segments):
        if seg_pos.shape[0] == 0:
            continue
        dists = torch.cdist(ego_pts.unsqueeze(0), seg_pos.unsqueeze(0)).squeeze(0)
        avg_min_dist = float(dists.min(dim=1)[0].mean().item())
        if avg_min_dist < best_dist:
            best_dist = avg_min_dist
            best_idx = idx
    return segments[best_idx]

def _sanitize_path_geometry(
    pos: torch.Tensor,
    valid: torch.Tensor,
    arc: torch.Tensor | None,
    interval: float,
    ego_xy: torch.Tensor | None = None,
    ego_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    v_idx = valid.nonzero(as_tuple=True)[0]
    if len(v_idx) == 0:
        return (
            torch.zeros((0, 2), dtype=pos.dtype, device=pos.device),
            torch.zeros(0, dtype=pos.dtype, device=pos.device),
        )

    clean_pos = pos[v_idx]
    clean_arc = arc[v_idx] if arc is not None else None
    segments = _segment_path_geometry(clean_pos, clean_arc)
    if not segments:
        return clean_pos[:1], torch.zeros(1, dtype=pos.dtype, device=pos.device)

    merged_segments: list[tuple[torch.Tensor, torch.Tensor]] = [segments[0]]
    for next_seg in segments[1:]:
        merged = _merge_overlapping_segments(merged_segments[-1], next_seg, interval)
        if merged is None:
            merged_segments.append(next_seg)
        else:
            merged_segments[-1] = merged

    rep_pos, rep_arc = _select_representative_segment(merged_segments, ego_xy, ego_valid)
    rep_pos = dedupe(rep_pos)
    return rep_pos, normalize_arc(rep_pos, rep_arc)

def infer_route_from_ego(
    ego_xy: torch.Tensor,
    ego_valid: torch.Tensor,
    path_pos: torch.Tensor,
    path_valid: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    P = path_pos.shape[0]
    is_route = torch.zeros(P, dtype=torch.bool)
    if not ego_valid.any():
        return is_route

    ego_pts = ego_xy[ego_valid]
    best_dist = float('inf')
    best_k = -1

    for k in range(P):
        if candidate_mask is not None and not bool(candidate_mask[k]):
            continue
        pv = path_valid[k]
        if not pv.any():
            continue
        p_pts = path_pos[k][pv]
        dists = torch.cdist(ego_pts.unsqueeze(0), p_pts.unsqueeze(0)).squeeze(0)
        avg_min_dist = dists.min(dim=1)[0].mean().item()
        if avg_min_dist < best_dist:
            best_dist = avg_min_dist
            best_k = k

    if best_k >= 0:
        is_route[best_k] = True
    return is_route

def resample_paths(
    sp: dict,
    interval: float,
    P_points: int,
    *,
    ego_xy: torch.Tensor | None = None,
    ego_valid: torch.Tensor | None = None,
) -> dict:
    P = sp['valid'].shape[0]
    src_pos = torch.stack([sp['x'], sp['y']], dim=-1)
    src_valid = sp['valid']
    src_arc = sp.get('arc_length')

    pos = torch.zeros((P, P_points, 2), dtype=torch.float32)
    dirs = torch.zeros((P, P_points, 2), dtype=torch.float32)
    valid = torch.zeros((P, P_points), dtype=torch.bool)
    arc = torch.zeros((P, P_points, 1), dtype=torch.float32)
    cost = torch.zeros(P, dtype=torch.float32)

    for k in range(P):
        clean_pos, _clean_arc = _sanitize_path_geometry(
            src_pos[k],
            src_valid[k],
            src_arc[k] if src_arc is not None else None,
            interval=interval,
            ego_xy=ego_xy,
            ego_valid=ego_valid,
        )
        clean_dirs = point_dirs(clean_pos)
        pieces = resample_split(
            clean_pos,
            clean_dirs,
            torch.ones(clean_pos.shape[0], dtype=torch.bool, device=clean_pos.device),
            W_target=P_points, interval=interval, mode='truncate',
        )
        if not pieces:
            continue
        p, d, v = pieces[0]
        pos[k] = p
        dirs[k] = d
        valid[k] = v
        n_valid = int(v.sum().item())
        if n_valid > 0:
            arc[k, :n_valid, 0] = torch.arange(n_valid, dtype=torch.float32) * interval

    return dict(
        near_link=torch.zeros((P, P_points), dtype=torch.int32),
        pos_xy=pos, dir_xy=dirs,
        arc_length=arc,
        is_route=sp['on_route'].bool(),
        cost=cost,
        valid=valid,
    )

def finalize_path_slots(paths: dict) -> dict:
    has_valid = paths['valid'].any(dim=-1)
    route = paths['is_route'] & has_valid
    route_idx = sorted(route.nonzero(as_tuple=True)[0].tolist(),
                       key=lambda k: float(paths['cost'][k]))
    rest_idx = [k for k in range(has_valid.shape[0]) if k not in route_idx]
    order = torch.tensor(route_idx + rest_idx, dtype=torch.long)
    out = {key: t[order] for key, t in paths.items()}
    out['cost'] = torch.where(out['is_route'] & out['valid'].any(dim=-1),
                              out['cost'], torch.zeros_like(out['cost']))
    return out

def assign_near_link(paths: dict, links: dict) -> torch.Tensor:
    P, P_points = paths['valid'].shape
    near = torch.zeros((P, P_points), dtype=torch.int32)

    link_ids = links['id'].to(torch.int32)
    link_valid = links['valid']
    pts_list: list[torch.Tensor] = []
    dir_list: list[torch.Tensor] = []
    fid_list: list[torch.Tensor] = []
    for i in range(link_ids.shape[0]):
        fid = int(link_ids[i].item())
        if fid <= 0:
            continue
        m = link_valid[i]
        vp = links['pos_xy'][i][m].to(torch.float32)
        if vp.shape[0] == 0:
            continue
        pts_list.append(vp)
        dir_list.append(links['dir_xy'][i][m].to(torch.float32))
        fid_list.append(torch.full((vp.shape[0],), fid, dtype=torch.int32))
    if not pts_list:
        return near
    link_pts = torch.cat(pts_list, dim=0)
    link_dir = torch.cat(dir_list, dim=0)
    link_dir = link_dir / link_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    link_fid = torch.cat(fid_list, dim=0)

    for k in range(P):
        v_idx = paths['valid'][k].nonzero(as_tuple=True)[0]
        if v_idx.numel() == 0:
            continue
        q = paths['pos_xy'][k][v_idx].to(torch.float32)
        d = torch.cdist(q, link_pts,
                        compute_mode='donot_use_mm_for_euclid_dist')
        min_d, nn = d.min(dim=1)
        q_dir = paths['dir_xy'][k][v_idx].to(torch.float32)
        q_dir = q_dir / q_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cos_align = (q_dir * link_dir[nn]).sum(dim=-1)
        gate = (min_d <= LINK_TRUST_RADIUS_M) & (cos_align >= NEAR_LINK_COS_MIN)
        near[k, v_idx] = torch.where(gate, link_fid[nn], torch.zeros_like(link_fid[nn]))
    return near
