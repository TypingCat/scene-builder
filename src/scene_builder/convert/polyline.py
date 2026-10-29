# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import torch

LANE_INTERVAL = 0.5
LINE_INTERVAL = 0.5
MARKER_INTERVAL = 0.5


def _interp_along_arc(pos: torch.Tensor, dirs: torch.Tensor, target_arc: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    seg = (pos[1:] - pos[:-1]).norm(dim=-1)
    arc = torch.cat([torch.zeros(1, dtype=pos.dtype), seg.cumsum(0)])
    V = pos.shape[0]
    idx_hi = torch.searchsorted(arc, target_arc, right=False).clamp(1, V - 1)
    idx_lo = (idx_hi - 1).clamp(min=0)
    a_lo = arc[idx_lo]
    a_hi = arc[idx_hi]
    ratio = ((target_arc - a_lo) / (a_hi - a_lo).clamp(min=1e-9)).clamp(0, 1).unsqueeze(-1)
    pos_out = pos[idx_lo] + (pos[idx_hi] - pos[idx_lo]) * ratio
    dir_out = dirs[idx_lo] + (dirs[idx_hi] - dirs[idx_lo]) * ratio
    dir_out = dir_out / dir_out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return pos_out, dir_out


def arc_length(pos: torch.Tensor) -> torch.Tensor:
    if pos.shape[0] == 0:
        return torch.zeros(0, dtype=pos.dtype, device=pos.device)
    if pos.shape[0] == 1:
        return torch.zeros(1, dtype=pos.dtype, device=pos.device)
    seg = (pos[1:] - pos[:-1]).norm(dim=-1)
    return torch.cat([torch.zeros(1, dtype=pos.dtype, device=pos.device), seg.cumsum(0)])


def point_dirs(pos: torch.Tensor) -> torch.Tensor:
    V = pos.shape[0]
    if V == 0:
        return torch.zeros((0, 2), dtype=pos.dtype, device=pos.device)
    if V == 1:
        return torch.tensor([[1.0, 0.0]], dtype=pos.dtype, device=pos.device)
    seg = pos[1:] - pos[:-1]
    seg_dir = seg / seg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dirs = torch.zeros((V, 2), dtype=pos.dtype, device=pos.device)
    dirs[0] = seg_dir[0]
    dirs[-1] = seg_dir[-1]
    if V > 2:
        dirs[1:-1] = seg_dir[:-1] + seg_dir[1:]
        dirs[1:-1] = dirs[1:-1] / dirs[1:-1].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return dirs


def normalize_arc(pos: torch.Tensor, arc: torch.Tensor | None) -> torch.Tensor:
    if arc is None or arc.shape[0] != pos.shape[0] or pos.shape[0] == 0:
        return arc_length(pos)
    arc = arc.to(dtype=pos.dtype, device=pos.device)
    arc = arc - arc[0]
    if pos.shape[0] > 1:
        if bool((torch.diff(arc) < -1e-6).any()) or float(arc[-1].item()) <= 0.0:
            return arc_length(pos)
    return arc


def sample_polyline(pos: torch.Tensor, interval: float, arc: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    if pos.shape[0] == 0:
        return pos, torch.zeros(0, dtype=pos.dtype, device=pos.device)
    if pos.shape[0] == 1:
        return pos.clone(), torch.zeros(1, dtype=pos.dtype, device=pos.device)
    arc = normalize_arc(pos, arc)
    total = float(arc[-1].item())
    if total <= 0.0:
        return pos[:1].clone(), torch.zeros(1, dtype=pos.dtype, device=pos.device)
    G = int(total / interval) + 1
    target_arc = torch.clamp(torch.arange(G, dtype=pos.dtype, device=pos.device) * interval, max=total)
    if target_arc.numel() == 0 or float(target_arc[-1].item()) < total:
        target_arc = torch.cat([target_arc, torch.tensor([total], dtype=pos.dtype, device=pos.device)])
    sample_pos, _ = _interp_along_arc(pos, point_dirs(pos), target_arc)
    return sample_pos, target_arc


def dedupe(pos: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    if pos.shape[0] <= 1:
        return pos
    keep = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
    keep[1:] = (pos[1:] - pos[:-1]).norm(dim=-1) > eps
    deduped = pos[keep]
    return deduped if deduped.shape[0] > 0 else pos[:1]


def resample_split(pos: torch.Tensor, dirs: torch.Tensor, valid: torch.Tensor,
                   W_target: int, interval: float, *, mode: str = 'split'
                   ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    v_idx = valid.nonzero(as_tuple=True)[0]
    if len(v_idx) == 0:
        return []
    vp = pos[v_idx]
    vd = dirs[v_idx]

    def _empty():
        return (torch.zeros((W_target, 2), dtype=pos.dtype),
                torch.zeros((W_target, 2), dtype=dirs.dtype),
                torch.zeros(W_target, dtype=torch.bool))

    def _point_piece():
        op, od, ov = _empty()
        op[0], od[0], ov[0] = vp[0], vd[0], True
        return op, od, ov

    if len(v_idx) < 2:
        return [_point_piece()]

    total = (vp[1:] - vp[:-1]).norm(dim=-1).sum().item()
    if total < interval and mode != 'fit_single':
        return [_point_piece()]

    if mode == 'fit_single':
        G = W_target
        target_arc = torch.linspace(0, total, G, dtype=pos.dtype)
    else:
        G = int(total / interval) + 1
        target_arc = torch.clamp(torch.arange(G, dtype=pos.dtype) * interval, max=total)
    ip, id_ = _interp_along_arc(vp, vd, target_arc)

    if mode in ('truncate', 'fit_single'):
        n = min(G, W_target)
        op, od, ov = _empty()
        op[:n], od[:n], ov[:n] = ip[:n], id_[:n], True
        return [(op, od, ov)]

    n_pieces = max(1, (G + W_target - 1) // W_target)
    base, rem = divmod(G, n_pieces)
    pieces = []
    start = 0
    for i in range(n_pieces):
        size = base + (1 if i < rem else 0)
        end = start + size
        op, od, ov = _empty()
        op[:size], od[:size], ov[:size] = ip[start:end], id_[start:end], True
        pieces.append((op, od, ov))
        start = end
    return pieces


def resample_split_batch(src_indices: torch.Tensor, roads_pos: torch.Tensor,
                         roads_dir: torch.Tensor, roads_valid: torch.Tensor,
                         W_target: int, interval: float, *, mode: str = 'split'):
    all_pos, all_dir, all_valid = [], [], []
    parents: list[int] = []
    piece_idxs: list[int] = []
    for si in src_indices.tolist():
        pieces = resample_split(roads_pos[si], roads_dir[si], roads_valid[si],
                                W_target, interval, mode=mode)
        for pi, (p, d, v) in enumerate(pieces):
            all_pos.append(p)
            all_dir.append(d)
            all_valid.append(v)
            parents.append(si)
            piece_idxs.append(pi)
    if not all_pos:
        return (torch.zeros((0, W_target, 2), dtype=roads_pos.dtype),
                torch.zeros((0, W_target, 2), dtype=roads_dir.dtype),
                torch.zeros((0, W_target), dtype=torch.bool), [], [])
    return torch.stack(all_pos), torch.stack(all_dir), torch.stack(all_valid), parents, piece_idxs


def piece_centroid(pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return pos[0]
    vc = valid.sum().clamp(min=1).to(pos.dtype)
    return (pos * valid.unsqueeze(-1)).sum(dim=0) / vc


def piece_endpoints(pos: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v_idx = valid.nonzero(as_tuple=True)[0]
    if len(v_idx) == 0:
        return pos[0], pos[0]
    return pos[v_idx[0]], pos[v_idx[-1]]


def sort_by_distance(ent_pos: torch.Tensor, ref_pts: torch.Tensor, max_count: int) -> torch.Tensor:
    if ent_pos.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)
    if ref_pts.dim() == 1:
        dist = (ent_pos - ref_pts.unsqueeze(0)).norm(dim=-1)
    else:
        dist = (ent_pos.unsqueeze(1) - ref_pts.unsqueeze(0)).norm(dim=-1).min(dim=1)[0]
    _, order = dist.sort()
    return order[:max_count]


def sort_by_encounter(agent_pos: torch.Tensor, agent_valid: torch.Tensor,
                      ego_pos: torch.Tensor, ego_valid: torch.Tensor,
                      max_count: int, always_first: torch.Tensor) -> torch.Tensor:
    if agent_pos.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)
    d = (agent_pos - ego_pos.unsqueeze(0)).norm(dim=-1)
    covalid = agent_valid & ego_valid.unsqueeze(0)
    dist = torch.where(covalid, d, torch.full_like(d, float('inf'))).min(dim=1)[0]
    dist = torch.where(always_first, torch.full_like(dist, -1.0), dist)
    _, order = dist.sort()
    return order[:max_count]


def distance_truncate(cat: dict, ego_pos: torch.Tensor | None, max_count: int) -> dict:
    num_pieces = cat['pos'].shape[0]
    if num_pieces == 0:
        return cat
    if ego_pos is not None:
        order = sort_by_distance(cat['centroid'], ego_pos, max_count)
    else:
        order = torch.arange(min(num_pieces, max_count), dtype=torch.long)
    return {k: (v[order] if isinstance(v, torch.Tensor) and v.shape[0] == num_pieces else v)
            for k, v in cat.items()}
