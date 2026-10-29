# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import torch

from ..dataspec.schema import empty_markers
from ..dataspec.spec import DS, MarkerType


def stop_line_markers(stop_xy: torch.Tensor, n_kept: int, id_allocator: list, W_target: int) -> dict:
    rows_xy: list[torch.Tensor] = []
    rows_id: list[int] = []
    for slot in range(n_kept):
        xy = stop_xy[slot]
        if not bool((xy != 0).any()):
            continue
        rows_id.append(id_allocator[0])
        id_allocator[0] += 1
        rows_xy.append(xy)

    k = len(rows_xy)
    pos = torch.zeros((k, W_target, 2), dtype=torch.float32)
    valid = torch.zeros((k, W_target), dtype=torch.bool)
    centroid = torch.zeros((k, 2), dtype=torch.float32)
    for i in range(k):
        pos[i, 0] = rows_xy[i]
        valid[i, 0] = True
        centroid[i] = rows_xy[i]

    return dict(
        pos=pos,
        dir=torch.zeros((k, W_target, 2), dtype=torch.float32),
        valid=valid,
        feat_id=torch.tensor(rows_id, dtype=torch.int32),
        centroid=centroid,
        marker_type=torch.full((k,), int(MarkerType.stop_line), dtype=torch.int32),
    )


def concat_marker_cats(a: dict, b: dict) -> dict:
    num_a = a['pos'].shape[0]
    num_b = b['pos'].shape[0]

    def _zeros_like(ref: torch.Tensor, num: int) -> torch.Tensor:
        return torch.zeros((num, *ref.shape[1:]), dtype=ref.dtype)

    out = {}
    for k in {*a.keys(), *b.keys()}:
        va, vb = a.get(k), b.get(k)
        if va is None:
            va = _zeros_like(vb, num_a)
        elif vb is None:
            vb = _zeros_like(va, num_b)
        out[k] = torch.cat([va, vb], dim=0)
    return out


def fill_markers(cat: dict) -> dict:
    markers = empty_markers(M=DS.num_markers, M_points=DS.num_marker_points)
    n = cat['pos'].shape[0]
    markers['id'][:n] = cat['feat_id']
    markers['type'][:n] = cat['marker_type']
    markers['pos_xy'][:n] = cat['pos']
    markers['dir_xy'][:n] = cat['dir']
    markers['valid'][:n] = cat['valid']
    return markers
