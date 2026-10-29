# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import torch

TIME_KEYS = ('log_trajectory', 'log_traffic_light')


def _fit(v, num_frames: int):
    if not isinstance(v, torch.Tensor) or v.dim() < 2:
        return v
    cur = v.shape[-1]
    if cur >= num_frames:
        return v[..., :num_frames].clone()
    tail = torch.zeros((*v.shape[:-1], num_frames - cur), dtype=v.dtype, device=v.device)
    return torch.cat([v, tail], dim=-1)


def fit_frames(raw: dict, num_frames: int) -> dict:
    out = dict(raw)
    for key in TIME_KEYS:
        out[key] = {k: _fit(v, num_frames) for k, v in raw[key].items()}
    return out
