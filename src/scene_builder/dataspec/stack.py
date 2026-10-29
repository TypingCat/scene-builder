# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import numpy as np

from .spec import FIELDS


def stack(scenes: list[dict]) -> dict:
    if not scenes:
        raise ValueError("stack: 장면이 없다")
    out: dict = {}
    for group, fields in FIELDS.items():
        out[group] = {}
        for name in fields:
            items = [s[group][name] for s in scenes]
            if isinstance(items[0], np.ndarray):
                out[group][name] = np.stack(items)
            else:
                import torch
                out[group][name] = torch.stack(items)
    out['meta'] = [dict(s['meta']) for s in scenes]
    return out
