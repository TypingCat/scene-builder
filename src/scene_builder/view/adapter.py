# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from ..dataspec.spec import FIELDS


def _tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.array(x))
    return x


def as_view(scene: dict) -> SimpleNamespace:
    groups = {g: SimpleNamespace(**{f: _tensor(scene[g][f]) for f in fields})
              for g, fields in FIELDS.items()}
    return SimpleNamespace(**groups, meta=dict(scene.get('meta') or {}))
