# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import torch

MEAS_SIGMA_M = 0.013
JERK_SIGMA = 0.1


def derive(pos: torch.Tensor, valid: torch.Tensor, dt: float
           ) -> tuple[torch.Tensor, torch.Tensor]:
    _pos, vel, acc = smooth_track(pos, valid, dt)
    keep = valid.unsqueeze(-1)
    return torch.where(keep, vel, torch.zeros_like(vel)), torch.where(keep, acc, torch.zeros_like(acc))


def smooth_track(pos: torch.Tensor, valid: torch.Tensor, dt: float
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, F = valid.shape
    dev, out_dtype = pos.device, pos.dtype
    N = B * 2
    z = pos.double().permute(0, 2, 1).reshape(N, F)
    m = valid.unsqueeze(1).expand(B, 2, F).reshape(N, F)

    A = torch.tensor([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]],
                     dtype=torch.float64, device=dev)
    g = torch.tensor([[dt ** 3 / 6.0], [dt * dt / 2.0], [dt]],
                     dtype=torch.float64, device=dev)
    Q = (g @ g.T) * JERK_SIGMA ** 2
    R = MEAS_SIGMA_M ** 2
    H = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=dev)
    eye = torch.eye(3, dtype=torch.float64, device=dev)

    first = m.float().argmax(dim=1)
    x = torch.zeros(N, 3, dtype=torch.float64, device=dev)
    x[:, 0] = z[torch.arange(N, device=dev), first]
    P = torch.diag(torch.tensor([1e4, 1e2, 1e2], dtype=torch.float64,
                                device=dev)).expand(N, 3, 3).clone()

    xf = torch.zeros(N, F, 3, dtype=torch.float64, device=dev)
    Pf = torch.zeros(N, F, 3, 3, dtype=torch.float64, device=dev)
    xp = torch.zeros_like(xf)
    Pp = torch.zeros_like(Pf)
    for t in range(F):
        if t > 0:
            x = x @ A.T
            P = A @ P @ A.T + Q
        xp[:, t], Pp[:, t] = x, P
        S = (P[:, 0, 0] + R).unsqueeze(-1)
        K = P[:, :, 0] / S
        y = (z[:, t] - x[:, 0]).unsqueeze(-1)
        upd = m[:, t].unsqueeze(-1)
        x = torch.where(upd, x + K * y, x)
        P = torch.where(upd.unsqueeze(-1), (eye - K.unsqueeze(-1) * H) @ P, P)
        xf[:, t], Pf[:, t] = x, P

    xs = xf.clone()
    for t in range(F - 2, -1, -1):
        C = Pf[:, t] @ A.T @ torch.linalg.inv(Pp[:, t + 1])
        xs[:, t] = xf[:, t] + (C @ (xs[:, t + 1] - xp[:, t + 1]).unsqueeze(-1)).squeeze(-1)

    est = xs.reshape(B, 2, F, 3).permute(0, 2, 3, 1).to(out_dtype)
    usable = (valid.sum(dim=1) >= 2).reshape(B, 1, 1)
    est_pos = torch.where(usable, est[:, :, 0], pos)
    return est_pos, torch.where(usable, est[:, :, 1], torch.zeros_like(pos)), \
        torch.where(usable, est[:, :, 2], torch.zeros_like(pos))
