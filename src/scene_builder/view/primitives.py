# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import plotly.graph_objects as go
import torch

from .theme import PALETTE


class Figure(go.Figure):
    def __init__(
        self,
        window_size: int = 600,
        window_margin: int = 20,
        viewport_radius: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.update_layout(
            template='plotly_white',
            dragmode='pan',
            showlegend=False,
            width=window_size,
            height=window_size,
            uirevision='scene',
            xaxis=dict(range=[-viewport_radius, viewport_radius]),
            yaxis=dict(range=[-viewport_radius, viewport_radius], scaleanchor='x', scaleratio=1),
            margin=dict(l=window_margin, r=window_margin, t=window_margin, b=window_margin),
        )


_COORD_NDIGITS = 2


def _coords(t: torch.Tensor) -> list[float]:
    return [round(v, _COORD_NDIGITS) for v in t.flatten().tolist()]


def create_markers(point: torch.Tensor, marker: dict | None = None) -> go.Scatter:
    return go.Scatter(
        x=_coords(point[..., 0]),
        y=_coords(point[..., 1]),
        mode='markers',
        marker=marker,
        hoverinfo='skip',
    )


def _nan_join(points: list[torch.Tensor]) -> torch.Tensor:
    if not points:
        return torch.empty(0, 2)
    sep = torch.full((1, 2), float('nan'), device=points[0].device)
    parts = [item for p in points for item in (p, sep)]
    return torch.cat(parts, dim=0)


def create_lines(points: list[torch.Tensor], line: dict | None = None) -> go.Scatter:
    xy = _nan_join(points)
    return go.Scatter(
        x=_coords(xy[..., 0]),
        y=_coords(xy[..., 1]),
        mode='lines',
        line=line,
        hoverinfo='skip',
    )


def create_polygons(polys: list[torch.Tensor], fillcolor: str, line_width: float = 0.0,
                    line_color: str | None = None) -> go.Scatter:
    xy = _nan_join(polys)
    return go.Scatter(
        x=_coords(xy[..., 0]),
        y=_coords(xy[..., 1]),
        mode='lines',
        fill='toself',
        fillcolor=fillcolor,
        line=dict(width=line_width, color=line_color or fillcolor),
        hoverinfo='skip',
    )


def _box_corners(
    pos_xy: torch.Tensor,
    dir_xy: torch.Tensor,
    size_xyz: torch.Tensor,
) -> list[torch.Tensor]:
    lat_dir_xy = torch.stack([-dir_xy[..., 1], dir_xy[..., 0]], dim=-1)
    rad_x = size_xyz[..., 0:1] / 2.
    rad_y = size_xyz[..., 1:2] / 2.

    pos_left_rear_xy = pos_xy - dir_xy * rad_x + lat_dir_xy * rad_y
    pos_left_front_xy = pos_xy + dir_xy * rad_x + lat_dir_xy * rad_y
    pos_right_front_xy = pos_xy + dir_xy * rad_x - lat_dir_xy * rad_y
    pos_right_rear_xy = pos_xy - dir_xy * rad_x - lat_dir_xy * rad_y
    return [pos_left_rear_xy, pos_left_front_xy, pos_right_front_xy, pos_right_rear_xy]


def boxes_to_polygons(pos_xy: torch.Tensor, dir_xy: torch.Tensor, size_xyz: torch.Tensor) -> list[torch.Tensor]:
    corners = _box_corners(pos_xy=pos_xy, dir_xy=dir_xy, size_xyz=size_xyz)
    return [torch.stack([p0, p1, p2, p3, p0]) for p0, p1, p2, p3 in zip(*corners)]


def heading_segments(pos_xy: torch.Tensor, dir_xy: torch.Tensor, r0: float, r1: float) -> list[torch.Tensor]:
    unit = dir_xy / dir_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    start = pos_xy + unit * r0
    end = pos_xy + unit * r1
    return [torch.stack([s, e]) for s, e in zip(start, end)]


def rail_segments(pos_xy: torch.Tensor, half_width: float) -> list[torch.Tensor]:
    n = int(pos_xy.shape[0])
    if n < 2:
        return []
    segs: list[torch.Tensor] = []
    for i in range(n):
        ia = 0 if i == 0 else i - 1
        ib = n - 1 if i + 1 == n else i + 1
        tangent = pos_xy[ib] - pos_xy[ia]
        tn = tangent.norm()
        if float(tn) < 1e-6:
            continue
        normal = torch.stack([-tangent[1], tangent[0]]) / tn
        offset = normal * half_width
        segs.append(torch.stack([pos_xy[i] - offset, pos_xy[i] + offset], dim=0))
    return segs


def frame_indices(last: int, stride: int) -> list[int]:
    idxs = {0, last}
    i = last
    while i >= 0:
        idxs.add(i)
        i -= stride
    return sorted(x for x in idxs if 0 <= x <= last)


def add_time_slider(figure: Figure, frame_groups: list[tuple[int, list]], *,
                    frame_unit: float, last_valid: int, label_every_s: float = 1.0) -> Figure:
    num_base = len(figure.data)
    counts = [len(tr) for _, tr in frame_groups]
    active = 0

    for k, (_, traces) in enumerate(frame_groups):
        vis = (k == active)
        for tr in traces:
            tr.visible = vis
            figure.add_trace(tr)

    steps = []
    prev_bucket = None
    for k, (f, _) in enumerate(frame_groups):
        visibility = [True] * num_base
        for j, c in enumerate(counts):
            visibility += [k == j] * c
        bucket = round(f * frame_unit / label_every_s)
        label = ''
        if bucket != prev_bucket:
            label = f'{bucket * label_every_s:.0f}s'
            if f > last_valid:
                label = f"<span style='color:{PALETTE.red}'>{label}</span>"
        prev_bucket = bucket
        steps.append(dict(method='update', args=[{'visible': visibility}], label=label))

    figure.update_layout(sliders=[dict(
        active=active,
        steps=steps,
        currentvalue=dict(visible=False),
        ticklen=3,
        minorticklen=3,
        tickwidth=1,
        pad=dict(t=30, b=0),
    )])
    return figure
