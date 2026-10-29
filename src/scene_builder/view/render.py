# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

from plotly.subplots import make_subplots

from ..dataspec.spec import DS
from .adapter import as_view
from .layers import plot_scene

NOW = 10


def _title(scene: dict) -> str:
    sid = str((scene.get('meta') or {}).get('scene_id', ''))
    v = as_view(scene)
    n_agents = int(v.agents.valid.any(dim=-1).sum())
    n_links = int(v.links.valid.any(dim=-1).sum())
    n_lights = int(v.traffics.valid.any(dim=-1).sum())
    return f'{sid[:16]} · agents {n_agents} · links {n_links} · lights {n_lights}'


def scene_figure(scene: dict, *, slider: bool, stride: int = 5, radius: int = 50, now: int = NOW):
    fig = plot_scene(
        as_view(scene),
        now=now,
        frame_unit=DS.frame_unit,
        slider=slider,
        frame_stride=stride,
        viewport_radius=radius,
    )
    fig.update_layout(title=dict(text=_title(scene), font=dict(size=12)),
                      margin=dict(t=40))
    return fig


def write_html(scene: dict, path: str, *, stride: int = 5, radius: int = 50) -> None:
    scene_figure(scene, slider=True, stride=stride, radius=radius).write_html(
        path, include_plotlyjs=True, full_html=True)


def write_png(scene: dict, path: str, *, radius: int = 50, now: int = NOW, scale: float = 2.0) -> None:
    scene_figure(scene, slider=False, radius=radius, now=now).write_image(path, scale=scale)


def write_grid(scenes: list[dict], path: str, *, cols: int = 3, radius: int = 50,
               now: int = NOW, cell_px: int = 420, scale: float = 2.0) -> None:
    rows = (len(scenes) + cols - 1) // cols
    grid = make_subplots(rows=rows, cols=cols, subplot_titles=[_title(s) for s in scenes],
                         horizontal_spacing=0.02, vertical_spacing=0.05)
    for k, scene in enumerate(scenes):
        r, c = k // cols + 1, k % cols + 1
        fig = scene_figure(scene, slider=False, radius=radius, now=now)
        for tr in fig.data:
            grid.add_trace(tr, row=r, col=c)
        xr = fig.layout.xaxis.range
        yr = fig.layout.yaxis.range
        axis_id = '' if k == 0 else str(k + 1)
        grid.update_xaxes(range=xr, showticklabels=False, row=r, col=c)
        grid.update_yaxes(range=yr, showticklabels=False, row=r, col=c,
                          scaleanchor=f'x{axis_id}', scaleratio=1)
    grid.update_layout(template='plotly_white', showlegend=False,
                       width=cell_px * cols, height=cell_px * rows,
                       margin=dict(l=10, r=10, t=40, b=10))
    grid.update_annotations(font_size=10)
    grid.write_image(path, scale=scale)
