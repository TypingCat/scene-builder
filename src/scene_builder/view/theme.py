# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

from types import SimpleNamespace

from ..dataspec.spec import LaneType, MarkerType, TrafficState

PALETTE = SimpleNamespace(
    background='#FFFFFF',
    ink='#171A1F',
    gray='#8B919A',
    surface='#D8D4CC',
    purple='#6B73E8',
    blue='#4A79F2',
    sky='#86EBDD',
    butter='#F2C14E',
    red='#FF8A80',
    green='#72C98F',
)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


LANE_STYLE = {
    LaneType.unknown:  (PALETTE.gray,   1.0, None,      1.0),
    LaneType.center:   (PALETTE.butter, 1.5, None,      1.0),
    LaneType.solid:    (PALETTE.gray,   1.0, None,      1.0),
    LaneType.dashed:   (PALETTE.gray,   1.0, '6px,6px', 0.7),
    LaneType.edge:     (PALETTE.butter, 1.5, None,      1.0),
    LaneType.boundary: (PALETTE.red,    2.0, None,      1.0),
    LaneType.median:   (PALETTE.red,    2.0, None,      1.0),
}

TRAFFIC_COLOR = {
    TrafficState.unknown: PALETTE.gray,
    TrafficState.go:      PALETTE.green,
    TrafficState.caution: PALETTE.butter,
    TrafficState.stop:    PALETTE.red,
}

AGENT = SimpleNamespace(
    fill=PALETTE.gray,
    center_fill=PALETTE.gray,
    center_border=PALETTE.background,
    ego_center_border=PALETTE.sky,
    center_px=7,
    center_border_px=1.0,
    tick_color=PALETTE.background,
    ego_tick_color=PALETTE.sky,
    tick_r0_m=0.5,
    tick_r1_m=1.5,
    tick_px=1.5,
)

SIGNAL = SimpleNamespace(
    ring_color=PALETTE.gray,
    ring_px=8,
    tick_r0_m=0.45,
    tick_r1_m=0.90,
)

LINK = SimpleNamespace(base_alpha=0.3, base_px=2, state_px=3)

MARKER = SimpleNamespace(point_px=4)

ROUTE = SimpleNamespace(color=PALETTE.sky, other_color=PALETTE.gray, half_width_factor=0.25, fallback_half_width_m=0.45, width_px=1.5)


POLYGON_MARKER_TYPES = {MarkerType.crosswalk, MarkerType.speed_bump, MarkerType.driveway}
