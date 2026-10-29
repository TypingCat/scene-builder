# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import torch

from ..dataspec.spec import LaneType, LinkSection, LinkType, MarkerType, TrafficState


class RoadType:
    unknown = 0
    lane_center_freeway = 1
    lane_center_surface_street = 2
    lane_center_bike_lane = 3
    road_line_broken_single_white = 4
    road_line_solid_single_white = 5
    road_line_solid_double_white = 6
    road_line_broken_single_yellow = 7
    road_line_broken_double_yellow = 8
    road_line_solid_single_yellow = 9
    road_line_solid_double_yellow = 10
    road_line_passing_double_yellow = 11
    road_edge_boundary = 12
    road_edge_median = 13
    stop_sign = 14
    crosswalk = 15
    speed_bump = 16
    driveway = 17

    @staticmethod
    def is_lane_center(t: torch.Tensor) -> torch.Tensor:
        return (t >= RoadType.lane_center_freeway) & (t <= RoadType.lane_center_bike_lane)

    @staticmethod
    def is_road_line(t: torch.Tensor) -> torch.Tensor:
        return (t >= RoadType.road_line_broken_single_white) & (t <= RoadType.road_line_passing_double_yellow)

    @staticmethod
    def is_road_edge(t: torch.Tensor) -> torch.Tensor:
        return (t >= RoadType.road_edge_boundary) & (t <= RoadType.road_edge_median)

    @staticmethod
    def is_marker(t: torch.Tensor) -> torch.Tensor:
        return (t >= RoadType.stop_sign) & (t <= RoadType.driveway)


def _map(t: torch.Tensor, table: dict[int, int]) -> torch.Tensor:
    out = torch.zeros_like(t)
    for dst, srcs in table.items():
        mask = torch.zeros_like(t, dtype=torch.bool)
        for s in srcs:
            mask |= t == s
        out = torch.where(mask, out.new_full([], dst), out)
    return out


def traffic_state_from_waymo(state: torch.Tensor) -> torch.Tensor:
    return _map(state, {
        TrafficState.go: (3, 6),
        TrafficState.caution: (2, 5, 8),
        TrafficState.stop: (1, 4, 7),
    })


def link_type_from_road_type(t: torch.Tensor) -> torch.Tensor:
    return _map(t, {
        LinkType.vehicle: (RoadType.lane_center_freeway, RoadType.lane_center_surface_street),
        LinkType.cyclist: (RoadType.lane_center_bike_lane,),
    })


def link_section_from_road_type(t: torch.Tensor) -> torch.Tensor:
    return _map(t, {LinkSection.freeway: (RoadType.lane_center_freeway,)})


def lane_type_from_road_type(t: torch.Tensor) -> torch.Tensor:
    return _map(t, {
        LaneType.dashed: (RoadType.road_line_broken_single_white,
                          RoadType.road_line_broken_single_yellow,
                          RoadType.road_line_broken_double_yellow),
        LaneType.solid: (RoadType.road_line_solid_single_white,
                         RoadType.road_line_solid_double_white),
        LaneType.center: (RoadType.road_line_solid_single_yellow,
                          RoadType.road_line_solid_double_yellow,
                          RoadType.road_line_passing_double_yellow),
        LaneType.edge: (RoadType.road_edge_boundary,),
        LaneType.median: (RoadType.road_edge_median,),
    })


def marker_type_from_road_type(t: torch.Tensor) -> torch.Tensor:
    return _map(t, {
        MarkerType.stop_sign: (RoadType.stop_sign,),
        MarkerType.crosswalk: (RoadType.crosswalk,),
        MarkerType.speed_bump: (RoadType.speed_bump,),
        MarkerType.driveway: (RoadType.driveway,),
    })
