# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import torch

from .spec import DS


def empty_agents(A: int, F: int) -> dict:
    return dict(
        id=torch.zeros(A, dtype=torch.int32),
        type=torch.zeros(A, dtype=torch.int32),
        is_ego=torch.zeros(A, dtype=torch.bool),
        size_xyz=torch.zeros((A, 3), dtype=torch.float32),
        time_s=torch.zeros((A, F, 1), dtype=torch.float32),
        pos_xy=torch.zeros((A, F, 2), dtype=torch.float32),
        dir_xy=torch.zeros((A, F, 2), dtype=torch.float32),
        vel_xy=torch.zeros((A, F, 2), dtype=torch.float32),
        acc_xy=torch.zeros((A, F, 2), dtype=torch.float32),
        valid=torch.zeros((A, F), dtype=torch.bool),
    )


def empty_links(
    I: int,
    I_points: int,
    I_connections: int,
    I_adjacents: int,
    I_boundaries: int,
) -> dict:
    return dict(
        id=torch.zeros(I, dtype=torch.int32),
        type=torch.zeros(I, dtype=torch.int32),
        section=torch.zeros(I, dtype=torch.int32),
        speed_limit=torch.zeros(I, dtype=torch.float32),
        pos_xy=torch.zeros((I, I_points, 2), dtype=torch.float32),
        dir_xy=torch.zeros((I, I_points, 2), dtype=torch.float32),
        valid=torch.zeros((I, I_points), dtype=torch.bool),
        entry_links=torch.zeros((I, I_connections), dtype=torch.int32),
        exit_links=torch.zeros((I, I_connections), dtype=torch.int32),
        left_links=torch.zeros((I, I_adjacents), dtype=torch.int32),
        right_links=torch.zeros((I, I_adjacents), dtype=torch.int32),
        left_lanes=torch.zeros((I, I_boundaries), dtype=torch.int32),
        right_lanes=torch.zeros((I, I_boundaries), dtype=torch.int32),
    )


def empty_lanes(L: int, L_points: int) -> dict:
    return dict(
        id=torch.zeros(L, dtype=torch.int32),
        type=torch.zeros(L, dtype=torch.int32),
        pos_xy=torch.zeros((L, L_points, 2), dtype=torch.float32),
        dir_xy=torch.zeros((L, L_points, 2), dtype=torch.float32),
        valid=torch.zeros((L, L_points), dtype=torch.bool),
    )


def empty_markers(M: int, M_points: int) -> dict:
    return dict(
        id=torch.zeros(M, dtype=torch.int32),
        type=torch.zeros(M, dtype=torch.int32),
        pos_xy=torch.zeros((M, M_points, 2), dtype=torch.float32),
        dir_xy=torch.zeros((M, M_points, 2), dtype=torch.float32),
        valid=torch.zeros((M, M_points), dtype=torch.bool),
    )


def empty_traffics(T: int, F: int) -> dict:
    return dict(
        id=torch.zeros(T, dtype=torch.int32),
        type=torch.zeros(T, dtype=torch.int32),
        target_link=torch.zeros(T, dtype=torch.int32),
        dir_xy=torch.zeros((T, 2), dtype=torch.float32),
        pos_xy=torch.zeros((T, 2), dtype=torch.float32),
        state=torch.zeros((T, F), dtype=torch.int32),
        time_s=torch.zeros((T, F, 1), dtype=torch.float32),
        valid=torch.zeros((T, F), dtype=torch.bool),
    )


def empty_paths(P: int, P_points: int) -> dict:
    return dict(
        near_link=torch.zeros((P, P_points), dtype=torch.int32),
        pos_xy=torch.zeros((P, P_points, 2), dtype=torch.float32),
        dir_xy=torch.zeros((P, P_points, 2), dtype=torch.float32),
        arc_length=torch.zeros((P, P_points, 1), dtype=torch.float32),
        is_route=torch.zeros(P, dtype=torch.bool),
        cost=torch.zeros(P, dtype=torch.float32),
        valid=torch.zeros((P, P_points), dtype=torch.bool),
    )


def empty_goal() -> dict:
    return dict(
        pos_xy=torch.zeros(2, dtype=torch.float32),
        dir_xy=torch.zeros(2, dtype=torch.float32),
        valid=torch.zeros((), dtype=torch.bool),
    )


def empty_meta() -> dict:
    return dict(scene_id='', map_id=None, anchor_xy=None)


def empty_scene() -> dict:
    return dict(
        agents=empty_agents(A=DS.num_agents, F=DS.num_frames),
        links=empty_links(
            I=DS.num_links, I_points=DS.num_link_points,
            I_connections=DS.num_link_connections,
            I_adjacents=DS.num_link_adjacents,
            I_boundaries=DS.num_link_boundaries,
        ),
        lanes=empty_lanes(L=DS.num_lanes, L_points=DS.num_lane_points),
        traffics=empty_traffics(T=DS.num_traffics, F=DS.num_frames),
        markers=empty_markers(M=DS.num_markers, M_points=DS.num_marker_points),
        paths=empty_paths(P=DS.num_paths, P_points=DS.num_path_points),
        goal=empty_goal(),
        meta=empty_meta(),
    )
