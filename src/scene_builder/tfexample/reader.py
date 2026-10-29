# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

from typing import Iterator

import numpy as np
import tensorflow as tf
import torch

from ..convert.codes import RoadType
from ..dataspec.spec import DS
from .roadgraph import assemble_roadgraph

HZ = 10
PAST = 10
FUTURE = 80
_NUM_AGENTS = 128
_NUM_LIGHTS = 16
_NUM_RG_POINTS = 30000
_NUM_PATHS = 45
_NUM_PATH_POINTS = 800
_TIMES = {'past': PAST, 'current': 1, 'future': FUTURE}

_ROAD_TYPE = np.zeros(256, dtype=np.int32)
for _src, _dst in {
    1: RoadType.lane_center_freeway,
    2: RoadType.lane_center_surface_street,
    3: RoadType.lane_center_bike_lane,
    6: RoadType.road_line_broken_single_white,
    7: RoadType.road_line_solid_single_white,
    8: RoadType.road_line_solid_double_white,
    9: RoadType.road_line_broken_single_yellow,
    10: RoadType.road_line_broken_double_yellow,
    11: RoadType.road_line_solid_single_yellow,
    12: RoadType.road_line_solid_double_yellow,
    13: RoadType.road_line_passing_double_yellow,
    15: RoadType.road_edge_boundary,
    16: RoadType.road_edge_median,
    17: RoadType.stop_sign,
    18: RoadType.crosswalk,
    19: RoadType.speed_bump,
    20: RoadType.driveway,
}.items():
    _ROAD_TYPE[_src] = _dst


def _features() -> dict:
    F = tf.io.FixedLenFeature
    f = {
        'scenario/id': F([], tf.string),
        'roadgraph_samples/xyz': F([_NUM_RG_POINTS, 3], tf.float32),
        'roadgraph_samples/dir': F([_NUM_RG_POINTS, 3], tf.float32),
        'roadgraph_samples/id': F([_NUM_RG_POINTS, 1], tf.int64),
        'roadgraph_samples/type': F([_NUM_RG_POINTS, 1], tf.int64),
        'roadgraph_samples/valid': F([_NUM_RG_POINTS, 1], tf.int64),
        'state/id': F([_NUM_AGENTS], tf.float32),
        'state/type': F([_NUM_AGENTS], tf.float32),
        'state/is_sdc': F([_NUM_AGENTS], tf.int64),
        'path_samples/xyz': F([_NUM_PATHS, _NUM_PATH_POINTS, 3], tf.float32),
        'path_samples/valid': F([_NUM_PATHS, _NUM_PATH_POINTS], tf.int64),
        'path_samples/arc_length': F([_NUM_PATHS, _NUM_PATH_POINTS], tf.float32),
        'path_samples/on_route': F([_NUM_PATHS, 1], tf.int64),
    }
    for t, n in _TIMES.items():
        for name in ('x', 'y', 'bbox_yaw', 'length', 'width', 'height'):
            f[f'state/{t}/{name}'] = F([_NUM_AGENTS, n], tf.float32)
        f[f'state/{t}/valid'] = F([_NUM_AGENTS, n], tf.int64)
        for name in ('state', 'valid', 'id'):
            f[f'traffic_light_state/{t}/{name}'] = F([n, _NUM_LIGHTS], tf.int64)
        for name in ('x', 'y'):
            f[f'traffic_light_state/{t}/{name}'] = F([n, _NUM_LIGHTS], tf.float32)
    return f


def _along_time(ex: dict, prefix: str, name: str, axis: int) -> np.ndarray:
    return np.concatenate([ex[f'{prefix}/{t}/{name}'].numpy() for t in _TIMES], axis=axis)


def _trajectory(ex: dict) -> dict:
    def get(name):
        return torch.from_numpy(_along_time(ex, 'state', name, axis=-1))
    return dict(
        x=get('x'), y=get('y'), yaw=get('bbox_yaw'),
        valid=torch.from_numpy(_along_time(ex, 'state', 'valid', axis=-1).astype(bool)),
        length=get('length'), width=get('width'), height=get('height'),
    )


def _agents_meta(ex: dict) -> dict:
    ids = ex['state/id'].numpy().astype(np.int32)
    types = ex['state/type'].numpy().astype(np.int32)
    pad = ids < 0
    ids[pad] = 0
    types[pad] = 0
    is_sdc = ex['state/is_sdc'].numpy() == 1
    return dict(ids=torch.from_numpy(ids), object_types=torch.from_numpy(types),
                is_sdc=torch.from_numpy(is_sdc))


def fill_end_dirs(dir_xyz: np.ndarray, rg_id: np.ndarray, rg_valid: np.ndarray) -> np.ndarray:
    out = dir_xyz.copy()
    n = np.linalg.norm(out[:, :2], axis=-1)
    for i in np.flatnonzero(rg_valid & (n < 1e-6)):
        j = i - 1
        while j >= 0 and rg_valid[j] and rg_id[j] == rg_id[i]:
            if np.linalg.norm(out[j, :2]) >= 1e-6:
                out[i] = out[j]
                break
            j -= 1
    return out


def _road_points(ex: dict) -> dict:
    xyz = ex['roadgraph_samples/xyz'].numpy()
    ids = ex['roadgraph_samples/id'].numpy().squeeze(-1).astype(np.int32)
    types = _ROAD_TYPE[np.clip(ex['roadgraph_samples/type'].numpy().squeeze(-1).astype(np.int32), 0, 255)]
    valid = ex['roadgraph_samples/valid'].numpy().squeeze(-1).astype(bool)
    dirs = fill_end_dirs(ex['roadgraph_samples/dir'].numpy(), ids, valid)
    return dict(x=xyz[:, 0], y=xyz[:, 1], dir_x=dirs[:, 0], dir_y=dirs[:, 1],
                types=types, ids=ids, valid=valid)


def _traffic_lights(ex: dict) -> dict:
    def get(name, dtype):
        return torch.from_numpy(_along_time(ex, 'traffic_light_state', name, axis=0).astype(dtype).T)
    return dict(
        x=get('x', np.float32), y=get('y', np.float32),
        state=get('state', np.int32), lane_ids=get('id', np.int32),
        valid=get('valid', bool),
    )


def _paths(ex: dict) -> dict:
    K, Q = DS.num_paths, DS.num_path_points
    xyz = ex['path_samples/xyz'].numpy()
    valid = ex['path_samples/valid'].numpy().astype(bool)
    arc = ex['path_samples/arc_length'].numpy()
    on_route = ex['path_samples/on_route'].numpy().squeeze(-1).astype(np.int32)

    selected = np.argsort(-(on_route * 1_000_000 + valid.sum(axis=-1)))[:K]

    out_x = np.zeros((K, Q), dtype=np.float32)
    out_y = np.zeros((K, Q), dtype=np.float32)
    out_valid = np.zeros((K, Q), dtype=bool)
    out_arc = np.zeros((K, Q), dtype=np.float32)
    out_on_route = np.zeros(K, dtype=np.int32)
    for i, k in enumerate(selected):
        v = valid[k]
        if not v.any():
            continue
        last = np.nonzero(v)[0][-1]
        if last + 1 > Q:
            idx = np.round(np.linspace(0, last, Q)).astype(int)
        else:
            idx = np.arange(last + 1)
        n = len(idx)
        out_x[i, :n] = xyz[k, idx, 0]
        out_y[i, :n] = xyz[k, idx, 1]
        out_valid[i, :n] = v[idx]
        out_arc[i, :n] = arc[k, idx]
        out_on_route[i] = on_route[k]
    return dict(x=torch.from_numpy(out_x), y=torch.from_numpy(out_y),
                valid=torch.from_numpy(out_valid), arc_length=torch.from_numpy(out_arc),
                on_route=torch.from_numpy(out_on_route))


def _ego_xy(meta: dict, traj: dict) -> tuple[float, float] | None:
    if not bool(meta['is_sdc'].any()):
        return None
    k = int(meta['is_sdc'].nonzero(as_tuple=True)[0][0])
    valid = traj['valid'][k]
    if not bool(valid.any()):
        return None
    t = PAST if bool(valid[PAST]) else int(valid.nonzero(as_tuple=True)[0][0])
    return float(traj['x'][k, t]), float(traj['y'][k, t])


def iter_scenes(tfrecord_path: str) -> Iterator[dict]:
    features = _features()
    for record in tf.data.TFRecordDataset(tfrecord_path):
        ex = tf.io.parse_single_example(record, features)
        traj = _trajectory(ex)
        meta = _agents_meta(ex)
        roads, lane_graph = assemble_roadgraph(_road_points(ex), ego_xy=_ego_xy(meta, traj))
        yield dict(
            log_trajectory=traj,
            object_metadata=meta,
            log_traffic_light=_traffic_lights(ex),
            sdc_paths=_paths(ex),
            roads=roads,
            lane_graph=lane_graph,
            _scene_id=ex['scenario/id'].numpy().decode('utf-8'),
            _source_hz=HZ,
        )
