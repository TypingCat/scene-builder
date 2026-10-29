# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import json
import os
import pickle
import struct
from contextlib import contextmanager

import numpy as np

_INT64 = 8
_HEAD = 4 * _INT64
_FORMAT_VERSION = 1
_BUFFER_SIZE = 1024 * 1024

SCENES_EXT = '.scenes'
INDEX_NAME = 'index.pkl'
PART_EXT = '.part'
STAGE_BUILT = 'built'


def make_stamp(dataspec: dict) -> dict:
    import torch
    return {
        'format_version': _FORMAT_VERSION,
        'stage': STAGE_BUILT,
        'pickle_protocol': pickle.HIGHEST_PROTOCOL,
        'torch_version': str(torch.__version__),
        'dataspec': dict(dataspec),
    }


@contextmanager
def _publish(path: str):
    tmp = f'{path}{PART_EXT}.{os.getpid()}'
    try:
        with open(tmp, 'wb') as f:
            yield f
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def _read_header(f) -> tuple[int, dict, int]:
    raw = f.read(_HEAD)
    if len(raw) < _HEAD:
        raise ValueError(f"장면 샤드가 아니다: {getattr(f, 'name', '')}")
    header_size, version, count, stamp_len = struct.unpack('<qqqq', raw)
    if stamp_len <= 0 or header_size != _HEAD + stamp_len or count < 0:
        raise ValueError(f"장면 샤드가 아니다: {getattr(f, 'name', '')}")
    if version != _FORMAT_VERSION:
        raise ValueError(f"샤드 형식 v{version} 은 읽을 수 없다 (v{_FORMAT_VERSION} 만 읽는다)")
    stamp = json.loads(f.read(stamp_len).decode('utf-8'))
    return count, stamp, header_size


def read_count(shard_path: str) -> int:
    with open(shard_path, 'rb') as f:
        return _read_header(f)[0]


def read_stamp(shard_path: str) -> dict:
    with open(shard_path, 'rb') as f:
        return _read_header(f)[1]


def check_stamp(stamp: dict, *, dataspec: dict) -> None:
    if stamp.get('stage') != STAGE_BUILT:
        raise ValueError(f"샤드 stage 가 다르다: {stamp.get('stage')!r} != {STAGE_BUILT!r}")
    got = stamp.get('dataspec', {})
    mismatch = {k: (v, got.get(k)) for k, v in dataspec.items() if got.get(k) != v}
    if mismatch:
        raise ValueError("샤드 dataspec 이 기대와 다르다 (키: (기대, 샤드)): " + repr(mismatch))


def write_scene_shard(scenes: list[dict], path: str, *, stamp: dict, filename: str) -> str:
    os.makedirs(path, exist_ok=True)
    blobs = [pickle.dumps(s, protocol=pickle.HIGHEST_PROTOCOL) for s in scenes]
    n = len(scenes)

    stamp = dict(stamp)
    if scenes:
        stamp['payload_keys'] = {
            k: sorted(v) if isinstance(v, dict) else None for k, v in scenes[0].items()}
    stamp_bytes = json.dumps(stamp, ensure_ascii=False, sort_keys=True).encode('utf-8')
    head = struct.pack('<qqqq', _HEAD + len(stamp_bytes), _FORMAT_VERSION, n, len(stamp_bytes)) + stamp_bytes

    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = len(head) + _INT64 * (n + 1)
    for j, blob in enumerate(blobs):
        offsets[j + 1] = offsets[j] + len(blob)

    with _publish(os.path.join(path, filename)) as f:
        f.write(head)
        f.write(offsets.tobytes())
        for blob in blobs:
            f.write(blob)
    return filename


def write_scene_index(path: str) -> None:
    shard_files = sorted(f for f in os.listdir(path) if f.endswith(SCENES_EXT))
    shard_sizes = [read_count(os.path.join(path, name)) for name in shard_files]
    with _publish(os.path.join(path, INDEX_NAME)) as f:
        pickle.dump({'shard_sizes': shard_sizes, 'shard_names': shard_files}, f)


class SceneDataset:
    def __init__(self, path: str, *, expect_dataspec: dict | None = None, loads=pickle.loads):
        path = os.path.expanduser(path)
        self.path = path
        self._loads = loads
        with open(os.path.join(path, INDEX_NAME), 'rb') as f:
            index = pickle.load(f)

        self._shard_paths = [os.path.join(path, name) for name in index['shard_names']]
        self._shard_sizes = list(index['shard_sizes'])
        self._shardpoints = np.cumsum([0] + self._shard_sizes)
        self._offsets_cache: dict[int, np.ndarray] = {}

        self.stamp = read_stamp(self._shard_paths[0]) if self._shard_paths else None
        if expect_dataspec is not None and self.stamp is not None:
            check_stamp(self.stamp, dataspec=expect_dataspec)

    @property
    def shard_paths(self) -> list[str]:
        return list(self._shard_paths)

    @property
    def shard_sizes(self) -> list[int]:
        return list(self._shard_sizes)

    def __len__(self):
        return int(self._shardpoints[-1])

    def locate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(f"장면 번호 {index} 가 범위 [0, {len(self)}) 밖이다")
        shard_id = int(np.searchsorted(self._shardpoints, index, side='right') - 1)
        return shard_id, index - int(self._shardpoints[shard_id])

    def __getitem__(self, index: int) -> dict:
        shard_id, local_idx = self.locate(int(index))
        offsets = self._get_offsets(shard_id)
        start = int(offsets[local_idx])
        length = int(offsets[local_idx + 1]) - start
        with open(self._shard_paths[shard_id], 'rb', buffering=_BUFFER_SIZE) as fh:
            fh.seek(start)
            return self._loads(fh.read(length))

    def _get_offsets(self, shard_id: int) -> np.ndarray:
        if shard_id not in self._offsets_cache:
            with open(self._shard_paths[shard_id], 'rb') as f:
                n, _stamp, offsets_pos = _read_header(f)
                f.seek(offsets_pos)
                self._offsets_cache[shard_id] = np.frombuffer(
                    f.read(_INT64 * (n + 1)), dtype=np.int64).copy()
        return self._offsets_cache[shard_id]
