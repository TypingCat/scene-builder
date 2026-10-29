# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import io
import pickle
import struct

import numpy as np

_INT64 = 8
_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
_LEGACY_PROTOCOL = 1001

_DTYPES = {
    'DoubleStorage': np.dtype('<f8'), 'FloatStorage': np.dtype('<f4'),
    'HalfStorage': np.dtype('<f2'),
    'LongStorage': np.dtype('<i8'), 'IntStorage': np.dtype('<i4'),
    'ShortStorage': np.dtype('<i2'), 'CharStorage': np.dtype('i1'),
    'ByteStorage': np.dtype('u1'), 'BoolStorage': np.dtype('?'),
}


class _StorageStream(pickle.Unpickler):
    def __init__(self, f):
        super().__init__(f, encoding='latin1')
        self.slots: dict[str, np.dtype] = {}

    def find_class(self, module: str, name: str):
        if module.startswith('torch'):
            return name
        return super().find_class(module, name)

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == 'storage'):
            raise ValueError(f"장면 storage 참조 형식을 모른다: {pid!r:.80}")
        stype = pid[1] if isinstance(pid[1], str) else getattr(pid[1], '__name__', '')
        dtype = _DTYPES.get(stype)
        if dtype is None:
            raise ValueError(f"numpy 로 옮길 수 없는 storage 타입이다: {stype!r}")
        key = str(pid[2])
        self.slots[key] = dtype
        return key


def _load_from_bytes(blob: bytes) -> np.ndarray:
    f = io.BytesIO(blob)
    up = _StorageStream(f)
    try:
        magic, protocol, sys_info = up.load(), up.load(), up.load()
        ok = magic == _LEGACY_MAGIC and protocol == _LEGACY_PROTOCOL
    except Exception as e:
        raise ValueError(
            f"장면 storage 가 레거시 torch.save 가 아니다({e!r}) — torch 직렬화 형식이 "
            f"바뀌었다면 이 디코더를 고칠 것.") from e
    if not ok:
        raise ValueError(
            f"장면 storage 가 레거시 torch.save 가 아니다(magic={magic!r} "
            f"protocol={protocol!r}) — torch 직렬화 형식이 바뀌었다면 이 디코더를 고칠 것.")
    if not (isinstance(sys_info, dict) and sys_info.get('little_endian', True)):
        raise ValueError("빅엔디언 장면이다 — 이 디코더는 리틀엔디언만 읽는다.")
    root = up.load()
    keys = up.load()
    out: dict[str, np.ndarray] = {}
    for key in keys:
        (numel,) = struct.unpack('<q', f.read(_INT64))
        dtype = up.slots[str(key)]
        raw = f.read(numel * dtype.itemsize)
        if len(raw) != numel * dtype.itemsize:
            raise ValueError(f"장면 storage 본문이 짧다: {key!r}")
        out[str(key)] = np.frombuffer(raw, dtype=dtype)
    return out[root]


def _rebuild_tensor_v2(storage, offset, size, stride, *_rest) -> np.ndarray:
    flat = np.asarray(storage)[int(offset):]
    shape = tuple(int(s) for s in size)
    if not shape:
        return flat[:1].reshape(())
    strides = tuple(int(s) * flat.dtype.itemsize for s in stride)
    return np.lib.stride_tricks.as_strided(flat, shape=shape, strides=strides)


class _SceneUnpickler(pickle.Unpickler):
    _STUBS = {('torch._utils', '_rebuild_tensor_v2'): _rebuild_tensor_v2,
              ('torch.storage', '_load_from_bytes'): _load_from_bytes}

    def find_class(self, module: str, name: str):
        stub = self._STUBS.get((module, name))
        if stub is not None:
            return stub
        if module.startswith('torch'):
            raise ValueError(
                f"장면 페이로드에 모르는 torch 이름이 있다: {module}.{name} — "
                f"이 디코더가 아는 것은 {sorted(m + '.' + n for m, n in self._STUBS)} 뿐이다.")
        return super().find_class(module, name)


def loads(blob: bytes) -> dict:
    return _SceneUnpickler(io.BytesIO(blob)).load()
