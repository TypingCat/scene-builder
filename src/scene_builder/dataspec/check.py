# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shard import STAGE_BUILT
from .spec import DATASPEC, ENUMS, FIELDS, GROUPS, META_KEYS, REFERENCES, shape_of

ITEMS = {
    1: '키',
    2: '형상과 타입',
    3: '스탬프',
    4: '열거형',
    5: '참조',
    6: '자차',
    7: '방향',
    8: '경로 순서',
    9: '정차 자세',
}

DIR_TOL = 1e-3


@dataclass(frozen=True)
class Problem:
    item: int
    where: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.item} {ITEMS[self.item]}] {self.where}: {self.detail}"


def _np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def check_stamp_item(stamp: dict) -> list[Problem]:
    out = []
    if stamp.get('stage') != STAGE_BUILT:
        out.append(Problem(3, 'stamp.stage', f"{stamp.get('stage')!r} != 'built'"))
    got = stamp.get('dataspec', {})
    for k, v in DATASPEC.items():
        if got.get(k) != v:
            out.append(Problem(3, f'stamp.dataspec.{k}', f'{got.get(k)!r} != {v!r}'))
    return out


def _check_keys(scene: dict) -> list[Problem]:
    out = []
    for g in GROUPS:
        if g not in scene:
            out.append(Problem(1, g, '최상위 키가 없다'))
    for g, fields in FIELDS.items():
        if g not in scene:
            continue
        for f in fields:
            if f not in scene[g]:
                out.append(Problem(1, f'{g}.{f}', '하위 필드가 없다'))
    if 'meta' in scene:
        for k in META_KEYS:
            if k not in scene['meta']:
                out.append(Problem(1, f'meta.{k}', '메타 키가 없다'))
    return out


def _check_shapes(arr: dict) -> list[Problem]:
    out = []
    for (g, f), a in arr.items():
        want_shape = shape_of(g, f)
        want_dtype = FIELDS[g][f][1]
        if tuple(a.shape) != want_shape:
            out.append(Problem(2, f'{g}.{f}', f'형상 {tuple(a.shape)} != {want_shape}'))
        if a.dtype.name != want_dtype:
            out.append(Problem(2, f'{g}.{f}', f'자료형 {a.dtype.name} != {want_dtype}'))
    return out


def _check_enums(arr: dict) -> list[Problem]:
    out = []
    for (g, f), n in ENUMS.items():
        a = arr.get((g, f))
        if a is None:
            continue
        bad = (a < 0) | (a >= n)
        if bad.any():
            out.append(Problem(4, f'{g}.{f}', f'범위 [0, {n}) 밖의 값 {sorted(set(a[bad].tolist()))[:5]}'))
    return out


def _slot_used(arr: dict, g: str) -> np.ndarray:
    v = arr[(g, 'valid')]
    return v.reshape(v.shape[0], -1).any(axis=1)


def _check_refs(arr: dict) -> list[Problem]:
    out = []
    ids = {}
    for g in ('links', 'lanes'):
        if (g, 'id') in arr and (g, 'valid') in arr:
            ids[g] = set(arr[(g, 'id')][_slot_used(arr, g)].tolist())
    for (g, f), target in REFERENCES.items():
        a = arr.get((g, f))
        if a is None or target not in ids:
            continue
        vals = set(a[a != 0].tolist())
        missing = vals - ids[target]
        if missing:
            out.append(Problem(5, f'{g}.{f}',
                               f'{target}.id 에 없는 값 {len(missing)}개 (예: {sorted(missing)[:5]})'))
    return out


def _check_ego(arr: dict) -> list[Problem]:
    a = arr.get(('agents', 'is_ego'))
    if a is None:
        return []
    out = []
    if not bool(a[0]):
        out.append(Problem(6, 'agents.is_ego[0]', '슬롯 0 이 자차가 아니다'))
    others = np.flatnonzero(a[1:]) + 1
    if others.size:
        out.append(Problem(6, 'agents.is_ego', f'슬롯 0 밖의 자차 {others[:5].tolist()}'))
    return out


def _check_dirs(arr: dict) -> list[Problem]:
    out = []
    pairs = [
        ('agents', arr.get(('agents', 'valid'))),
        ('links', arr.get(('links', 'valid'))),
        ('lanes', arr.get(('lanes', 'valid'))),
        ('markers', arr.get(('markers', 'valid'))),
        ('paths', arr.get(('paths', 'valid'))),
    ]
    for g, valid in pairs:
        d = arr.get((g, 'dir_xy'))
        if d is None or valid is None:
            continue
        n = np.linalg.norm(d[valid].astype(np.float64), axis=-1)
        bad = np.abs(n - 1.0) > DIR_TOL
        if g == 'markers':
            bad &= n != 0.0
        if bad.any():
            out.append(Problem(7, f'{g}.dir_xy', f'단위벡터가 아닌 유효 자리 {int(bad.sum())}/{n.size}'))
    if ('traffics', 'dir_xy') in arr and ('traffics', 'valid') in arr:
        used = _slot_used(arr, 'traffics')
        n = np.linalg.norm(arr[('traffics', 'dir_xy')][used].astype(np.float64), axis=-1)
        bad = np.abs(n - 1.0) > DIR_TOL
        if bad.any():
            out.append(Problem(7, 'traffics.dir_xy', f'단위벡터가 아닌 사용 슬롯 {int(bad.sum())}/{n.size}'))
    if ('goal', 'valid') in arr and bool(arr[('goal', 'valid')]):
        n = float(np.linalg.norm(arr[('goal', 'dir_xy')].astype(np.float64)))
        if abs(n - 1.0) > DIR_TOL:
            out.append(Problem(7, 'goal.dir_xy', f'노름 {n:.4f}'))
    return out


def _check_path_order(arr: dict) -> list[Problem]:
    r = arr.get(('paths', 'is_route'))
    c = arr.get(('paths', 'cost'))
    if r is None or c is None:
        return []
    out = []
    k = int(r.sum())
    if not r[:k].all():
        out.append(Problem(8, 'paths.is_route', f'루트 슬롯이 앞에 모여 있지 않다 {r.astype(int).tolist()}'))
    elif k > 1 and (np.diff(c[:k]) < 0).any():
        out.append(Problem(8, 'paths.cost', f'루트 슬롯의 cost 가 오름차순이 아니다 {c[:k].tolist()}'))
    return out


def _check_goal(arr: dict) -> list[Problem]:
    v = arr.get(('goal', 'valid'))
    if v is None or bool(v):
        return []
    out = []
    for f in ('pos_xy', 'dir_xy'):
        a = arr.get(('goal', f))
        if a is not None and np.any(a != 0):
            out.append(Problem(9, f'goal.{f}', 'valid 가 거짓인데 값이 0 이 아니다'))
    return out


def check_scene(scene: dict) -> list[Problem]:
    problems = _check_keys(scene)
    arr = {}
    for g, fields in FIELDS.items():
        for f in fields:
            if g in scene and f in scene[g]:
                arr[(g, f)] = _np(scene[g][f])
    shape_problems = _check_shapes(arr)
    problems += shape_problems
    if shape_problems:
        return problems
    problems += _check_enums(arr)
    problems += _check_refs(arr)
    problems += _check_ego(arr)
    problems += _check_dirs(arr)
    problems += _check_path_order(arr)
    problems += _check_goal(arr)
    return problems
