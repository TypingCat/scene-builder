# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

"""Route and maneuver measures from converted scenes."""

from __future__ import annotations

import math

import numpy as np

_TURNS = ('GoStraight', 'TurnLeft', 'TurnRight', 'UTurn')
ROUTE_TAGS = (*_TURNS, 'Mapless', 'Pivot', 'NoRoute')
_STOPPED_MPS = 0.5
_STRAIGHT_RAD = math.radians(45.0)
_UTURN_RAD = math.radians(135.0)
_ONROAD_M = 5.0
_RUN_CAP_M = 40.0
_PIVOT_EDGE_M = 10.0
_OFFROUTE_M = 30.0
_CHOICE_SEEDS = 6
_REVERSE_SPEED_MPS = 0.1
_REVERSE_MERGE_S = 0.5
_REVERSE_MIN_ARC_M = 0.5


def _turn_class(delta: float) -> str:
    """Classify the change in heading with 45- and 135-degree boundaries."""
    d = (delta + math.pi) % (2 * math.pi) - math.pi
    if abs(d) < _STRAIGHT_RAD:
        return 'GoStraight'
    if abs(d) >= _UTURN_RAD:
        return 'UTurn'
    return 'TurnLeft' if d > 0 else 'TurnRight'


def _speeds(t, pos) -> np.ndarray:
    """Frame speeds from adjacent positions, repeating the first measured speed."""
    v = np.hypot(*np.diff(pos, axis=0).T) / np.maximum(np.diff(t), 1e-6)
    return np.concatenate([v[:1], v])


def _pivot_share(path, slot, ego) -> float:
    """Share of the clip covered by a reversal of direction involving backing up."""
    import torch

    from .convert.track import smooth_track

    t, pos, dirs, heading = ego
    n = len(t)
    if n < 3:
        return 0.0
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return 0.0
    g = np.rint((t - t[0]) / dt).astype(int)
    track = torch.zeros((1, int(g[-1]) + 1, 2), dtype=torch.float64)
    valid = torch.zeros((1, int(g[-1]) + 1), dtype=torch.bool)
    track[0, g] = torch.from_numpy(np.asarray(pos, dtype=np.float64))
    valid[0, g] = True
    est, vel, _a = smooth_track(track, valid, dt)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(est[0, g].numpy(), axis=0).T))])
    along = (vel[0, g].numpy() * dirs).sum(axis=-1)
    spans = []
    for i in np.flatnonzero(along < -_REVERSE_SPEED_MPS):
        distance = -along[i] * dt
        if spans and t[i] - t[spans[-1][1]] <= _REVERSE_MERGE_S + 1e-6:
            spans[-1][1] = i
            spans[-1][2] += distance
        else:
            spans.append([i, i, distance])
    spans = [(i, j) for i, j, distance in spans if distance >= _REVERSE_MIN_ARC_M]
    if not spans:
        return 0.0
    valid = path['valid'][slot]
    rp = path['pos_xy'][slot, valid]
    rd = path['dir_xy'][slot, valid]
    share = 0.0
    for i0, i1 in spans:
        before = int(np.searchsorted(arc, arc[i0] - _PIVOT_EDGE_M))
        after = min(n - 1, int(np.searchsorted(arc, arc[i1] + _PIVOT_EDGE_M)))
        ka = int(np.argmin(np.hypot(*(rp - pos[before]).T)))
        kb = int(np.argmin(np.hypot(*(rp - pos[after]).T)))
        if (math.hypot(*(rp[ka] - pos[before])) <= _ONROAD_M
                and math.hypot(*(rp[kb] - pos[after])) <= _ONROAD_M):
            angle = math.atan2(rd[kb][1], rd[kb][0]) - math.atan2(rd[ka][1], rd[ka][0])
        else:
            angle = heading[after] - heading[before]
        if _turn_class(angle) == 'UTurn':
            share += (after - before + 1) / n
    return min(share, 1.0)


def _turns(scene, pos, slot):
    """Measure choices at route branches using the link graph."""
    path = scene['paths']
    valid = path['valid'][slot]
    rp = path['pos_xy'][slot, valid].astype(np.float64)
    rl = path['near_link'][slot, valid]
    if len(rp) < 2:
        return {}
    links = scene['links']
    lv = links['valid']
    row_of = {int(links['id'][r]): r for r in np.flatnonzero(lv.any(-1))}
    ldir = links['dir_xy']
    exits, lefts, rights = (links[name] for name in ('exit_links', 'left_links', 'right_links'))

    def tangent_end(r: int) -> float:
        k = np.flatnonzero(lv[r])[-1]
        return math.atan2(ldir[r, k, 1], ldir[r, k, 0])

    def branch(lid: int) -> bool:
        r = row_of.get(lid)
        return r is not None and int((exits[r] > 0).sum()) >= 2

    def choice(A: int) -> tuple[bool, bool]:
        """Report distinct turn options and whether a referenced exit is unavailable."""
        seeds = {A}
        stack = [A]
        while stack:
            a = stack.pop()
            for nb in (*lefts[row_of[a]], *rights[row_of[a]]):
                nb = int(nb)
                if nb and nb in row_of and nb not in seeds and len(seeds) < _CHOICE_SEEDS:
                    seeds.add(nb)
                    stack.append(nb)
        classes: set[str] = set()
        unknown = False
        for s0 in seeds:
            ta = tangent_end(row_of[s0])
            for b in exits[row_of[s0]]:
                b = int(b)
                if not b:
                    continue
                if b not in row_of:
                    unknown = True
                    continue
                classes.add(_turn_class(tangent_end(row_of[b]) - ta))
        return len(classes) >= 2, unknown

    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(rp, axis=0).T))])

    fa = np.full(len(pos), np.nan)
    j = 0
    for i in range(len(pos)):
        hi = min(len(rp), j + 200)
        dd = np.hypot(rp[j:hi, 0] - pos[i, 0], rp[j:hi, 1] - pos[i, 1])
        k = j + int(dd.argmin())
        if dd.min() <= _OFFROUTE_M:
            j = k
            fa[i] = arc[k]
    chain: list[list[int]] = []
    for i, lid in enumerate(rl):
        lid = int(lid)
        if lid == 0:
            continue
        if chain and chain[-1][0] == lid:
            chain[-1][2] = i
        else:
            chain.append([lid, i, i])
    out = {tag: 0.0 for tag in _TURNS}
    for i in range(len(chain) - 1):
        A = chain[i][0]
        if A not in row_of or not branch(A):
            continue
        k0, k1e = chain[i + 1][1], chain[i + 1][2]
        a0, a1 = arc[k0], arc[k1e]
        has, unk = choice(A)
        if not (has or unk):
            continue
        k1 = int(np.searchsorted(arc, min(a1, a0 + _RUN_CAP_M)))
        k1 = min(max(k1, k0 + 1), len(rp) - 1)
        h1 = math.atan2(*(rp[k1] - rp[k1 - 1])[::-1])
        tag = _turn_class(h1 - tangent_end(row_of[A]))
        inside = float(((fa >= a0) & (fa <= a1)).mean())
        out[tag] = max(out[tag], inside)
    return out


def _route_slot(path):
    candidates = np.flatnonzero(path['is_route'] & path['valid'].any(axis=1))
    return int(candidates[0]) if candidates.size else None


def route_tags(scene, ego, on_map):
    out = dict.fromkeys(ROUTE_TAGS)
    if ego is None:
        return out
    out['Mapless'] = float((~on_map).mean())
    t, pos, *_ = ego
    path = scene['paths']
    slot = _route_slot(path)
    if slot is not None and len(t) >= 3:
        out.update(_turns(scene, pos, slot))
        if out['UTurn'] is not None:
            out['Pivot'] = 0.0 if out['UTurn'] > 0 else _pivot_share(path, slot, ego)
    if len(t) >= 3:
        out['NoRoute'] = 0.0 if slot is not None else float((_speeds(t, pos) >= _STOPPED_MPS).mean())
    return out
