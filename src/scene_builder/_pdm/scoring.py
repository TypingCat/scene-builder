# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

from dataclasses import fields
import math

import torch

from . import harness
from .deploy import Deployment
from .dtypes import feature
from .navsim import NAVSIM_2025_PLANNING as navsim
from .sim.closed_loop import rollout_to_output

_METRICS = {
    'EPDMS': 'epdms',
    'NC': 'no_at_fault_collisions',
    'DAC': 'drivable_area_compliance',
    'DDC': 'driving_direction_compliance',
    'TLC': 'traffic_light_compliance',
    'EP': 'ego_progress',
    'TTC': 'time_to_collision_within_bound',
    'LK': 'lane_keeping',
    'HC': 'history_comfort',
    'EC': 'extended_comfort',
}
_GROUPS = {
    'agents': feature.Agent, 'links': feature.Link, 'lanes': feature.Lane,
    'markers': feature.Marker, 'traffics': feature.Traffic, 'paths': feature.Path,
    'goal': feature.Goal,
}
_CONTROL_HZ = 50.0


def _device(name: str) -> torch.device:
    if name == 'cpu':
        return torch.device(name)
    if name.startswith('cuda:') and name[5:].isdigit():
        target = torch.device(name)
        if target.index < torch.cuda.device_count():
            return target
        raise ValueError(f'사용할 수 없는 CUDA 장치다: {name}')
    raise ValueError('채점 장치는 cpu 또는 cuda:N을 지정해야 한다.')


def _batch(scene: dict, device: torch.device) -> feature.Scene:
    groups = {}
    for name, group in _GROUPS.items():
        values = {}
        for field in fields(group):
            tensor = torch.tensor(scene[name][field.name], device=device)
            if tensor.dtype == torch.int32:
                tensor = tensor.long()
            values[field.name] = tensor.unsqueeze(0)
        groups[name] = group(**values)
    return feature.Scene(**groups)


class PDMScorer:
    def __init__(self, deploy, *, frame_unit: float, device: torch.device):
        self.deploy = deploy
        self.device = device
        self.spec = harness.sim_spec_from_manifest(deploy.manifest)
        if not math.isclose(self.spec.frame_unit, frame_unit, rel_tol=0, abs_tol=1e-6):
            raise ValueError('PDM과 장면의 프레임 간격이 다르다.')
        if self.spec.num_past_frames < 1 or self.spec.num_future_frames < 1:
            raise ValueError('PDM 배포물의 과거·미래 프레임 수가 각각 1 이상이어야 한다.')

    @classmethod
    def embedded(cls, dataspec: dict, device: str):
        from .embedded import EmbeddedPlanner

        return cls(EmbeddedPlanner(), frame_unit=dataspec['frame_unit'], device=_device(device))

    @classmethod
    def from_deployment(cls, folder: str, dataspec: dict, device: str):
        target = _device(device)
        deploy = Deployment(folder, dataspec=dataspec, device=target)
        return cls(deploy, frame_unit=dataspec['frame_unit'], device=target)

    @torch.inference_mode()
    def score(self, scene: dict) -> dict[str, float | None]:
        scores = dict.fromkeys((*_METRICS, 'SLC'))
        agents, paths = scene['agents'], scene['paths']
        valid = agents['valid'][agents['is_ego']]
        now = self.spec.now
        if (len(valid) != 1 or now >= valid.shape[1] or not valid[0, now]
                or not (paths['is_route'] & (paths['valid'].sum(axis=1) >= 2)).any()):
            return scores
        future = valid[0, now + 1:now + 1 + self.spec.num_future_frames]
        if not future.any():
            return scores
        steps = int(future.nonzero()[0][-1]) + 1
        rollout = harness.rollout(self.deploy, _batch(scene, self.device),
                                  control_hz=_CONTROL_HZ, num_steps=steps)
        output = rollout_to_output(rollout, self.spec)
        configs = harness.navsim_configs(self.deploy.manifest, control_hz=_CONTROL_HZ,
                                         num_future_frames=steps)
        metrics = navsim.score_rollout(output, rollout.scene_t0, configs)
        for tag, key in _METRICS.items():
            metric = metrics[f'{navsim.PREFIX}/{key}']
            value = float(metric.tensor.item())
            if bool(metric.valid.item()) and math.isfinite(value):
                scores[tag] = value
        return scores
