from __future__ import annotations
import dataclasses
import torch
from scene_builder._pdm.dtypes import config as sim_config, feature
from scene_builder._pdm.sim.closed_loop import RolloutResult, SimSpec, closed_loop_rollout


_OUTPUT_FIELDS = ('pos_xy', 'dir_xy', 'size_xyz', 'valid', 'is_ego', 'type', 'batch_valid')


def sim_spec_from_manifest(manifest: dict) -> SimSpec:
    """Read the frame window and simulation settings from the manifest."""
    missing = [k for k in ('frame_unit', 'num_input_frames', 'num_output_frames',
                           'controller', 'dynamics') if manifest.get(k) is None]
    if missing:
        raise ValueError(f'PDM manifest에 필요한 시뮬레이션 설정이 없다: {missing}')

    num_past = int(manifest['num_input_frames'])
    return SimSpec(
        frame_unit=float(manifest['frame_unit']),
        now=num_past - 1,
        num_past_frames=num_past,
        num_future_frames=int(manifest['num_output_frames']),
        controller=sim_config.ControllerConfig.from_dict(manifest['controller']),
        dynamics=sim_config.DynamicsConfig.from_dict(manifest['dynamics']),
    )


def _to_output(plan: dict) -> feature.Output:
    """브리지 dict 을 `feature.Output` 으로 포장한다."""
    unknown = set(plan) - set(_OUTPUT_FIELDS)
    if unknown:
        raise ValueError(f"브리지가 모르는 Output 필드를 돌려줬다: {sorted(unknown)}")
    return feature.Output(**{k: plan[k] for k in _OUTPUT_FIELDS if plan.get(k) is not None})


def plan_fn(deploy):
    """`closed_loop_rollout` 이 요구하는 `Callable[[Scene], Output]` 을 만든다."""
    def _plan(scene: feature.Scene) -> feature.Output:
        return _to_output(deploy.plan(scene))
    return _plan


@dataclasses.dataclass(frozen=True)
class _Dataloader:
    """NAVSIM 채점이 configs.dataloader 에서 읽는 표면 — frame_unit 뿐."""
    frame_unit: float


@dataclasses.dataclass(frozen=True)
class _Trainer:
    """NAVSIM 채점이 configs.trainer 에서 읽는 표면 — 폐루프 EC 페어링 주기."""
    control_hz: float
    validation_replan_hz: float


@dataclasses.dataclass(frozen=True)
class NavsimConfigs:
    """Settings used by the closed-loop metrics."""
    dataloader: _Dataloader
    time: sim_config.TimeConfig
    drivable_area: sim_config.DrivableAreaConfig
    dynamics: sim_config.DynamicsConfig
    controller: sim_config.ControllerConfig
    trainer: _Trainer


def navsim_configs(manifest: dict, *, control_hz: float,
                   num_future_frames: int | None = None) -> NavsimConfigs:
    """Configure scoring for the realized rollout window."""
    spec = sim_spec_from_manifest(manifest)
    return NavsimConfigs(
        dataloader=_Dataloader(frame_unit=spec.frame_unit),
        time=sim_config.TimeConfig(
            num_past_frames=spec.num_past_frames,
            num_future_frames=(spec.num_future_frames if num_future_frames is None
                               else num_future_frames)),
        drivable_area=sim_config.DrivableAreaConfig(),
        dynamics=spec.dynamics,
        controller=spec.controller,
        trainer=_Trainer(control_hz=control_hz, validation_replan_hz=replan_hz(spec)),
    )


def replan_hz(spec: SimSpec) -> float:
    """Compute the replanning frequency from the frame interval."""
    return 1.0 / spec.frame_unit


def rollout(deploy, scene: feature.Scene, *, control_hz: float, seed: int = 0,
            **kwargs) -> RolloutResult:
    """Run closed-loop simulation with the supplied planner."""
    spec = sim_spec_from_manifest(deploy.manifest)
    torch.manual_seed(seed)
    return closed_loop_rollout(scene, plan_fn(deploy), spec,
                               replan_hz=replan_hz(spec), control_hz=control_hz,
                               **kwargs)
