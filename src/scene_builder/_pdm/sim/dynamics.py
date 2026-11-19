from __future__ import annotations
import math
import typing
import torch
from scene_builder._pdm.dtypes import feature
from scene_builder._pdm.dtypes.config import DynamicsConfig


X, Y, YAW, V = 0, 1, 2, 3


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class SimState(typing.NamedTuple):
    """Agent dynamics state threading through the rollout.

    core: [N, 4] — (x, y, yaw, v). controller 가 그대로 소비하는 view.
    ring: [N, R] — 조향 지연 큐(oldest-first, 고정 폭). per-agent 지연은 ``AgentParams.
          ring_idx`` 로 읽는 위치가 갈린다 — ring 폭은 전 agent 공통(정적 형상).
    """
    core: torch.Tensor
    ring: torch.Tensor


class AgentParams(typing.NamedTuple):
    """per-agent plant/controller 파라미터. 전부 [N] (ring_idx 만 [N, 1] long).

    클로저가 아니라 텐서 pytree 로 compiled region 에 들어간다 — 프리셋 identity 가드가
    없으므로 배치가 바뀌어도 재컴파일되지 않는다.
    """
    wheelbase: torch.Tensor
    lr: torch.Tensor
    max_steer: torch.Tensor
    max_accel: torch.Tensor
    max_brake: torch.Tensor
    drag_coef: torch.Tensor
    v_max: torch.Tensor
    pm_a_max: torch.Tensor
    is_point_mass: torch.Tensor
    ring_idx: torch.Tensor


def ring_slots(control_hz: float, steer_delay_max_s: float) -> int:
    """조향 지연 ring 의 고정 폭. 최대 지연(steer_delay_max_s)까지 담고 +1 은 지연 0
    (방금 넣은 명령을 그대로 읽는 passthrough) 슬롯."""
    return max(1, round(steer_delay_max_s * control_hz)) + 1


def _lerp(anchor: tuple[float, float], t: torch.Tensor) -> torch.Tensor:
    return anchor[0] + (anchor[1] - anchor[0]) * t


def params_from_agents(
    size_xyz: torch.Tensor,
    agent_type: torch.Tensor,
    control_hz: float,
    dc: DynamicsConfig,
) -> AgentParams:
    """agent 의 크기·타입 → plant 파라미터. bicycle 앵커는 타입별 독립(vehicle=길이보간,
    cyclist=단일 앵커); point-mass(ped/unknown)는 아래 pm_a_max/v_max 로만 실현된다."""
    device = size_xyz.device
    length = size_xyz[:, 0]

    veh, cyc = dc.vehicle, dc.cyclist
    t = ((length - veh['anchor_length_m'][0])
         / (veh['anchor_length_m'][1] - veh['anchor_length_m'][0])).clamp(0.0, 1.0)
    veh_wb = torch.minimum(_lerp(veh['wheelbase_m'], t),
                           veh['wheelbase_l_ratio'] * length).clamp(min=veh['wheelbase_min_m'])
    veh_delay = _lerp(veh['delay_s'], t)
    veh_accel = _lerp(veh['accel'], t)
    veh_brake = _lerp(veh['brake'], t)


    is_cyc = agent_type == feature.AGENT_TYPE.cyclist
    wheelbase = torch.where(is_cyc, torch.full_like(veh_wb, cyc['wheelbase_m']), veh_wb)
    delay_s = torch.where(is_cyc, torch.full_like(veh_delay, cyc['delay_s']), veh_delay
                          ).clamp(0.0, veh['steer_delay_max_s'])
    max_accel = torch.where(is_cyc, torch.full_like(veh_accel, cyc['accel']), veh_accel)
    max_brake = torch.where(is_cyc, torch.full_like(veh_brake, cyc['brake']), veh_brake)
    max_steer = torch.where(is_cyc, torch.full_like(veh_wb, cyc['max_steer']),
                            torch.full_like(veh_wb, veh['max_steer']))
    drag_coef = torch.where(is_cyc, torch.full_like(veh_wb, cyc['drag_coef']),
                            torch.full_like(veh_wb, veh['drag_coef']))


    R = ring_slots(control_hz, veh['steer_delay_max_s'])
    steps = (delay_s * control_hz).round().long().clamp(0, R - 1)
    ring_idx = (R - 1 - steps)[:, None]

    v_max = torch.tensor(dc.max_vel_by_type(), device=device)[agent_type]
    pm_a_max = torch.tensor(dc.max_acc_by_type(), device=device)[agent_type]
    is_pm = ((agent_type == feature.AGENT_TYPE.pedestrian)
             | (agent_type == feature.AGENT_TYPE.unknown))

    return AgentParams(
        wheelbase=wheelbase,
        lr=0.5 * wheelbase,
        max_steer=max_steer,
        max_accel=max_accel,
        max_brake=max_brake,
        drag_coef=drag_coef,
        v_max=v_max,
        pm_a_max=pm_a_max,
        is_point_mass=is_pm,
        ring_idx=ring_idx,
    )


def tree_where(done: torch.Tensor, old, new):
    """done-freeze 단일 헬퍼: (NamedTuple/tuple of) 텐서 리프별
    ``torch.where(done, old, new)``. freeze 는 core 뿐 아니라 ring·ctrl_state 까지
    전체에 적용돼야 한다 — 완료 샘플의 hidden state 가 계속 변하면 안 된다."""
    if isinstance(old, torch.Tensor):
        shape = (done.shape[0],) + (1,) * (old.dim() - 1)
        return torch.where(done.view(shape), old, new)
    items = [tree_where(done, o, m) for o, m in zip(old, new)]
    return type(old)(*items) if hasattr(old, '_fields') else type(old)(items)


def init_state(pos: torch.Tensor, yaw: torch.Tensor, speed: torch.Tensor,
               slots: int) -> SimState:
    """log 상태 → SimState. ring 은 zero-init(신규 engage); 주행 중 이어받기는
    ``warm_start`` 가 첫 컨트롤러 명령으로 프리필한다."""
    core = torch.stack([pos[:, 0], pos[:, 1], yaw, speed], dim=-1)
    return SimState(core, core.new_zeros(core.shape[0], slots))


def kinematic_step(state: SimState, delta_cmd: torch.Tensor, accel_frac: torch.Tensor,
                   brake_frac: torch.Tensor, p: AgentParams, dt: float) -> SimState:
    """One semi-implicit Euler step. 위치·yaw 는 갱신 전 속도로, 속도는 마지막에.

    지연 조향: 최신 명령을 ring 에 밀어넣고 per-agent 오프셋에서 읽는다 — ``ring_idx``
    가 R-1 이면 방금 넣은 명령(지연 0 passthrough), 작을수록 오래된 명령.
    """
    ring = torch.cat([state.ring[:, 1:], delta_cmd[:, None]], dim=1)
    delta = ring.gather(1, p.ring_idx).squeeze(1)
    x, y, yaw, v = state.core.unbind(dim=-1)


    dv = accel_frac * p.max_accel - brake_frac * p.max_brake - p.drag_coef * v
    v_new = torch.clamp(v + dv * dt, torch.zeros_like(v), p.v_max)


    yaw_rate = (v / p.wheelbase) * torch.tan(delta)
    yaw_mid = yaw + 0.5 * yaw_rate * dt

    x_new = x + v * torch.cos(yaw_mid) * dt
    y_new = y + v * torch.sin(yaw_mid) * dt
    yaw_new = wrap_angle(yaw + yaw_rate * dt)

    return SimState(torch.stack([x_new, y_new, yaw_new, v_new], dim=-1), ring)


def readout(state: SimState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pos [N,2], dir [N,2], vel [N,2]) — 세계좌표. no-slip 이라 vel ∥ heading."""
    pos = state.core[:, :2]
    yaw = state.core[:, YAW]
    dir_ = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    vel = state.core[:, V, None] * dir_
    return pos, dir_, vel


def warm_start(state: SimState, delta0: torch.Tensor) -> SimState:
    """ring 전체를 첫 컨트롤러 명령으로 프리필. rollout 은 주행 중 이어받기라
    zero-init 이면 매 윈도 서두에 지연창만큼 조향 0 아티팩트가 박힌다."""
    return SimState(state.core, delta0[:, None].expand(-1, state.ring.shape[1]).clone())


def point_mass_step(
    pos: torch.Tensor,
    vel: torch.Tensor,
    target_pos: torch.Tensor,
    target_ok: torch.Tensor,
    a_max: torch.Tensor,
    v_max: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """홀로노믹 2차 적분 + 포화. 목표점을 향하되 가속·속도 상한으로 저역통과된다 —
    forecast 가 순간이동해도 실현 궤적은 물리적으로 bounded. forecast 무효 프레임은
    제자리를 목표로 삼아 감속·정지한다. heading 은 여기서 만들지 않는다(호출부가 forecast
    dir 를 그대로 통과시킨다 — 속도 유도 heading 은 정지 agent 의 방향을 잃는다)."""
    tgt = torch.where(target_ok[..., None], target_pos, pos)
    dv = (tgt - pos) / dt - vel
    n = dv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dv = dv * (n.clamp(max=(a_max * dt)[..., None]) / n)
    v_new = vel + dv
    s = v_new.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v_new = v_new * (s.clamp(max=v_max[..., None]) / s)
    return pos + v_new * dt, v_new
