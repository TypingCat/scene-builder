from __future__ import annotations
import math
import typing
import torch
from scene_builder._pdm.dtypes.config import ControllerConfig
from scene_builder._pdm.sim.dynamics import AgentParams


X, Y, YAW, V = 0, 1, 2, 3


CtrlState = typing.Tuple[torch.Tensor, ...]


def plan_profile(
    path: torch.Tensor,
    path_valid: torch.Tensor,
    frame_unit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan-invariant pure-pursuit inputs — depend only on the held plan, so
    compute once per replan instead of every control substep.

    v_profile: per-point speed ||path[i+1]-path[i]||/frame_unit, last repeats.
    arc: cumulative arc length from the plan start.
    last_valid: last valid index per row.

    무효 점은 zero-fill 이라 그 점이 낀 구간의 변위는 프레임 원점까지의 거리가 되어버린다
    (수십 m / frame_unit = 수백 m/s 의 목표속도 → 종방향 full throttle). 양끝이 모두 유효한
    구간만 속도·호길이에 반영한다.
    """
    N, P, _ = path.shape
    device = path.device
    seg = (path[:, 1:] - path[:, :-1]).norm(dim=-1)
    seg = seg * (path_valid[:, :-1] & path_valid[:, 1:]).to(seg.dtype)
    v_profile = torch.cat([seg / frame_unit, seg[:, -1:] / frame_unit], dim=1)
    idx = torch.arange(P, device=device)
    last_valid = (path_valid.long() * idx).argmax(dim=1)
    arc = torch.cat([torch.zeros(N, 1, device=device), seg.cumsum(dim=1)], dim=1)
    return v_profile, arc, last_valid


def longitudinal_command(
    v: torch.Tensor,
    target_v: torch.Tensor,
    integ: torch.Tensor,
    at_end: torch.Tensor,
    dt_ctrl: float,
    p: AgentParams,
    cp: ControllerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Speed-error PI. 컨트롤러는 플랜트 내부 drag 를 모른다(실차 계약) — 적분항이 그
    정상상태 부하를 흡수해 속도가 처지지 않게 한다. net = 목표 구동가속 [m/s²]; 부호로
    accel/brake 를 가른다. 경로 끝에서는 제동을 강제하고 적분기를 리셋한다.
    """
    vel_err = target_v - v
    integ = integ + vel_err * dt_ctrl
    i_lim = p.max_accel / max(cp.ki_speed, 1e-6)
    integ = torch.clamp(integ, -i_lim, i_lim)
    net = cp.kp_speed * vel_err + cp.ki_speed * integ
    zero = torch.zeros_like(net)
    one = torch.ones_like(net)
    accel_frac = torch.clamp(net / p.max_accel, zero, one)
    brake_frac = torch.clamp(-net / p.max_brake, zero, one)

    accel_frac = torch.where(at_end, torch.zeros_like(accel_frac), accel_frac)
    brake_frac = torch.where(
        at_end,
        torch.maximum(brake_frac, torch.full_like(brake_frac, cp.end_brake_frac)),
        brake_frac,
    )
    integ = torch.where(at_end, torch.zeros_like(integ), integ)
    return accel_frac, brake_frac, integ


def init_ctrl_state(ego: torch.Tensor) -> CtrlState:
    """(integ,) — [N] 종방향 속도오차 적분."""
    return (ego.new_zeros(ego.shape[0]),)


def pure_pursuit_control(
    path: torch.Tensor,
    path_valid: torch.Tensor,
    ego: torch.Tensor,
    ctrl_state: CtrlState,
    *,
    profile: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    params: AgentParams,
    dt_ctrl: float,
    cp: ControllerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, CtrlState]:
    """Batched pure pursuit(횡) + speed-error PI(종). 기하 기준점은 후축이고, 목표 속도는
    공간적으로 가장 가까운 plan 점의 속도다(시간 인덱싱 없음).
    """
    N, P, _ = path.shape
    device = path.device
    INF = torch.finfo(path.dtype).max
    p = params
    (integ,) = ctrl_state
    x, y, yaw, v = ego[:, X], ego[:, Y], ego[:, YAW], ego[:, V]

    v_profile, arc, last_valid = profile


    rx = x - p.lr * torch.cos(yaw)
    ry = y - p.lr * torch.sin(yaw)

    d2 = (path[..., 0] - rx[:, None]) ** 2 + (path[..., 1] - ry[:, None]) ** 2
    d2 = d2.masked_fill(~path_valid, INF)
    c_i = d2.argmin(dim=1)

    target_v = v_profile.gather(1, c_i[:, None]).squeeze(1)

    Ld = torch.clamp(cp.look_ahead_k_s * v.clamp(min=0.0),
                     cp.look_ahead_min_m, cp.look_ahead_max_m)


    target_arc = arc.gather(1, c_i[:, None]) + Ld[:, None]
    reached = (arc >= target_arc) & path_valid
    has = reached.any(dim=1)
    t_i = torch.where(has, reached.float().argmax(dim=1), last_valid)
    t_i = torch.minimum(t_i, last_valid)

    tx = path[torch.arange(N, device=device), t_i, 0]
    ty = path[torch.arange(N, device=device), t_i, 1]
    dx = tx - rx
    dy = ty - ry
    L = torch.hypot(dx, dy)

    beta = torch.atan2(dy, dx) - yaw
    beta = (beta + math.pi) % (2.0 * math.pi) - math.pi
    alpha = torch.atan2(p.wheelbase * torch.sin(beta), 0.5 * L)
    delta = torch.clamp(alpha, -p.max_steer, p.max_steer)
    delta = torch.where(L < 1e-3, torch.zeros_like(delta), delta)


    no_ref = ~path_valid.any(dim=1)
    delta = torch.where(no_ref, torch.zeros_like(delta), delta)


    at_end = c_i >= last_valid
    accel_frac, brake_frac, integ = longitudinal_command(
        v, target_v, integ, at_end, dt_ctrl, p, cp)

    return delta, accel_frac, brake_frac, (integ,)
