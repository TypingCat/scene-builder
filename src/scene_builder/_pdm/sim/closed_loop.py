from __future__ import annotations
import dataclasses
import torch
from scene_builder._pdm.dtypes import config, feature
from scene_builder._pdm.sim.controller import (
    CtrlState, init_ctrl_state, plan_profile, pure_pursuit_control,
)
from scene_builder._pdm.sim.dynamics import (
    V, AgentParams, SimState, init_state, kinematic_step, params_from_agents,
    point_mass_step, readout, ring_slots, tree_where, warm_start,
)
from scene_builder._pdm.sim.preprocessing import compute_scene_transform, transform_scene


@dataclasses.dataclass(frozen=True)
class SimSpec:
    """Time window, controller settings and dynamics for closed-loop simulation."""
    frame_unit: float
    now: int
    num_past_frames: int
    num_future_frames: int
    controller: config.ControllerConfig
    dynamics: config.DynamicsConfig

def _align_by_id(t: torch.Tensor, src_id: torch.Tensor, dst_id: torch.Tensor) -> torch.Tensor:
    """per-agent 텐서 ``t`` [B, A_src, ...] 를 ``src_id`` 순서 → ``dst_id`` 순서로 재배열.

    플래너 Output 은 ``representation`` 의 ego-거리순(per-scene) 정렬이라 rollout 의
    ``scene_t0`` 축과 다르고 replan tick 마다 흔들린다("tick drift"). agent ``id`` 로
    되맞춰 cross-agent 소비(non_ego forecast advance)를 정합. dst i ← src j where
    ``src_id[j]==dst_id[i]``; 미매칭=0 (no-grad; 실 id>0 유일, padding=0 은 valid-mask 로 무해)."""
    B, A_dst = dst_id.shape
    match = dst_id[:, :, None] == src_id[:, None, :]
    has = match.any(dim=2)
    j = match.float().argmax(dim=2)
    tail = (1,) * (t.dim() - 2)
    idx = j.reshape(B, A_dst, *tail).expand(B, A_dst, *t.shape[2:])
    out = t.gather(1, idx)
    return torch.where(has.reshape(B, A_dst, *tail), out, torch.zeros_like(out))


def _shift_scene_time(scene: feature.Scene, shift: int) -> feature.Scene:
    """Shift temporal scene fields and mark out-of-bounds frames invalid."""
    F = scene.agents.pos_xy.shape[2]
    device = scene.agents.valid.device
    src = torch.arange(F, device=device) + shift
    in_bounds = (src >= 0) & (src < F)
    clamped = src.clamp(0, F - 1)
    ib = in_bounds[None, None, :]

    new_agents = scene.agents.replace(
        time_s=scene.agents.time_s[:, :, clamped],
        pos_xy=scene.agents.pos_xy[:, :, clamped],
        dir_xy=scene.agents.dir_xy[:, :, clamped],
        vel_xy=scene.agents.vel_xy[:, :, clamped],
        acc_xy=scene.agents.acc_xy[:, :, clamped],
        valid=scene.agents.valid[:, :, clamped] & ib,
    )
    new_traffics = scene.traffics.replace(
        state=scene.traffics.state[:, :, clamped],
        time_s=scene.traffics.time_s[:, :, clamped],
        valid=scene.traffics.valid[:, :, clamped] & ib,
    )
    return scene.replace(agents=new_agents, traffics=new_traffics)


def advance_scene(
    scene: feature.Scene,
    ego_past: dict,
    ego_idx: torch.Tensor,
    step: int,
    spec: SimSpec,
    *,
    ego_future: dict | None = None,
    nonego_past: dict | None = None,
    nonego_future: dict | None = None,
    nonego_ok: torch.Tensor | None = None,
) -> feature.Scene:
    """Build the tick-``step`` Scene: log agents shifted by ``step``, ego past
    slots (local 0..F_now) overwritten by the realized ego history.

    ``step==0`` with ``ego_past`` equal to the scene's own ego past is a no-op
    (the open-loop-equivalence gate). ``ego_future``/``nonego_future`` are passed
    only by the *harvest* call (replay collect), not by the rollout-internal
    replan (which leaves the future as shift-log context for the planner).

    harvest: ``ego_future`` overwrites ego future [P:P+Sf] with the realized
    rollout **continuation** (not shift-log → no seam jump, HOLE A 해소).
    non_ego forecast (``nonego_past`` given): only non-ego slots' past [:P] and
    future [P:P+Ff] are overwritten by the imagined trajectory; the ego slot is
    untouched here (``ego_past``/``ego_future`` own it exclusively).
    **Truncation (C5)**: frames beyond the imagined horizon are marked invalid
    (NOT log-shift-filled) — ego beyond Sf, overwritten non-ego beyond Ff.
    """
    P = spec.num_past_frames
    shifted = _shift_scene_time(scene, step)
    a = shifted.agents
    B, A = a.valid.shape[:2]
    bi = torch.arange(B, device=a.valid.device)

    pos = a.pos_xy.clone()
    dir_ = a.dir_xy.clone()
    vel = a.vel_xy.clone()
    acc = a.acc_xy.clone()
    val = a.valid.clone()

    if nonego_past is not None:


        nonego_slot = a.valid.new_ones((B, A), dtype=torch.bool)
        nonego_slot[bi, ego_idx] = False
        if nonego_ok is not None:
            nonego_slot = nonego_slot & nonego_ok
        m = nonego_slot[:, :, None, None]

        def _ow(base, sl, new):
            base[:, :, sl] = torch.where(m, new, base[:, :, sl])

        _ow(pos, slice(None, P), nonego_past["pos"])
        _ow(dir_, slice(None, P), nonego_past["dir"])
        _ow(vel, slice(None, P), nonego_past["vel"])
        _ow(acc, slice(None, P), nonego_past["acc"])
        if nonego_future is not None:
            Ff = nonego_future["pos"].shape[2]
            _ow(pos, slice(P, P + Ff), nonego_future["pos"])
            _ow(dir_, slice(P, P + Ff), nonego_future["dir"])
            _ow(vel, slice(P, P + Ff), nonego_future["vel"])
            _ow(acc, slice(P, P + Ff), nonego_future["acc"])

            val[:, :, P + Ff:] = val[:, :, P + Ff:] & ~nonego_slot[:, :, None]


    pos[bi, ego_idx, :P] = ego_past["pos"]
    dir_[bi, ego_idx, :P] = ego_past["dir"]
    vel[bi, ego_idx, :P] = ego_past["vel"]
    acc[bi, ego_idx, :P] = ego_past["acc"]
    val[bi, ego_idx, :P] = True

    if ego_future is not None:

        Sf = ego_future["pos"].shape[1]
        pos[bi, ego_idx, P:P + Sf] = ego_future["pos"]
        dir_[bi, ego_idx, P:P + Sf] = ego_future["dir"]
        vel[bi, ego_idx, P:P + Sf] = ego_future["vel"]
        acc[bi, ego_idx, P:P + Sf] = ego_future["acc"]
        ev = ego_future.get("valid")
        val[bi, ego_idx, P:P + Sf] = True if ev is None else ev
        val[bi, ego_idx, P + Sf:] = False

    return shifted.replace(
        agents=a.replace(pos_xy=pos, dir_xy=dir_, vel_xy=vel, acc_xy=acc, valid=val)
    )


@dataclasses.dataclass
class RolloutResult:
    """Realized closed-loop ego trajectory + the T0-frame scene to score against.

    All ego_* are realized (simulated) values over S = num_steps ticks, T0 frame.
    Non-ego agents / map / paths live in ``scene_t0`` (log-replay) unless
    ``forecast_nonego`` — then the realized non-ego world is in ``nonego_*``.
    ``valid`` masks ticks at/after each sample's last-valid ego frame (done-freeze).
    """
    scene_t0: feature.Scene
    ego_idx: torch.Tensor
    ego_pos: torch.Tensor
    ego_dir: torch.Tensor
    ego_v: torch.Tensor
    ego_vel: torch.Tensor
    ego_yaw_rate: torch.Tensor
    valid: torch.Tensor
    F_now: int
    dt: float
    num_plan_calls: int = 0


    nonego_pos: torch.Tensor | None = None
    nonego_dir: torch.Tensor | None = None
    nonego_vel: torch.Tensor | None = None
    nonego_acc: torch.Tensor | None = None
    nonego_valid: torch.Tensor | None = None


    plan_ego_pos: torch.Tensor | None = None
    plan_ego_dir: torch.Tensor | None = None
    plan_ego_valid: torch.Tensor | None = None
    plan_issue_done: torch.Tensor | None = None


    plan_nonego_pos: torch.Tensor | None = None
    plan_nonego_dir: torch.Tensor | None = None
    plan_nonego_valid: torch.Tensor | None = None

    plan_belief_logits: torch.Tensor | None = None


def _seed_state(pos, dir_, vel, slots: int) -> tuple[SimState, CtrlState]:
    """log 상태 [N, 2] × 3 → plant 초기상태. ring 은 zero-init(신규 engage);
    rollout 서두의 ego 는 ``warm_start`` 가 첫 명령으로 프리필한다."""
    yaw = torch.atan2(dir_[:, 1], dir_[:, 0])
    state = init_state(pos, yaw, vel.norm(dim=-1), slots)
    return state, init_ctrl_state(state.core)


def _track_plan_frame(plan, plan_valid, profile, state: SimState, ctrl_state,
                      freeze, ctrl_per_data, params: AgentParams, dt_ctrl: float,
                      cp: config.ControllerConfig):
    """One data frame of plan tracking: ``ctrl_per_data`` controller + dynamics
    substeps with freeze. The controller targets the spatial-nearest plan point
    (no time index) and carries its PI integrator in ``ctrl_state``. Rows are
    agents (folded [B·A_r]) — ``params`` carries their per-agent plant constants.
    """
    for _ in range(ctrl_per_data):
        delta, af, bf, new_cs = pure_pursuit_control(
            plan, plan_valid, state.core, ctrl_state,
            profile=profile, params=params, dt_ctrl=dt_ctrl, cp=cp)
        new_state = kinematic_step(state, delta, af, bf, params, dt_ctrl)

        state = tree_where(freeze, state, new_state)
        ctrl_state = tree_where(freeze, ctrl_state, new_cs)
    return state, ctrl_state


_track_plan_frame_compiled = torch.compile(_track_plan_frame, dynamic=False)


def track_plan_frame(plan, plan_valid, profile, state: SimState, ctrl_state,
                     freeze, ctrl_per_data, params: AgentParams, dt_ctrl: float,
                     cp: config.ControllerConfig):
    """``_track_plan_frame`` 공개 seam — compiled 는 CUDA 에서만 성립한다(inductor).

    rollout 밖에서 plan 을 plant 로 실현하는 소비자(EC 채점)를 위한 진입점.
    """
    track = _track_plan_frame_compiled if plan.is_cuda else _track_plan_frame
    return track(plan, plan_valid, profile, state, ctrl_state, freeze,
                 ctrl_per_data, params, dt_ctrl, cp)


def _data_per_replan(spec: SimSpec, replan_hz: float) -> int:
    """replan 1회당 data frame 수 — replan 케이던스의 단일 출처."""
    return max(1, round((1.0 / replan_hz) / spec.frame_unit))


def closed_loop_rollout(
    full_scene: feature.Scene,
    plan_fn,
    spec: SimSpec,
    *,
    replan_hz: float,
    control_hz: float = 50.0,
    num_steps: int | None = None,
    transform=None,
    forecast_nonego: bool = False,
    track_plan: bool = True,


    capture_plans: bool = True,

    on_plan_call=None,

) -> RolloutResult:
    """Roll every model-driven agent forward under its plant; the rest replay from log.

    Matches deployment: the planner replans at ``replan_hz`` and the controller
    tracks that one plan at ``control_hz`` until the next replan — exactly the
    deploy inference/control split. Realized states are recorded at the data rate
    (``frame_unit``) so they line up with log agents and metrics. Runs until each
    sample's last-valid ego frame ``F_end`` (collision-agnostic).

    Plants (``dynamics.py``, params from each agent's size/type):
      - vehicle/cyclist: kinematic bicycle tracking its reference with pure pursuit.
      - pedestrian/unknown: point mass — its forecast integrated under accel/speed caps.
    The ego's reference is its own plan (``track_plan``) or the log future
    (``track_plan=False``); a non-ego's reference is its held forecast. Either way
    the stored trajectory is plant-realized — never raw log-shift, never raw
    forecast splice. Agents with no forecast (``forecast_nonego=False``, or slots
    whose forecast is missing) keep their log states.

    ``plan_fn`` is called only when an ego plan (``track_plan``) or a non-ego
    forecast (``forecast_nonego``) is needed; pure log-track + log-replay skips it.
    """
    device = full_scene.agents.valid.device
    F_now = spec.now
    P = spec.num_past_frames
    dt_data = spec.frame_unit
    dt_ctrl = 1.0 / control_hz
    data_per_replan = _data_per_replan(spec, replan_hz)
    ctrl_per_data = max(1, round(dt_data / dt_ctrl))
    B, A = full_scene.agents.valid.shape[:2]
    bi = torch.arange(B, device=device)


    tf0, valid0 = compute_scene_transform(full_scene, spec.now)
    if transform is not None:
        tf0 = transform
    scene_t0 = transform_scene(full_scene, tf0, valid0)

    ego_idx = scene_t0.agents.is_ego.float().argmax(dim=1)
    scene_t0_id = scene_t0.agents.id
    ag = scene_t0.agents


    hist_pos = ag.pos_xy[bi, ego_idx, :P].clone()
    hist_dir = ag.dir_xy[bi, ego_idx, :P].clone()
    hist_vel = ag.vel_xy[bi, ego_idx, :P].clone()
    hist_acc = ag.acc_xy[bi, ego_idx, :P].clone()


    A_r = A if forecast_nonego else 1
    if forecast_nonego:
        r_pos, r_dir, r_vel, r_valid = ag.pos_xy, ag.dir_xy, ag.vel_xy, ag.valid
        r_size, r_type = ag.size_xyz, ag.type
        ego_row = ego_idx
    else:
        r_pos = ag.pos_xy[bi, ego_idx][:, None]
        r_dir = ag.dir_xy[bi, ego_idx][:, None]
        r_vel = ag.vel_xy[bi, ego_idx][:, None]
        r_valid = ag.valid[bi, ego_idx][:, None]
        r_size = ag.size_xyz[bi, ego_idx][:, None]
        r_type = ag.type[bi, ego_idx][:, None]
        ego_row = torch.zeros_like(ego_idx)
    N = B * A_r
    R = ring_slots(control_hz, spec.dynamics.vehicle['steer_delay_max_s'])
    params = params_from_agents(r_size.reshape(N, 3), r_type.reshape(N).long(), control_hz, spec.dynamics)
    is_pm = params.is_point_mass.reshape(B, A_r).clone()


    is_pm[bi, ego_row] = False


    if forecast_nonego:
        ne_pos = ag.pos_xy[:, :, :P].clone()
        ne_dir = ag.dir_xy[:, :, :P].clone()
        ne_vel = ag.vel_xy[:, :, :P].clone()
        ne_acc = ag.acc_xy[:, :, :P].clone()
        ne_fpos = ne_fdir = ne_fvalid = None
        pm_pos = r_pos[:, :, F_now].clone()
        pm_vel = r_vel[:, :, F_now].clone()


    F_src = ag.valid.shape[2]
    fidx = torch.arange(F_src, device=device)
    F_end_a = (r_valid.long() * (fidx + 1)).argmax(dim=2)
    F_end = F_end_a[bi, ego_row]
    if num_steps is None:
        num_steps = int((F_end - F_now).clamp(min=0).max().item())

    state, ctrl_state = _seed_state(
        r_pos[:, :, F_now].reshape(N, 2), r_dir[:, :, F_now].reshape(N, 2),
        r_vel[:, :, F_now].reshape(N, 2), R)


    done_a = F_now >= F_end_a
    live_a = r_valid[:, :, F_now].clone()
    freeze = (done_a | ~live_a).reshape(N)
    done = done_a[bi, ego_row]
    rec_pos, rec_dir, rec_v, rec_vel, rec_valid = [], [], [], [], []

    plan_ec_pos, plan_ec_dir, plan_ec_valid, plan_ec_done = [], [], [], []

    plan_ne_pos, plan_ne_dir, plan_ne_valid = [], [], []

    plan_belief_logits = []

    plan = plan_valid = profile = None
    num_plan_calls = 0


    F_plan = spec.num_future_frames
    log_ego_pos = scene_t0.agents.pos_xy[bi, ego_idx]
    log_ego_valid = scene_t0.agents.valid[bi, ego_idx]

    def _replan(hp, hd, hv, ha, k, ne_past=None):


        ne_fpos = ne_fdir = ne_fvalid = None
        plan = plan_valid = None
        if track_plan or forecast_nonego:
            ego_past = {"pos": hp, "dir": hd, "vel": hv, "acc": ha}
            sc_k = advance_scene(scene_t0, ego_past, ego_idx, k, spec, nonego_past=ne_past)
            out_k = plan_fn(sc_k)

            out_id = (out_k.extra or {}).get("agent_id")
            if capture_plans:


                bl = (out_k.extra or {}).get("belief_logits")
                if bl is not None:
                    if out_id is not None and out_id.numel() > 0:
                        bl = _align_by_id(bl, out_id, scene_t0_id)
                    plan_belief_logits.append(bl)
            if track_plan:
                eo = out_k.is_ego.any(dim=-1).float().argmax(dim=1)
                plan = out_k.pos_xy[bi, eo, P:]
                plan_valid = out_k.valid[bi, eo, P:]
                if capture_plans:


                    plan_ec_pos.append(out_k.pos_xy[bi, eo, P - 1:])
                    plan_ec_dir.append(out_k.dir_xy[bi, eo, P - 1:])
                    plan_ec_valid.append(plan_valid)
                    plan_ec_done.append(done.clone())
            if forecast_nonego:


                ne_fpos = out_k.pos_xy[:, :, P:]
                ne_fdir = out_k.dir_xy[:, :, P:]
                ne_fvalid = out_k.valid[:, :, P:]


                if out_id is not None and out_id.numel() > 0:
                    ne_fpos = _align_by_id(ne_fpos, out_id, scene_t0_id)
                    ne_fdir = _align_by_id(ne_fdir, out_id, scene_t0_id)
                    ne_fvalid = _align_by_id(ne_fvalid, out_id, scene_t0_id)
                if capture_plans:


                    ec_pos = out_k.pos_xy[:, :, P - 1:]
                    ec_dir = out_k.dir_xy[:, :, P - 1:]
                    ec_valid = out_k.valid[:, :, P:]
                    if out_id is not None and out_id.numel() > 0:
                        ec_pos = _align_by_id(ec_pos, out_id, scene_t0_id)
                        ec_dir = _align_by_id(ec_dir, out_id, scene_t0_id)
                        ec_valid = _align_by_id(ec_valid, out_id, scene_t0_id)
                    plan_ne_pos.append(ec_pos)
                    plan_ne_dir.append(ec_dir)
                    plan_ne_valid.append(ec_valid)
        if not track_plan:


            src = F_now + k + 1 + torch.arange(F_plan, device=device)
            src_c = src.clamp(max=F_src - 1)
            plan = log_ego_pos[:, src_c]
            plan_valid = log_ego_valid[:, src_c] & (src < F_src)[None]


        if forecast_nonego:
            plan_all = ne_fpos.clone()
            valid_all = ne_fvalid.clone()
        else:
            plan_all = plan[:, None].clone()
            valid_all = plan_valid[:, None].clone()
        plan_all[bi, ego_row] = plan
        valid_all[bi, ego_row] = plan_valid
        return plan, plan_valid, plan_all, valid_all, ne_fpos, ne_fdir, ne_fvalid

    for k in range(num_steps):

        if k % data_per_replan == 0:
            ne_past = ({"pos": ne_pos[:, :, -P:], "dir": ne_dir[:, :, -P:],
                        "vel": ne_vel[:, :, -P:], "acc": ne_acc[:, :, -P:]}
                       if forecast_nonego else None)
            plan, plan_valid, plan_all, valid_all, ne_fpos, ne_fdir, ne_fvalid = _replan(
                hist_pos[:, -P:], hist_dir[:, -P:], hist_vel[:, -P:], hist_acc[:, -P:], k,
                ne_past=ne_past)
            if track_plan or forecast_nonego:
                num_plan_calls += 1
                if on_plan_call is not None:
                    on_plan_call()

            ref = plan_all.reshape(N, -1, 2)
            ref_valid = valid_all.reshape(N, -1)

            profile = plan_profile(ref, ref_valid, dt_data)

        if k > 0 and forecast_nonego:


            now_valid = r_valid[:, :, F_now + k].clone()
            now_valid[bi, ego_row] = True
            newly = now_valid & ~live_a
            s_new, c_new = _seed_state(
                r_pos[:, :, F_now + k].reshape(N, 2), r_dir[:, :, F_now + k].reshape(N, 2),
                r_vel[:, :, F_now + k].reshape(N, 2), R)
            keep = ~newly.reshape(N)
            state = tree_where(keep, state, s_new)
            ctrl_state = tree_where(keep, ctrl_state, c_new)
            pm_pos = torch.where(newly[..., None], r_pos[:, :, F_now + k], pm_pos)
            pm_vel = torch.where(newly[..., None], r_vel[:, :, F_now + k], pm_vel)
            live_a = now_valid
            freeze = (done_a | ~live_a).reshape(N)

        if k == 0:


            delta0 = pure_pursuit_control(
                ref, ref_valid, state.core, ctrl_state,
                profile=profile, params=params, dt_ctrl=dt_ctrl, cp=spec.controller)[0]
            state = warm_start(state, delta0)


        track = _track_plan_frame_compiled if ref.is_cuda else _track_plan_frame
        state, ctrl_state = track(ref, ref_valid, profile, state, ctrl_state,
                                  freeze, ctrl_per_data, params, dt_ctrl, spec.controller)

        bpos, bdir, bvel = readout(state)
        core = state.core.reshape(B, A_r, 4)

        if forecast_nonego:


            fi = k % data_per_replan
            tgt, tgt_ok = ne_fpos[:, :, fi], ne_fvalid[:, :, fi]
            new_pp, new_pv = point_mass_step(
                pm_pos, pm_vel, tgt, tgt_ok,
                params.pm_a_max.reshape(B, A_r), params.v_max.reshape(B, A_r), dt_data)


            live = (is_pm & live_a & ~done_a)[..., None]
            pm_pos = torch.where(live, new_pp, pm_pos)
            pm_vel = torch.where(live, new_pv, pm_vel)

            fdir = ne_fdir[:, :, fi]
            fdir = fdir / fdir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            pm_dir = torch.where(live & tgt_ok[..., None], fdir, ne_dir[:, :, -1])


            src = (F_now + k + 1) if (F_now + k + 1) < F_src else (F_src - 1)
            pm3 = is_pm[..., None]
            npos_ne = torch.where(pm3, pm_pos, bpos.reshape(B, A_r, 2))
            ndir_ne = torch.where(pm3, pm_dir, bdir.reshape(B, A_r, 2))
            nvel_ne = torch.where(pm3, pm_vel, bvel.reshape(B, A_r, 2))
            s3 = live_a[..., None]
            npos_ne = torch.where(s3, npos_ne, r_pos[:, :, src])
            ndir_ne = torch.where(s3, ndir_ne, r_dir[:, :, src])
            nvel_ne = torch.where(s3, nvel_ne, r_vel[:, :, src])


            nacc_ne = (nvel_ne - ne_vel[:, :, -1]) / dt_data

            ne_pos = torch.cat([ne_pos, npos_ne[:, :, None]], dim=2)
            ne_dir = torch.cat([ne_dir, ndir_ne[:, :, None]], dim=2)
            ne_vel = torch.cat([ne_vel, nvel_ne[:, :, None]], dim=2)
            ne_acc = torch.cat([ne_acc, nacc_ne[:, :, None]], dim=2)

            npos, ndir, nvel = (npos_ne[bi, ego_row], ndir_ne[bi, ego_row],
                                nvel_ne[bi, ego_row])
        else:
            npos, ndir, nvel = bpos, bdir, bvel

        rec_pos.append(npos)
        rec_dir.append(ndir)
        rec_v.append(core[bi, ego_row, V].clone())
        rec_vel.append(nvel)
        rec_valid.append(~done)

        new_acc = (nvel - hist_vel[:, -1]) / dt_data
        hist_pos = torch.cat([hist_pos, npos[:, None]], dim=1)
        hist_dir = torch.cat([hist_dir, ndir[:, None]], dim=1)
        hist_vel = torch.cat([hist_vel, nvel[:, None]], dim=1)
        hist_acc = torch.cat([hist_acc, new_acc[:, None]], dim=1)

        done_a = done_a | ((F_now + k + 1) >= F_end_a)
        freeze = (done_a | ~live_a).reshape(N)
        done = done_a[bi, ego_row]

    ego_pos = torch.stack(rec_pos, dim=1)
    ego_dir = torch.stack(rec_dir, dim=1)
    ego_v = torch.stack(rec_v, dim=1)
    ego_vel = torch.stack(rec_vel, dim=1)
    valid = torch.stack(rec_valid, dim=1)


    yaw = torch.atan2(ego_dir[..., 1], ego_dir[..., 0])
    yaw0 = torch.atan2(hist_dir[:, P - 1, 1], hist_dir[:, P - 1, 0])[:, None]
    yaw_prev = torch.cat([yaw0, yaw[:, :-1]], dim=1)
    dyaw = (yaw - yaw_prev + torch.pi) % (2 * torch.pi) - torch.pi
    ego_yaw_rate = dyaw / dt_data

    ne_fields = {}
    if forecast_nonego:
        S = ne_pos.shape[2] - P
        vsrc = torch.arange(P + S, device=device).clamp(max=F_src - 1)
        ne_fields = dict(
            nonego_pos=ne_pos, nonego_dir=ne_dir, nonego_vel=ne_vel, nonego_acc=ne_acc,
            nonego_valid=scene_t0.agents.valid[:, :, vsrc],
        )

    plan_ec_fields = {}
    if plan_ec_pos:
        plan_ec_fields = dict(
            plan_ego_pos=torch.stack(plan_ec_pos, dim=1),
            plan_ego_dir=torch.stack(plan_ec_dir, dim=1),
            plan_ego_valid=torch.stack(plan_ec_valid, dim=1),
            plan_issue_done=torch.stack(plan_ec_done, dim=1),
        )
    if plan_ne_pos:
        plan_ec_fields.update(
            plan_nonego_pos=torch.stack(plan_ne_pos, dim=1),
            plan_nonego_dir=torch.stack(plan_ne_dir, dim=1),
            plan_nonego_valid=torch.stack(plan_ne_valid, dim=1),
        )
    if plan_belief_logits:
        plan_ec_fields.update(
            plan_belief_logits=torch.stack(plan_belief_logits, dim=1),
        )

    return RolloutResult(
        scene_t0=scene_t0, ego_idx=ego_idx,
        ego_pos=ego_pos, ego_dir=ego_dir, ego_v=ego_v, ego_vel=ego_vel,
        ego_yaw_rate=ego_yaw_rate,
        valid=valid, F_now=F_now, dt=dt_data,
        num_plan_calls=num_plan_calls,
        **ne_fields,
        **plan_ec_fields,
    )


def rollout_to_output(rollout: RolloutResult, spec: SimSpec) -> feature.Output:
    """Pack a closed-loop rollout into a standard ``feature.Output`` for scoring.

    The realized ego trajectory replaces the ego's FUTURE slots; every other slot
    (and the ego's past) stays as log-replay from ``scene_t0``. Built over the full
    rollout window ``P + S`` in the T0 frame — frame-consistent with ``scene_t0``, so
    the same open-loop validation metric scores closed-loop just by reading the
    realized ego from this Output (with a configs whose ``num_future_frames == S``).
    Mirrors ``scripts.evaluation._scene_to_gt_output``, but overwrites the ego future.
    """
    agents = rollout.scene_t0.agents
    P = spec.num_past_frames
    S = rollout.ego_pos.shape[1]
    idx = list(range(P + S))
    B, A = agents.valid.shape[:2]
    device = agents.valid.device
    bi = torch.arange(B, device=device)
    ego = rollout.ego_idx

    pos = agents.pos_xy[:, :, idx].clone()
    dir_ = agents.dir_xy[:, :, idx].clone()
    valid = agents.valid[:, :, idx].clone()
    size_xyz = agents.size_xyz.unsqueeze(2).expand(B, A, P + S, 3)
    agent_type = agents.type.unsqueeze(-1).expand(B, A, P + S)
    is_ego = agents.is_ego.unsqueeze(-1).expand(B, A, P + S)


    pos[bi, ego, P:] = rollout.ego_pos
    dir_[bi, ego, P:] = rollout.ego_dir
    valid[bi, ego, P:] = rollout.valid


    if rollout.nonego_pos is not None:
        ne = agents.is_ego.new_ones((B, A), dtype=torch.bool)
        ne[bi, ego] = False
        m = ne[:, :, None, None]
        pos = torch.where(m, rollout.nonego_pos[:, :, :P + S], pos)
        dir_ = torch.where(m, rollout.nonego_dir[:, :, :P + S], dir_)
        valid = torch.where(ne[:, :, None], rollout.nonego_valid[:, :, :P + S], valid)


    ego_vel = agents.vel_xy[bi, ego][:, idx].clone()
    ego_vel[:, P:] = rollout.ego_vel


    ego_acc = agents.acc_xy[bi, ego][:, idx].clone()
    previous_speed = torch.cat([
        ego_vel[:, P - 1:P].norm(dim=-1),
        rollout.ego_v[:, :-1],
    ], dim=1)
    realized_lon_acc = (rollout.ego_v - previous_speed) / rollout.dt
    ego_acc[:, P:] = realized_lon_acc[..., None] * rollout.ego_dir

    ego_yaw_rate = torch.zeros(B, P + S, dtype=pos.dtype, device=device)
    ego_yaw_rate[:, P:] = rollout.ego_yaw_rate
    previous_yaw_rate = torch.cat([
        ego_yaw_rate[:, P - 1:P],
        rollout.ego_yaw_rate[:, :-1],
    ], dim=1)
    ego_yaw_accel = torch.zeros_like(ego_yaw_rate)
    ego_yaw_accel[:, P:] = (
        rollout.ego_yaw_rate - previous_yaw_rate
    ) / rollout.dt


    extra = {
        "agent_id": agents.id,
        "ego_vel": ego_vel,
        "ego_acc": ego_acc,
        "ego_yaw_rate": ego_yaw_rate,
        "ego_yaw_accel": ego_yaw_accel,
    }
    if rollout.plan_ego_pos is not None:
        extra.update(
            plan_ego_pos=rollout.plan_ego_pos,
            plan_ego_dir=rollout.plan_ego_dir,
            plan_ego_valid=rollout.plan_ego_valid,
            plan_issue_done=rollout.plan_issue_done,
        )
    if rollout.plan_nonego_pos is not None:
        extra.update(
            plan_nonego_pos=rollout.plan_nonego_pos,
            plan_nonego_dir=rollout.plan_nonego_dir,
            plan_nonego_valid=rollout.plan_nonego_valid,
        )
    if rollout.plan_belief_logits is not None:
        extra.update(plan_belief_logits=rollout.plan_belief_logits)

    return feature.Output(
        pos_xy=pos, dir_xy=dir_, size_xyz=size_xyz,
        valid=valid, is_ego=is_ego, type=agent_type,
        batch_valid=agents.is_ego.any(dim=1),
        extra=extra,
    )
