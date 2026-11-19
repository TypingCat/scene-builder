from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from scene_builder._pdm.dtypes import config, dtype, feature
from scene_builder._pdm.harness import NavsimConfigs
from scene_builder._pdm.sim.closed_loop import track_plan_frame
from scene_builder._pdm.sim.controller import (
    init_ctrl_state,
    plan_profile,
    pure_pursuit_control,
)
from scene_builder._pdm.sim.dynamics import (
    init_state,
    params_from_agents,
    readout,
    ring_slots,
    warm_start,
)
from scene_builder._pdm.sim.drivable_area import (
    DrivableAreaPolygons,
    build_drivable_area_polygons,
)
from scene_builder._pdm.utils.geometry import pick_ego_route


STAGE_GAP_SEC: float = 0.5


EC_ACCEL_THRESHOLD = 0.7


EC_JERK_THRESHOLD = 0.5


EC_YAW_RATE_THRESHOLD = 0.1


EC_YAW_ACCEL_THRESHOLD = 0.1


PREFIX = "NAVSIM_2025_PLANNING"


DDC_COMPLIANCE_THRESHOLD = 2.0


DDC_VIOLATION_THRESHOLD = 6.0


WEIGHTED_KEYS = [
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "extended_comfort",
]


MULTIPLIER_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
]


def score_rollout(
    output: feature.Output,
    scene: feature.Scene,
    configs: NavsimConfigs,
) -> dict[str, dtype.Metric]:
    """Score a single rollout. EC 는 rollout 의 per-replan plan 쌍에서 계산(폐루프) —
    plan 이 없는 Output(open-loop 단일 rollout)이면 `_run_all_metrics` 의 placeholder 1.0."""
    device = scene.agents.pos_xy.device
    zeros_valid = torch.zeros(1, dtype=torch.bool, device=device)
    eb = _extract_navsim_ego_batch(output, configs)

    if eb is None or not eb.has_ego.any():
        zero = torch.zeros(1, device=device)
        result: dict[str, dtype.Metric] = {}
        for key in MULTIPLIER_KEYS + WEIGHTED_KEYS:
            result[f"{PREFIX}/{key}"] = dtype.Metric(tensor=zero.clone(), valid=zeros_valid)
        result[f"{PREFIX}/multiplier_product"] = dtype.Metric(tensor=zero.clone(), valid=zeros_valid)
        result[f"{PREFIX}/epdms"] = dtype.Metric(tensor=zero.clone(), valid=zeros_valid)
        return result

    sample_valid = eb.has_ego
    gt_eb = _extract_gt_ego_batch(scene, configs)
    metrics = _run_all_metrics(eb, gt_eb, scene, configs)
    model_masked = _masked_progress(metrics)


    if gt_eb is not None and gt_eb.has_ego.any():
        gt_metrics = _run_all_metrics(gt_eb, gt_eb, scene, configs)
        norm_constant = torch.maximum(model_masked, _masked_progress(gt_metrics))
        metrics = _apply_human_penalty_filter(metrics, gt_metrics)
    else:
        norm_constant = model_masked


    rollout_ec = _compute_rollout_ec(output, configs)
    if rollout_ec is not None:
        metrics["extended_comfort"] = rollout_ec

    metrics = _aggregate_epdms_with_ec(metrics, norm_constant)

    result: dict[str, dtype.Metric] = {}
    for key in MULTIPLIER_KEYS + WEIGHTED_KEYS:
        result[f"{PREFIX}/{key}"] = dtype.Metric(
            tensor=metrics[key], valid=sample_valid,
        )
    result[f"{PREFIX}/multiplier_product"] = dtype.Metric(
        tensor=metrics["multiplier_product"], valid=sample_valid,
    )

    result[f"{PREFIX}/ego_progress_raw"] = dtype.Metric(
        tensor=metrics["ego_progress_raw"], valid=sample_valid,
    )
    result[f"{PREFIX}/ego_progress_masked"] = dtype.Metric(
        tensor=model_masked, valid=sample_valid,
    )

    result[f"{PREFIX}/speed_limit_compliance"] = dtype.Metric(
        tensor=metrics["speed_limit_compliance"], valid=sample_valid,
    )
    result[f"{PREFIX}/epdms"] = dtype.Metric(
        tensor=metrics["epdms"], valid=sample_valid,
    )
    return result


def _aggregate_epdms_with_ec(
    metrics: dict[str, torch.Tensor],
    norm_constant: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute EPDMS including EC in weighted average (5 weights, sum=16).

    Identical to OPEN_LOOP's _aggregate_epdms but includes EC weight.

    norm_constant: proposal 집합의 masked progress 최대값 [m]. 이하이면 (정지 씬)
    전 proposal 이 1.0.
    """
    device = metrics["no_at_fault_collisions"].device

    multiplier = (
        metrics["no_at_fault_collisions"]
        * metrics["drivable_area_compliance"]
        * metrics["driving_direction_compliance"]
        * metrics["traffic_light_compliance"]
    )

    ep_score = _normalize_progress(metrics["ego_progress_raw"], norm_constant)
    metrics["ego_progress"] = ep_score


    weights = torch.tensor(
        [EP_WEIGHT, TTC_WEIGHT, LK_WEIGHT, HC_WEIGHT, EC_WEIGHT],
        device=device,
    )
    values = torch.stack([
        ep_score,
        metrics["time_to_collision_within_bound"],
        metrics["lane_keeping"],
        metrics["history_comfort"],
        metrics["extended_comfort"],
    ], dim=-1)
    weighted_avg = (values * weights).sum(dim=-1) / weights.sum()

    epdms = multiplier * weighted_avg

    metrics["multiplier_product"] = multiplier
    metrics["epdms"] = epdms
    return metrics


def _two_frame_ec_from_fields(
    a_vel: torch.Tensor, a_heading: torch.Tensor, a_valid: torch.Tensor,
    b_vel: torch.Tensor, b_heading: torch.Tensor, b_valid: torch.Tensor,
    *,
    dt: float,
) -> torch.Tensor:
    """Two-frame EC core on overlap-aligned fields (A,B already sliced to the SAME
    real-world window). Kinematic-feature RMS of A−B vs thresholds → [B] (0.0/1.0).

    a_vel/b_vel: [B, S_overlap, 2]; a_heading/b_heading, a_valid/b_valid: [B, S_overlap].
    """

    a_accel, a_jerk, a_yaw_rate, a_yaw_accel = _extract_ec_features(a_vel, a_heading, a_valid, dt)
    b_accel, b_jerk, b_yaw_rate, b_yaw_accel = _extract_ec_features(b_vel, b_heading, b_valid, dt)


    both_valid = a_valid & b_valid
    valid_count = both_valid.float().sum(dim=1).clamp(min=1)

    def _rms(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a - b
        masked = (diff ** 2) * both_valid.float()
        return torch.sqrt(masked.sum(dim=1) / valid_count + EPS)

    rms_accel = _rms(a_accel, b_accel)
    rms_jerk = _rms(a_jerk, b_jerk)
    rms_yaw_rate = _rms(a_yaw_rate, b_yaw_rate)
    rms_yaw_accel = _rms(a_yaw_accel, b_yaw_accel)


    ec = (
        (rms_accel <= EC_ACCEL_THRESHOLD)
        & (rms_jerk <= EC_JERK_THRESHOLD)
        & (rms_yaw_rate <= EC_YAW_RATE_THRESHOLD)
        & (rms_yaw_accel <= EC_YAW_ACCEL_THRESHOLD)
    )

    return ec.float()


def _two_frame_ec_aligned(
    a_vel: torch.Tensor, a_heading: torch.Tensor, a_valid: torch.Tensor,
    b_vel: torch.Tensor, b_heading: torch.Tensor, b_valid: torch.Tensor,
    *,
    gap_sec: float,
    dt: float,
) -> torch.Tensor:
    """Align two predictions' FULL future fields at gap_sec, then score two-frame EC.

    overlap_start = round(gap_sec / dt); A_overlap = A_future[overlap_start:],
    B_overlap = B_future[:-overlap_start] (same real-world window). Returns ones
    when the overlap is too short (<3) to estimate jerk. a_vel/b_vel: [B, F, 2];
    a_heading/b_heading, a_valid/b_valid: [B, F].
    """
    overlap_start = round(gap_sec / dt)
    assert overlap_start >= 1, f"gap_sec={gap_sec} too small for dt={dt}"

    B = a_vel.shape[0]
    device = a_vel.device
    S_overlap = a_vel.shape[1] - overlap_start
    if S_overlap < 3:
        return torch.ones(B, device=device)

    return _two_frame_ec_from_fields(
        a_vel[:, overlap_start:], a_heading[:, overlap_start:], a_valid[:, overlap_start:],
        b_vel[:, :-overlap_start], b_heading[:, :-overlap_start], b_valid[:, :-overlap_start],
        dt=dt,
    )


def _extract_ec_features(
    vel: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract kinematic features for EC: acceleration, jerk, yaw_rate, yaw_accel.

    Velocity from data (dir * speed), not diff(pos).
    Invalid timesteps forward-filled to prevent Savgol boundary spikes.

    Returns: (accel_mag [B, S], jerk_mag [B, S], yaw_rate [B, S], yaw_accel [B, S])
    """
    S = vel.shape[1]


    ff_vel = vel.clone()
    ff_heading = heading.clone()
    for t in range(1, S):
        mask = ~valid[:, t]
        ff_vel[:, t][mask] = ff_vel[:, t - 1][mask]
        ff_heading[:, t][mask] = ff_heading[:, t - 1][mask]

    def _round8(v: torch.Tensor) -> torch.Tensor:
        return torch.round(v * 1e8) / 1e8


    calc_dtype = torch.float64


    raw_accel = torch.diff(ff_vel.norm(dim=-1).to(calc_dtype), dim=1) / dt
    raw_accel = torch.cat([raw_accel[:, :1], raw_accel], dim=1)
    accel_mag = _round8(_hc_savgol_filter_1d(
        raw_accel.abs(), window_length=8, poly_order=2,
    ))


    jerk_mag = _round8(_hc_savgol_filter_1d(
        accel_mag, window_length=15, poly_order=2, deriv=1, delta=dt,
    ))


    unwrapped = _phase_unwrap(ff_heading.to(calc_dtype))
    yaw_rate = _round8(_hc_savgol_filter_1d(
        unwrapped, window_length=5, poly_order=2, deriv=1, delta=dt,
    ))
    yaw_accel = _round8(_hc_savgol_filter_1d(
        unwrapped, window_length=5, poly_order=2, deriv=2, delta=dt,
    ))

    return accel_mag, jerk_mag, yaw_rate, yaw_accel


def _run_all_metrics(
    eb: _NavsimEgoBatch,
    gt_eb: _NavsimEgoBatch | None,
    scene: feature.Scene,
    configs: NavsimConfigs,
) -> dict[str, torch.Tensor]:
    """Run all sub-metrics and return raw scores."""
    device = eb.device


    drivable_polygons = _build_drivable_polygons_for_ego(
        eb, scene.links, scene.lanes, configs,
    )
    dac_score, offroad_mask, multiple_lanes_mask = (
        _compute_drivable_area_compliance(
            eb,
            scene.links,
            scene.lanes,
            configs,
            polygons=drivable_polygons,
        )
    )
    lateral_fault_area = offroad_mask | multiple_lanes_mask


    nc_score = _compute_no_at_fault_collisions(
        eb, gt_eb, lateral_fault_area,
    )
    ddc_score = _compute_driving_direction_compliance(
        eb,
        scene.paths,
        scene.links,
        lanes=scene.lanes,
        configs=configs,
        polygons=drivable_polygons,
    )
    tlc_score = _compute_traffic_light_compliance(
        eb,
        scene.traffics,
        scene.links,
        configs,
        paths=scene.paths,
        lanes=scene.lanes,
        polygons=drivable_polygons,
    )


    ep_raw = _compute_ego_progress(eb, scene.paths)
    ttc_score = _compute_ttc(
        eb,
        gt_eb,
        lateral_fault_area,
        scene.lanes,
        polygons=drivable_polygons,
        configs=configs,
    )
    lk_score = _compute_lane_keeping(
        eb,
        gt_eb,
        scene.paths,
        scene.links,
        scene.lanes,
        configs,
        polygons=drivable_polygons,
    )
    hc_score = _compute_history_comfort(eb)
    ec_score = torch.full((eb.B,), _compute_extended_comfort(), device=device)


    slc_score = _compute_speed_limit_compliance(
        eb,
        scene.links,
        scene.lanes,
        configs,
        polygons=drivable_polygons,
    )

    return {
        "no_at_fault_collisions": nc_score,
        "drivable_area_compliance": dac_score,
        "driving_direction_compliance": ddc_score,
        "traffic_light_compliance": tlc_score,
        "ego_progress_raw": ep_raw,
        "time_to_collision_within_bound": ttc_score,
        "lane_keeping": lk_score,
        "history_comfort": hc_score,
        "extended_comfort": ec_score,
        "speed_limit_compliance": slc_score,
    }


HUMAN_FILTER_KEYS = MULTIPLIER_KEYS + [
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
]


def _apply_human_penalty_filter(
    model_m: dict[str, torch.Tensor],
    gt_m: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Transfer GT's passing metrics to model: where GT failed (==0), exempt model (→1)."""
    for key in HUMAN_FILTER_KEYS:
        mask = gt_m[key] == 0.0
        model_m[key] = torch.where(mask, torch.ones_like(model_m[key]), model_m[key])
    return model_m


def _compute_no_at_fault_collisions(
    eb: _NavsimEgoBatch,
    gt_eb: _NavsimEgoBatch | None,
    ego_area_violation: torch.Tensor,
) -> torch.Tensor:
    """NAVSIM no-at-fault-collision score, vectorized by track.

    Collision type follows the reference decision order: stopped ego, stopped
    track, active rear, active front-bumper, then active lateral.  Objects
    intersecting ego at the proposal initial state are ignored, and only the
    first subsequent collision with each track is classified.
    """
    B, S = eb.B, eb.num_future_frames
    device = eb.device
    if S == 0:
        return torch.ones(B, device=device)
    if gt_eb is None:
        return torch.ones(B, device=device)

    assert eb.B == gt_eb.B and eb.num_future_frames == gt_eb.num_future_frames \
        and eb.dt == gt_eb.dt, "eb/gt_eb 정렬 불일치 (B/S/dt)"

    A = gt_eb.all_pos_future.shape[1]
    if A == 0:
        return torch.ones(B, device=device)


    ego_rot = _heading_to_rot(eb.ego_heading_future)
    other_rot = _heading_to_rot(gt_eb.all_heading_future)
    other_valid = (
        gt_eb.all_valid_future
        & ~gt_eb.ego_mask.unsqueeze(-1)
    ).permute(0, 2, 1)
    other_valid &= eb.ego_valid_future.unsqueeze(-1)
    sat_dist = _sat_signed_distance(
        eb.ego_pos_future.unsqueeze(2).expand(-1, -1, A, -1),
        ego_rot.unsqueeze(2).expand(-1, -1, A, -1, -1),
        eb.ego_size_future.unsqueeze(2).expand(-1, -1, A, -1),
        gt_eb.all_pos_future.permute(0, 2, 1, 3),
        other_rot.permute(0, 2, 1, 3, 4),
        gt_eb.all_size_future.permute(0, 2, 1, 3),
    )
    colliding = (sat_dist <= 0.0) & other_valid


    initial_heading = torch.atan2(
        eb.ego_dir_past[:, -1, 1], eb.ego_dir_past[:, -1, 0],
    )
    initial_ego_rot = _heading_to_rot(initial_heading)
    initial_other_rot = _heading_to_rot(gt_eb.all_heading_initial)
    initial_sat = _sat_signed_distance(
        eb.ego_pos_past[:, -1, None].expand(-1, A, -1),
        initial_ego_rot[:, None].expand(-1, A, -1, -1),
        eb.ego_size_future[:, 0, None].expand(-1, A, -1),
        gt_eb.all_pos_initial,
        initial_other_rot,
        gt_eb.all_size_initial,
    )
    initially_collided = (
        (initial_sat <= 0.0)
        & gt_eb.all_valid_initial
        & ~gt_eb.ego_mask
        & eb.ego_valid_past[:, -1, None]
    )
    colliding &= ~initially_collided[:, None]


    has_collision = colliding.any(dim=1)
    first_time = colliding.float().argmax(dim=1)
    gather_time = first_time[:, None, :]

    ego_stopped_all = (
        eb.ego_speed_future <= NC_STOPPED_SPEED_THRESHOLD
    )[:, :, None].expand(-1, -1, A)


    first_valid_time = gt_eb.all_valid_future.float().argmax(dim=-1)
    first_future_speed = gt_eb.all_speed_future.gather(
        2, first_valid_time[..., None],
    ).squeeze(-1)
    track_speed = torch.where(
        gt_eb.all_valid_initial,
        gt_eb.all_speed_initial,
        first_future_speed,
    )
    is_agent = (
        (gt_eb.all_type_future[:, :, 0] == feature.AGENT_TYPE.vehicle)
        | (gt_eb.all_type_future[:, :, 0] == feature.AGENT_TYPE.pedestrian)
        | (gt_eb.all_type_future[:, :, 0] == feature.AGENT_TYPE.cyclist)
    )
    track_stopped = (
        (~is_agent)
        | (track_speed <= NC_STOPPED_SPEED_THRESHOLD)
    )[:, None].expand(-1, S, -1)


    ego_dir_norm = (
        eb.ego_dir_future
        / eb.ego_dir_future.norm(dim=-1, keepdim=True).clamp(min=EPS)
    )
    ego_rear = (
        eb.ego_pos_future
        - ego_dir_norm
        * eb.ego_size_future[..., :1]
        * NC_REAR_AXLE_TO_CENTER_RATIO
    )
    to_other = (
        gt_eb.all_pos_future.permute(0, 2, 1, 3)
        - ego_rear.unsqueeze(2)
    )
    relative_cos = (
        (to_other * ego_dir_norm.unsqueeze(2)).sum(dim=-1)
        / to_other.norm(dim=-1).clamp(min=EPS)
    )
    active_rear = relative_cos < math.cos(math.radians(150.0))


    ego_corners = _build_ego_corners(
        eb.ego_pos_future, eb.ego_dir_future, eb.ego_size_future,
    )
    active_front = _segment_intersects_obbs(
        ego_corners[..., 0, :],
        ego_corners[..., 1, :],
        gt_eb.all_pos_future.permute(0, 2, 1, 3),
        other_rot.permute(0, 2, 1, 3, 4),
        gt_eb.all_size_future.permute(0, 2, 1, 3),
    )

    first_ego_stopped = ego_stopped_all.gather(1, gather_time).squeeze(1)
    first_track_stopped = track_stopped.gather(1, gather_time).squeeze(1)
    first_rear = active_rear.gather(1, gather_time).squeeze(1)
    first_front = active_front.gather(1, gather_time).squeeze(1)
    area = (
        ego_area_violation[:, :S]
        if ego_area_violation.shape[1] >= S
        else F.pad(ego_area_violation, (0, S - ego_area_violation.shape[1]))
    )
    first_area = area[:, :, None].expand(-1, -1, A).gather(
        1, gather_time,
    ).squeeze(1)


    stopped_track_collision = ~first_ego_stopped & first_track_stopped
    front_collision = (
        ~first_ego_stopped
        & ~first_track_stopped
        & ~first_rear
        & first_front
    )
    lateral_collision = (
        ~first_ego_stopped
        & ~first_track_stopped
        & ~first_rear
        & ~first_front
    )
    at_fault = has_collision & (
        stopped_track_collision
        | front_collision
        | (lateral_collision & first_area)
    )

    all_type = gt_eb.all_type_future.gather(
        2, first_time[..., None],
    ).squeeze(-1)
    is_agent_type = (
        (all_type == feature.AGENT_TYPE.vehicle) |
        (all_type == feature.AGENT_TYPE.pedestrian) |
        (all_type == feature.AGENT_TYPE.cyclist)
    )
    collision_score = torch.where(
        at_fault, torch.where(is_agent_type, 0.0, 0.5), 1.0,
    )
    return collision_score.amin(dim=1)


def _build_drivable_polygons_for_ego(
    eb: _NavsimEgoBatch,
    links: feature.Link,
    lanes: feature.Lane | None,
    configs: NavsimConfigs | None,
) -> DrivableAreaPolygons | None:
    """Build the shared local polygon set used by DAC and DDC."""
    area_cfg = (
        configs.drivable_area
        if configs is not None
        else config.DrivableAreaConfig()
    )
    center_count = eb.ego_valid_future.sum(dim=1, keepdim=True)
    center_xy = (
        (eb.ego_pos_future * eb.ego_valid_future[..., None]).sum(dim=1)
        / center_count.clamp(min=1)
    )
    center_xy = torch.where(
        center_count > 0,
        center_xy,
        eb.ego_pos_past[:, -1],
    )
    return build_drivable_area_polygons(
        links,
        lanes,
        center_xy=center_xy,
        max_links=area_cfg.max_links,
        max_lanes=area_cfg.max_lanes,
        default_half_width_m=area_cfg.default_half_width_m,
        min_half_width_m=area_cfg.min_half_width_m,
        max_half_width_m=area_cfg.max_half_width_m,
        link_chunk=area_cfg.link_chunk,
    )


def _compute_drivable_area_compliance(
    eb: _NavsimEgoBatch,
    links: feature.Link,
    lanes: feature.Lane,
    configs: NavsimConfigs,
    *,
    polygons: DrivableAreaPolygons | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NAVSIM DAC: every ego corner must lie in the drivable polygon union.

    The source dataspec has no roadblock polygons, so the shared reconstruction
    builds them from vehicle links and their referenced left/right boundaries.
    The returned ``multiple_lanes`` and ``offroad`` masks are shared with
    NC/TTC for the reference lateral-fault attribution rule.
    """
    B, S = eb.B, eb.num_future_frames
    device = eb.device
    area_cfg = configs.drivable_area
    if polygons is None:
        polygons = _build_drivable_polygons_for_ego(
            eb, links, lanes, configs,
        )
    if polygons is None or polygons.B != B:
        offroad = torch.zeros(B, S, dtype=torch.bool, device=device)
        multiple_lanes = torch.zeros_like(offroad)
        score = torch.zeros(B, device=device)
        return score, offroad, multiple_lanes


    initial_pos = eb.ego_pos_past[:, -1:]
    initial_dir = eb.ego_dir_past[:, -1:]
    initial_size = eb.ego_size_future[:, :1]
    initial_valid = eb.ego_valid_past[:, -1:]
    all_pos = torch.cat([initial_pos, eb.ego_pos_future], dim=1)
    all_dir = torch.cat([initial_dir, eb.ego_dir_future], dim=1)
    all_size = torch.cat([initial_size, eb.ego_size_future], dim=1)
    all_valid = torch.cat([initial_valid, eb.ego_valid_future], dim=1)
    corners = _build_ego_corners(all_pos, all_dir, all_size)
    membership = polygons.contains_each(
        corners,
        polygon_chunk=area_cfg.polygon_chunk,
    )
    corner_count = membership.sum(dim=2)
    all_offroad = (~membership.any(dim=-1).all(dim=-1)) & all_valid
    all_multiple_lanes = (
        ((corner_count > 0).sum(dim=-1) > 1)
        & (corner_count != 4).all(dim=-1)
        & all_valid
    )
    score = (~all_offroad.any(dim=-1)).to(eb.ego_pos_future.dtype)
    return score, all_offroad[:, 1:], all_multiple_lanes[:, 1:]


def _compute_ttc(
    eb: _NavsimEgoBatch,
    gt_eb: "_NavsimEgoBatch | None",
    ego_area_violation: torch.Tensor,
    lanes: "feature.Lane | None" = None,
    *,
    polygons: DrivableAreaPolygons | None = None,
    configs: NavsimConfigs | None = None,
) -> torch.Tensor:
    """NAVSIM TTC from projected ego boxes against future observations.

    For each proposal state, the ego box is translated at its current heading
    and speed over offsets ``[0, 3, 6, 9]``.  Intersections are classified with
    nuPlan's 30-degree ahead / 150-degree behind cones.  A first non-fault
    collision permanently excludes that track from later TTC checks.

    Returns [B] score (0.0 or 1.0).
    """
    B, S = eb.B, eb.num_future_frames
    device = eb.device
    dtype_ = eb.ego_pos_future.dtype
    scores = torch.ones(B, dtype=dtype_, device=device)

    if gt_eb is None:
        return scores
    assert eb.B == gt_eb.B and eb.num_future_frames == gt_eb.num_future_frames \
        and eb.dt == gt_eb.dt, "eb/gt_eb 정렬 불일치 (B/S/dt)"

    A = gt_eb.all_pos_future.shape[1]
    if S == 0 or A == 0:
        return scores


    offsets = torch.arange(
        0,
        int(FUTURE_COLLISION_HORIZON * 10),
        3,
        device=device,
    )
    if offsets.numel() == 0:
        return scores


    initial_size = eb.ego_size_future[:, :1]
    ego_pos = torch.cat([eb.ego_pos_past[:, -1:], eb.ego_pos_future], dim=1)
    ego_dir = torch.cat([eb.ego_dir_past[:, -1:], eb.ego_dir_future], dim=1)
    ego_speed = torch.cat(
        [eb.ego_speed_past[:, -1:], eb.ego_speed_future],
        dim=1,
    )
    ego_size = torch.cat([initial_size, eb.ego_size_future], dim=1)
    ego_valid = torch.cat(
        [eb.ego_valid_past[:, -1:], eb.ego_valid_future],
        dim=1,
    )
    ego_dir = ego_dir / ego_dir.norm(dim=-1, keepdim=True).clamp(min=EPS)

    other_pos_states = torch.cat(
        [gt_eb.all_pos_initial[:, :, None], gt_eb.all_pos_future],
        dim=2,
    )
    other_heading_states = torch.cat(
        [gt_eb.all_heading_initial[:, :, None], gt_eb.all_heading_future],
        dim=2,
    )
    other_valid_states = torch.cat(
        [gt_eb.all_valid_initial[:, :, None], gt_eb.all_valid_future],
        dim=2,
    )
    other_size_states = torch.cat(
        [gt_eb.all_size_initial[:, :, None], gt_eb.all_size_future],
        dim=2,
    )

    num_states = S + 1
    max_offset = int(offsets[-1])
    num_sources = num_states - max_offset
    if num_sources <= 0:
        return scores
    source_indices = torch.arange(num_sources, device=device)
    target_indices = source_indices[:, None] + offsets[None]
    flat_target = target_indices.reshape(-1)
    O = offsets.numel()

    source_pos = ego_pos[:, source_indices]
    source_dir = ego_dir[:, source_indices]
    source_speed = ego_speed[:, source_indices]
    source_size = ego_size[:, source_indices]
    source_valid = ego_valid[:, source_indices]
    source_heading = torch.atan2(source_dir[..., 1], source_dir[..., 0])
    source_rot = _heading_to_rot(source_heading)

    delta_t = offsets.to(dtype_) * eb.dt
    projected_pos = (
        source_pos[:, :, None]
        + source_dir[:, :, None]
        * source_speed[:, :, None, None]
        * delta_t[None, None, :, None]
    )

    other_pos = other_pos_states.index_select(2, flat_target).reshape(
        B, A, num_sources, O, 2,
    ).permute(0, 2, 3, 1, 4)
    other_heading = other_heading_states.index_select(2, flat_target).reshape(
        B, A, num_sources, O,
    ).permute(0, 2, 3, 1)
    other_valid = other_valid_states.index_select(2, flat_target).reshape(
        B, A, num_sources, O,
    ).permute(0, 2, 3, 1)
    first_valid_idx = other_valid_states.to(torch.long).argmax(dim=2)
    unique_size = other_size_states.gather(
        2,
        first_valid_idx[..., None, None].expand(B, A, 1, 2),
    ).squeeze(2)
    other_size = unique_size[:, None, None].expand(
        -1, num_sources, O, -1, -1,
    )
    other_valid &= ~gt_eb.ego_mask[:, None, None]


    projected_rot = source_rot[:, :, None, None].expand(-1, -1, O, A, -1, -1)
    sat_dist = _sat_signed_distance(
        projected_pos[:, :, :, None].expand(-1, -1, -1, A, -1),
        projected_rot,
        source_size[:, :, None, None].expand(-1, -1, O, A, -1),
        other_pos,
        _heading_to_rot(other_heading),
        other_size,
    )
    colliding = (
        (sat_dist <= 0.0)
        & other_valid
        & source_valid[:, :, None, None]
    )


    initial_area = torch.zeros(B, dtype=torch.bool, device=device)
    in_intersection = torch.zeros(
        B, num_states, dtype=torch.bool, device=device,
    )
    if polygons is not None and polygons.B == B:
        area_cfg = (
            configs.drivable_area
            if configs is not None
            else config.DrivableAreaConfig()
        )
        initial_corners = _build_ego_corners(
            ego_pos[:, :1], ego_dir[:, :1], ego_size[:, :1],
        )
        initial_membership = polygons.contains_each(
            initial_corners,
            polygon_chunk=area_cfg.polygon_chunk,
        )
        initial_count = initial_membership.sum(dim=2)
        initial_offroad = ~initial_membership.any(dim=-1).all(dim=-1)
        initial_multiple = (
            ((initial_count > 0).sum(dim=-1) > 1)
            & (initial_count != 4).all(dim=-1)
        )
        initial_area = (initial_offroad | initial_multiple).squeeze(1)


        ego_rear = (
            ego_pos
            - ego_dir
            * ego_size[..., :1]
            * NC_REAR_AXLE_TO_CENTER_RATIO
        )
        rear_membership = polygons.contains_each(
            ego_rear,
            polygon_chunk=area_cfg.polygon_chunk,
        )
        intersection_polygons = (
            (polygons.link_section == feature.LinkSection.intersection)
            & polygons.valid.any(dim=-1)
        )
        in_intersection = (
            rear_membership & intersection_polygons[:, None]
        ).any(dim=-1)

    future_area = (
        ego_area_violation[:, :S]
        if ego_area_violation.shape[1] >= S
        else F.pad(ego_area_violation, (0, S - ego_area_violation.shape[1]))
    )
    area_states = torch.cat([initial_area[:, None], future_area], dim=1)


    source_rear = (
        source_pos
        - source_dir
        * source_size[..., :1]
        * NC_REAR_AXLE_TO_CENTER_RATIO
    )
    relative = other_pos - source_rear[:, :, None, None]
    relative_cos = (
        (relative * source_dir[:, :, None, None]).sum(dim=-1)
        / relative.norm(dim=-1).clamp(min=EPS)
    )
    ahead = relative_cos > math.cos(math.radians(30.0))
    behind = relative_cos < math.cos(math.radians(150.0))
    source_area = area_states[:, source_indices]
    source_intersection = in_intersection[:, source_indices]
    at_fault_position = ahead | (
        (source_area | source_intersection)[:, :, None, None]
        & ~behind
    )


    initial_heading = torch.atan2(
        eb.ego_dir_past[:, -1, 1], eb.ego_dir_past[:, -1, 0],
    )
    initial_ego_rot = _heading_to_rot(initial_heading)
    initial_other_rot = _heading_to_rot(gt_eb.all_heading_initial)
    initial_sat = _sat_signed_distance(
        eb.ego_pos_past[:, -1, None].expand(-1, A, -1),
        initial_ego_rot[:, None].expand(-1, A, -1, -1),
        eb.ego_size_future[:, 0, None].expand(-1, A, -1),
        gt_eb.all_pos_initial,
        initial_other_rot,
        gt_eb.all_size_initial,
    )
    initially_collided = (
        (initial_sat <= 0.0)
        & gt_eb.all_valid_initial
        & ~gt_eb.ego_mask
        & eb.ego_valid_past[:, -1, None]
    )


    ignored_tracks = gt_eb.ego_mask.clone() | initially_collided
    stopped = source_speed < STOPPED_SPEED_THRESHOLD
    for source_idx in range(num_sources):
        for offset_idx in range(O):
            eligible = (
                colliding[:, source_idx, offset_idx]
                & ~ignored_tracks
                & ~stopped[:, source_idx, None]
            )
            fault = eligible & at_fault_position[:, source_idx, offset_idx]
            scores = torch.where(
                fault.any(dim=-1),
                torch.zeros_like(scores),
                scores,
            )
            ignored_tracks |= eligible & ~at_fault_position[:, source_idx, offset_idx]

    return scores


def _compute_driving_direction_compliance(
    eb: _NavsimEgoBatch,
    paths: feature.Path,
    links: feature.Link,
    compliance_threshold: float = DDC_COMPLIANCE_THRESHOLD,
    violation_threshold: float = DDC_VIOLATION_THRESHOLD,
    *,
    lanes: feature.Lane | None = None,
    configs: NavsimConfigs | None = None,
    polygons: DrivableAreaPolygons | None = None,
) -> torch.Tensor:
    """NAVSIM DDC from distance travelled outside on-route lane polygons.

    The reference marks a state as oncoming when the ego center is not inside
    any on-route lane/lane-connector polygon.  Travel at such states is summed
    over a trailing one-second horizon, except inside intersections.

    Returns [B] score (0.0, 0.5, or 1.0).
    """
    B = eb.B
    device = eb.device
    dtype_ = eb.ego_pos_future.dtype

    if paths is None or paths.valid.numel() == 0:
        return torch.zeros(B, device=device)
    if links is None or links.valid.numel() == 0:
        return torch.zeros(B, device=device)

    _, _, route_valid, near_link, has_route = pick_ego_route(paths)
    if polygons is None:
        polygons = _build_drivable_polygons_for_ego(
            eb, links, lanes, configs,
        )
    if polygons is None or polygons.B != B:
        return torch.zeros(B, device=device)

    route_polygons = polygons.matches_link_ids(
        near_link, route_valid,
    )
    has_ddc = has_route & route_polygons.any(dim=-1)


    all_pos = torch.cat(
        [eb.ego_pos_past[:, -1:], eb.ego_pos_future],
        dim=1,
    )
    all_valid = torch.cat(
        [eb.ego_valid_past[:, -1:], eb.ego_valid_future],
        dim=1,
    )
    area_cfg = (
        configs.drivable_area
        if configs is not None
        else config.DrivableAreaConfig()
    )
    center_membership = polygons.contains_each(
        all_pos,
        polygon_chunk=area_cfg.polygon_chunk,
    )
    is_oncoming = (
        ~(center_membership & route_polygons[:, None]).any(dim=-1)
        & all_valid
    )


    intersection_polygons = (
        (polygons.link_section == feature.LinkSection.intersection)
        & polygons.valid.any(dim=-1)
    )
    in_intersection = (
        center_membership & intersection_polygons[:, None]
    ).any(dim=-1)

    oncoming_progress = torch.zeros(
        all_pos.shape[:2], dtype=dtype_, device=device,
    )
    segment_valid = all_valid[:, 1:] & all_valid[:, :-1]
    step_distance = (all_pos[:, 1:] - all_pos[:, :-1]).norm(dim=-1)
    oncoming_progress[:, 1:] = torch.where(
        segment_valid, step_distance, torch.zeros_like(step_distance),
    )
    oncoming_progress = torch.where(
        is_oncoming & ~in_intersection,
        oncoming_progress,
        torch.zeros_like(oncoming_progress),
    )


    horizon_steps = max(0, int(DDC_HORIZON / eb.dt))
    padded = F.pad(oncoming_progress, (horizon_steps, 0))
    windows = padded.unfold(-1, horizon_steps + 1, 1)
    max_oncoming = windows.sum(dim=-1).max(dim=-1).values

    scores = torch.ones(B, dtype=dtype_, device=device)
    scores = torch.where(
        max_oncoming >= violation_threshold,
        torch.zeros_like(scores),
        scores,
    )
    scores = torch.where(
        (max_oncoming >= compliance_threshold) & (max_oncoming < violation_threshold),
        torch.full_like(scores, 0.5), scores,
    )
    return torch.where(has_ddc, scores, torch.zeros_like(scores))


def _compute_traffic_light_compliance(
    eb: _NavsimEgoBatch,
    traffics: feature.Traffic,
    links: feature.Link,
    configs: NavsimConfigs,
    *,
    paths: feature.Path | None = None,
    lanes: feature.Lane | None = None,
    polygons: DrivableAreaPolygons | None = None,
) -> torch.Tensor:
    """NAVSIM TLC from ego-box intersection with red route-link polygons.

    At every proposal state (initial plus future), the reference inserts each
    red on-route lane connector polygon into the occupancy map and fails a
    proposal when its ego polygon intersects one.  It does not require a
    stop-line crossing or heading alignment.

    Returns [B] score (0.0 or 1.0).
    """
    B, S = eb.B, eb.num_future_frames
    device = eb.device
    dtype_ = eb.ego_pos_future.dtype
    scores = torch.ones(B, dtype=dtype_, device=device)

    if traffics is None or traffics.valid.numel() == 0:
        return scores
    if traffics.state.numel() == 0 or traffics.target_link.numel() == 0:
        return scores
    if paths is None or paths.valid.numel() == 0:
        return scores
    if links is None or links.valid.numel() == 0:
        return scores

    _, _, route_valid, near_link, has_route = pick_ego_route(paths)
    if polygons is None:
        polygons = _build_drivable_polygons_for_ego(
            eb, links, lanes, configs,
        )
    if polygons is None or polygons.B != B:
        return scores

    route_polygons = (
        polygons.matches_link_ids(near_link, route_valid)
        & has_route[:, None]
    )
    target_link = traffics.target_link.to(polygons.link_id.dtype)
    traffic_to_route_polygon = (
        (target_link[:, :, None] == polygons.link_id[:, None])
        & (target_link[:, :, None] > 0)
        & route_polygons[:, None]
        & polygons.valid.any(dim=-1)[:, None]
    )
    if not traffic_to_route_polygon.any():
        return scores

    initial_size = eb.ego_size_future[:, :1]
    all_pos = torch.cat([eb.ego_pos_past[:, -1:], eb.ego_pos_future], dim=1)
    all_dir = torch.cat([eb.ego_dir_past[:, -1:], eb.ego_dir_future], dim=1)
    all_size = torch.cat([initial_size, eb.ego_size_future], dim=1)
    all_valid = torch.cat(
        [eb.ego_valid_past[:, -1:], eb.ego_valid_future],
        dim=1,
    )
    ego_corners = _build_ego_corners(all_pos, all_dir, all_size)
    ego_intersects = polygons.intersects_boxes(
        ego_corners,
        polygon_chunk=configs.drivable_area.polygon_chunk,
    )


    num_states = S + 1
    traffic_steps = traffics.state.shape[-1]
    state_indices = (
        eb.num_past_frames - 1
        + torch.arange(num_states, device=device)
    )
    state_in_range = (state_indices >= 0) & (state_indices < traffic_steps)
    state_indices = state_indices.clamp(min=0, max=max(traffic_steps - 1, 0))
    red_at_state = (
        (traffics.state[..., state_indices] == feature.TrafficState.stop)
        & traffics.valid[..., state_indices]
        & state_in_range[None, None]
    )

    active_red_polygons = (
        red_at_state.permute(0, 2, 1)[..., None]
        & traffic_to_route_polygon[:, None]
    ).any(dim=2)
    violation = ego_intersects & active_red_polygons & all_valid[..., None]
    return torch.where(
        violation.any(dim=(1, 2)),
        torch.zeros_like(scores),
        scores,
    )


def _compute_speed_limit_compliance(
    eb: _NavsimEgoBatch,
    links: feature.Link,
    lanes: feature.Lane | None,
    configs: NavsimConfigs,
    *,
    polygons: DrivableAreaPolygons | None = None,
) -> torch.Tensor:
    """nuPlan `speed_limit_compliance` — duration-weighted mean overspeed.

    Reference (`speed_limit_compliance.py`): per frame the violation depth is
    ``max(0, ego_speed - max(speed limits of the lanes ego is assigned to))``
    and the score is
    ``max(0, 1 - sum(depth)*dt / (SLC_MAX_OVERSPEED_THRESHOLD * duration))``.
    The ``dt`` cancels, leaving ``sum(depth) / (threshold * (num_states - 1))``.

    Assignment is by ego *center pose* containment, mirroring the reference's
    pose-based `get_route`; a frame with no containing link is not judged.
    Links whose ``speed_limit`` is 0 carry no posted limit — the reference skips
    them (`if all(speed_limits)`), so they must not act as a 0 m/s limit.

    Returns [B] score in [0, 1].
    """
    B, S = eb.B, eb.num_future_frames
    device = eb.device
    dtype_ = eb.ego_pos_future.dtype
    scores = torch.ones(B, dtype=dtype_, device=device)

    if links is None or links.valid.numel() == 0 or links.speed_limit.numel() == 0:
        return scores
    if polygons is None:
        polygons = _build_drivable_polygons_for_ego(eb, links, lanes, configs)
    if polygons is None or polygons.B != B:
        return scores


    pos = torch.cat([eb.ego_pos_past[:, -1:], eb.ego_pos_future], dim=1)
    speed = torch.cat([eb.ego_speed_past[:, -1:], eb.ego_speed_future], dim=1)
    valid = torch.cat([eb.ego_valid_past[:, -1:], eb.ego_valid_future], dim=1)


    link_match = (
        (polygons.link_id[..., None] == links.id[:, None])
        & links.valid.any(dim=-1)[:, None]
        & (polygons.link_id[..., None] > 0)
    )
    limit_per_link = links.speed_limit.to(dtype_)[:, None].expand_as(link_match)
    posted = torch.where(link_match, limit_per_link, torch.zeros_like(limit_per_link))
    polygon_limit = posted.amax(dim=-1)
    has_limit = polygon_limit > 0.0

    inside = polygons.contains_each(
        pos, polygon_chunk=configs.drivable_area.polygon_chunk,
    )
    applicable = inside & has_limit[:, None]

    frame_limit = torch.where(
        applicable, polygon_limit[:, None].expand_as(applicable),
        torch.zeros_like(applicable, dtype=dtype_),
    ).amax(dim=-1)
    judged = applicable.any(dim=-1) & valid

    depth = torch.where(
        judged, (speed - frame_limit).clamp(min=0.0),
        torch.zeros_like(frame_limit),
    )

    num_intervals = max(S, 1)
    loss = depth.sum(dim=1) / (SLC_MAX_OVERSPEED_THRESHOLD * num_intervals)
    return (1.0 - loss).clamp(min=0.0)


def _compute_lane_keeping(
    eb: _NavsimEgoBatch,
    gt_eb: _NavsimEgoBatch | None,
    paths: feature.Path,
    links: feature.Link,
    lanes: feature.Lane | None,
    configs: NavsimConfigs,
    *,
    polygons: DrivableAreaPolygons | None = None,
) -> torch.Tensor:
    """NAVSIM LK from consecutive center-to-route deviations over 0.5 m.

    Proposal states are ``[initial, future...]``.  Intersection states are
    skipped without incrementing or resetting the consecutive exceedance
    count, matching the reference loop exactly.

    Returns [B] score (0.0 or 1.0).
    """
    B, S = eb.B, eb.num_future_frames
    device = eb.device
    dtype_ = eb.ego_pos_future.dtype

    if paths is None or paths.valid.numel() == 0:
        return torch.zeros(B, dtype=dtype_, device=device)

    route_pos, _, route_valid, _, has_route = pick_ego_route(paths)
    has_centerline = (
        has_route
        & (route_valid[:, :-1] & route_valid[:, 1:]).any(dim=-1)
    )
    all_pos = torch.cat(
        [eb.ego_pos_past[:, -1:], eb.ego_pos_future],
        dim=1,
    )
    all_valid = torch.cat(
        [eb.ego_valid_past[:, -1:], eb.ego_valid_future],
        dim=1,
    )
    _, lateral_deviation = _project_onto_polyline(
        all_pos, route_pos, route_valid,
    )
    exceeds = lateral_deviation > LANE_KEEPING_DEVIATION_LIMIT
    continuous_steps_required = max(
        1,
        int(math.ceil(LANE_KEEPING_HORIZON / eb.dt)),
    )

    in_intersection = torch.zeros(
        B, S + 1, dtype=torch.bool, device=device,
    )
    if polygons is None and links is not None and links.valid.numel() > 0:
        polygons = _build_drivable_polygons_for_ego(
            eb, links, lanes, configs,
        )
    if polygons is not None and polygons.B == B:
        membership = polygons.contains_each(
            all_pos,
            polygon_chunk=configs.drivable_area.polygon_chunk,
        )
        intersection_polygons = (
            (polygons.link_section == feature.LinkSection.intersection)
            & polygons.valid.any(dim=-1)
        )
        in_intersection = (
            membership & intersection_polygons[:, None]
        ).any(dim=-1)


    active = all_valid & ~in_intersection
    active_exceeds = active & exceeds
    active_resets = active & ~exceeds


    cumsum_exc = active_exceeds.long().cumsum(dim=-1)
    cumsum_at_reset = torch.where(active_resets, cumsum_exc, torch.zeros_like(cumsum_exc))
    max_at_reset = cumsum_at_reset.cummax(dim=-1).values
    consecutive = cumsum_exc - max_at_reset

    scores = (
        consecutive.max(dim=-1).values < continuous_steps_required
    ).to(dtype_)
    return torch.where(has_centerline, scores, torch.zeros_like(scores))


def _compute_history_comfort(
    eb: _NavsimEgoBatch,
) -> torch.Tensor:
    """NAVSIM history comfort over padded human history + proposal states.

    This mirrors ``PDMScorer._calculate_history_comfort`` and
    ``pdm_comfort_metrics.ego_is_comfortable``:

    * sample cached human history from -1.5 s through -0.2 s;
    * append proposal state zero and every future state;
    * smooth stored rear-axle acceleration with SciPy-compatible SavGol;
    * evaluate all six comfort metrics over the *entire* padded trajectory;
    * use strict, eight-decimal-rounded threshold comparisons.
    """
    B = eb.B
    device = eb.device
    dt = eb.dt
    signal_dt = HC_HISTORY_SAMPLE_INTERVAL
    P = eb.num_past_frames

    if P == 0:
        return torch.ones(B, device=device)

    heading_past = torch.atan2(
        eb.ego_dir_past[..., 1], eb.ego_dir_past[..., 0],
    )


    full_velocity = torch.cat(
        [eb.ego_vel_past, eb.ego_vel_future], dim=1,
    )
    fallback_acceleration = torch.diff(
        full_velocity, dim=1, prepend=full_velocity[:, :1],
    ) / dt
    if fallback_acceleration.shape[1] > 1:
        fallback_acceleration[:, 0] = fallback_acceleration[:, 1]
    acceleration_past = getattr(
        eb, "ego_acc_past", fallback_acceleration[:, :P],
    )
    acceleration_future = getattr(
        eb, "ego_acc_future", fallback_acceleration[:, P:],
    )

    full_heading = torch.cat(
        [heading_past, eb.ego_heading_future], dim=1,
    )
    fallback_yaw_delta = torch.diff(
        full_heading, dim=1, prepend=full_heading[:, :1],
    )
    fallback_yaw_delta = (
        fallback_yaw_delta + math.pi
    ) % (2.0 * math.pi) - math.pi
    fallback_yaw_rate = fallback_yaw_delta / dt
    fallback_yaw_accel = torch.diff(
        fallback_yaw_rate,
        dim=1,
        prepend=fallback_yaw_rate[:, :1],
    ) / dt
    yaw_rate_past = getattr(
        eb, "ego_yaw_rate_past", fallback_yaw_rate[:, :P],
    )
    yaw_rate_future = getattr(
        eb, "ego_yaw_rate_future", fallback_yaw_rate[:, P:],
    )
    yaw_accel_past = getattr(
        eb, "ego_yaw_accel_past", fallback_yaw_accel[:, :P],
    )
    yaw_accel_future = getattr(
        eb, "ego_yaw_accel_future", fallback_yaw_accel[:, P:],
    )


    history_steps = int(round(HC_HISTORY_HORIZON / HC_HISTORY_SAMPLE_INTERVAL))
    current_index = P - 1
    history_start = max(0, current_index - history_steps)
    history_stop = max(history_start, current_index - 1)

    def _pad(
        past: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                past[:, history_start:history_stop],
                past[:, -1:],
                future,
            ],
            dim=1,
        )

    rear_acceleration = _pad(
        acceleration_past, acceleration_future,
    )
    stored_yaw_rate = _pad(yaw_rate_past, yaw_rate_future)
    stored_yaw_accel = _pad(yaw_accel_past, yaw_accel_future)
    heading = _pad(heading_past, eb.ego_heading_future)
    valid = _pad(eb.ego_valid_past, eb.ego_valid_future)

    T = heading.shape[-1]
    if T < 4:


        return torch.ones(B, device=device)


    rear_acceleration = rear_acceleration.clone()
    stored_yaw_rate = stored_yaw_rate.clone()
    stored_yaw_accel = stored_yaw_accel.clone()
    heading = heading.clone()
    for time_idx in range(1, T):
        missing = ~valid[:, time_idx]
        rear_acceleration[:, time_idx][missing] = (
            rear_acceleration[:, time_idx - 1][missing]
        )
        stored_yaw_rate[:, time_idx][missing] = (
            stored_yaw_rate[:, time_idx - 1][missing]
        )
        stored_yaw_accel[:, time_idx][missing] = (
            stored_yaw_accel[:, time_idx - 1][missing]
        )
        heading[:, time_idx][missing] = heading[:, time_idx - 1][missing]

    calc_dtype = torch.float64
    rear_acceleration = rear_acceleration.to(calc_dtype)
    stored_yaw_rate = stored_yaw_rate.to(calc_dtype)
    stored_yaw_accel = stored_yaw_accel.to(calc_dtype)
    heading = heading.to(calc_dtype)

    direction = torch.stack([heading.cos(), heading.sin()], dim=-1)
    rear_lon = (rear_acceleration * direction).sum(dim=-1)


    center_lon = rear_lon + HC_REAR_AXLE_TO_CENTER_M * (
        stored_yaw_rate.square() + stored_yaw_accel
    )


    center_lat = torch.zeros_like(rear_lon)

    def _round8(value: torch.Tensor) -> torch.Tensor:
        return torch.round(value * 1e8) / 1e8

    lon_accel = _round8(_hc_savgol_filter_1d(
        center_lon, window_length=8, poly_order=2,
    ))
    lat_accel = _round8(_hc_savgol_filter_1d(
        center_lat, window_length=8, poly_order=2,
    ))
    rear_accel_magnitude = _round8(_hc_savgol_filter_1d(
        rear_lon.abs(),
        window_length=8,
        poly_order=2,
    ))
    jerk_magnitude = _round8(_hc_savgol_filter_1d(
        rear_accel_magnitude,
        window_length=15,
        poly_order=2,
        deriv=1,
        delta=signal_dt,
    ))
    lon_jerk = _round8(_hc_savgol_filter_1d(
        lon_accel,
        window_length=15,
        poly_order=2,
        deriv=1,
        delta=signal_dt,
    ))

    unwrapped_heading = _phase_unwrap(heading)


    yaw_rate = _round8(_hc_savgol_filter_1d(
        unwrapped_heading,
        window_length=5,
        poly_order=2,
        deriv=1,
        delta=signal_dt,
    ))
    yaw_accel = _round8(_hc_savgol_filter_1d(
        unwrapped_heading,
        window_length=5,
        poly_order=3,
        deriv=2,
        delta=signal_dt,
    ))

    def _within(
        metric: torch.Tensor,
        lower: float,
        upper: float,
    ) -> torch.Tensor:
        return (((metric > lower) & (metric < upper)) | ~valid).all(dim=-1)

    comfortable = (
        _within(lon_accel, MIN_LON_ACCEL, MAX_LON_ACCEL)
        & _within(lat_accel, -MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL)
        & _within(jerk_magnitude, -MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK)
        & _within(lon_jerk, -MAX_ABS_LON_JERK, MAX_ABS_LON_JERK)
        & _within(yaw_accel, -MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL)
        & _within(yaw_rate, -MAX_ABS_YAW_RATE, MAX_ABS_YAW_RATE)
    )
    return comfortable.to(dtype=eb.ego_pos_future.dtype)


def _projected_progress(
    pos: torch.Tensor,
    valid: torch.Tensor,
    poly_pos: torch.Tensor,
    poly_valid: torch.Tensor,
) -> torch.Tensor:
    """폴리라인 위 start→end 투영 진행거리 [B].

    endpoint 는 마지막 valid 프레임 (invalid trailing frames 의 (0,0) 방지).
    """
    B, S = valid.shape
    bi = torch.arange(B, device=pos.device)
    last_valid = (S - 1 - valid.flip(-1).float().argmax(dim=-1)).clamp(min=0)
    start_end = torch.stack([pos[:, 0], pos[bi, last_valid]], dim=1)
    arc, _ = _project_onto_polyline(start_end, poly_pos, poly_valid)
    return (arc[:, 1] - arc[:, 0]).clamp(min=0.0)


def _compute_ego_progress(
    eb: _NavsimEgoBatch,
    paths: feature.Path,
) -> torch.Tensor:
    """EP raw: initial state부터 horizon 끝까지 route 진행거리 [m].

    정규화 분모는 proposal 집합의 masked progress 최대값 — `_aggregate_epdms_with_ec`
    가 담당한다. NAVSIM proposal state 의 index 0은 현재(initial) 상태이므로 마지막
    past frame을 반드시 포함한다. route 부재면 0.
    """
    if paths is None or paths.valid.numel() == 0:
        return torch.zeros(eb.B, device=eb.device)

    route_pos, _, route_valid, _, has_route = pick_ego_route(paths)
    pos = torch.cat([eb.ego_pos_past[:, -1:], eb.ego_pos_future], dim=1)
    valid = torch.cat([eb.ego_valid_past[:, -1:], eb.ego_valid_future], dim=1)
    raw = _projected_progress(
        pos, valid, route_pos, route_valid)
    return raw * has_route.to(raw.dtype)


def _normalize_progress(
    raw: torch.Tensor, norm_constant: torch.Tensor,
) -> torch.Tensor:
    """raw progress [m] → [0, 1]. norm_constant 가 임계 이하면 (정지 씬) 1.0."""
    return torch.where(
        norm_constant > PROGRESS_DISTANCE_THRESHOLD,
        (raw / norm_constant.clamp(min=EPS)).clamp(0.0, 1.0),
        torch.ones_like(raw),
    )


def _masked_progress(metrics: dict[str, torch.Tensor]) -> torch.Tensor:
    """proposal 의 EP 분모 기여분 = raw progress × multiplicative 점수.

    human-penalty 필터 이전 값을 써야 한다 — 레퍼런스의 정규화 상수는 필터를 받지 않는다.
    """
    return metrics["ego_progress_raw"] * (
        metrics["no_at_fault_collisions"]
        * metrics["drivable_area_compliance"]
        * metrics["driving_direction_compliance"]
        * metrics["traffic_light_compliance"]
    )


def _compute_extended_comfort() -> float:
    """EC: placeholder for single-rollout path (two-frame EC computed in _compute_two_frame_ec)."""
    return 1.0


@dataclass
class _NavsimEgoBatch:
    """Extracted ego & agent data for NAVSIM scoring."""
    device: torch.device
    B: int
    A: int
    num_past_frames: int
    num_future_frames: int
    dt: float
    stride: int


    ego_pos_past: torch.Tensor
    ego_dir_past: torch.Tensor
    ego_vel_past: torch.Tensor
    ego_acc_past: torch.Tensor
    ego_yaw_rate_past: torch.Tensor
    ego_yaw_accel_past: torch.Tensor
    ego_speed_past: torch.Tensor
    ego_valid_past: torch.Tensor

    ego_pos_future: torch.Tensor
    ego_dir_future: torch.Tensor
    ego_vel_future: torch.Tensor
    ego_acc_future: torch.Tensor
    ego_yaw_rate_future: torch.Tensor
    ego_yaw_accel_future: torch.Tensor
    ego_speed_future: torch.Tensor
    ego_size_future: torch.Tensor
    ego_valid_future: torch.Tensor
    ego_heading_future: torch.Tensor


    all_pos_future: torch.Tensor
    all_dir_future: torch.Tensor
    all_size_future: torch.Tensor
    all_speed_future: torch.Tensor
    all_valid_future: torch.Tensor
    all_valid_past: torch.Tensor
    all_type_future: torch.Tensor
    all_heading_future: torch.Tensor
    all_pos_initial: torch.Tensor
    all_size_initial: torch.Tensor
    all_speed_initial: torch.Tensor
    all_heading_initial: torch.Tensor
    all_valid_initial: torch.Tensor
    ego_mask: torch.Tensor

    has_ego: torch.Tensor


def _extract_navsim_ego_batch(
    output: feature.Output,
    configs: NavsimConfigs,
) -> _NavsimEgoBatch | None:
    """Extract ego and agent data from Output."""
    pos = output.pos_xy
    dir_xy = output.dir_xy
    valid = output.valid
    size = output.size_xyz[..., :2]
    is_ego = output.is_ego
    agent_type = output.type

    B, A, S = pos.shape[:3]
    device = pos.device
    if A == 0 or S == 0:
        return None

    dt = configs.dataloader.frame_unit
    stride = 1
    num_past_frames = configs.time.num_past_frames
    num_future_frames = S - num_past_frames
    if num_future_frames <= 0:
        return None


    dpos = torch.diff(pos, dim=2, prepend=pos[:, :, :1]) / dt
    speed = dpos.norm(dim=-1)


    ego_any = is_ego.any(dim=-1)
    has_ego = ego_any.any(dim=1)
    ego_idx = ego_any.float().argmax(dim=1)
    bi = torch.arange(B, device=device)


    ego_mask = torch.zeros(B, A, dtype=torch.bool, device=device)
    ego_mask[bi, ego_idx] = True


    ego_vel_full = dpos[bi, ego_idx]
    extra = output.extra or {}
    extra_vel = extra.get("ego_vel")
    if (
        isinstance(extra_vel, torch.Tensor)
        and extra_vel.shape == ego_vel_full.shape
    ):
        ego_vel_full = extra_vel.to(device=device, dtype=pos.dtype)

    ego_acc_full = torch.diff(
        ego_vel_full, dim=1, prepend=ego_vel_full[:, :1],
    ) / dt
    if S > 1:
        ego_acc_full[:, 0] = ego_acc_full[:, 1]
    extra_acc = extra.get("ego_acc")
    if (
        isinstance(extra_acc, torch.Tensor)
        and extra_acc.shape == ego_acc_full.shape
    ):
        ego_acc_full = extra_acc.to(device=device, dtype=pos.dtype)

    ego_heading_full = torch.atan2(
        dir_xy[bi, ego_idx, :, 1], dir_xy[bi, ego_idx, :, 0],
    )
    heading_delta = torch.diff(
        ego_heading_full, dim=1, prepend=ego_heading_full[:, :1],
    )
    heading_delta = (
        heading_delta + math.pi
    ) % (2.0 * math.pi) - math.pi
    ego_yaw_rate_full = heading_delta / dt
    extra_yaw_rate = extra.get("ego_yaw_rate")
    if (
        isinstance(extra_yaw_rate, torch.Tensor)
        and extra_yaw_rate.shape == ego_yaw_rate_full.shape
    ):
        ego_yaw_rate_full = extra_yaw_rate.to(
            device=device, dtype=pos.dtype,
        )
    ego_yaw_accel_full = torch.diff(
        ego_yaw_rate_full, dim=1, prepend=ego_yaw_rate_full[:, :1],
    ) / dt
    extra_yaw_accel = extra.get("ego_yaw_accel")
    if (
        isinstance(extra_yaw_accel, torch.Tensor)
        and extra_yaw_accel.shape == ego_yaw_accel_full.shape
    ):
        ego_yaw_accel_full = extra_yaw_accel.to(
            device=device, dtype=pos.dtype,
        )


    ego_pos_past = pos[bi, ego_idx, :num_past_frames]
    ego_dir_past = dir_xy[bi, ego_idx, :num_past_frames]
    ego_vel_past = ego_vel_full[:, :num_past_frames]
    ego_acc_past = ego_acc_full[:, :num_past_frames]
    ego_yaw_rate_past = ego_yaw_rate_full[:, :num_past_frames]
    ego_yaw_accel_past = ego_yaw_accel_full[:, :num_past_frames]
    ego_speed_past = speed[bi, ego_idx, :num_past_frames]
    ego_valid_past = valid[bi, ego_idx, :num_past_frames]


    ego_pos_future = pos[bi, ego_idx, num_past_frames::stride]
    ego_dir_future = dir_xy[bi, ego_idx, num_past_frames::stride]
    ego_heading_future = torch.atan2(ego_dir_future[..., 1], ego_dir_future[..., 0])
    ego_vel_future = ego_vel_full[:, num_past_frames::stride]
    ego_acc_future = ego_acc_full[:, num_past_frames::stride]
    ego_yaw_rate_future = ego_yaw_rate_full[:, num_past_frames::stride]
    ego_yaw_accel_future = ego_yaw_accel_full[:, num_past_frames::stride]
    ego_speed_future = speed[bi, ego_idx, num_past_frames::stride]
    ego_size_future = size[bi, ego_idx, num_past_frames::stride]
    ego_valid_future = valid[bi, ego_idx, num_past_frames::stride]


    heading = torch.atan2(dir_xy[..., 1], dir_xy[..., 0])


    all_pos_future = pos[:, :, num_past_frames::stride]
    all_dir_future = dir_xy[:, :, num_past_frames::stride]
    all_size_future = size[:, :, num_past_frames::stride]
    all_speed_future = speed[:, :, num_past_frames::stride]
    all_valid_future = valid[:, :, num_past_frames::stride]
    all_valid_past = valid[:, :, :num_past_frames]
    all_heading_future = heading[:, :, num_past_frames::stride]
    all_type_future = agent_type[:, :, num_past_frames::stride]
    all_pos_initial = pos[:, :, num_past_frames - 1]
    all_size_initial = size[:, :, num_past_frames - 1]
    all_speed_initial = speed[:, :, num_past_frames - 1]
    all_heading_initial = heading[:, :, num_past_frames - 1]
    all_valid_initial = valid[:, :, num_past_frames - 1]

    return _NavsimEgoBatch(
        device=device, B=B, A=A, num_past_frames=num_past_frames, num_future_frames=num_future_frames, dt=dt, stride=stride,
        ego_pos_past=ego_pos_past, ego_dir_past=ego_dir_past,
        ego_vel_past=ego_vel_past, ego_acc_past=ego_acc_past,
        ego_yaw_rate_past=ego_yaw_rate_past, ego_yaw_accel_past=ego_yaw_accel_past,
        ego_speed_past=ego_speed_past, ego_valid_past=ego_valid_past,
        ego_pos_future=ego_pos_future, ego_dir_future=ego_dir_future,
        ego_vel_future=ego_vel_future, ego_acc_future=ego_acc_future,
        ego_yaw_rate_future=ego_yaw_rate_future, ego_yaw_accel_future=ego_yaw_accel_future,
        ego_speed_future=ego_speed_future, ego_size_future=ego_size_future,
        ego_valid_future=ego_valid_future, ego_heading_future=ego_heading_future,
        all_pos_future=all_pos_future, all_dir_future=all_dir_future,
        all_size_future=all_size_future, all_speed_future=all_speed_future,
        all_valid_future=all_valid_future,
        all_valid_past=all_valid_past,
        all_type_future=all_type_future, all_heading_future=all_heading_future,
        all_pos_initial=all_pos_initial, all_size_initial=all_size_initial,
        all_speed_initial=all_speed_initial,
        all_heading_initial=all_heading_initial,
        all_valid_initial=all_valid_initial,
        ego_mask=ego_mask, has_ego=has_ego,
    )


def _extract_gt_ego_batch(
    scene: feature.Scene,
    configs: NavsimConfigs,
) -> _NavsimEgoBatch | None:
    """Extract GT ego trajectory from scene.agents → _NavsimEgoBatch.

    Agent fields differ from Output:
      type [B,A], is_ego [B,A], size_xyz [B,A,3] → need expansion.
      pos_xy/dir_xy/valid [B,A,F,...] → slice to past+future.
    """


    agents = scene.agents
    step_indices = configs.time.idx()
    pos = agents.pos_xy[:, :, step_indices]
    dir_xy = agents.dir_xy[:, :, step_indices]
    source_vel = agents.vel_xy[:, :, step_indices]
    source_acc = agents.acc_xy[:, :, step_indices]
    valid = agents.valid[:, :, step_indices]

    B, A, F_total = valid.shape[:3]
    device = pos.device
    if A == 0 or F_total == 0:
        return None

    dt = configs.dataloader.frame_unit
    stride = 1
    num_past_frames = configs.time.num_past_frames
    num_future_frames = F_total - num_past_frames
    if num_future_frames <= 0:
        return None


    S = F_total
    is_ego_3d = agents.is_ego.unsqueeze(-1).expand(-1, -1, S)
    agent_type = agents.type.unsqueeze(-1).expand(-1, -1, S)
    size_2d = agents.size_xyz[..., :2]
    size_3d = size_2d.unsqueeze(2).expand(-1, -1, S, -1)


    dpos = torch.diff(pos, dim=2, prepend=pos[:, :, :1]) / dt
    speed = dpos.norm(dim=-1)
    track_speed = source_vel.norm(dim=-1)


    ego_any = is_ego_3d.any(dim=-1)
    has_ego = ego_any.any(dim=1)
    ego_idx = ego_any.float().argmax(dim=1)
    bi = torch.arange(B, device=device)

    ego_mask = torch.zeros(B, A, dtype=torch.bool, device=device)
    ego_mask[bi, ego_idx] = True

    ego_vel_full = source_vel[bi, ego_idx]
    ego_acc_full = source_acc[bi, ego_idx]


    ego_yaw_rate_full = torch.zeros(
        B, F_total, dtype=pos.dtype, device=device,
    )
    ego_yaw_accel_full = torch.zeros_like(ego_yaw_rate_full)


    ego_pos_past = pos[bi, ego_idx, :num_past_frames]
    ego_dir_past = dir_xy[bi, ego_idx, :num_past_frames]
    ego_vel_past = ego_vel_full[:, :num_past_frames]
    ego_acc_past = ego_acc_full[:, :num_past_frames]
    ego_yaw_rate_past = ego_yaw_rate_full[:, :num_past_frames]
    ego_yaw_accel_past = ego_yaw_accel_full[:, :num_past_frames]
    ego_speed_past = speed[bi, ego_idx, :num_past_frames]
    ego_valid_past = valid[bi, ego_idx, :num_past_frames]


    ego_pos_future = pos[bi, ego_idx, num_past_frames::stride]
    ego_dir_future = dir_xy[bi, ego_idx, num_past_frames::stride]
    ego_vel_future = ego_vel_full[:, num_past_frames::stride]
    ego_acc_future = ego_acc_full[:, num_past_frames::stride]
    ego_yaw_rate_future = ego_yaw_rate_full[:, num_past_frames::stride]
    ego_yaw_accel_future = ego_yaw_accel_full[:, num_past_frames::stride]
    ego_speed_future = speed[bi, ego_idx, num_past_frames::stride]
    ego_size_future = size_3d[bi, ego_idx, num_past_frames::stride]
    ego_valid_future = valid[bi, ego_idx, num_past_frames::stride]

    heading = torch.atan2(dir_xy[..., 1], dir_xy[..., 0])
    ego_heading_future = heading[bi, ego_idx, num_past_frames::stride]


    all_pos_future = pos[:, :, num_past_frames::stride]
    all_dir_future = dir_xy[:, :, num_past_frames::stride]
    all_size_future = size_3d[:, :, num_past_frames::stride]
    all_speed_future = track_speed[:, :, num_past_frames::stride]
    all_valid_future = valid[:, :, num_past_frames::stride]
    all_valid_past = valid[:, :, :num_past_frames]
    all_heading_future = heading[:, :, num_past_frames::stride]
    all_type_future = agent_type[:, :, num_past_frames::stride]
    all_pos_initial = pos[:, :, num_past_frames - 1]
    all_size_initial = size_3d[:, :, num_past_frames - 1]
    all_speed_initial = track_speed[:, :, num_past_frames - 1]
    all_heading_initial = heading[:, :, num_past_frames - 1]
    all_valid_initial = valid[:, :, num_past_frames - 1]

    return _NavsimEgoBatch(
        device=device, B=B, A=A, num_past_frames=num_past_frames, num_future_frames=num_future_frames, dt=dt, stride=stride,
        ego_pos_past=ego_pos_past, ego_dir_past=ego_dir_past,
        ego_vel_past=ego_vel_past, ego_acc_past=ego_acc_past,
        ego_yaw_rate_past=ego_yaw_rate_past, ego_yaw_accel_past=ego_yaw_accel_past,
        ego_speed_past=ego_speed_past, ego_valid_past=ego_valid_past,
        ego_pos_future=ego_pos_future, ego_dir_future=ego_dir_future,
        ego_vel_future=ego_vel_future, ego_acc_future=ego_acc_future,
        ego_yaw_rate_future=ego_yaw_rate_future, ego_yaw_accel_future=ego_yaw_accel_future,
        ego_speed_future=ego_speed_future, ego_size_future=ego_size_future,
        ego_valid_future=ego_valid_future, ego_heading_future=ego_heading_future,
        all_pos_future=all_pos_future, all_dir_future=all_dir_future,
        all_size_future=all_size_future, all_speed_future=all_speed_future,
        all_valid_future=all_valid_future,
        all_valid_past=all_valid_past,
        all_type_future=all_type_future, all_heading_future=all_heading_future,
        all_pos_initial=all_pos_initial, all_size_initial=all_size_initial,
        all_speed_initial=all_speed_initial,
        all_heading_initial=all_heading_initial,
        all_valid_initial=all_valid_initial,
        ego_mask=ego_mask, has_ego=has_ego,
    )


def _build_ego_corners(
    pos: torch.Tensor,
    dir_xy: torch.Tensor,
    size_xyz: torch.Tensor,
) -> torch.Tensor:
    """Build 4 corner coordinates from center, direction, and size.

    Returns: [..., 4, 2] corners (FL, FR, RL, RR).
    """
    d_norm = dir_xy.norm(dim=-1, keepdim=True).clamp(min=EPS)
    fwd = dir_xy / d_norm
    lat = torch.stack([-fwd[..., 1], fwd[..., 0]], dim=-1)
    half_L = size_xyz[..., 0:1] * 0.5
    half_W = size_xyz[..., 1:2] * 0.5
    return torch.stack([
        pos + fwd * half_L + lat * half_W,
        pos + fwd * half_L - lat * half_W,
        pos - fwd * half_L - lat * half_W,
        pos - fwd * half_L + lat * half_W,
    ], dim=-2)


def _project_onto_polyline(
    points: torch.Tensor,
    polyline: torch.Tensor,
    poly_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project points onto polyline, returning arc-length and lateral distance.

    Returns:
        arc_length: [B, N] — cumulative distance along polyline to nearest point
        lateral_dist: [B, N] — perpendicular distance to nearest segment
    """
    B, N, _ = points.shape
    M = polyline.shape[1]
    device = points.device

    if M < 2:
        return (
            torch.zeros(B, N, device=device),
            torch.full((B, N), 1e6, device=device),
        )


    seg_start = polyline[:, :-1]
    seg_end = polyline[:, 1:]
    seg_vec = seg_end - seg_start
    seg_len = seg_vec.norm(dim=-1).clamp(min=EPS)
    seg_dir = seg_vec / seg_len.unsqueeze(-1)


    seg_valid = poly_valid[:, :-1] & poly_valid[:, 1:]


    cum_len = torch.zeros(B, M, device=device)
    cum_len[:, 1:] = seg_len.cumsum(dim=1)


    diff = points.unsqueeze(2) - seg_start.unsqueeze(1)
    t = (diff * seg_dir.unsqueeze(1)).sum(dim=-1)
    t_clamped = t.clamp(min=0.0)
    t_clamped = torch.minimum(t_clamped, seg_len.unsqueeze(1))


    closest = seg_start.unsqueeze(1) + seg_dir.unsqueeze(1) * t_clamped.unsqueeze(-1)
    dist = (points.unsqueeze(2) - closest).norm(dim=-1)


    dist = dist.masked_fill(~seg_valid.unsqueeze(1), 1e6)


    min_dist, min_seg = dist.min(dim=2)


    bi = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    arc_at_seg = cum_len[:, :-1]
    arc_start = arc_at_seg[bi, min_seg]
    t_at_min = t_clamped[bi, torch.arange(N, device=device).unsqueeze(0).expand(B, N), min_seg]

    arc_length = arc_start + t_at_min
    return arc_length, min_dist


def _heading_to_rot(heading: torch.Tensor) -> torch.Tensor:
    """Convert heading [...] to rotation matrix [..., 2, 2]."""
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    row0 = torch.stack([cos_h, -sin_h], dim=-1)
    row1 = torch.stack([sin_h, cos_h], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _sat_signed_distance(
    pos_a: torch.Tensor, rot_a: torch.Tensor, size_a: torch.Tensor,
    pos_b: torch.Tensor, rot_b: torch.Tensor, size_b: torch.Tensor,
) -> torch.Tensor:
    """SAT signed distance. Positive = separated, negative = overlap."""
    BA_pos = torch.einsum("...ij,...j->...i", rot_a.transpose(-2, -1), pos_b - pos_a)
    BA_rot = torch.einsum("...ij,...jk->...ik", rot_a.transpose(-2, -1), rot_b)
    half_a = size_a * 0.5
    half_b = size_b * 0.5
    BA0 = BA_rot[..., 0, 0].abs() * half_b[..., 0] + BA_rot[..., 0, 1].abs() * half_b[..., 1]
    BA1 = BA_rot[..., 1, 0].abs() * half_b[..., 0] + BA_rot[..., 1, 1].abs() * half_b[..., 1]
    BA0_margin = BA_pos[..., 0].abs() - (BA0 + half_a[..., 0])
    BA1_margin = BA_pos[..., 1].abs() - (BA1 + half_a[..., 1])
    AB_pos = torch.einsum("...ij,...j->...i", BA_rot.transpose(-2, -1), BA_pos)
    AB0 = BA_rot[..., 0, 0].abs() * half_a[..., 0] + BA_rot[..., 1, 0].abs() * half_a[..., 1]
    AB1 = BA_rot[..., 0, 1].abs() * half_a[..., 0] + BA_rot[..., 1, 1].abs() * half_a[..., 1]
    AB0_margin = AB_pos[..., 0].abs() - (AB0 + half_b[..., 0])
    AB1_margin = AB_pos[..., 1].abs() - (AB1 + half_b[..., 1])
    return torch.maximum(
        torch.maximum(BA0_margin, BA1_margin),
        torch.maximum(AB0_margin, AB1_margin),
    )


def _segment_intersects_obbs(
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
    box_pos: torch.Tensor,
    box_rot: torch.Tensor,
    box_size: torch.Tensor,
) -> torch.Tensor:
    """Closed line-segment/OBB intersection for front-bumper classification."""
    inverse_rot = box_rot.transpose(-1, -2)
    start_local = torch.einsum(
        "...ij,...j->...i",
        inverse_rot,
        segment_start.unsqueeze(2) - box_pos,
    )
    end_local = torch.einsum(
        "...ij,...j->...i",
        inverse_rot,
        segment_end.unsqueeze(2) - box_pos,
    )
    delta = end_local - start_local
    half = box_size * 0.5

    parallel = delta.abs() <= 1e-12
    safe_delta = torch.where(parallel, torch.ones_like(delta), delta)
    t1 = (-half - start_local) / safe_delta
    t2 = (half - start_local) / safe_delta
    near = torch.minimum(t1, t2)
    far = torch.maximum(t1, t2)

    outside_parallel = parallel & (start_local.abs() > half)
    near = torch.where(
        parallel,
        torch.full_like(near, float("-inf")),
        near,
    )
    far = torch.where(
        parallel,
        torch.where(
            outside_parallel,
            torch.full_like(far, float("-inf")),
            torch.full_like(far, float("inf")),
        ),
        far,
    )
    t_min = near.amax(dim=-1)
    t_max = far.amin(dim=-1)
    return (t_min <= t_max) & (t_max >= 0.0) & (t_min <= 1.0)


def _phase_unwrap(headings: torch.Tensor) -> torch.Tensor:
    """Phase-unwrap heading angles so successive differences <= pi."""
    two_pi = 2.0 * math.pi
    adjustments = torch.zeros_like(headings)
    adjustments[..., 1:] = torch.cumsum(
        torch.round(torch.diff(headings, dim=-1) / two_pi), dim=-1
    )
    return headings - two_pi * adjustments


def _hc_savgol_filter_1d(
    x: torch.Tensor,
    *,
    window_length: int,
    poly_order: int,
    deriv: int = 0,
    delta: float = 1.0,
) -> torch.Tensor:
    """Torch equivalent of SciPy ``savgol_filter(..., mode="interp")``.

    NAVSIM HC relies on two details that the general helper above intentionally
    does not provide: an even eight-sample smoothing window and polynomial
    interpolation at both ends.  Keep this HC-specific implementation separate
    so EC/KF behavior is unchanged.
    """
    T = x.shape[-1]
    window_length = min(window_length, T)
    if poly_order >= window_length:
        raise ValueError(
            f"poly_order={poly_order} must be less than "
            f"window_length={window_length}"
        )
    if deriv > poly_order:
        return torch.zeros_like(x)

    orig_shape = x.shape
    flat = x.reshape(-1, T)
    dtype_ = flat.dtype
    device = flat.device
    W = window_length
    half = W // 2


    position = float(half) if W % 2 else float(half) - 0.5
    sample_x = (
        torch.arange(W, dtype=dtype_, device=device) - position
    )
    powers = torch.arange(poly_order + 1, device=device)
    design = sample_x[None] ** powers[:, None]
    target = torch.zeros(poly_order + 1, dtype=dtype_, device=device)
    target[deriv] = math.factorial(deriv) / (delta ** deriv)
    dot_coefficients = torch.linalg.pinv(design) @ target

    windows = flat.unfold(-1, W, 1)
    interior = (windows * dot_coefficients).sum(dim=-1)
    output = torch.zeros_like(flat)
    center_offset = (W - 1) // 2
    output[
        :, center_offset:center_offset + interior.shape[-1]
    ] = interior


    edge_x = torch.arange(W, dtype=dtype_, device=device)
    edge_design = edge_x[:, None] ** powers[None]
    edge_pinv = torch.linalg.pinv(edge_design)

    def _edge_values(
        samples: torch.Tensor,
        evaluation_x: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = edge_pinv @ samples.T
        derivative_design = torch.zeros(
            evaluation_x.numel(),
            poly_order + 1,
            dtype=dtype_,
            device=device,
        )
        for degree in range(deriv, poly_order + 1):
            scale = (
                math.factorial(degree)
                / math.factorial(degree - deriv)
                / (delta ** deriv)
            )
            derivative_design[:, degree] = (
                scale * evaluation_x ** (degree - deriv)
            )
        return (derivative_design @ coefficients).T

    if half:
        low_x = torch.arange(half, dtype=dtype_, device=device)
        high_x = torch.arange(
            W - half, W, dtype=dtype_, device=device,
        )
        output[:, :half] = _edge_values(flat[:, :W], low_x)
        output[:, T - half:] = _edge_values(flat[:, -W:], high_x)

    return output.reshape(orig_shape)


EP_WEIGHT = 5.0


TTC_WEIGHT = 5.0


LK_WEIGHT = 2.0


HC_WEIGHT = 2.0


EC_WEIGHT = 2.0


DDC_HORIZON = 1.0


LANE_KEEPING_DEVIATION_LIMIT = 0.5


LANE_KEEPING_HORIZON = 2.0


PROGRESS_DISTANCE_THRESHOLD = 5.0


STOPPED_SPEED_THRESHOLD = 5e-3


NC_STOPPED_SPEED_THRESHOLD = 5e-2


NC_REAR_AXLE_TO_CENTER_RATIO = 1.461 / 5.176


FUTURE_COLLISION_HORIZON = 1.0


SLC_MAX_OVERSPEED_THRESHOLD = 2.23


MAX_ABS_MAG_JERK = 8.37


MAX_ABS_LAT_ACCEL = 4.89


MAX_LON_ACCEL = 2.40


MIN_LON_ACCEL = -4.05


MAX_ABS_YAW_ACCEL = 1.93


MAX_ABS_LON_JERK = 4.13


MAX_ABS_YAW_RATE = 0.95


HC_HISTORY_HORIZON = 1.5


HC_HISTORY_SAMPLE_INTERVAL = 0.1


HC_REAR_AXLE_TO_CENTER_M = 1.461


EPS = 1e-6


def _ec_pair_stride(configs: NavsimConfigs) -> int:
    """#replans whose issue-time gap == STAGE_GAP_SEC (for rollout-EC pairing).

    Fail-fast if the replan rate can't express the 0.5s gap as an integer count of
    replans — then the pair-axis stride and the within-plan overlap (round(0.5/dt))
    would denote different real-world windows and EC would be meaningless.
    validation_replan_hz 10→5, 2→1 (exact); 3→1.5 raises.
    """
    replan_hz = configs.trainer.validation_replan_hz
    ratio = STAGE_GAP_SEC * replan_hz
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > 1e-6:
        raise ValueError(
            f"validation_replan_hz={replan_hz} cannot express a {STAGE_GAP_SEC}s EC gap "
            f"as an integer number of replans (got {ratio}); rollout EC is undefined."
        )
    return stride


def _simulate_rollout_plans(
    output: feature.Output, configs: NavsimConfigs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """per-replan plan 을 plant(컨트롤러+dynamics)로 실현 → (pos, dir) [B, N, 1+F, 2].

    각 plan 은 발행 시점의 실현 ego 상태에서 출발해 자기 지평 끝까지 추종된다 — rollout
    과 달리 중간에 다음 plan 으로 갈아타지 않는다. B·N 개 plan 을 agent 축으로 접어 한
    번의 프레임 루프로 굴린다. 반환 index 0 은 발행 시점 상태(속도 seed).
    """
    extra = output.extra
    pos = extra["plan_ego_pos"]
    plan_valid = extra["plan_ego_valid"]
    B, N, F_plan = pos.shape[0], pos.shape[1], pos.shape[2] - 1
    device = pos.device
    dt = configs.dataloader.frame_unit
    control_hz = configs.trainer.control_hz
    ctrl_per_data = max(1, round(control_hz * dt))
    dt_ctrl = dt / ctrl_per_data
    data_per_replan = max(1, round((1.0 / configs.trainer.validation_replan_hz) / dt))

    bi = torch.arange(B, device=device)
    ego = output.is_ego.any(dim=-1).float().argmax(dim=1)
    P = configs.time.num_past_frames


    hist_pos = output.pos_xy[bi, ego, P - 1:]
    hist_dir = output.dir_xy[bi, ego, P - 1:]
    hist_vel = extra["ego_vel"][:, P - 1:]
    issue = (torch.arange(N, device=device) * data_per_replan).clamp(
        max=hist_pos.shape[1] - 1)
    i_pos = hist_pos[:, issue]
    i_dir = hist_dir[:, issue]
    i_speed = hist_vel[:, issue].norm(dim=-1)

    rows = B * N
    slots = ring_slots(control_hz, configs.dynamics.vehicle['steer_delay_max_s'])
    state = init_state(
        i_pos.reshape(rows, 2),
        torch.atan2(i_dir[..., 1], i_dir[..., 0]).reshape(rows),
        i_speed.reshape(rows), slots,
    )
    ctrl_state = init_ctrl_state(state.core)
    params = params_from_agents(
        output.size_xyz[bi, ego, 0][:, None].expand(B, N, 3).reshape(rows, 3),
        output.type[bi, ego, 0][:, None].expand(B, N).reshape(rows).long(),
        control_hz, configs.dynamics,
    )

    ref = pos[:, :, 1:].reshape(rows, F_plan, 2)
    ref_valid = plan_valid.reshape(rows, F_plan)
    profile = plan_profile(ref, ref_valid, dt)
    freeze = torch.zeros(rows, dtype=torch.bool, device=device)

    delta0 = pure_pursuit_control(
        ref, ref_valid, state.core, ctrl_state, profile=profile,
        params=params, dt_ctrl=dt_ctrl, cp=configs.controller)[0]
    state = warm_start(state, delta0)

    sim_pos, sim_dir = [], []
    for _ in range(F_plan):
        state, ctrl_state = track_plan_frame(
            ref, ref_valid, profile, state, ctrl_state, freeze,
            ctrl_per_data, params, dt_ctrl, configs.controller)
        step_pos, step_dir, _ = readout(state)
        sim_pos.append(step_pos.reshape(B, N, 2))
        sim_dir.append(step_dir.reshape(B, N, 2))

    return (torch.cat([i_pos[:, :, None], torch.stack(sim_pos, dim=2)], dim=2),
            torch.cat([i_dir[:, :, None], torch.stack(sim_dir, dim=2)], dim=2))


def _compute_rollout_ec(
    output: feature.Output, configs: NavsimConfigs,
) -> torch.Tensor | None:
    """per-sample [B] rollout EC — 연속 replan plan 쌍(STAGE_GAP_SEC 간격)의 two-frame EC 평균.

    plan 이 부착되지 않은 Output(open-loop)이면 None — 호출자가 규약에 맞게 처리한다.
    """
    extra = output.extra or {}
    if "plan_ego_pos" not in extra:
        return None

    valid = extra["plan_ego_valid"]
    issue_done = extra["plan_issue_done"]
    device = valid.device
    B, N = valid.shape[:2]
    dt = configs.dataloader.frame_unit
    stride = _ec_pair_stride(configs)

    if N <= stride:
        return torch.ones(B, device=device)


    pos, dir_ = _simulate_rollout_plans(output, configs)


    vel = (torch.diff(pos, dim=2, prepend=pos[:, :, :1]) / dt)[:, :, 1:]
    heading = torch.atan2(dir_[..., 1], dir_[..., 0])[:, :, 1:]


    P = N - stride
    a, b = slice(0, P), slice(stride, stride + P)

    def _merge(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(B * P, *t.shape[2:])

    ec_pairs = _two_frame_ec_aligned(
        _merge(vel[:, a]), _merge(heading[:, a]), _merge(valid[:, a]),
        _merge(vel[:, b]), _merge(heading[:, b]), _merge(valid[:, b]),
        gap_sec=STAGE_GAP_SEC, dt=dt,
    ).reshape(B, P)


    live = (~issue_done[:, a]) & (~issue_done[:, b])
    count = live.float().sum(dim=1)
    return torch.where(
        count > 0,
        (ec_pairs * live.float()).sum(dim=1) / count.clamp(min=1),
        torch.ones(B, device=device),
    )
