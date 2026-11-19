#include "planner.h"
#include "scorer.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace torch::indexing;

namespace {
struct PlannerProfileTimer {
    bool enabled;
    std::chrono::steady_clock::time_point last;
    PlannerProfileTimer() {
        const char* flag = std::getenv("PDM_PROFILE");
        enabled = (flag && flag[0] == '1');
        if (enabled) {
            last = std::chrono::steady_clock::now();
        }
    }
    void mark(const char* label) {
        if (!enabled) return;
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - last).count();
        std::fprintf(stderr, "    [prof2] %-26s %7.3f ms\n", label, ms);
        last = now;
    }
};
}


constexpr float WHEEL_BASE_RATIO = 0.6f;
constexpr float MAX_STEERING = static_cast<float>(3.14159265358979323846) / 3.0f;
constexpr float EBRAKE_TIME_THRESHOLD = 2.0f;
constexpr float EBRAKE_SPEED_THRESHOLD = 5.0f;
constexpr float EBRAKE_MAX_ACCEL = 2.40f;
constexpr float EBRAKE_MIN_ACCEL = -4.05f;


torch::Tensor interpolate_path(
    const torch::Tensor& path_pos,
    const torch::Tensor& path_arc_in,
    const torch::Tensor& query_arc
) {

    auto N = path_pos.size(1);
    auto path_arc = std::get<0>(torch::cummax(path_arc_in,  1));
    auto query = query_arc.clamp(
        path_arc.index({Slice(), Slice(None, 1)}),
        path_arc.index({Slice(), Slice(-1, None)})
    );
    auto idx = torch::searchsorted(path_arc.contiguous(), query.contiguous());
    idx = idx.clamp(1, N - 1);
    auto idx_lo = idx - 1;

    auto arc_lo = torch::gather(path_arc, 1, idx_lo);
    auto arc_hi = torch::gather(path_arc, 1, idx);
    auto pos_lo = torch::gather(path_pos, 1, idx_lo.unsqueeze(-1).expand({-1, -1, 2}));
    auto pos_hi = torch::gather(path_pos, 1, idx.unsqueeze(-1).expand({-1, -1, 2}));

    auto t = (query - arc_lo) / (arc_hi - arc_lo).clamp_min(1e-6f);
    return pos_lo + t.unsqueeze(-1) * (pos_hi - pos_lo);
}

torch::Tensor interpolate_heading(
    const torch::Tensor& path_heading,
    const torch::Tensor& path_arc_in,
    const torch::Tensor& query_arc
) {

    auto dh = normalize_angle(torch::diff(path_heading,  1,  -1));
    auto unwrapped = torch::cat({
        path_heading.index({Slice(), Slice(None, 1)}),
        path_heading.index({Slice(), Slice(None, 1)}) + torch::cumsum(dh,  -1),
    },  -1);

    auto N = unwrapped.size(1);
    auto path_arc = std::get<0>(torch::cummax(path_arc_in,  1));
    auto query = query_arc.clamp(
        path_arc.index({Slice(), Slice(None, 1)}),
        path_arc.index({Slice(), Slice(-1, None)})
    );
    auto idx = torch::searchsorted(path_arc.contiguous(), query.contiguous());
    idx = idx.clamp(1, N - 1);
    auto idx_lo = idx - 1;

    auto arc_lo = torch::gather(path_arc, 1, idx_lo);
    auto arc_hi = torch::gather(path_arc, 1, idx);
    auto h_lo = torch::gather(unwrapped, 1, idx_lo);
    auto h_hi = torch::gather(unwrapped, 1, idx);

    auto t = (query - arc_lo) / (arc_hi - arc_lo).clamp_min(1e-6f);
    return normalize_angle(h_lo + t * (h_hi - h_lo));
}

CenterlineResult extract_centerline(const Scene& scene) {

    const auto& paths = scene.paths;
    auto B = paths.pos_xy.size(0);
    auto device = paths.pos_xy.device();

    auto any_valid = paths.valid.any( -1);
    auto path_score = paths.is_route.to(torch::kFloat32) * 2.0f + any_valid.to(torch::kFloat32);
    auto route_idx = path_score.argmax( 1);

    auto bi = torch::arange(B, device);
    return {
        paths.pos_xy.index({bi, route_idx}),
        paths.dir_xy.index({bi, route_idx}),
        paths.arc_length.index({bi, route_idx}).squeeze(-1),
        paths.valid.index({bi, route_idx}),
    };
}


LateralPathsResult generate_lateral_paths(
    const torch::Tensor& center_pos,
    const torch::Tensor& center_dir,
    const torch::Tensor& center_arc,
    const torch::Tensor& center_valid,
    const std::vector<float>& offsets
) {


    auto perp = torch::stack({
        -center_dir.index({Slice(), Slice(), 1}),
         center_dir.index({Slice(), Slice(), 0}),
    },  -1);

    std::vector<torch::Tensor> all_pos_vec;
    all_pos_vec.push_back(center_pos);
    for (float offset : offsets) {
        all_pos_vec.push_back(center_pos + offset * perp);
    }
    auto L = static_cast<int64_t>(all_pos_vec.size());
    auto all_pos = torch::stack(all_pos_vec,  1);
    auto all_valid = center_valid.unsqueeze(1).expand({-1, L, -1});

    auto seg = torch::diff(all_pos,  1,  -2);
    auto seg_len = seg.norm(2,  -1);
    auto all_arc = torch::cat({
        torch::zeros_like(seg_len.index({Slice(), Slice(), Slice(None, 1)})),
        seg_len.cumsum( -1),
    },  -1);

    auto heading = torch::atan2(
        seg.index({Slice(), Slice(), Slice(), 1}),
        seg.index({Slice(), Slice(), Slice(), 0})
    );
    auto all_heading = torch::cat({
        heading.index({Slice(), Slice(), Slice(None, 1)}),
        heading,
    },  -1);

    return {all_pos, all_arc, all_valid, all_heading};
}


std::tuple<torch::Tensor, torch::Tensor> project_agents_onto_paths(
    const torch::Tensor& agent_pos,
    const torch::Tensor& agent_valid,
    const torch::Tensor& path_pos,
    const torch::Tensor& path_arc,
    const torch::Tensor& path_valid
) {


    auto diff = agent_pos.unsqueeze(1).unsqueeze(3) - path_pos.unsqueeze(2);
    auto dist = diff.norm(2,  -1);
    dist = dist.masked_fill(~path_valid.unsqueeze(2), 1e6f);
    dist = dist.masked_fill(~agent_valid.unsqueeze(1).unsqueeze(-1), 1e6f);

    auto min_result = torch::min(dist,  -1);
    auto nearest_dist = std::get<0>(min_result);
    auto nearest_idx = std::get<1>(min_result);

    auto A = agent_pos.size(1);
    auto agent_arc = torch::gather(
        path_arc.unsqueeze(2).expand({-1, -1, A, -1}),
        3,
        nearest_idx.unsqueeze(-1)
    ).squeeze(-1);

    return {agent_arc, nearest_dist};
}

LeadingAgentResult find_leading_agent(
    const torch::Tensor& ego_progress,
    const torch::Tensor& agent_arc,
    const torch::Tensor& agent_lateral_dist,
    const torch::Tensor& agent_speed,
    const torch::Tensor& agent_size,
    const torch::Tensor& agent_valid,
    const torch::Tensor& path_length,
    int P_lon,
    const torch::Tensor& ego_front_overhang,
    float vehicle_half_len,
    float corridor_half_width
) {

    auto B = ego_progress.size(0);
    auto P = ego_progress.size(1);
    auto device = ego_progress.device();


    auto lat_idx = torch::div(torch::arange(P, torch::TensorOptions().device(device).dtype(torch::kLong)), P_lon, "trunc");

    auto prop_agent_arc = agent_arc.index({Slice(), lat_idx});
    auto prop_agent_lat = agent_lateral_dist.index({Slice(), lat_idx});

    auto in_corridor = prop_agent_lat < corridor_half_width;
    auto ahead_mask = (prop_agent_arc > ego_progress.unsqueeze(-1))
                    & agent_valid.unsqueeze(1)
                    & in_corridor;
    auto dist_ahead = (prop_agent_arc - ego_progress.unsqueeze(-1)).masked_fill(~ahead_mask, 1e6f);

    auto min_result = torch::min(dist_ahead,  -1);
    auto min_dist = std::get<0>(min_result);
    auto min_idx = std::get<1>(min_result);
    auto has_leader = min_dist < 1e5f;

    auto bi = torch::arange(B, device).unsqueeze(1).expand({-1, P});
    auto lead_speed = agent_speed.index({bi, min_idx});
    auto agent_half_len = agent_size.index({bi, min_idx, 0}) * 0.5f;
    auto lead_arc = torch::gather(prop_agent_arc, 2, min_idx.unsqueeze(-1)).squeeze(-1);


    auto arc_diff = lead_arc - ego_progress;
    auto efo = ego_front_overhang.unsqueeze(-1);
    auto polygon_dist_approx = (arc_diff - agent_half_len - efo).clamp_min(0.0f);

    auto path_len = path_length.index({Slice(), lat_idx});
    auto lead_progress = torch::where(has_leader, ego_progress + polygon_dist_approx, path_len);
    auto lead_velocity = torch::where(has_leader, lead_speed, torch::zeros_like(lead_speed));
    auto lead_rear_len = torch::where(
        has_leader,
        torch::zeros_like(agent_half_len),
        torch::full_like(agent_half_len, vehicle_half_len)
    );

    return {lead_progress, lead_velocity, lead_rear_len};
}


std::tuple<torch::Tensor, torch::Tensor> simulate_idm_loop(
    const torch::Tensor& ego_progress_init,
    const torch::Tensor& ego_velocity_init,
    const torch::Tensor& all_arc,
    const torch::Tensor& all_lat,
    const torch::Tensor& all_a_speed,
    const torch::Tensor& agent_sizes,
    const torch::Tensor& all_a_valid,
    const torch::Tensor& path_length,
    const torch::Tensor& path_len_bp,
    const torch::Tensor& ego_front_overhang,
    const torch::Tensor& tl_arc_bp,
    const torch::Tensor& tl_on_bp,
    const torch::Tensor& tl_red,
    float corridor_half_width,
    float vehicle_half_len,
    int P_lon,
    int S,
    float dt,
    BatchIDMPolicy& idm
) {
    auto B = ego_progress_init.size(0);
    auto P = ego_progress_init.size(1);
    auto device = ego_progress_init.device();
    auto f32 = torch::TensorOptions().device(device).dtype(torch::kFloat32);


    bool has_traffics = tl_arc_bp.defined() && tl_arc_bp.numel() > 0;

    auto ego_progress = ego_progress_init.clone();
    auto ego_velocity = ego_velocity_init.clone();


    auto progress_out = torch::empty({B, P, S}, f32);
    auto velocity_out = torch::empty({B, P, S}, f32);

    for (int64_t s = 0; s < S; ++s) {
        auto lead = find_leading_agent(
            ego_progress,
            all_arc.index({Slice(), Slice(), Slice(), s}),
            all_lat.index({Slice(), Slice(), Slice(), s}),
            all_a_speed.index({Slice(), Slice(), s}),
            agent_sizes,
            all_a_valid.index({Slice(), Slice(), s}),
            path_length, P_lon, ego_front_overhang,
            vehicle_half_len, corridor_half_width
        );
        auto lp = lead.progress;
        auto lv = lead.velocity;
        auto lr = lead.rear_len;

        if (has_traffics) {
            auto red_s = tl_on_bp & tl_red.index({Slice(), None, Slice(), s});
            auto ahead = (tl_arc_bp > ego_progress.unsqueeze(-1)) & red_s;
            auto tl_dist = (tl_arc_bp - ego_progress.unsqueeze(-1)).masked_fill(~ahead, 1e6f);
            auto tl_min_result = torch::min(tl_dist,  -1);
            auto tl_min_dist = std::get<0>(tl_min_result);
            auto tl_min_idx = std::get<1>(tl_min_result);
            auto red_lp = tl_arc_bp.gather(2, tl_min_idx.unsqueeze(-1)).squeeze(-1);
            auto use_red = (tl_min_dist < 1e5f) & (red_lp < lp);
            lp = torch::where(use_red, red_lp, lp);
            lv = torch::where(use_red, torch::zeros_like(lv), lv);
            lr = torch::where(use_red, torch::zeros_like(lr), lr);
        }

        auto idm_result = idm.propagate(
            ego_progress, ego_velocity, lp, lv, lr, dt, path_len_bp);
        ego_progress = std::get<0>(idm_result);
        ego_velocity = std::get<1>(idm_result);

        progress_out.index_put_({Slice(), Slice(), s}, ego_progress);
        velocity_out.index_put_({Slice(), Slice(), s}, ego_velocity);
    }

    return {progress_out, velocity_out};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> resimulate_loop(
    const torch::Tensor& ref_poses,
    const torch::Tensor& vel_profile,
    const torch::Tensor& curv_profile,
    const torch::Tensor& ego_pos_init,
    const torch::Tensor& ego_heading_init,
    const torch::Tensor& ego_speed_init,
    const torch::Tensor& wheel_base,
    float dt,
    const LQRConfig& lqr_config,
    BatchLQRTracker& lqr
) {
    auto N = ref_poses.size(0);
    auto S = ref_poses.size(1);
    auto device = ref_poses.device();

    auto tau_a = lqr_config.accel_time_constant;
    auto tau_s = lqr_config.steering_time_constant;
    auto horizon = lqr_config.tracking_horizon;

    auto pos = ego_pos_init.clone();
    auto heading = ego_heading_init.clone();
    auto speed = ego_speed_init.clone();
    auto accel = torch::zeros({N}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto steering = torch::zeros({N}, torch::TensorOptions().device(device).dtype(torch::kFloat32));


    auto f32 = torch::TensorOptions().device(device).dtype(torch::kFloat32);
    auto positions = torch::empty({N, S, 2}, f32);
    auto headings = torch::empty({N, S}, f32);
    auto speeds = torch::empty({N, S}, f32);

    for (int64_t s = 0; s < S; ++s) {
        auto ref_pose_s = ref_poses.index({Slice(), s});

        auto ref_idx = std::min(s + static_cast<int64_t>(horizon), S - 2);
        auto ref_vel = vel_profile.index({Slice(), ref_idx});

        auto curv_start = s;
        auto curv_end = std::min(s + static_cast<int64_t>(horizon), S - 1);
        auto curv_len = curv_end - curv_start;
        auto ref_curv = torch::zeros({N, horizon}, f32);
        if (curv_len > 0) {
            ref_curv.index_put_({Slice(), Slice(None, curv_len)},
                curv_profile.index({Slice(), Slice(curv_start, curv_end)}));
        }
        if (curv_len < horizon) {
            auto last_curv = curv_profile.index({Slice(), std::min(curv_end, S - 2)});
            ref_curv.index_put_({Slice(), Slice(curv_len, None)}, last_curv.unsqueeze(-1));
        }

        auto track_result = lqr.track(
            pos.index({Slice(), 0}), pos.index({Slice(), 1}),
            heading, speed, steering,
            ref_pose_s, ref_vel, ref_curv
        );
        auto accel_cmd = std::get<0>(track_result);
        auto steer_rate_cmd = std::get<1>(track_result);

        float alpha_a = dt / (dt + tau_a);
        accel = accel + alpha_a * (accel_cmd - accel);

        auto ideal_steering = steering + steer_rate_cmd * dt;
        float alpha_s = dt / (dt + tau_s);
        steering = steering + alpha_s * (ideal_steering - steering);
        steering = steering.clamp(-MAX_STEERING, MAX_STEERING);

        auto cos_h = heading.cos();
        auto sin_h = heading.sin();
        pos = pos + speed.unsqueeze(-1) * torch::stack({cos_h, sin_h},  -1) * dt;
        heading = normalize_angle(heading + speed * steering.tan() / wheel_base * dt);
        speed = (speed + accel * dt).clamp_min(0.0f);

        positions.index_put_({Slice(), s, Slice()}, pos);
        headings.index_put_({Slice(), s}, heading);
        speeds.index_put_({Slice(), s}, speed);
    }

    return {positions, headings, speeds};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> resimulate_from_poses(
    const torch::Tensor& ref_poses,
    const torch::Tensor& ego_pos_init,
    const torch::Tensor& ego_heading_init,
    const torch::Tensor& ego_speed_init,
    const torch::Tensor& wheel_base,
    float dt,
    const LQRConfig& lqr_config
) {
    auto vel_curv = extract_velocity_curvature_profiles(
        ref_poses, dt, lqr_config.jerk_penalty, lqr_config.curvature_rate_penalty
    );
    BatchLQRTracker lqr(
        lqr_config.q_lateral, lqr_config.r_lateral,
        lqr_config.q_longitudinal, lqr_config.r_longitudinal,
        lqr_config.tracking_horizon,
        lqr_config.stopping_proportional_gain, lqr_config.stopping_velocity,
        wheel_base, dt
    );
    return resimulate_loop(
        ref_poses, std::get<0>(vel_curv), std::get<1>(vel_curv),
        ego_pos_init, ego_heading_init, ego_speed_init,
        wheel_base, dt, lqr_config, lqr);
}

std::tuple<torch::Tensor, torch::Tensor> simulate_lqr_bicycle(
    const torch::Tensor& ego_pos_init,
    const torch::Tensor& ego_heading_init,
    const torch::Tensor& ego_speed_init,
    const torch::Tensor& progress_seq,
    const torch::Tensor& path_pos,
    const torch::Tensor& path_arc,
    const torch::Tensor& path_heading,
    const torch::Tensor& wheel_base,
    float dt,
    int P_lon,
    const LQRConfig& lqr_config
) {

    auto B = progress_seq.size(0);
    auto P = progress_seq.size(1);
    auto S = progress_seq.size(2);
    auto L = path_pos.size(1);
    auto N = path_pos.size(2);

    auto path_pos_flat = path_pos.reshape({B * L, N, 2});
    auto path_arc_flat = path_arc.reshape({B * L, N});
    auto path_heading_flat = path_heading.reshape({B * L, N});


    auto prog_for_interp = progress_seq.reshape({B, L, P_lon, S}).reshape({B * L, P_lon * S});
    auto ref_pos_flat = interpolate_path(path_pos_flat, path_arc_flat, prog_for_interp);
    auto ref_heading_flat = interpolate_heading(path_heading_flat, path_arc_flat, prog_for_interp);

    auto ref_pos = ref_pos_flat.reshape({B, L, P_lon, S, 2}).reshape({B, P, S, 2});
    auto ref_heading = ref_heading_flat.reshape({B, L, P_lon, S}).reshape({B, P, S});


    auto ref_poses = torch::cat({ref_pos, ref_heading.unsqueeze(-1)},  -1).reshape({B * P, S, 3});


    auto BP = B * P;
    auto init_pos = ego_pos_init.unsqueeze(1).expand({-1, P, -1}).reshape({BP, 2});
    auto init_heading = ego_heading_init.unsqueeze(1).expand({-1, P}).reshape({BP});
    auto init_speed = ego_speed_init.unsqueeze(1).expand({-1, P}).reshape({BP});

    auto sim_result = resimulate_from_poses(
        ref_poses, init_pos, init_heading, init_speed,
        wheel_base, dt, lqr_config
    );

    auto positions = std::get<0>(sim_result).reshape({B, P, S, 2});
    auto headings = std::get<1>(sim_result).reshape({B, P, S});

    return {positions, headings};
}

torch::Tensor generate_brake_trajectory(
    const Scene& scene,
    const PlannerConfig& configs
) {

    auto S = configs.time.num_future_frames;
    auto dt = configs.frame_unit;

    auto ego_mask = scene.agents.is_ego;
    auto ego_idx = ego_mask.to(torch::kFloat32).argmax( 1);
    auto B = ego_mask.size(0);
    auto device = ego_mask.device();
    auto bi = torch::arange(B, device);

    auto now = configs.time.now();
    auto pos = scene.agents.pos_xy.index({bi, ego_idx, now}).clone();
    auto ego_dir = scene.agents.dir_xy.index({bi, ego_idx, now});
    auto heading = torch::atan2(
        ego_dir.index({Slice(), 1}),
        ego_dir.index({Slice(), 0})
    );
    auto speed = scene.agents.vel_xy.index({bi, ego_idx, now}).norm(2,  -1);


    torch::Tensor accel;
    if (now > 0) {
        auto prev_speed = scene.agents.vel_xy.index({bi, ego_idx, now - 1}).norm(2,  -1);
        accel = (speed - prev_speed) / dt;
    } else {
        accel = torch::zeros_like(speed);
    }


    auto u_high = -10.0f * speed;
    auto a_high = u_high.clamp(EBRAKE_MIN_ACCEL, EBRAKE_MAX_ACCEL);
    auto v_high = 1.1f * (speed + a_high);

    auto u_low = -4.0f * speed - 1.0f * accel;
    auto v_low = u_low.clamp(EBRAKE_MIN_ACCEL, EBRAKE_MAX_ACCEL);

    auto is_high = speed > 0.2f;
    auto correcting_velocity = torch::where(is_high, v_high, v_low).clamp_min(0.0f);

    auto times = torch::arange(1, S + 1, torch::TensorOptions().device(device).dtype(speed.dtype())) * dt;
    auto dir_xy = torch::stack({heading.cos(), heading.sin()},  -1);
    auto displacements = correcting_velocity.unsqueeze(-1) * times.unsqueeze(0);
    auto positions = pos.unsqueeze(1) + displacements.unsqueeze(-1) * dir_xy.unsqueeze(1);

    auto heading_expanded = heading.unsqueeze(-1).expand({-1, S});
    auto speeds = correcting_velocity.unsqueeze(-1).expand({-1, S});
    return torch::stack({
        positions.index({Slice(), Slice(), 0}),
        positions.index({Slice(), Slice(), 1}),
        heading_expanded,
        speeds,
    },  -1);
}


ProposalResult generate_proposals(
    const Scene& scene,
    const PlannerConfig& configs
) {
    PlannerProfileTimer prof;
    const auto& tc = configs.time;
    const auto& mc = configs.model;
    auto device = scene.agents.pos_xy.device();
    auto dtype = scene.agents.pos_xy.dtype();

    auto S_internal = tc.num_future_frames;
    auto dt = configs.frame_unit;
    auto B = scene.agents.pos_xy.size(0);


    auto centerline = extract_centerline(scene);
    auto& center_pos = centerline.pos;
    auto& center_dir = centerline.dir;
    auto& center_arc = centerline.arc;
    auto& center_valid = centerline.valid;

    auto lateral_paths = generate_lateral_paths(
        center_pos, center_dir, center_arc, center_valid, mc.lateral_offsets
    );
    auto& path_pos = lateral_paths.pos;
    auto& path_arc = lateral_paths.arc;
    auto& path_valid = lateral_paths.valid;
    auto& path_heading = lateral_paths.heading;
    auto L = path_pos.size(1);


    auto fractions = mc.speed_limit_fraction;
    auto P_lon = static_cast<int>(fractions.size());
    auto P = L * P_lon;


    auto bi_sl = torch::arange(B, device);
    auto route_score = scene.paths.is_route.to(torch::kFloat32) * 2.0f
                     + scene.paths.valid.any( -1).to(torch::kFloat32);
    auto route_idx = route_score.argmax( 1);

    auto route_link_id = scene.paths.near_link.index({bi_sl, route_idx}).select(1, 0);

    auto link_match = (scene.links.id == route_link_id.unsqueeze(1));
    auto has_match = link_match.any( 1);
    auto match_idx = link_match.to(torch::kFloat32).argmax( 1);
    auto link_speed = torch::where(
        has_match,
        scene.links.speed_limit.index({bi_sl, match_idx}),
        torch::zeros({B}, torch::TensorOptions().device(device).dtype(torch::kFloat32))
    );
    auto speed_limit = torch::where(
        link_speed > 0.0f,
        link_speed,
        torch::full({B}, mc.target_speed, torch::TensorOptions().device(device).dtype(torch::kFloat32))
    );

    auto fractions_t = torch::tensor(fractions, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    auto target_vels = (speed_limit.unsqueeze(-1) * fractions_t.unsqueeze(0));
    target_vels = target_vels.unsqueeze(1).expand({-1, L, -1}).reshape({B, P}).contiguous();


    auto ego_mask = scene.agents.is_ego;
    auto ego_idx = ego_mask.to(torch::kFloat32).argmax( 1);
    auto bi = torch::arange(B, device);

    auto ego_length = scene.agents.size_xyz.index({bi, ego_idx, 0});
    auto wheel_base = (ego_length * WHEEL_BASE_RATIO).mean();

    auto now = tc.now();
    auto ego_dir = scene.agents.dir_xy.index({bi, ego_idx, now});
    auto rac = ego_length * mc.rear_axle_to_center_ratio;
    auto ego_pos = scene.agents.pos_xy.index({bi, ego_idx, now}) - rac.unsqueeze(-1) * ego_dir;
    auto ego_speed_init = (scene.agents.vel_xy.index({bi, ego_idx, now}) * ego_dir).sum( -1).clamp_min(0.0f);
    auto ego_heading = torch::atan2(
        ego_dir.index({Slice(), 1}),
        ego_dir.index({Slice(), 0})
    );


    auto ego_dist = (path_pos - ego_pos.unsqueeze(1).unsqueeze(2)).norm(2,  -1);
    ego_dist = ego_dist.masked_fill(~path_valid, 1e6f);
    auto nearest_idx = ego_dist.argmin( -1);
    auto ego_progress_per_path = torch::gather(path_arc, 2, nearest_idx.unsqueeze(-1)).squeeze(-1);

    auto ego_progress = ego_progress_per_path.unsqueeze(-1).expand({-1, -1, P_lon}).reshape({B, P});
    auto ego_velocity = ego_speed_init.unsqueeze(-1).expand({-1, P}).contiguous();


    auto nonego_mask = ~scene.agents.is_ego;
    auto& agent_sizes = scene.agents.size_xyz;
    auto path_length = std::get<0>(torch::max(path_arc,  -1));

    auto A = scene.agents.pos_xy.size(1);
    auto cv = predict_agents_cv(scene, tc, dt);
    auto& all_a_pos = cv.pos;
    auto all_a_valid = cv.valid & nonego_mask.unsqueeze(-1);
    auto all_a_speed = (cv.vel * ego_dir.unsqueeze(1).unsqueeze(2)).sum( -1).clamp_min(0.0f);

    auto AS = A * S_internal;
    auto proj_result = project_agents_onto_paths(
        all_a_pos.reshape({B, AS, 2}),
        all_a_valid.reshape({B, AS}),
        path_pos, path_arc, path_valid
    );
    auto all_arc = std::get<0>(proj_result).reshape({B, L, A, S_internal});
    auto all_lat = std::get<1>(proj_result).reshape({B, L, A, S_internal});


    auto lat_idx = torch::div(torch::arange(P, torch::TensorOptions().device(device).dtype(torch::kLong)), P_lon, "trunc");
    bool has_traffics = scene.traffics.valid.defined()
                     && scene.traffics.valid.size(1) > 0;

    torch::Tensor tl_arc_bp, tl_on_bp, tl_red;
    float corridor_half_width = mc.corridor_half_width;

    if (has_traffics) {
        auto ref_t = now;
        auto tl_pos_ref = scene.traffics.pos_xy;
        auto tl_valid_ref = scene.traffics.valid.index({Slice(), Slice(), ref_t});
        auto tl_proj = project_agents_onto_paths(
            tl_pos_ref, tl_valid_ref, path_pos, path_arc, path_valid
        );
        auto tl_arc_proj = std::get<0>(tl_proj);
        auto tl_lat_proj = std::get<1>(tl_proj);
        auto tl_on_path = tl_lat_proj < corridor_half_width;

        tl_arc_bp = tl_arc_proj.index({Slice(), lat_idx});
        tl_on_bp = tl_on_path.index({Slice(), lat_idx});

        auto tl_state_now = scene.traffics.state.index({Slice(), Slice(), ref_t});
        auto tl_valid_now = scene.traffics.valid.index({Slice(), Slice(), ref_t});
        auto tl_red_now = (tl_state_now == STOP_STATE) & tl_valid_now;
        tl_red = tl_red_now.unsqueeze(-1).expand({-1, -1, S_internal});
    }

    prof.mark("setup+paths+TL");


    auto path_len_bp = path_length.unsqueeze(-1).expand({-1, -1, P_lon}).reshape({B, P}).contiguous();
    auto ego_front_overhang = ego_length * (0.5f + mc.rear_axle_to_center_ratio);

    BatchIDMPolicy idm(target_vels, mc.min_gap, mc.headway_time,
                       mc.idm_accel_max, mc.decel_desired);
    auto idm_out = simulate_idm_loop(
        ego_progress, ego_velocity,
        all_arc, all_lat, all_a_speed, agent_sizes, all_a_valid,
        path_length, path_len_bp, ego_front_overhang,
        tl_arc_bp, tl_on_bp, tl_red,
        corridor_half_width,  2.5f,
        P_lon, S_internal, dt, idm
    );
    auto progress_seq = std::get<0>(idm_out);
    auto velocity_seq = std::get<1>(idm_out);

    prof.mark("IDM loop");


    auto sim_result = simulate_lqr_bicycle(
        ego_pos, ego_heading, ego_speed_init, progress_seq,
        path_pos, path_arc, path_heading,
        wheel_base, dt, P_lon, mc.lqr
    );
    prof.mark("simulate_lqr_bicycle");
    auto sim_pos = std::get<0>(sim_result);
    auto sim_heading = std::get<1>(sim_result);


    auto fwd = torch::stack({sim_heading.cos(), sim_heading.sin()},  -1);
    auto rac_bcast = (ego_length * mc.rear_axle_to_center_ratio)
        .reshape({B, 1, 1, 1});
    auto center_pos_out = sim_pos + rac_bcast * fwd;
    auto proposals = torch::cat({center_pos_out, sim_heading.unsqueeze(-1)},  -1);

    return {proposals, velocity_seq};
}


CVPrediction predict_agents_cv(
    const Scene& scene,
    const TimeConfig& tc,
    float dt
) {
    using namespace torch::indexing;
    auto now = tc.now();
    auto S = tc.num_future_frames;

    auto pos_now = scene.agents.pos_xy.index({Slice(), Slice(), now});
    auto vel_now = scene.agents.vel_xy.index({Slice(), Slice(), now});
    auto dir_now = scene.agents.dir_xy.index({Slice(), Slice(), now});
    auto valid_now = scene.agents.valid.index({Slice(), Slice(), now});

    auto dt_offsets = torch::arange(1, S + 1, torch::TensorOptions().device(pos_now.device()).dtype(pos_now.dtype())) * dt;


    auto pred_pos = pos_now.unsqueeze(2) + vel_now.unsqueeze(2) * dt_offsets.reshape({1, 1, S, 1});
    auto pred_vel = vel_now.unsqueeze(2).expand_as(pred_pos).contiguous();
    auto pred_dir = dir_now.unsqueeze(2).expand_as(pred_pos).contiguous();
    auto pred_valid = valid_now.unsqueeze(2).expand({-1, -1, S}).contiguous();

    return {pred_pos, pred_vel, pred_dir, pred_valid};
}

Output build_output(
    const torch::Tensor& proposals_10hz,
    const torch::Tensor& best_idx,
    const Scene& scene,
    const PlannerConfig& configs,
    const torch::Tensor& batch_valid
) {
    auto S_fut = configs.time.num_future_frames;
    auto num_past = configs.time.num_past_frames;
    auto dt = configs.frame_unit;

    const auto& agents = scene.agents;
    auto B = agents.pos_xy.size(0);
    auto A = agents.pos_xy.size(1);
    auto device = agents.pos_xy.device();
    auto bi = torch::arange(B, device);


    auto best_traj = proposals_10hz.index({bi, best_idx});

    auto ego_fut_pos = best_traj.index({Slice(), Slice(), Slice(None, 2)}).unsqueeze(1);
    auto ego_fut_heading = best_traj.index({Slice(), Slice(), 2});
    auto ego_fut_dir = torch::stack({
        ego_fut_heading.cos(), ego_fut_heading.sin()
    },  -1).unsqueeze(1);


    auto past_pos = agents.pos_xy.index({Slice(), Slice(), Slice(None, num_past)});
    auto past_dir = agents.dir_xy.index({Slice(), Slice(), Slice(None, num_past)});
    auto past_valid = agents.valid.index({Slice(), Slice(), Slice(None, num_past)});


    auto cv = predict_agents_cv(scene, configs.time, dt);

    auto ego_agent_idx = agents.is_ego.to(torch::kFloat32).argmax( 1);

    auto fut_pos = cv.pos.clone();
    auto fut_dir = cv.dir.clone();
    fut_pos.index_put_({bi, ego_agent_idx}, ego_fut_pos.squeeze(1));
    fut_dir.index_put_({bi, ego_agent_idx}, ego_fut_dir.squeeze(1));

    auto out_pos = torch::cat({past_pos, fut_pos},  2);
    auto out_dir = torch::cat({past_dir, fut_dir},  2);

    auto S_out = num_past + S_fut;
    auto out_size = agents.size_xyz.unsqueeze(2).expand({-1, -1, S_out, -1});
    auto fut_valid = cv.valid.clone();
    fut_valid.index_put_({bi, ego_agent_idx}, true);
    auto out_valid = torch::cat({past_valid, fut_valid},  2);
    auto out_is_ego = agents.is_ego.unsqueeze(-1).expand({-1, -1, S_out});
    auto out_type = agents.type.unsqueeze(-1).expand({-1, -1, S_out});

    return Output{out_pos, out_dir, out_size, out_valid, out_is_ego, out_type,
                  batch_valid};
}

std::tuple<torch::Tensor, torch::Tensor> run_pdm_closed(
    const Scene& scene,
    const PlannerConfig& configs
) {
    torch::NoGradGuard no_grad;

    auto result = generate_proposals(scene, configs);
    auto& ref_proposals = result.proposals;
    auto& ref_speeds = result.speeds;
    auto proposals = torch::cat({ref_proposals, ref_speeds.unsqueeze(-1)},  -1);

    auto B = proposals.size(0);
    auto device = proposals.device();
    auto bi = torch::arange(B, device);

    auto scorer_result = score_proposals(proposals, scene, configs);
    auto best_idx = scorer_result.scores.argmax( 1);
    auto best = proposals.index({bi, best_idx});


    auto dt = configs.frame_unit;
    auto best_collision_step = scorer_result.first_at_fault_step.index({bi, best_idx});
    auto ego_agent_idx = scene.agents.is_ego.to(torch::kFloat32).argmax( 1);
    auto ego_vel_now = scene.agents.vel_xy.index({bi, ego_agent_idx, configs.time.now()});
    auto ego_speed_now = ego_vel_now.norm( -1);
    auto needs_brake = (best_collision_step.to(torch::kFloat32) * dt <= EBRAKE_TIME_THRESHOLD)
                     & (ego_speed_now <= EBRAKE_SPEED_THRESHOLD);
    auto brake = generate_brake_trajectory(scene, configs);
    best = torch::where(needs_brake.unsqueeze(1).unsqueeze(2), brake, best);


    return {best.index({Slice(), Slice(), Slice(None, 3)}).contiguous(),
            best.index({Slice(), Slice(), 3}).contiguous()};
}

Output run_pdm_full(const Scene& scene, const PlannerConfig& configs) {
    torch::NoGradGuard no_grad;

    auto result = generate_proposals(scene, configs);
    auto proposals = torch::cat({result.proposals, result.speeds.unsqueeze(-1)}, -1);

    auto B = proposals.size(0);
    auto device = proposals.device();
    auto bi = torch::arange(B, device);

    auto scorer_result = score_proposals(proposals, scene, configs);
    auto best_idx = scorer_result.scores.argmax(1);


    auto dt = configs.frame_unit;
    auto best_collision_step = scorer_result.first_at_fault_step.index({bi, best_idx});
    auto ego_agent_idx = scene.agents.is_ego.to(torch::kFloat32).argmax(1);
    auto ego_vel_now = scene.agents.vel_xy.index({bi, ego_agent_idx, configs.time.now()});
    auto ego_speed_now = torch::norm(ego_vel_now, 2, -1);
    auto needs_brake = (best_collision_step.to(torch::kFloat32) * dt <= EBRAKE_TIME_THRESHOLD)
                     & (ego_speed_now <= EBRAKE_SPEED_THRESHOLD);

    auto best = proposals.index({bi, best_idx});
    auto brake = generate_brake_trajectory(scene, configs);


    auto proposals_2 = torch::stack({best, brake}, 1);
    auto zero_idx = torch::zeros({B}, torch::TensorOptions().device(device).dtype(torch::kLong));
    auto one_idx = torch::ones({B}, torch::TensorOptions().device(device).dtype(torch::kLong));
    auto final_idx = torch::where(needs_brake, one_idx, zero_idx);

    auto has_ego = scene.agents.is_ego.any( 1);
    auto has_centerline = extract_centerline(scene).valid.any( -1);
    auto bv = has_ego & has_centerline;
    return build_output(proposals_2, final_idx, scene, configs, bv);
}
