#include "control.h"
#include <cmath>

using namespace torch::indexing;


torch::Tensor normalize_angle(const torch::Tensor& angle) {
    return torch::atan2(torch::sin(angle), torch::cos(angle));
}


BatchIDMPolicy::BatchIDMPolicy(
    torch::Tensor target_velocities,
    float min_gap,
    float headway_time,
    float accel_max,
    float decel_max)
    : target_velocities_(std::move(target_velocities)),
      min_gap_(min_gap),
      headway_time_(headway_time),
      accel_max_(accel_max),
      decel_max_(decel_max) {}

std::tuple<torch::Tensor, torch::Tensor> BatchIDMPolicy::propagate(
    const torch::Tensor& progress,
    const torch::Tensor& velocity,
    const torch::Tensor& lead_progress,
    const torch::Tensor& lead_velocity,
    const torch::Tensor& lead_rear_len,
    float dt,
    const std::optional<torch::Tensor>& path_length) {

    auto v = velocity;
    auto v_target = target_velocities_.to(v.device(), v.scalar_type());


    float sqrt_ab = std::sqrt(accel_max_ * decel_max_);
    auto s_star = min_gap_
        + v * headway_time_
        + v * (v - lead_velocity) / (2.0f * sqrt_ab + 1e-6f);


    auto s_alpha = (lead_progress - progress - lead_rear_len).clamp_min(min_gap_);


    auto v_ratio = torch::pow(v / v_target.clamp_min(1e-6f), 10);
    auto s_ratio = torch::pow(s_star / s_alpha, 2);
    auto accel = accel_max_ * (1.0f - v_ratio - s_ratio);
    accel = accel.clamp(-decel_max_, accel_max_);

    auto next_progress = progress + dt * v;
    if (path_length.has_value()) {
        next_progress = torch::clamp_max(next_progress, path_length.value());
    }
    auto next_velocity = (velocity + dt * accel).clamp_min(0.0f);

    return {next_progress, next_velocity};
}


namespace {

constexpr double kInitialCurvaturePenalty = 1e-10;
constexpr double kSolveRidge = 1e-2;
constexpr double kDegenerateDispThreshold = 1e-4;

torch::Tensor make_banded_diff_matrix(int n, torch::Device device) {
    auto mat = torch::zeros({n, n + 1}, torch::dtype(torch::kFloat64).device(device));
    auto eye = torch::eye(n, torch::dtype(torch::kFloat64).device(device));
    mat.index_put_({Slice(), Slice(1, None)}, eye);
    mat.index_put_({Slice(), Slice(None, -1)},
                   mat.index({Slice(), Slice(None, -1)}) - eye);
    return mat;
}

std::tuple<torch::Tensor, torch::Tensor> fit_velocity_acceleration(
    const torch::Tensor& xy_disp,
    const torch::Tensor& heading,
    const torch::Tensor& step_weight,
    float dt,
    float jerk_penalty) {

    int64_t B = xy_disp.size(0);
    int64_t M = xy_disp.size(1);
    auto device = xy_disp.device();


    auto y = xy_disp.reshape({B, -1}).to(torch::kFloat64);

    auto cos_h = heading.cos().to(torch::kFloat64);
    auto sin_h = heading.sin().to(torch::kFloat64);


    auto A_col = torch::zeros({B, 2 * M}, torch::dtype(torch::kFloat64).device(device));
    A_col.index_put_({Slice(), Slice(0, None, 2)}, cos_h);
    A_col.index_put_({Slice(), Slice(1, None, 2)}, sin_h);


    auto A = A_col.unsqueeze(-1).expand({-1, -1, M}) * (dt * dt);
    A = A.clone();
    A.index_put_({Slice(), Slice(), 0}, A_col * dt);


    auto upper_mask = torch::triu(torch::ones({M, M}, torch::dtype(torch::kBool).device(device)), 1);
    auto upper_mask_2m = upper_mask.repeat_interleave(2, 0);
    A.index_put_({Slice(), upper_mask_2m}, 0.0);


    auto w2m = step_weight.to(torch::kFloat64).repeat_interleave(2, -1);
    auto w_sqrt = w2m.sqrt();
    A = A * w_sqrt.unsqueeze(-1);
    y = y * w_sqrt;


    torch::Tensor R;
    if (M > 2) {
        auto banded = make_banded_diff_matrix(M - 2, device);
        R = torch::cat({torch::zeros({M - 2, 1}, torch::dtype(torch::kFloat64).device(device)),
                        banded}, 1);
    } else {
        R = torch::zeros({1, M}, torch::dtype(torch::kFloat64).device(device));
    }

    auto AT = A.transpose(1, 2);
    auto ATA = torch::bmm(AT, A);
    auto RTR = torch::matmul(R.t(), R).unsqueeze(0);
    auto ATy = torch::bmm(AT, y.unsqueeze(-1)).squeeze(-1);
    auto eye = torch::eye(M, torch::dtype(torch::kFloat64).device(device)).unsqueeze(0);

    auto lhs = ATA + jerk_penalty * RTR + kSolveRidge * eye;
    auto x = torch::linalg_solve(lhs, ATy);

    return std::make_tuple(
        x.index({Slice(), 0}).to(torch::kFloat32),
        x.index({Slice(), Slice(1, None)}).to(torch::kFloat32));
}

std::tuple<torch::Tensor, torch::Tensor> fit_curvature(
    const torch::Tensor& heading_disp,
    const torch::Tensor& velocity,
    const torch::Tensor& step_weight,
    float dt,
    float curvature_rate_penalty) {

    int64_t B = heading_disp.size(0);
    int64_t M = heading_disp.size(1);
    auto device = heading_disp.device();

    auto y = heading_disp.to(torch::kFloat64);
    auto vel = velocity.to(torch::kFloat64);
    auto v_dt2 = vel * (dt * dt);


    auto A = torch::tril(torch::ones({M, M}, torch::dtype(torch::kFloat64).device(device)))
                 .unsqueeze(0).expand({B, -1, -1}).clone();
    A.index_put_({Slice(), Slice(), 0}, vel * dt);
    A.index_put_({Slice(), Slice(), Slice(1, None)},
                 A.index({Slice(), Slice(), Slice(1, None)}) * v_dt2.unsqueeze(2));


    auto w = step_weight.to(torch::kFloat64);
    auto w_sqrt = w.sqrt();
    A = A * w_sqrt.unsqueeze(-1);
    y = y * w_sqrt;


    auto Q = curvature_rate_penalty * torch::eye(M, torch::dtype(torch::kFloat64).device(device));
    Q.index_put_({0, 0}, kInitialCurvaturePenalty);

    auto AT = A.transpose(1, 2);
    auto ATA = torch::bmm(AT, A);
    auto ATy = torch::bmm(AT, y.unsqueeze(-1)).squeeze(-1);
    auto eye = torch::eye(M, torch::dtype(torch::kFloat64).device(device)).unsqueeze(0);

    auto lhs = ATA + Q.unsqueeze(0) + kSolveRidge * eye;
    auto x = torch::linalg_solve(lhs, ATy);

    return std::make_tuple(
        x.index({Slice(), 0}).to(torch::kFloat32),
        x.index({Slice(), Slice(1, None)}).to(torch::kFloat32));
}

}


std::tuple<torch::Tensor, torch::Tensor> extract_velocity_curvature_profiles(
    const torch::Tensor& poses,
    float dt,
    float jerk_penalty,
    float curvature_rate_penalty) {

    auto pose_diff = torch::diff(poses, 1, 1);
    auto xy_disp = pose_diff.index({"...", Slice(None, 2)});
    auto heading_disp = normalize_angle(pose_diff.index({"...", 2}));
    auto heading_profile = poses.index({Slice(), Slice(None, -1), 2});


    auto disp_norm = xy_disp.norm(2, -1);
    auto step_weight = (disp_norm > kDegenerateDispThreshold).to(torch::kFloat32);


    auto [init_vel, accel_profile] = fit_velocity_acceleration(
        xy_disp, heading_profile, step_weight, dt, jerk_penalty);

    auto cumsum_a = accel_profile.cumsum(-1) * dt;
    auto vel_profile = torch::cat({init_vel.unsqueeze(-1),
                                   init_vel.unsqueeze(-1) + cumsum_a}, -1);


    auto curv_weight = step_weight * (vel_profile.abs() > kDegenerateDispThreshold).to(torch::kFloat32);


    auto [init_curv, curv_rate_profile] = fit_curvature(
        heading_disp, vel_profile, curv_weight, dt, curvature_rate_penalty);

    auto cumsum_cr = curv_rate_profile.cumsum(-1) * dt;
    auto curv_profile = torch::cat({init_curv.unsqueeze(-1),
                                    init_curv.unsqueeze(-1) + cumsum_cr}, -1);

    return {vel_profile, curv_profile};
}


BatchLQRTracker::BatchLQRTracker(
    const std::vector<float>& q_lateral,
    const std::vector<float>& r_lateral,
    float q_longitudinal,
    float r_longitudinal,
    int tracking_horizon,
    float stopping_proportional_gain,
    float stopping_velocity,
    const torch::Tensor& wheel_base,
    float dt)
    : q_lon_(q_longitudinal),
      r_lon_(r_longitudinal),
      horizon_(tracking_horizon),
      stopping_gain_(stopping_proportional_gain),
      stopping_vel_(stopping_velocity),
      wheel_base_(wheel_base),
      dt_(dt) {


    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(wheel_base.device());
    q_lat_ = torch::diag(torch::tensor(
        {q_lateral[0], q_lateral[1], q_lateral[2]}, opts));
    r_lat_ = torch::diag(torch::tensor(
        {r_lateral[0]}, opts));


    in_mat_ = torch::tensor({{0.0f}, {0.0f}, {dt}}, opts);
}

std::tuple<torch::Tensor, torch::Tensor> BatchLQRTracker::track(
    const torch::Tensor& ego_x,
    const torch::Tensor& ego_y,
    const torch::Tensor& ego_heading,
    const torch::Tensor& ego_speed,
    const torch::Tensor& ego_steering,
    const torch::Tensor& ref_pose,
    const torch::Tensor& ref_velocity,
    const torch::Tensor& ref_curvature) {

    auto& q_lat = q_lat_;
    auto& r_lat = r_lat_;


    auto dx = ego_x - ref_pose.index({Slice(), 0});
    auto dy = ego_y - ref_pose.index({Slice(), 1});
    auto h_ref = ref_pose.index({Slice(), 2});
    auto lat_err = -dx * h_ref.sin() + dy * h_ref.cos();
    auto head_err = normalize_angle(ego_heading - h_ref);
    auto lat_state = torch::stack({lat_err, head_err, ego_steering}, -1);


    auto stop_accel = -stopping_gain_ * (ego_speed - ref_velocity);
    auto stop_steer_rate = torch::zeros_like(stop_accel);


    double B_lon = horizon_ * dt_;
    auto state_err = ego_speed - ref_velocity;
    double inv_lon = -1.0 / (B_lon * q_lon_ * B_lon + r_lon_);
    auto lqr_accel = inv_lon * B_lon * q_lon_ * state_err;


    auto k = torch::arange(horizon_, torch::dtype(torch::kFloat32).device(ego_x.device()));
    auto vel_profile = ego_speed.unsqueeze(-1)
                       + lqr_accel.unsqueeze(-1) * k * dt_;


    auto lqr_steer_rate = solve_lateral(
        lat_state, vel_profile, ref_curvature, q_lat, r_lat);


    auto should_stop = ((ref_velocity <= stopping_vel_) & (ego_speed <= stopping_vel_))
                           .to(torch::kFloat32);
    auto accel_cmd = should_stop * stop_accel + (1.0 - should_stop) * lqr_accel;
    auto steer_rate_cmd = should_stop * stop_steer_rate + (1.0 - should_stop) * lqr_steer_rate;

    return {accel_cmd, steer_rate_cmd};
}

torch::Tensor BatchLQRTracker::solve_lateral(
    const torch::Tensor& lat_state,
    const torch::Tensor& vel_profile,
    const torch::Tensor& curv_profile,
    const torch::Tensor& Q,
    const torch::Tensor& R) {

    int64_t B = lat_state.size(0);
    auto device = lat_state.device();
    auto opts32 = torch::dtype(torch::kFloat32).device(device);

    auto I3 = torch::eye(3, opts32);
    auto& in_mat = in_mat_;


    auto A = I3.unsqueeze(0).expand({B, -1, -1}).clone();
    auto B_mat = torch::zeros({B, 3, 1}, opts32);
    auto g = torch::zeros({B, 3}, opts32);

    for (int k = 0; k < horizon_; ++k) {
        auto v_k = vel_profile.index({Slice(), k});
        auto kappa_k = curv_profile.index({Slice(), k});

        auto F_k = I3.unsqueeze(0).expand({B, -1, -1}).clone();
        F_k.index_put_({Slice(), 0, 1}, v_k * dt_);
        F_k.index_put_({Slice(), 1, 2}, v_k * dt_ / wheel_base_);

        auto d_k = torch::zeros({B, 3}, opts32);
        d_k.index_put_({Slice(), 1}, -v_k * kappa_k * dt_);

        A = torch::bmm(F_k, A);
        B_mat = torch::bmm(F_k, B_mat) + in_mat.unsqueeze(0);
        g = torch::bmm(F_k, g.unsqueeze(-1)).squeeze(-1) + d_k;
    }


    auto se = torch::bmm(A, lat_state.unsqueeze(-1)).squeeze(-1) + g;
    se.index_put_({Slice(), 1}, normalize_angle(se.index({Slice(), 1})));
    se.index_put_({Slice(), 2}, normalize_angle(se.index({Slice(), 2})));


    auto BT = B_mat.transpose(1, 2);
    auto BT_Q = torch::matmul(BT, Q.unsqueeze(0));
    auto denom = torch::bmm(BT_Q, B_mat) + R.unsqueeze(0);
    auto numer = torch::bmm(BT_Q, se.unsqueeze(-1));

    auto u = -numer / denom;
    return u.squeeze(-1).squeeze(-1);
}
