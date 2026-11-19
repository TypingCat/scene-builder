#pragma once
#include <torch/torch.h>
#include <tuple>
#include <optional>


torch::Tensor normalize_angle(const torch::Tensor& angle);


class BatchIDMPolicy {
public:
    BatchIDMPolicy(
        torch::Tensor target_velocities,
        float min_gap,
        float headway_time,
        float accel_max,
        float decel_max);

    std::tuple<torch::Tensor, torch::Tensor> propagate(
        const torch::Tensor& progress,
        const torch::Tensor& velocity,
        const torch::Tensor& lead_progress,
        const torch::Tensor& lead_velocity,
        const torch::Tensor& lead_rear_len,
        float dt,
        const std::optional<torch::Tensor>& path_length = std::nullopt);

private:
    torch::Tensor target_velocities_;
    float min_gap_;
    float headway_time_;
    float accel_max_;
    float decel_max_;
};


std::tuple<torch::Tensor, torch::Tensor> extract_velocity_curvature_profiles(
    const torch::Tensor& poses,
    float dt,
    float jerk_penalty,
    float curvature_rate_penalty);


class BatchLQRTracker {
public:
    BatchLQRTracker(
        const std::vector<float>& q_lateral,
        const std::vector<float>& r_lateral,
        float q_longitudinal,
        float r_longitudinal,
        int tracking_horizon,
        float stopping_proportional_gain,
        float stopping_velocity,
        const torch::Tensor& wheel_base,
        float dt);

    std::tuple<torch::Tensor, torch::Tensor> track(
        const torch::Tensor& ego_x,
        const torch::Tensor& ego_y,
        const torch::Tensor& ego_heading,
        const torch::Tensor& ego_speed,
        const torch::Tensor& ego_steering,
        const torch::Tensor& ref_pose,
        const torch::Tensor& ref_velocity,
        const torch::Tensor& ref_curvature);

private:
    torch::Tensor solve_lateral(
        const torch::Tensor& lat_state,
        const torch::Tensor& vel_profile,
        const torch::Tensor& curv_profile,
        const torch::Tensor& Q,
        const torch::Tensor& R);

    float q_lon_;
    float r_lon_;
    torch::Tensor q_lat_;
    torch::Tensor r_lat_;
    torch::Tensor in_mat_;
    int horizon_;
    float stopping_gain_;
    float stopping_vel_;
    torch::Tensor wheel_base_;
    float dt_;
};
