#pragma once

#include "public_types.h"

#include <optional>
#include <string>
#include <vector>

namespace LaneTypeConst {
inline torch::Tensor is_edge(const torch::Tensor& t) {
    return (t == edge) | (t == boundary) | (t == median);
}
}

constexpr int64_t STOP_STATE = TrafficStateConst::stop;

struct LQRConfig {
    std::vector<float> q_lateral;
    std::vector<float> r_lateral;
    float q_longitudinal = 10.0f;
    float r_longitudinal = 1.0f;
    int tracking_horizon = 10;
    float jerk_penalty = 1e-4f;
    float curvature_rate_penalty = 1e-2f;
    float stopping_proportional_gain = 0.5f;
    float stopping_velocity = 0.2f;
    float accel_time_constant = 0.2f;
    float steering_time_constant = 0.05f;
};

struct PreprocessConfig {
    int num_agents = 0;
    int num_traffics = 0;
};


struct DrivableAreaConfig {
    int max_links = 64;
    int max_lanes = 64;
    float default_half_width_m = 1.5f;
    float min_half_width_m = 0.5f;
    float max_half_width_m = 8.0f;
    int link_chunk = 4;
    int polygon_chunk = 8;
};

struct ModelConfig {
    std::string name;
    bool use_correction = false;
    int hidden_dim = 512;
    float target_speed = 15.0f;
    std::vector<float> speed_limit_fraction;
    std::vector<float> lateral_offsets;
    float rear_axle_to_center_ratio = 0.28f;
    float corridor_half_width = 2.0f;
    float min_gap = 1.0f;
    float headway_time = 1.5f;
    float idm_accel_max = 1.5f;
    float decel_desired = 3.0f;
    int centerline_samples = 120;
    float centerline_interval = 1.0f;
    float correction_horizon = 2.0f;
    LQRConfig lqr;
};

struct TimeConfig {
    int num_past_frames = 10;
    int num_future_frames = 40;

    int now() const { return num_past_frames - 1; }
    int total_frames() const { return num_past_frames + num_future_frames; }
};

struct PlannerConfig {
    TimeConfig time;
    PreprocessConfig preprocess;
    ModelConfig model;
    DrivableAreaConfig drivable_area;
    float frame_unit = 0.1f;
};
