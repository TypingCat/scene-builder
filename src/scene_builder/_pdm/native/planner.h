#pragma once
#include "types.h"
#include "control.h"
#include <torch/torch.h>
#include <tuple>
#include <vector>


struct CenterlineResult {
    torch::Tensor pos;
    torch::Tensor dir;
    torch::Tensor arc;
    torch::Tensor valid;
};

struct LateralPathsResult {
    torch::Tensor pos;
    torch::Tensor arc;
    torch::Tensor valid;
    torch::Tensor heading;
};

struct LeadingAgentResult {
    torch::Tensor progress;
    torch::Tensor velocity;
    torch::Tensor rear_len;
};

struct CVPrediction {
    torch::Tensor pos;
    torch::Tensor vel;
    torch::Tensor dir;
    torch::Tensor valid;
};

struct ProposalResult {
    torch::Tensor proposals;
    torch::Tensor speeds;
};


torch::Tensor interpolate_path(
    const torch::Tensor& path_pos,
    const torch::Tensor& path_arc,
    const torch::Tensor& query_arc);


torch::Tensor interpolate_heading(
    const torch::Tensor& path_heading,
    const torch::Tensor& path_arc,
    const torch::Tensor& query_arc);


CenterlineResult extract_centerline(const Scene& scene);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> resimulate_from_poses(
    const torch::Tensor& ref_poses,
    const torch::Tensor& ego_pos_init,
    const torch::Tensor& ego_heading_init,
    const torch::Tensor& ego_speed_init,
    const torch::Tensor& wheel_base,
    float dt,
    const LQRConfig& lqr_config);


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
    BatchLQRTracker& lqr);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> resimulate_from_poses_graph(
    const torch::Tensor& ref_poses,
    const torch::Tensor& ego_pos_init,
    const torch::Tensor& ego_heading_init,
    const torch::Tensor& ego_speed_init,
    const torch::Tensor& wheel_base,
    float dt,
    const LQRConfig& lqr_config);


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
    BatchIDMPolicy& idm);


std::tuple<torch::Tensor, torch::Tensor> simulate_idm_loop_graph(
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
    const torch::Tensor& target_vels,
    float min_gap, float headway_time, float accel_max, float decel_max,
    float corridor_half_width, float vehicle_half_len,
    int P_lon, int S, float dt);


torch::Tensor generate_brake_trajectory(
    const Scene& scene,
    const PlannerConfig& configs);


CVPrediction predict_agents_cv(
    const Scene& scene,
    const TimeConfig& tc,
    float dt);


ProposalResult generate_proposals(
    const Scene& scene,
    const PlannerConfig& configs);


std::tuple<torch::Tensor, torch::Tensor> run_pdm_closed(
    const Scene& scene,
    const PlannerConfig& configs);


Output build_output(
    const torch::Tensor& proposals_10hz,
    const torch::Tensor& best_idx,
    const Scene& scene,
    const PlannerConfig& configs,
    const torch::Tensor& batch_valid);


Output run_pdm_full(const Scene& scene, const PlannerConfig& configs);
