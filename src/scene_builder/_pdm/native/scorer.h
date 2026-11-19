#pragma once
#include "types.h"
#include <torch/torch.h>


struct ScorerResult {
    torch::Tensor scores;
    torch::Tensor first_at_fault_step;

    torch::Tensor nc_score;
    torch::Tensor ttc_score;
    torch::Tensor dac_score;
    torch::Tensor ddc_score;
    torch::Tensor tlc_score;
    torch::Tensor ep_score;
    torch::Tensor hc_score;
    torch::Tensor lk_score;
};


ScorerResult score_proposals(
    const torch::Tensor& proposals,
    const Scene& scene,
    const PlannerConfig& configs);


torch::Tensor first_at_fault_step(
    const torch::Tensor& proposals,
    const Scene& scene,
    const PlannerConfig& configs);


struct ScorerPrep {
    torch::Tensor dac_score;
    torch::Tensor dac_offroad;
    torch::Tensor multiple_lanes;
    torch::Tensor in_intersection;
    torch::Tensor t0_in_intersection;
    torch::Tensor ddc_on_route;
    torch::Tensor ddc_intersection;
    torch::Tensor has_ddc;
    torch::Tensor tlc_violation;
};


ScorerPrep score_proposals_prep(
    const torch::Tensor& proposals,
    const Scene& scene,
    const PlannerConfig& configs);


ScorerResult score_proposals_core(
    const torch::Tensor& proposals,
    const Scene& scene,
    const ScorerPrep& prep,
    const PlannerConfig& configs);


ScorerResult score_proposals_graph(
    const torch::Tensor& proposals,
    const Scene& scene,
    const PlannerConfig& configs);
