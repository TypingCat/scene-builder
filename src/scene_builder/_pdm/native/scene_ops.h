#pragma once

#include "public_types.h"

#include <torch/torch.h>

#include <algorithm>
#include <limits>

namespace runtime {

using namespace torch::indexing;

struct Transform {
    torch::Tensor inverse_matrix;
};

inline torch::Tensor align_batch(torch::Tensor tensor, int batch_rank, int target_batch_rank) {
    int extra = target_batch_rank - batch_rank;
    TORCH_CHECK(extra >= 0, "target batch rank < matrix batch rank");
    for (int i = 0; i < extra; ++i) {
        tensor = tensor.unsqueeze(batch_rank);
    }
    return tensor;
}

inline Transform transform_from_pose(const torch::Tensor& pos_xy, const torch::Tensor& dir_xy) {
    auto inverse_matrix = torch::stack({
        dir_xy.index({torch::indexing::Ellipsis, 0}),
        dir_xy.index({torch::indexing::Ellipsis, 1}),
        -pos_xy.index({torch::indexing::Ellipsis, 0}) * dir_xy.index({torch::indexing::Ellipsis, 0})
            - pos_xy.index({torch::indexing::Ellipsis, 1}) * dir_xy.index({torch::indexing::Ellipsis, 1}),
        -dir_xy.index({torch::indexing::Ellipsis, 1}),
        dir_xy.index({torch::indexing::Ellipsis, 0}),
        pos_xy.index({torch::indexing::Ellipsis, 0}) * dir_xy.index({torch::indexing::Ellipsis, 1})
            - pos_xy.index({torch::indexing::Ellipsis, 1}) * dir_xy.index({torch::indexing::Ellipsis, 0}),
    }, -1);
    return Transform{inverse_matrix.reshape({inverse_matrix.size(0), 2, 3})};
}

inline torch::Tensor get_pos_xy(const Transform& transform, const torch::Tensor& pos_xy) {
    auto rot = transform.inverse_matrix.index({Ellipsis, Slice(None, 2), Slice(None, 2)});
    auto trans = transform.inverse_matrix.index({Ellipsis, Slice(None, 2), 2});
    int batch_rank = rot.dim() - 2;
    int target_batch_rank = pos_xy.dim() - 1;
    rot = align_batch(rot, batch_rank, target_batch_rank);
    trans = align_batch(trans, batch_rank, target_batch_rank);
    return torch::matmul(pos_xy.unsqueeze(-2), rot.transpose(-1, -2)).squeeze(-2) + trans;
}

inline torch::Tensor get_dir_xy(const Transform& transform, const torch::Tensor& dir_xy) {
    auto rot = transform.inverse_matrix.index({Ellipsis, Slice(None, 2), Slice(None, 2)});
    int batch_rank = rot.dim() - 2;
    int target_batch_rank = dir_xy.dim() - 1;
    rot = align_batch(rot, batch_rank, target_batch_rank);
    return torch::matmul(dir_xy.unsqueeze(-2), rot.transpose(-1, -2)).squeeze(-2);
}

inline torch::Tensor get_pos_xy_world(const Transform& transform, const torch::Tensor& pos_xy_local) {
    auto rot = transform.inverse_matrix.index({Ellipsis, Slice(None, 2), Slice(None, 2)});
    auto trans = transform.inverse_matrix.index({Ellipsis, Slice(None, 2), 2});
    auto pos_origin = -torch::matmul(trans.unsqueeze(-2), rot).squeeze(-2);
    int batch_rank = rot.dim() - 2;
    int target_batch_rank = pos_xy_local.dim() - 1;
    rot = align_batch(rot, batch_rank, target_batch_rank);
    pos_origin = align_batch(pos_origin, batch_rank, target_batch_rank);
    return torch::matmul(pos_xy_local.unsqueeze(-2), rot).squeeze(-2) + pos_origin;
}

inline torch::Tensor get_dir_xy_world(const Transform& transform, const torch::Tensor& dir_xy_local) {
    auto rot = transform.inverse_matrix.index({Ellipsis, Slice(None, 2), Slice(None, 2)});
    int batch_rank = rot.dim() - 2;
    int target_batch_rank = dir_xy_local.dim() - 1;
    rot = align_batch(rot, batch_rank, target_batch_rank);
    return torch::matmul(dir_xy_local.unsqueeze(-2), rot).squeeze(-2);
}

inline std::pair<Transform, torch::Tensor> compute_scene_transform(const Scene& scene, int num_input_frames) {
    int t_now = num_input_frames - 1;
    auto ego_mask = scene.agents.is_ego & scene.agents.valid.index({Ellipsis, t_now});
    auto scene_valid = ego_mask.any(1);
    auto ego_idx = torch::argmax(ego_mask.to(torch::kFloat), 1);
    auto batch = torch::arange(scene.agents.pos_xy.size(0), ego_idx.options());
    auto origin_pos = scene.agents.pos_xy.index({batch, ego_idx, t_now});
    auto origin_dir = scene.agents.dir_xy.index({batch, ego_idx, t_now});
    return {transform_from_pose(origin_pos, origin_dir), scene_valid};
}

template <typename Store>
inline Store transform_posdir_store(Store store, const Transform& transform, const torch::Tensor& valid) {
    if (!store.valid.defined() || store.valid.numel() == 0) {
        return store;
    }
    auto new_valid = store.valid & valid.unsqueeze(1).unsqueeze(1);
    store.pos_xy = get_pos_xy(transform, store.pos_xy)
                 * new_valid.unsqueeze(-1).to(store.pos_xy.dtype());
    store.dir_xy = get_dir_xy(transform, store.dir_xy);
    store.valid = new_valid;
    return store;
}

inline Agent transform_agent_store(Agent store, const Transform& transform, const torch::Tensor& valid) {
    if (!store.valid.defined() || store.valid.numel() == 0) {
        return store;
    }
    auto new_valid = store.valid & valid.unsqueeze(1).unsqueeze(1);
    store.pos_xy = get_pos_xy(transform, store.pos_xy)
                 * new_valid.unsqueeze(-1).to(store.pos_xy.dtype());
    store.dir_xy = get_dir_xy(transform, store.dir_xy);
    store.vel_xy = get_dir_xy(transform, store.vel_xy);
    if (store.acc_xy.defined() && store.acc_xy.numel() > 0) {
        store.acc_xy = get_dir_xy(transform, store.acc_xy);
    }
    store.valid = new_valid;
    return store;
}

inline Traffic transform_traffic_store(Traffic store, const Transform& transform, const torch::Tensor& valid) {
    if (!store.valid.defined() || store.valid.numel() == 0) {
        return store;
    }
    auto new_valid = store.valid & valid.unsqueeze(1).unsqueeze(1);

    auto slot_valid = new_valid.any(2,  true).to(store.pos_xy.dtype());
    store.pos_xy = get_pos_xy(transform, store.pos_xy) * slot_valid;
    if (store.dir_xy.defined() && store.dir_xy.numel() > 0) {
        store.dir_xy = get_dir_xy(transform, store.dir_xy);
    }
    store.valid = new_valid;
    return store;
}

inline Scene transform_scene(Scene scene, const Transform& transform, const torch::Tensor& valid) {
    Transform t{transform.inverse_matrix.unsqueeze(1)};
    scene.agents = transform_agent_store(scene.agents, t, valid);
    scene.links = transform_posdir_store(scene.links, t, valid);
    scene.lanes = transform_posdir_store(scene.lanes, t, valid);
    scene.markers = transform_posdir_store(scene.markers, t, valid);
    scene.paths = transform_posdir_store(scene.paths, t, valid);
    scene.traffics = transform_traffic_store(scene.traffics, t, valid);
    return scene;
}

inline Output transform_output_to_global(Output output, const Transform& transform) {
    if (output.pos_xy.defined() && output.pos_xy.numel() > 0) {
        output.pos_xy = get_pos_xy_world(transform, output.pos_xy);
    }
    if (output.dir_xy.defined() && output.dir_xy.numel() > 0) {
        output.dir_xy = get_dir_xy_world(transform, output.dir_xy);
    }
    return output;
}

inline Agent recover_agent_dirs(Agent agents, double eps = 1e-6) {
    if (!agents.valid.defined() || agents.valid.numel() == 0) {
        return agents;
    }

    auto dir_xy = agents.dir_xy;
    auto pos_xy = agents.pos_xy;
    auto valid = agents.valid;

    auto dir_norm = torch::norm(dir_xy, 2, -1);
    auto zero_mask = (dir_norm < eps) & valid;

    auto diff = torch::zeros_like(pos_xy);
    diff.index_put_({Ellipsis, Slice(1, None), Slice()},
                    pos_xy.index({Ellipsis, Slice(1, None), Slice()})
                    - pos_xy.index({Ellipsis, Slice(None, -1), Slice()}));
    auto diff_norm = torch::norm(diff, 2, -1, true).clamp_min(eps);
    auto diff_unit = diff / diff_norm;
    auto w1 = diff_norm.squeeze(-1);

    auto neighbor_dir = torch::zeros_like(dir_xy);
    neighbor_dir.index_put_({Ellipsis, Slice(1, None), Slice()},
                            dir_xy.index({Ellipsis, Slice(None, -1), Slice()}));
    auto neighbor_norm = torch::norm(neighbor_dir, 2, -1, true).clamp_min(eps);
    auto neighbor_unit = neighbor_dir / neighbor_norm;
    auto w2 = neighbor_norm.squeeze(-1);

    auto w_sum = (w1 + w2).clamp_min(eps);
    auto result = (w1.unsqueeze(-1) * diff_unit + w2.unsqueeze(-1) * neighbor_unit) / w_sum.unsqueeze(-1);
    auto result_norm = torch::norm(result, 2, -1, true).clamp_min(eps);
    auto result_unit = result / result_norm;

    auto prev_valid = torch::zeros_like(valid);
    prev_valid.index_put_({Ellipsis, Slice(1, None)},
                          valid.index({Ellipsis, Slice(None, -1)}));
    auto is_first = valid & ~prev_valid & zero_mask;
    auto vel_norm = torch::norm(agents.vel_xy, 2, -1, true).clamp_min(eps);
    auto vel_unit = agents.vel_xy / vel_norm;
    result_unit = torch::where(is_first.unsqueeze(-1), vel_unit, result_unit);

    agents.dir_xy = torch::where(zero_mask.unsqueeze(-1), result_unit, dir_xy);
    return agents;
}

inline torch::Tensor gather_entities(const torch::Tensor& tensor, const torch::Tensor& indices) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return tensor;
    }
    auto idx = indices;
    for (int dim = 2; dim < tensor.dim(); ++dim) {
        idx = idx.unsqueeze(-1);
    }
    auto expand_sizes = indices.sizes().vec();
    for (int dim = 2; dim < tensor.dim(); ++dim) {
        expand_sizes.push_back(tensor.size(dim));
    }
    return tensor.gather(1, idx.expand(expand_sizes));
}


inline Output reorder_output_by_id(
    Output out, const torch::Tensor& src_id, const torch::Tensor& dst_id
) {
    if (!src_id.defined() || src_id.numel() == 0
        || !dst_id.defined() || dst_id.numel() == 0) {
        return out;
    }
    auto match = (dst_id.unsqueeze(2) == src_id.unsqueeze(1))
                 & (dst_id.unsqueeze(2) > 0);
    auto has = match.any(2);
    auto j = std::get<1>(match.to(torch::kInt).max(2));

    auto apply = [&](torch::Tensor& t) {
        if (!t.defined() || t.numel() == 0) return;
        t = gather_entities(t, j);
        auto mask = has;
        for (int d = 2; d < t.dim(); ++d) mask = mask.unsqueeze(-1);
        if (t.scalar_type() == torch::kBool) {
            t = t & mask;
        } else {
            t = torch::where(mask, t, torch::zeros_like(t));
        }
    };
    apply(out.pos_xy);
    apply(out.dir_xy);
    apply(out.size_xyz);
    apply(out.valid);
    apply(out.is_ego);
    apply(out.type);
    return out;
}

inline torch::Tensor pad_entities(const torch::Tensor& tensor, int64_t pad_size) {
    if (!tensor.defined()) {
        return tensor;
    }
    auto pad_shape = tensor.sizes().vec();
    pad_shape[1] = pad_size;
    return torch::cat({tensor, torch::zeros(pad_shape, tensor.options())}, 1);
}

inline Agent resize_agents(Agent agents, int num_agents) {
    if (num_agents <= 0 || !agents.valid.defined() || agents.valid.numel() == 0) {
        return agents;
    }

    auto B = agents.valid.size(0);
    auto A = agents.valid.size(1);
    if (A == num_agents) {
        return agents;
    }
    if (A > num_agents) {
        auto t_ref = agents.valid.size(2) / 2;
        auto ego_mask = agents.is_ego;
        auto ego_idx = ego_mask.to(torch::kFloat).argmax(1);
        auto bi = torch::arange(B, ego_idx.options());
        auto ego_pos = agents.pos_xy.index({bi, ego_idx, t_ref});
        auto agent_pos = agents.pos_xy.index({torch::indexing::Slice(), torch::indexing::Slice(), t_ref});
        auto dist = torch::norm(agent_pos - ego_pos.unsqueeze(1), 2, -1);
        dist.index_put_({ego_mask}, -1.0f);
        auto order = std::get<1>(dist.sort(1));
        auto selected = std::get<0>(order.index({Slice(), Slice(None, num_agents)}).sort(1));

        agents.id = gather_entities(agents.id, selected);
        agents.type = gather_entities(agents.type, selected);
        agents.time_s = gather_entities(agents.time_s, selected);
        agents.pos_xy = gather_entities(agents.pos_xy, selected);
        agents.dir_xy = gather_entities(agents.dir_xy, selected);
        agents.vel_xy = gather_entities(agents.vel_xy, selected);
        agents.acc_xy = gather_entities(agents.acc_xy, selected);
        agents.size_xyz = gather_entities(agents.size_xyz, selected);
        agents.is_ego = gather_entities(agents.is_ego, selected);
        agents.valid = gather_entities(agents.valid, selected);
        return agents;
    }

    int64_t pad_size = num_agents - A;
    agents.id = pad_entities(agents.id, pad_size);
    agents.type = pad_entities(agents.type, pad_size);
    agents.time_s = pad_entities(agents.time_s, pad_size);
    agents.pos_xy = pad_entities(agents.pos_xy, pad_size);
    agents.dir_xy = pad_entities(agents.dir_xy, pad_size);
    agents.vel_xy = pad_entities(agents.vel_xy, pad_size);
    agents.acc_xy = pad_entities(agents.acc_xy, pad_size);
    agents.size_xyz = pad_entities(agents.size_xyz, pad_size);
    agents.is_ego = pad_entities(agents.is_ego, pad_size);
    agents.valid = pad_entities(agents.valid, pad_size);
    return agents;
}

inline Traffic resize_traffics(Traffic traffics, int num_traffics, const torch::Tensor& ego_pos) {
    if (num_traffics <= 0 || !traffics.valid.defined() || traffics.valid.numel() == 0) {
        return traffics;
    }

    auto B = traffics.valid.size(0);
    auto T = traffics.valid.size(1);
    if (T == num_traffics) {
        return traffics;
    }
    if (T > num_traffics) {
        auto dist = torch::norm(traffics.pos_xy - ego_pos.unsqueeze(1), 2, -1);
        auto tl_valid = traffics.valid.any(2);
        dist.index_put_({~tl_valid}, std::numeric_limits<float>::infinity());
        auto order = std::get<1>(dist.sort(1));
        auto selected = std::get<0>(order.index({Slice(), Slice(None, num_traffics)}).sort(1));

        traffics.id = gather_entities(traffics.id, selected);
        traffics.type = gather_entities(traffics.type, selected);
        traffics.target_link = gather_entities(traffics.target_link, selected);
        traffics.dir_xy = gather_entities(traffics.dir_xy, selected);
        traffics.state = gather_entities(traffics.state, selected);
        traffics.time_s = gather_entities(traffics.time_s, selected);
        traffics.pos_xy = gather_entities(traffics.pos_xy, selected);
        traffics.valid = gather_entities(traffics.valid, selected);
        return traffics;
    }

    int64_t pad_size = num_traffics - T;
    traffics.id = pad_entities(traffics.id, pad_size);
    traffics.type = pad_entities(traffics.type, pad_size);
    traffics.target_link = pad_entities(traffics.target_link, pad_size);
    traffics.dir_xy = pad_entities(traffics.dir_xy, pad_size);
    traffics.state = pad_entities(traffics.state, pad_size);
    traffics.time_s = pad_entities(traffics.time_s, pad_size);
    traffics.pos_xy = pad_entities(traffics.pos_xy, pad_size);
    traffics.valid = pad_entities(traffics.valid, pad_size);
    return traffics;
}

template <typename PreprocessConfigLike>
inline std::pair<Scene, Transform> preprocess_scene(Scene scene, int num_input_frames, const PreprocessConfigLike& preprocess) {
    scene.agents = recover_agent_dirs(scene.agents);
    auto [transform, scene_valid] = compute_scene_transform(scene, num_input_frames);
    scene = transform_scene(scene, transform, scene_valid);

    if (preprocess.num_agents > 0) {
        scene.agents = resize_agents(scene.agents, preprocess.num_agents);
    }
    if (preprocess.num_traffics > 0 && scene.traffics.valid.defined() && scene.traffics.valid.numel() > 0) {
        auto t_ref = scene.agents.valid.size(2) / 2;
        auto ego_idx = scene.agents.is_ego.to(torch::kFloat).argmax(1);
        auto bi = torch::arange(scene.agents.pos_xy.size(0), ego_idx.options());
        auto ego_pos = scene.agents.pos_xy.index({bi, ego_idx, t_ref});
        scene.traffics = resize_traffics(scene.traffics, preprocess.num_traffics, ego_pos);
    }

    return {scene, transform};
}

}
