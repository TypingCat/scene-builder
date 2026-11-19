#include "scorer.h"
#include "planner.h"
#include "drivable_area.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>

using namespace torch::indexing;

namespace {


constexpr float EPS = 1e-6f;

constexpr float STOPPED_SPEED_THRESHOLD = 5e-3f;
constexpr float STOPPED_EGO_THRESHOLD = 5e-2f;
constexpr float FUTURE_COLLISION_HORIZON = 1.0f;

constexpr float DDC_COMPLIANCE_THRESHOLD = 2.0f;
constexpr float DDC_VIOLATION_THRESHOLD = 6.0f;
constexpr float DDC_HORIZON = 1.0f;
constexpr float DDC_LATERAL_GATE = 5.0f;

constexpr float LANE_KEEPING_DEVIATION_LIMIT = 0.5f;
constexpr float LANE_KEEPING_HORIZON = 2.0f;

constexpr float PROGRESS_DISTANCE_THRESHOLD = 5.0f;

constexpr float MAX_ABS_MAG_JERK = 8.37f;
constexpr float MAX_ABS_LAT_ACCEL = 4.89f;
constexpr float MAX_LON_ACCEL = 2.40f;
constexpr float MIN_LON_ACCEL = -4.05f;
constexpr float MAX_ABS_YAW_ACCEL = 1.93f;
constexpr float MAX_ABS_LON_JERK = 4.13f;
constexpr float MAX_ABS_YAW_RATE = 0.95f;
constexpr float HC_HISTORY_HORIZON = 1.5f;
constexpr double HC_REAR_AXLE_TO_CENTER_M = 1.461;


constexpr float NC_REAR_AXLE_TO_CENTER_RATIO = 1.461f / 5.176f;
constexpr int64_t LINK_SECTION_INTERSECTION = 1;
constexpr int64_t AGENT_TYPE_VEHICLE = 1;
constexpr int64_t AGENT_TYPE_PEDESTRIAN = 2;
constexpr int64_t AGENT_TYPE_CYCLIST = 3;

constexpr int64_t _MAX_NC_TTC_AGENTS = 32;
constexpr int64_t _OFFROAD_KNN_NEIGHBORS = 64;
constexpr float _OFFROAD_KNN_MAX_DIST = 10.0f;
constexpr float DAC_EDGE_GATE = 5.0f;

constexpr float _SENTINEL = 1e6f;


torch::Tensor heading_to_rot(const torch::Tensor& heading) {

    auto cos_h = torch::cos(heading);
    auto sin_h = torch::sin(heading);
    auto row0 = torch::stack({cos_h, -sin_h},  -1);
    auto row1 = torch::stack({sin_h, cos_h},  -1);
    return torch::stack({row0, row1},  -2);
}

torch::Tensor sat_signed_distance(
    const torch::Tensor& pos_a, const torch::Tensor& rot_a, const torch::Tensor& size_a,
    const torch::Tensor& pos_b, const torch::Tensor& rot_b, const torch::Tensor& size_b
) {

    auto rot_a_T = rot_a.transpose(-2, -1);
    auto BA_pos = torch::matmul(rot_a_T, (pos_b - pos_a).unsqueeze(-1)).squeeze(-1);
    auto BA_rot = torch::matmul(rot_a_T, rot_b);
    auto half_a = size_a * 0.5f;
    auto half_b = size_b * 0.5f;
    auto BA0 = BA_rot.index({"...", 0, 0}).abs() * half_b.index({"...", 0}) + BA_rot.index({"...", 0, 1}).abs() * half_b.index({"...", 1});
    auto BA1 = BA_rot.index({"...", 1, 0}).abs() * half_b.index({"...", 0}) + BA_rot.index({"...", 1, 1}).abs() * half_b.index({"...", 1});
    auto BA0_margin = BA_pos.index({"...", 0}).abs() - (BA0 + half_a.index({"...", 0}));
    auto BA1_margin = BA_pos.index({"...", 1}).abs() - (BA1 + half_a.index({"...", 1}));

    auto AB_pos = torch::matmul(BA_rot.transpose(-2, -1), BA_pos.unsqueeze(-1)).squeeze(-1);
    auto AB0 = BA_rot.index({"...", 0, 0}).abs() * half_a.index({"...", 0}) + BA_rot.index({"...", 1, 0}).abs() * half_a.index({"...", 1});
    auto AB1 = BA_rot.index({"...", 0, 1}).abs() * half_a.index({"...", 0}) + BA_rot.index({"...", 1, 1}).abs() * half_a.index({"...", 1});
    auto AB0_margin = AB_pos.index({"...", 0}).abs() - (AB0 + half_b.index({"...", 0}));
    auto AB1_margin = AB_pos.index({"...", 1}).abs() - (AB1 + half_b.index({"...", 1}));

    return torch::maximum(
        torch::maximum(BA0_margin, BA1_margin),
        torch::maximum(AB0_margin, AB1_margin)
    );
}

struct LaneSegments {
    torch::Tensor seg_a;
    torch::Tensor seg_b;
    torch::Tensor valid;
    int64_t M = 0;
};

LaneSegments build_lane_segments(const Lane& lanes) {
    auto pos = lanes.pos_xy;
    auto valid = lanes.valid;
    auto type = lanes.type;
    auto device = valid.device();
    int64_t B = valid.size(0);

    if (pos.numel() == 0 || pos.size(2) < 2) {
        auto z = torch::zeros({B, 0, 2}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto zv = torch::zeros({B, 0}, torch::TensorOptions().device(device).dtype(torch::kBool));
        return {z, z, zv, 0};
    }

    if (type.dim() < valid.dim()) {
        type = type.unsqueeze(-1).expand_as(valid);
    }
    auto is_barrier = (type == LaneTypeConst::boundary) | (type == LaneTypeConst::median);
    auto vm = valid & is_barrier;

    int64_t R = pos.size(1);
    int64_t W = pos.size(2);
    int64_t M = R * (W - 1);
    auto seg_a = pos.index({Slice(), Slice(), Slice(None, -1)}).reshape({B, M, 2});
    auto seg_b = pos.index({Slice(), Slice(), Slice(1, None)}).reshape({B, M, 2});
    auto seg_valid = (vm.index({Slice(), Slice(), Slice(None, -1)}) & vm.index({Slice(), Slice(), Slice(1, None)})).reshape({B, M});
    return {seg_a, seg_b, seg_valid, M};
}

torch::Tensor build_ego_corners(
    const torch::Tensor& pos,
    const torch::Tensor& dir_xy,
    const torch::Tensor& size_xyz);

LaneSegments compact_lane_segments(
    const LaneSegments& segs,
    const torch::Tensor& ego_pos,
    const torch::Tensor& ego_rot,
    const torch::Tensor& ego_size
) {
    if (segs.M <= 0) return segs;

    auto corners = build_ego_corners(
        ego_pos,
        ego_rot.index({"...", Slice(), 0}),
        ego_size
    );
    auto ego_min = corners.amin(torch::IntArrayRef({1, 2, 3}));
    auto ego_max = corners.amax(torch::IntArrayRef({1, 2, 3}));
    auto seg_min = torch::minimum(segs.seg_a, segs.seg_b);
    auto seg_max = torch::maximum(segs.seg_a, segs.seg_b);


    auto candidate = segs.valid
        & (seg_max.index({"...", 0}) >= ego_min.index({Slice(), None, 0}))
        & (seg_min.index({"...", 0}) <= ego_max.index({Slice(), None, 0}))
        & (seg_max.index({"...", 1}) >= ego_min.index({Slice(), None, 1}))
        & (seg_min.index({"...", 1}) <= ego_max.index({Slice(), None, 1}));

    auto max_candidates = candidate.sum( 1).amax().item<int64_t>();
    if (max_candidates == 0) {
        auto B = segs.seg_a.size(0);
        auto xy = segs.seg_a.new_empty({B, 0, 2});
        auto valid = candidate.new_empty({B, 0});
        return {xy, xy.clone(), valid, 0};
    }


    auto ranked = std::get<1>(candidate.to(torch::kInt8).topk(
        max_candidates,  1,  true,  false));
    auto gather_xy = ranked.unsqueeze(-1).expand({-1, -1, 2});
    return {
        torch::gather(segs.seg_a, 1, gather_xy),
        torch::gather(segs.seg_b, 1, gather_xy),
        torch::gather(candidate, 1, ranked),
        max_candidates,
    };
}

torch::Tensor segment_obb_collide(
    const torch::Tensor& ego_pos,
    const torch::Tensor& ego_rot,
    const torch::Tensor& ego_size,
    const torch::Tensor& seg_a,
    const torch::Tensor& seg_b,
    const torch::Tensor& seg_valid
) {
    auto rel_a = seg_a - ego_pos.unsqueeze(-2);
    auto rel_b = seg_b - ego_pos.unsqueeze(-2);
    auto rot_T = ego_rot.transpose(-1, -2).unsqueeze(-3);
    auto a_local = torch::matmul(rot_T, rel_a.unsqueeze(-1)).squeeze(-1);
    auto b_local = torch::matmul(rot_T, rel_b.unsqueeze(-1)).squeeze(-1);
    auto half = (ego_size * 0.5f).unsqueeze(-2);
    auto d = b_local - a_local;
    const float eps = 1e-12f;
    auto d_zero = d.abs() < eps;
    auto d_safe = torch::where(d_zero, torch::ones_like(d), d);
    auto inv_d = 1.0f / d_safe;
    auto t1 = (-half - a_local) * inv_d;
    auto t2 = (half - a_local) * inv_d;
    auto t_near = torch::minimum(t1, t2);
    auto t_far = torch::maximum(t1, t2);
    auto a_out = a_local.abs() > half;
    const float inf_f = std::numeric_limits<float>::infinity();
    auto pos_inf = torch::full_like(t_near, inf_f);
    auto neg_inf = torch::full_like(t_near, -inf_f);
    t_near = torch::where(d_zero, torch::where(a_out, pos_inf, neg_inf), t_near);
    t_far = torch::where(d_zero, torch::where(a_out, neg_inf, pos_inf), t_far);
    auto t_min = t_near.amax(-1);
    auto t_max = t_far.amin(-1);
    auto intersect = (t_min <= t_max) & (t_max >= 0.0f) & (t_min <= 1.0f);
    return intersect & seg_valid;
}

torch::Tensor build_ego_corners(
    const torch::Tensor& pos,
    const torch::Tensor& dir_xy,
    const torch::Tensor& size_xyz
) {

    auto d_norm = torch::norm(dir_xy, 2,  -1,  true).clamp_min(EPS);
    auto fwd = dir_xy / d_norm;
    auto lat = torch::stack({-fwd.index({"...", 1}), fwd.index({"...", 0})},  -1);
    auto half_L = size_xyz.index({"...", Slice(0, 1)}) * 0.5f;
    auto half_W = size_xyz.index({"...", Slice(1, 2)}) * 0.5f;
    return torch::stack({
        pos + fwd * half_L + lat * half_W,
        pos + fwd * half_L - lat * half_W,
        pos - fwd * half_L - lat * half_W,
        pos - fwd * half_L + lat * half_W,
    },  -2);
}

torch::Tensor phase_unwrap(const torch::Tensor& headings) {
    auto two_pi = 2.0f * static_cast<float>(3.14159265358979323846);
    auto adjustments = torch::zeros_like(headings);
    auto diff = torch::diff(headings,  1,  -1);
    adjustments.index_put_({"...", Slice(1, None)}, torch::cumsum(torch::round(diff / two_pi),  -1));
    return headings - two_pi * adjustments;
}

int savgol_window(float dt, float target_seconds = 1.5f) {
    int w = std::max(5, static_cast<int>(std::round(target_seconds / dt)));
    return (w % 2 == 1) ? w : w + 1;
}


struct SavgolKey {
    int window_length;
    int poly_order;
    int deriv;
    uint64_t delta_bits;
    int device_idx;
    int dtype;
    bool operator<(const SavgolKey& o) const {
        return std::tie(window_length, poly_order, deriv, delta_bits, device_idx, dtype)
             < std::tie(o.window_length, o.poly_order, o.deriv, o.delta_bits, o.device_idx, o.dtype);
    }
};

static std::map<SavgolKey, torch::Tensor> g_savgol_cache;
static std::mutex g_savgol_mtx;

static torch::Tensor get_savgol_kernel(
    int window_length, int poly_order, int deriv, float delta,
    const torch::Tensor& ref
) {
    uint64_t dbits = 0;
    std::memcpy(&dbits, &delta, sizeof(float));
    SavgolKey key{window_length, poly_order, deriv, dbits,
                  ref.is_cuda() ? ref.device().index() : -1,
                  static_cast<int>(ref.scalar_type())};

    std::lock_guard<std::mutex> lock(g_savgol_mtx);
    auto it = g_savgol_cache.find(key);
    if (it != g_savgol_cache.end()) return it->second;

    auto half_w = window_length / 2;
    auto opts = torch::TensorOptions().dtype(ref.dtype()).device(ref.device());

    auto j = torch::arange(-half_w, half_w + 1, opts);
    auto powers = torch::arange(poly_order + 1, opts);
    auto A = j.unsqueeze(1).pow(powers.unsqueeze(0));

    auto ATA = torch::matmul(A.t(), A);
    auto ATA_inv = torch::linalg_inv(ATA.to(torch::kFloat32)).to(ATA.dtype());
    auto pinv = torch::matmul(ATA_inv, A.t());

    float d_factorial = 1.0f;
    for (int i = 1; i <= deriv; ++i) d_factorial *= static_cast<float>(i);
    auto coeffs = pinv[deriv] * d_factorial / std::pow(delta, static_cast<float>(deriv));
    auto kernel = coeffs.flip(0).reshape({1, 1, -1}).contiguous();

    g_savgol_cache[key] = kernel;
    return kernel;
}


struct InterpSavgol {
    torch::Tensor dot;
    torch::Tensor edge_pinv;
    torch::Tensor design_low;
    torch::Tensor design_high;
};
static std::map<SavgolKey, InterpSavgol> g_interp_cache;

static InterpSavgol get_interp_savgol(
    int window_length, int poly_order, int deriv, float delta,
    const torch::Tensor& ref
) {
    uint64_t dbits = 0;
    std::memcpy(&dbits, &delta, sizeof(float));
    SavgolKey key{window_length, poly_order, deriv, dbits,
                  ref.is_cuda() ? ref.device().index() : -1,
                  static_cast<int>(ref.scalar_type())};

    std::lock_guard<std::mutex> lock(g_savgol_mtx);
    auto it = g_interp_cache.find(key);
    if (it != g_interp_cache.end()) return it->second;

    const int W = window_length;
    const int half = W / 2;
    auto opts = torch::TensorOptions().dtype(ref.dtype()).device(ref.device());
    auto powers = torch::arange(poly_order + 1, opts);


    double position = (W % 2) ? half : half - 0.5;
    auto sample_x = torch::arange(W, opts) - position;
    auto design = sample_x.unsqueeze(0).pow(powers.unsqueeze(1));
    auto target = (torch::arange(poly_order + 1, opts) == deriv).to(opts.dtype())
                * (std::tgamma(deriv + 1.0) / std::pow(delta, deriv));
    InterpSavgol out;
    out.dot = torch::linalg_pinv(design).matmul(target);

    auto edge_x = torch::arange(W, opts);
    out.edge_pinv = torch::linalg_pinv(edge_x.unsqueeze(1).pow(powers.unsqueeze(0)));

    auto build_design = [&](const torch::Tensor& eval_x) {
        auto d = torch::zeros({eval_x.numel(), poly_order + 1}, opts);
        for (int degree = deriv; degree <= poly_order; ++degree) {
            double scale = std::tgamma(degree + 1.0)
                         / std::tgamma(degree - deriv + 1.0)
                         / std::pow(delta, deriv);
            d.index_put_({Slice(), degree}, scale * eval_x.pow(degree - deriv));
        }
        return d;
    };
    out.design_low = build_design(torch::arange(half, opts));
    out.design_high = build_design(torch::arange(W - half, W, opts));

    g_interp_cache[key] = out;
    return out;
}


torch::Tensor savgol_filter_1d(
    const torch::Tensor& x,
    int window_length,
    int poly_order,
    int deriv = 0,
    float delta = 1.0f
) {
    auto T = x.size(-1);
    window_length = std::min(window_length, static_cast<int>(T));
    if (deriv > poly_order) return torch::zeros_like(x);

    auto orig_shape = x.sizes().vec();
    auto flat = x.reshape({-1, T});
    const int W = window_length;
    const int half = W / 2;
    auto coeff = get_interp_savgol(W, poly_order, deriv, delta, flat);

    auto windows = flat.unfold(-1, W, 1);
    auto interior = (windows * coeff.dot).sum(-1);
    auto output = torch::zeros_like(flat);
    int center_offset = (W - 1) / 2;
    output.index_put_({Slice(), Slice(center_offset, center_offset + interior.size(-1))},
                      interior);


    if (half > 0) {
        auto edge_values = [&](const torch::Tensor& samples, const torch::Tensor& design) {
            return design.matmul(coeff.edge_pinv.matmul(samples.t())).t();
        };
        output.index_put_({Slice(), Slice(None, half)},
                          edge_values(flat.index({Slice(), Slice(None, W)}), coeff.design_low));
        output.index_put_({Slice(), Slice(T - half, None)},
                          edge_values(flat.index({Slice(), Slice(T - W, None)}), coeff.design_high));
    }
    return output.reshape(orig_shape);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
pick_ego_route(const Path& paths) {
    auto pos = paths.pos_xy;
    auto dir_ = paths.dir_xy;
    auto valid = paths.valid;
    auto is_route = paths.is_route;
    auto near_link = paths.near_link;

    auto B = valid.size(0);
    auto device = pos.device();

    if (is_route.dim() >= valid.dim()) {
        is_route = is_route.any( -1);
    }

    auto route_count = (valid & is_route.unsqueeze(-1)).sum( -1);
    auto has_route = route_count.sum( -1) > 0;

    auto best_k = route_count.argmax( -1);
    auto bi = torch::arange(B, device);

    auto out_pos = pos.index({bi, best_k});
    auto out_dir = dir_.index({bi, best_k});
    auto out_valid = valid.index({bi, best_k}) & has_route.unsqueeze(-1);
    auto out_near = near_link.index({bi, best_k}).to(torch::kLong);
    return std::make_tuple(out_pos, out_dir, out_valid, out_near, has_route);
}


torch::Tensor link_in_route_mask(
    const torch::Tensor& query_ids,
    const torch::Tensor& route_near_link,
    const torch::Tensor& route_valid
) {
    auto match = (query_ids.unsqueeze(-1) == route_near_link.unsqueeze(1));
    match = match & route_valid.unsqueeze(1);
    auto query_valid = query_ids > 0;
    return match.any( -1) & query_valid;
}


std::tuple<torch::Tensor, torch::Tensor> nearest_link_point_dir(
    const torch::Tensor& query_pos,
    const torch::Tensor& link_pos,
    const torch::Tensor& link_dir,
    const torch::Tensor& link_valid,
    const torch::Tensor& link_mask
) {
    auto diff = query_pos.index({Slice(), Slice(), None, None, Slice()})
              - link_pos.index({Slice(), None, Slice(), Slice(), Slice()});
    auto dist2 = (diff * diff).sum( -1);

    auto valid_mask = link_valid.index({Slice(), None})
                    & link_mask.index({Slice(), None, Slice(), None});
    dist2 = dist2 + (~valid_mask).to(dist2.dtype()) * _SENTINEL;

    auto dist2_flat = dist2.flatten( 2);
    auto min_result = dist2_flat.min( -1);
    auto min_d2 = std::get<0>(min_result);
    auto min_idx = std::get<1>(min_result);

    auto link_dir_flat = link_dir.flatten( 1,  2);
    auto nearest_dir = link_dir_flat.gather(
        1, min_idx.unsqueeze(-1).expand({-1, -1, 2})
    );
    nearest_dir = nearest_dir / nearest_dir.norm(2, -1,  true).clamp_min(EPS);

    auto lateral = min_d2.clamp_min(EPS).sqrt();
    return {lateral, nearest_dir};
}


torch::Tensor lateral_dist_to_link_segments(
    const torch::Tensor& query_pos,
    const torch::Tensor& link_pos,
    const torch::Tensor& link_valid,
    const torch::Tensor& link_mask
) {
    auto seg_start = link_pos.index({Ellipsis, Slice(None, -1), Slice()});
    auto seg_end   = link_pos.index({Ellipsis, Slice(1, None), Slice()});
    auto seg_vec   = seg_end - seg_start;
    auto seg_sq    = (seg_vec * seg_vec).sum( -1).clamp_min(EPS);
    auto seg_valid = link_valid.index({Ellipsis, Slice(None, -1)})
                   & link_valid.index({Ellipsis, Slice(1, None)});
    auto seg_active = seg_valid & link_mask.unsqueeze(-1);

    auto q  = query_pos.index({Slice(), Slice(), None, None, Slice()});
    auto p0 = seg_start.unsqueeze(1);
    auto sv = seg_vec.unsqueeze(1);
    auto ssq = seg_sq.unsqueeze(1);

    auto diff = q - p0;
    auto t = ((diff * sv).sum( -1) / ssq).clamp(0.0f, 1.0f);
    auto closest = p0 + t.unsqueeze(-1) * sv;
    auto dist = (q - closest).norm(2,  -1);

    auto invalid = (~seg_active).unsqueeze(1).to(dist.dtype()) * _SENTINEL;
    dist = dist + invalid;
    return std::get<0>(dist.flatten( 2).min( -1));
}


std::tuple<torch::Tensor, torch::Tensor> project_onto_polyline(
    const torch::Tensor& points,
    const torch::Tensor& polyline,
    const torch::Tensor& poly_valid
) {
    auto B = points.size(0);
    auto N = points.size(1);
    auto M = polyline.size(1);
    auto device = points.device();

    if (M < 2) {
        return {torch::zeros({B, N}, device), torch::full({B, N}, _SENTINEL, device)};
    }

    auto seg_start = polyline.index({Slice(), Slice(None, -1)});
    auto seg_end = polyline.index({Slice(), Slice(1, None)});
    auto seg_vec = seg_end - seg_start;
    auto seg_len = seg_vec.norm(2, -1).clamp_min(EPS);
    auto seg_dir = seg_vec / seg_len.unsqueeze(-1);

    auto seg_valid = poly_valid.index({Slice(), Slice(None, -1)}) & poly_valid.index({Slice(), Slice(1, None)});

    auto cum_len = torch::zeros({B, M}, device);
    cum_len.index_put_({Slice(), Slice(1, None)}, seg_len.cumsum( 1));

    auto diff = points.unsqueeze(2) - seg_start.unsqueeze(1);
    auto t = (diff * seg_dir.unsqueeze(1)).sum( -1);
    auto t_clamped = torch::minimum(t.clamp_min(0.0f), seg_len.unsqueeze(1));

    auto closest = seg_start.unsqueeze(1) + seg_dir.unsqueeze(1) * t_clamped.unsqueeze(-1);
    auto dist = (points.unsqueeze(2) - closest).norm(2, -1);
    dist = dist.masked_fill(~seg_valid.unsqueeze(1), _SENTINEL);

    auto [min_dist, min_seg] = torch::min(dist,  2);

    auto bi = torch::arange(B, device).unsqueeze(1).expand({B, N});
    auto ni = torch::arange(N, device).unsqueeze(0).expand({B, N});
    auto arc_at_seg = cum_len.index({Slice(), Slice(None, -1)});
    auto arc_start = arc_at_seg.index({bi, min_seg});
    auto t_at_min = t_clamped.index({bi, ni, min_seg});

    return {arc_start + t_at_min, min_dist};
}


struct KnnFixed {
    torch::Tensor q_flat;
    torch::Tensor r_flat;
    torch::Tensor keep;
    int64_t k_actual = 0;
};


KnnFixed knn_bipartite_fixed(
    const torch::Tensor& src,
    const torch::Tensor& query,
    const torch::Tensor& valid_src,
    const torch::Tensor& valid_query,
    int k,
    float r = 0.0f
) {
    auto G = src.size(0);
    auto Ns = src.size(1);
    auto Nq = query.size(1);
    auto device = src.device();

    auto k_actual = std::min(k, static_cast<int>(Ns));
    if (k_actual <= 0) {
        auto empty_long = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(device));
        auto empty_bool = torch::empty({0}, torch::TensorOptions().dtype(torch::kBool).device(device));
        return {empty_long, empty_long, empty_bool, 0};
    }

    auto mask = valid_src.unsqueeze(-1).to(src.dtype());
    auto src_masked = src * mask;
    auto count = mask.sum(1,  true).clamp_min(1.0f);
    auto center = src_masked.sum(1,  true) / count;

    auto q_centered = query - center;
    auto s_centered = src - center;
    auto dist = torch::cdist(q_centered, s_centered);
    dist.masked_fill_(~valid_src.unsqueeze(1), INFINITY);
    dist.masked_fill_(~valid_query.unsqueeze(2), INFINITY);
    if (r > 0.0f) dist.masked_fill_(dist > r, INFINITY);

    auto [topk_dist, topk_idx] = torch::topk(dist, k_actual,  2,  false);

    auto src_offsets = torch::arange(G, torch::TensorOptions().dtype(torch::kLong).device(device)).reshape({-1, 1, 1}) * Ns;
    auto query_offsets = torch::arange(G, torch::TensorOptions().dtype(torch::kLong).device(device)).reshape({-1, 1, 1}) * Nq;
    auto nq_range = torch::arange(Nq, torch::TensorOptions().dtype(torch::kLong).device(device)).reshape({1, -1, 1});

    auto query_flat = (query_offsets + nq_range).expand({G, Nq, k_actual}).contiguous().reshape(-1);
    auto src_flat = (src_offsets + topk_idx.to(torch::kLong)).reshape(-1);

    auto topk_dist_flat = topk_dist.reshape(-1);
    auto keep = torch::isfinite(topk_dist_flat);
    if (r > 0.0f) keep = keep & (topk_dist_flat <= r);

    return {query_flat, src_flat, keep, k_actual};
}

torch::Tensor knn_bipartite_flat(
    const torch::Tensor& src,
    const torch::Tensor& query,
    const torch::Tensor& valid_src,
    const torch::Tensor& valid_query,
    int k,
    float r = 0.0f
) {
    auto G = src.size(0);
    auto Ns = src.size(1);
    auto Nq = query.size(1);
    auto device = src.device();
    auto empty = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kLong).device(device));

    auto k_actual = std::min(k, static_cast<int>(Ns));
    if (k_actual <= 0) return empty;


    auto mask = valid_src.unsqueeze(-1).to(src.dtype());
    auto src_masked = src * mask;
    auto count = mask.sum(1,  true).clamp_min(1.0f);
    auto center = src_masked.sum(1,  true) / count;

    auto q_centered = query - center;
    auto s_centered = src - center;
    auto dist = torch::cdist(q_centered, s_centered);
    dist.masked_fill_(~valid_src.unsqueeze(1), INFINITY);
    dist.masked_fill_(~valid_query.unsqueeze(2), INFINITY);
    if (r > 0.0f) dist.masked_fill_(dist > r, INFINITY);

    auto [topk_dist, topk_idx] = torch::topk(dist, k_actual,  2,  false);


    auto src_offsets = torch::arange(G, torch::TensorOptions().dtype(torch::kLong).device(device)).reshape({-1, 1, 1}) * Ns;
    auto query_offsets = torch::arange(G, torch::TensorOptions().dtype(torch::kLong).device(device)).reshape({-1, 1, 1}) * Nq;
    auto nq_range = torch::arange(Nq, torch::TensorOptions().dtype(torch::kLong).device(device)).reshape({1, -1, 1});

    auto query_flat = (query_offsets + nq_range).expand({G, Nq, k_actual});
    auto src_flat = src_offsets + topk_idx.to(torch::kLong);

    auto topk_dist_flat = topk_dist.reshape(-1);
    auto keep = torch::isfinite(topk_dist_flat);
    if (r > 0.0f) keep = keep & (topk_dist_flat <= r);

    auto q_out = query_flat.reshape(-1).index({keep});
    auto s_out = src_flat.reshape(-1).index({keep});

    if (q_out.numel() == 0) return empty;
    return torch::stack({q_out, s_out}, 0);
}

struct DacRoadCache {
    torch::Tensor road_pos_flat;
    torch::Tensor road_dir_flat;
    torch::Tensor edge_valid_flat;
    torch::Tensor interior_ref;
    torch::Tensor has_ref;
    int64_t B;
    int64_t M_road;
};

std::optional<DacRoadCache> precompute_dac_road_cache(
    const Link& links,
    const Lane& lanes,
    const torch::Tensor& ego_center
) {
    bool has_links = links.valid.numel() > 0;
    bool has_lanes = lanes.valid.numel() > 0;
    if (!has_links && !has_lanes) return std::nullopt;

    std::vector<torch::Tensor> parts_pos, parts_dir, parts_valid, parts_edge, parts_interior;

    if (has_links) {
        parts_pos.push_back(links.pos_xy);
        parts_dir.push_back(links.dir_xy);
        parts_valid.push_back(links.valid);
        parts_edge.push_back(torch::zeros_like(links.valid));
        parts_interior.push_back(links.valid);
    }

    if (has_lanes) {
        parts_pos.push_back(lanes.pos_xy);
        parts_dir.push_back(lanes.dir_xy);
        parts_valid.push_back(lanes.valid);
        auto lane_type = lanes.type;
        if (lane_type.dim() < lanes.valid.dim()) {
            lane_type = lane_type.unsqueeze(-1).expand_as(lanes.valid);
        }
        auto is_edge = LaneTypeConst::is_edge(lane_type) & lanes.valid;
        parts_edge.push_back(is_edge);
        parts_interior.push_back(~is_edge & lanes.valid);
    }

    auto road_pos = torch::cat(parts_pos,  1);
    auto road_dir = torch::cat(parts_dir,  1);
    auto road_valid = torch::cat(parts_valid,  1);
    auto edge_valid = torch::cat(parts_edge,  1);
    auto interior_valid = torch::cat(parts_interior,  1);

    if (road_pos.numel() == 0) return std::nullopt;


    if (ego_center.numel() > 0) {
        auto B_r = road_pos.size(0);
        auto R = road_pos.size(1);
        auto W = road_pos.size(2);
        if (R > 32) {
            auto entity_has_valid = road_valid.any( 2);
            auto masked_pos = road_pos * road_valid.unsqueeze(-1).to(road_pos.dtype());
            auto valid_count = road_valid.sum( 2).clamp_min(1).unsqueeze(-1);
            auto entity_center = masked_pos.sum( 2) / valid_count;
            auto dist_e = (entity_center - ego_center.unsqueeze(1)).norm(2, -1);
            dist_e = dist_e.masked_fill(~entity_has_valid, INFINITY);
            int64_t K = std::min(static_cast<int64_t>(32), R);
            auto [_, indices] = torch::topk(dist_e, K,  1,  false);
            indices = std::get<0>(torch::sort(indices,  1));
            auto idx_4d = indices.unsqueeze(-1).unsqueeze(-1).expand({-1, -1, W, 2});
            auto idx_3d = indices.unsqueeze(-1).expand({-1, -1, W});
            road_pos = road_pos.gather(1, idx_4d);
            road_dir = road_dir.gather(1, idx_4d);
            road_valid = road_valid.gather(1, idx_3d);
            edge_valid = edge_valid.gather(1, idx_3d);
            interior_valid = interior_valid.gather(1, idx_3d);
        }
    }

    auto B = road_valid.size(0);
    auto R = road_valid.size(1);
    auto W = road_valid.size(2);
    auto device = road_pos.device();

    int64_t M_road = R * W;
    auto road_pos_flat = road_pos.reshape({B, M_road, 2});
    auto road_dir_flat = road_dir.reshape({B, M_road, 2});
    auto edge_valid_flat = edge_valid.reshape({B, M_road});
    auto interior_valid_flat = interior_valid.reshape({B, M_road});

    auto road_pos_all = road_pos_flat.reshape({-1, 2});
    auto interior_ref = torch::zeros({B * M_road, 2}, torch::TensorOptions().device(device).dtype(road_pos.dtype()));
    auto has_ref = torch::zeros({B * M_road}, torch::TensorOptions().device(device).dtype(torch::kBool));

    auto ref_knn = knn_bipartite_fixed(
        road_pos_flat, road_pos_flat,
        interior_valid_flat, edge_valid_flat,
        1, _OFFROAD_KNN_MAX_DIST
    );

    if (ref_knn.k_actual > 0) {
        auto pos_i = road_pos_all.index_select(0, ref_knn.r_flat);
        auto pos_e = road_pos_all.index_select(0, ref_knn.q_flat);
        auto ref_vec = pos_i - pos_e;
        auto ref_norm = torch::norm(ref_vec, 2,  -1,  true).clamp_min(EPS);
        auto ref_unit = ref_vec / ref_norm;


        auto ref_unit_m = torch::where(ref_knn.keep.unsqueeze(-1), ref_unit, torch::zeros_like(ref_unit));
        auto q_expanded = ref_knn.q_flat.unsqueeze(-1).expand({-1, 2});
        interior_ref.scatter_(0, q_expanded, ref_unit_m);
        has_ref.scatter_(0, ref_knn.q_flat, ref_knn.keep);
    }

    return DacRoadCache{road_pos_flat, road_dir_flat, edge_valid_flat, interior_ref, has_ref, B, M_road};
}


torch::Tensor compute_ego_progress(
    const torch::Tensor& ego_pos_future,
    const torch::Tensor& center_pos,
    const torch::Tensor& center_valid
) {
    auto B = ego_pos_future.size(0);
    auto device = ego_pos_future.device();

    if (center_pos.size(1) < 2) return torch::zeros({B}, device);

    auto start_end = torch::cat({
        ego_pos_future.index({Slice(), Slice(0, 1)}),
        ego_pos_future.index({Slice(), Slice(-1, None)})
    },  1);
    auto [arc, _] = project_onto_polyline(start_end, center_pos, center_valid);
    return (arc.index({Slice(), 1}) - arc.index({Slice(), 0})).clamp_min(0.0f);
}


torch::Tensor compute_ddc(
    const torch::Tensor& all_pos,
    const torch::Tensor& all_valid,
    const torch::Tensor& on_route,
    const torch::Tensor& in_intersection,
    const torch::Tensor& has_ddc,
    float dt
) {
    auto B = all_pos.size(0);
    auto P = all_pos.size(1);
    auto S1 = all_pos.size(2);
    auto device = all_pos.device();

    auto is_oncoming = ~on_route & all_valid;

    auto oncoming_progress = torch::zeros({B, P, S1}, device);
    auto segment_valid = all_valid.index({Slice(), Slice(), Slice(1, None)})
                       & all_valid.index({Slice(), Slice(), Slice(None, -1)});
    auto step = (all_pos.index({Slice(), Slice(), Slice(1, None)})
               - all_pos.index({Slice(), Slice(), Slice(None, -1)})).norm(2, -1);
    oncoming_progress.index_put_({Slice(), Slice(), Slice(1, None)},
                                 torch::where(segment_valid, step, torch::zeros_like(step)));


    oncoming_progress = torch::where(is_oncoming & ~in_intersection,
                                     oncoming_progress, torch::zeros_like(oncoming_progress));

    int horizon_steps = std::max(0, static_cast<int>(DDC_HORIZON / dt));
    namespace F = torch::nn::functional;
    auto padded = F::pad(oncoming_progress, F::PadFuncOptions({horizon_steps, 0}));
    auto windows = padded.unfold( -1,  horizon_steps + 1,  1);
    auto max_oncoming = std::get<0>(windows.sum( -1).max( -1));

    auto scores = torch::ones({B, P}, device);
    scores = torch::where(max_oncoming >= DDC_VIOLATION_THRESHOLD, torch::zeros_like(scores), scores);
    scores = torch::where(
        (max_oncoming >= DDC_COMPLIANCE_THRESHOLD) & (max_oncoming < DDC_VIOLATION_THRESHOLD),
        torch::full_like(scores, 0.5f), scores
    );

    return torch::where(has_ddc.unsqueeze(-1).expand({B, P}), scores, torch::zeros_like(scores));
}


torch::Tensor compute_lk(
    const torch::Tensor& all_pos,
    const torch::Tensor& all_valid,
    const torch::Tensor& all_intersection,
    const Path& paths,
    float dt
) {
    auto B = all_pos.size(0);
    auto P = all_pos.size(1);
    auto S = all_pos.size(2);
    auto device = all_pos.device();

    if (paths.valid.numel() == 0) {
        return torch::ones({B, P}, device);
    }


    auto [route_pos, route_dir, route_valid, near_link_route, has_route] = pick_ego_route(paths);
    (void)route_dir; (void)near_link_route;
    auto has_lk = has_route
                & (route_valid.index({Slice(), Slice(None, -1)})
                   & route_valid.index({Slice(), Slice(1, None)})).any( -1);

    auto ego_flat = all_pos.reshape({B, P * S, 2});
    auto lateral_flat = lateral_dist_to_link_segments(
        ego_flat, route_pos.unsqueeze(1), route_valid.unsqueeze(1),
        torch::ones({B, 1}, torch::TensorOptions().dtype(torch::kBool).device(device))
    );

    lateral_flat = torch::where(
        has_lk.unsqueeze(-1), lateral_flat, torch::zeros_like(lateral_flat)
    );
    auto lateral_dist = lateral_flat.reshape({B, P, S});

    auto exceeds = lateral_dist > LANE_KEEPING_DEVIATION_LIMIT;
    int continuous_steps_required = std::max(1, static_cast<int>(std::ceil(LANE_KEEPING_HORIZON / dt)));


    auto active = all_valid & ~all_intersection;
    auto active_exceeds = active & exceeds;
    auto active_resets = active & ~exceeds;

    auto cumsum_exc = active_exceeds.to(torch::kLong).cumsum( -1);
    auto cumsum_at_reset = torch::where(active_resets, cumsum_exc, torch::zeros_like(cumsum_exc));
    auto max_at_reset = std::get<0>(torch::cummax(cumsum_at_reset,  -1));
    auto consecutive = cumsum_exc - max_at_reset;

    auto max_consecutive = std::get<0>(consecutive.max( -1));
    auto scores = (max_consecutive < continuous_steps_required).to(torch::kFloat32);

    return torch::where(
        has_lk.unsqueeze(-1).expand({B, P}), scores, torch::zeros_like(scores)
    );
}


torch::Tensor compute_comfort(
    const torch::Tensor& pos_full,
    const torch::Tensor& heading_full,
    int64_t num_past_frames,
    float dt
) {
    auto B = pos_full.size(0);
    auto device = pos_full.device();
    if (pos_full.size(1) < 3) return torch::ones({B}, device);


    auto pos = pos_full.to(torch::kFloat64);
    auto vel = torch::diff(pos, 1, 1, pos.index({Slice(), Slice(0, 1)})) / dt;
    auto accel = torch::diff(vel, 1, 1, vel.index({Slice(), Slice(0, 1)})) / dt;
    accel.index_put_({Slice(), 0}, accel.index({Slice(), 1}));

    auto heading = phase_unwrap(heading_full).to(torch::kFloat64);
    auto heading_delta = torch::diff(heading, 1, 1, heading.index({Slice(), Slice(0, 1)}));
    heading_delta = torch::remainder(heading_delta + 3.14159265358979323846, 2.0 * 3.14159265358979323846) - 3.14159265358979323846;
    auto yaw_rate = heading_delta / dt;
    auto yaw_accel = torch::diff(yaw_rate, 1, 1, yaw_rate.index({Slice(), Slice(0, 1)})) / dt;


    auto steps = static_cast<int64_t>(std::lround(HC_HISTORY_HORIZON / dt));
    auto start = std::max<int64_t>(0, (num_past_frames - 1) - steps);
    auto stop = std::max<int64_t>(start, num_past_frames - 2);
    auto window = [&](const torch::Tensor& t) {
        return torch::cat({t.index({Slice(), Slice(start, stop)}),
                           t.index({Slice(), Slice(num_past_frames - 1, None)})}, 1);
    };
    accel = window(accel);
    heading = window(heading);
    yaw_rate = window(yaw_rate);
    yaw_accel = window(yaw_accel);

    auto direction = torch::stack({heading.cos(), heading.sin()}, -1);
    auto rear_lon = (accel * direction).sum(-1);

    auto center_lon = rear_lon + HC_REAR_AXLE_TO_CENTER_M * (yaw_rate.square() + yaw_accel);

    auto lon_accel = savgol_filter_1d(center_lon, 8, 2);
    auto rear_accel_magnitude = savgol_filter_1d(rear_lon.abs(), 8, 2);
    auto jerk_mag = savgol_filter_1d(rear_accel_magnitude, 15, 2, 1, dt);
    auto lon_jerk = savgol_filter_1d(lon_accel, 15, 2, 1, dt);


    auto yaw_rate_s = savgol_filter_1d(heading, 5, 2, 1, dt);
    auto yaw_accel_s = savgol_filter_1d(heading, 5, 3, 2, dt);


    auto comfortable = (
        (lon_accel > MIN_LON_ACCEL) & (lon_accel < MAX_LON_ACCEL)
        & (jerk_mag.abs() < MAX_ABS_MAG_JERK)
        & (lon_jerk.abs() < MAX_ABS_LON_JERK)
        & (yaw_accel_s.abs() < MAX_ABS_YAW_ACCEL)
        & (yaw_rate_s.abs() < MAX_ABS_YAW_RATE)
    );

    return comfortable.all( -1).to(torch::kFloat32);
}


struct DacBlockResult {
    torch::Tensor score;
    torch::Tensor offroad;
    torch::Tensor multiple_lanes;
    torch::Tensor in_intersection;
    torch::Tensor t0_in_intersection;

    torch::Tensor ddc_on_route;
    torch::Tensor ddc_intersection;
    torch::Tensor has_ddc;
    torch::Tensor tlc_violation;
};


DacBlockResult compute_dac_block(
    int64_t B, int64_t P, int64_t S_fut,
    const torch::Tensor& ego_pos_p,
    const torch::Tensor& ego_dir_p,
    const torch::Tensor& ego_size_p,
    const Scene& scene,
    const DrivableAreaConfig& area_cfg,
    int64_t configs_time_now,
    c10::Device device
) {
    auto ego_center_dac = ego_pos_p.mean(torch::IntArrayRef({1, 2}));
    auto polygons = build_drivable_area_polygons(
        scene.links, scene.lanes, ego_center_dac, area_cfg);

    DacBlockResult out;
    if (polygons.defined && polygons.vertices.size(0) == B) {
        auto corners = build_ego_corners(ego_pos_p, ego_dir_p, ego_size_p);
        auto query = corners.reshape({B, P * S_fut * 4, 2});
        auto membership = points_in_polygons(
            query, polygons.vertices, polygons.valid, area_cfg.polygon_chunk);
        auto contained = membership.any(-1).reshape({B, P, S_fut, 4});
        auto offroad = ~contained.all(-1);


        auto corner_count = membership.reshape({B, P, S_fut, 4, -1}).sum(3);
        out.multiple_lanes = ((corner_count > 0).sum(-1) > 1)
                           & (corner_count != 4).all(-1);
        out.score = (~offroad.any( -1)).to(torch::kFloat32);
        out.offroad = offroad;


        auto rear = ego_pos_p - ego_dir_p * ego_size_p.index({"...", Slice(None, 1)})
                                          * NC_REAR_AXLE_TO_CENTER_RATIO;
        auto rear_membership = points_in_polygons(
            rear.reshape({B, P * S_fut, 2}),
            polygons.vertices, polygons.valid, area_cfg.polygon_chunk);
        auto is_intersection = (polygons.link_section == LINK_SECTION_INTERSECTION)
                             & polygons.valid.any(-1);
        out.in_intersection = (rear_membership & is_intersection.unsqueeze(1))
                                  .any(-1).reshape({B, P, S_fut});


        auto bi_t0 = torch::arange(B, device);
        auto ego_idx_t0 = scene.agents.is_ego.to(torch::kFloat32).argmax(1);
        auto t0 = configs_time_now;
        auto ego_pos_t0 = scene.agents.pos_xy.index({bi_t0, ego_idx_t0, t0});
        auto ego_dir_t0 = scene.agents.dir_xy.index({bi_t0, ego_idx_t0, t0});
        auto ego_len_t0 = scene.agents.size_xyz.index({bi_t0, ego_idx_t0, 0}).unsqueeze(-1);
        auto rear_t0 = ego_pos_t0 - ego_dir_t0 * ego_len_t0 * NC_REAR_AXLE_TO_CENTER_RATIO;
        auto t0_membership = points_in_polygons(
            rear_t0.unsqueeze(1), polygons.vertices, polygons.valid, area_cfg.polygon_chunk);
        out.t0_in_intersection = (t0_membership & is_intersection.unsqueeze(1))
                                     .any(-1).squeeze(1);


        auto [r_pos, r_dir, r_valid, r_near_link, r_has] = pick_ego_route(scene.paths);
        (void)r_pos; (void)r_dir;
        auto route_polygons = polygons_match_link_ids(polygons, r_near_link, r_valid);
        out.has_ddc = r_has & route_polygons.any(-1);
        auto centre_states = torch::cat(
            {ego_pos_t0.unsqueeze(1).unsqueeze(1).expand({B, P, 1, 2}), ego_pos_p},  2);
        auto centre_membership = points_in_polygons(
            centre_states.reshape({B, P * (S_fut + 1), 2}),
            polygons.vertices, polygons.valid, area_cfg.polygon_chunk);
        out.ddc_on_route = (centre_membership & route_polygons.unsqueeze(1))
                               .any(-1).reshape({B, P, S_fut + 1});
        out.ddc_intersection = (centre_membership & is_intersection.unsqueeze(1))
                                   .any(-1).reshape({B, P, S_fut + 1});


        out.tlc_violation = torch::zeros({B, P}, torch::TensorOptions().dtype(torch::kBool).device(device));
        if (scene.traffics.valid.defined() && scene.traffics.valid.numel() > 0
            && scene.traffics.target_link.defined() && scene.traffics.target_link.numel() > 0) {
            auto target_link = scene.traffics.target_link.to(polygons.link_id.scalar_type());
            auto traffic_to_route = (target_link.unsqueeze(-1) == polygons.link_id.unsqueeze(1))
                                  & (target_link.unsqueeze(-1) > 0)
                                  & route_polygons.unsqueeze(1)
                                  & polygons.valid.any(-1).unsqueeze(1);

            auto dir_states = torch::cat(
                {ego_dir_t0.unsqueeze(1).unsqueeze(1).expand({B, P, 1, 2}), ego_dir_p},  2);
            auto size_states = torch::cat(
                {ego_size_p.index({Slice(), Slice(), Slice(0, 1)}), ego_size_p},  2);
            auto corners = build_ego_corners(centre_states, dir_states, size_states);
            auto ego_hits = polygons_intersect_boxes(
                corners.reshape({B, P * (S_fut + 1), 4, 2}),
                polygons.vertices, polygons.valid, area_cfg.polygon_chunk)
                .reshape({B, P, S_fut + 1, -1});

            auto steps = scene.traffics.state.size(-1);
            auto idx = configs_time_now + torch::arange(S_fut + 1, device);
            auto in_range = (idx >= 0) & (idx < steps);
            idx = idx.clamp(0, std::max<int64_t>(steps - 1, 0));
            auto red = (scene.traffics.state.index_select(-1, idx) == STOP_STATE)
                     & scene.traffics.valid.index_select(-1, idx)
                     & in_range.unsqueeze(0).unsqueeze(0);
            auto active_red = (red.permute({0, 2, 1}).unsqueeze(-1)
                               & traffic_to_route.unsqueeze(1)).any(2);
            out.tlc_violation = (ego_hits & active_red.unsqueeze(1)).any(-1).any(-1);
        }
    } else {
        auto zeros_b = torch::zeros({B, P, S_fut}, torch::TensorOptions().dtype(torch::kBool).device(device));
        out.score = torch::ones({B, P}, device);
        out.offroad = zeros_b;
        out.multiple_lanes = zeros_b.clone();
        out.in_intersection = zeros_b.clone();
        auto zeros_b1 = torch::zeros({B, P, S_fut + 1}, torch::TensorOptions().dtype(torch::kBool).device(device));
        out.t0_in_intersection = torch::zeros({B}, torch::TensorOptions().dtype(torch::kBool).device(device));
        out.ddc_on_route = zeros_b1;
        out.ddc_intersection = zeros_b1.clone();
        out.has_ddc = torch::zeros({B}, torch::TensorOptions().dtype(torch::kBool).device(device));
        out.tlc_violation = torch::zeros({B, P}, torch::TensorOptions().dtype(torch::kBool).device(device));
    }
    return out;
}

}


namespace {


ScorerResult score_proposals_impl(
    const torch::Tensor& proposals,
    const Scene& scene,
    const PlannerConfig& configs,
    const ScorerPrep* prep_override,
    bool first_at_fault_only
) {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;

    auto tc = configs.time;
    auto dt = configs.frame_unit;
    auto B = proposals.size(0);
    auto P = proposals.size(1);
    auto S_fut = proposals.size(2);
    auto BP = B * P;
    auto device = proposals.device();

    auto now = tc.now();
    auto num_past_frames = tc.num_past_frames;
    auto A = scene.agents.pos_xy.size(1);


    auto ego_mask_ba = scene.agents.is_ego;
    auto ego_idx = ego_mask_ba.to(torch::kFloat32).argmax( 1);
    auto bi = torch::arange(B, device);

    auto ego_past_pos = scene.agents.pos_xy.index({bi, ego_idx}).index({Slice(), Slice(None, num_past_frames)});
    auto ego_size_2d = scene.agents.size_xyz.index({bi, ego_idx}).index({Slice(), Slice(None, 2)});


    auto ego_past_dir = scene.agents.dir_xy.index({bi, ego_idx}).index({Slice(), Slice(None, num_past_frames)});
    auto ego_past_heading = torch::atan2(ego_past_dir.index({"...", 1}), ego_past_dir.index({"...", 0}));
    auto ego_past_vel = scene.agents.vel_xy.index({bi, ego_idx}).index({Slice(), Slice(None, num_past_frames)});
    auto ego_past_speed = (ego_past_vel * ego_past_dir).sum( -1).clamp_min(0.0f);


    auto cv = predict_agents_cv(scene, tc, dt);
    auto all_pos_f = cv.pos.index({Slice(), Slice(), Slice(None, S_fut)});
    auto all_dir_f = cv.dir.index({Slice(), Slice(), Slice(None, S_fut)});
    auto all_valid_f = cv.valid.index({Slice(), Slice(), Slice(None, S_fut)});
    auto all_type_f = scene.agents.type.unsqueeze(-1).expand({-1, -1, S_fut});
    auto all_heading_f = torch::atan2(all_dir_f.index({"...", 1}), all_dir_f.index({"...", 0}));
    auto all_size_2d = scene.agents.size_xyz.index({Slice(), Slice(), Slice(None, 2)});


    auto all_speed_now = scene.agents.vel_xy.index({Slice(), Slice(), now}).norm(2, -1);
    auto all_is_agent = (scene.agents.type == AGENT_TYPE_VEHICLE)
                      | (scene.agents.type == AGENT_TYPE_PEDESTRIAN)
                      | (scene.agents.type == AGENT_TYPE_CYCLIST);


    if (A > _MAX_NC_TTC_AGENTS) {
        auto ref_pos = ego_past_pos.index({Slice(), -1});
        auto valid_any = all_valid_f.any( 2);
        auto valid_count = all_valid_f.sum( 2,  true).clamp_min(1);
        auto agent_mean = (all_pos_f * all_valid_f.unsqueeze(-1).to(all_pos_f.dtype())).sum( 2) / valid_count;
        auto dists = torch::cdist(ref_pos.unsqueeze(1), agent_mean).squeeze(1);
        dists.masked_fill_(~valid_any, INFINITY);
        dists.masked_fill_(ego_mask_ba, -INFINITY);
        auto [_, topk_idx] = torch::topk(dists, _MAX_NC_TTC_AGENTS,  1,  false);
        auto bi_k = bi.unsqueeze(1).expand_as(topk_idx);
        all_pos_f = all_pos_f.index({bi_k, topk_idx});
        all_dir_f = all_dir_f.index({bi_k, topk_idx});
        all_valid_f = all_valid_f.index({bi_k, topk_idx});
        all_type_f = all_type_f.index({bi_k, topk_idx});
        all_heading_f = all_heading_f.index({bi_k, topk_idx});
        all_size_2d = all_size_2d.index({bi_k, topk_idx});
        all_speed_now = all_speed_now.index({bi_k, topk_idx});
        all_is_agent = all_is_agent.index({bi_k, topk_idx});
        ego_mask_ba = ego_mask_ba.index({bi_k, topk_idx});
        A = _MAX_NC_TTC_AGENTS;
    }


    auto ego_pos_p = proposals.index({"...", Slice(None, 2)});
    auto ego_heading_p = proposals.index({"...", 2});
    auto ego_speed_p = proposals.index({"...", 3});
    auto ego_dir_p = torch::stack({ego_heading_p.cos(), ego_heading_p.sin()},  -1);
    auto ego_size_p = ego_size_2d.unsqueeze(1).unsqueeze(2).expand({-1, P, S_fut, -1});


    torch::Tensor dac_score, dac_offroad, dac_multiple_lanes, dac_in_intersection, dac_t0_in_intersection;
    torch::Tensor dac_ddc_on_route, dac_ddc_intersection, dac_has_ddc, dac_tlc_violation;
    if (prep_override) {
        dac_score = prep_override->dac_score;
        dac_offroad = prep_override->dac_offroad;
        dac_multiple_lanes = prep_override->multiple_lanes;
        dac_in_intersection = prep_override->in_intersection;
        dac_t0_in_intersection = prep_override->t0_in_intersection;
        dac_ddc_on_route = prep_override->ddc_on_route;
        dac_ddc_intersection = prep_override->ddc_intersection;
        dac_has_ddc = prep_override->has_ddc;
        dac_tlc_violation = prep_override->tlc_violation;
    } else {
        auto dac_r = compute_dac_block(B, P, S_fut, ego_pos_p, ego_dir_p, ego_size_p, scene, configs.drivable_area, now, device);
        dac_score = dac_r.score;
        dac_offroad = dac_r.offroad;
        dac_multiple_lanes = dac_r.multiple_lanes;
        dac_in_intersection = dac_r.in_intersection;
        dac_t0_in_intersection = dac_r.t0_in_intersection;
        dac_ddc_on_route = dac_r.ddc_on_route;
        dac_ddc_intersection = dac_r.ddc_intersection;
        dac_has_ddc = dac_r.has_ddc;
        dac_tlc_violation = dac_r.tlc_violation;
    }

    auto lateral_fault_area = dac_offroad | dac_multiple_lanes;


    torch::Tensor nc_score, first_at_fault_step;
    torch::Tensor has_any_collision_bpa, at_fault_first_bpa;

    if (A > 0 && S_fut > 0) {
        auto other_pos_sa = all_pos_f.permute({0, 2, 1, 3});
        auto other_heading_sa = all_heading_f.permute({0, 2, 1});
        auto other_size_sa = all_size_2d.unsqueeze(2).expand({-1, -1, S_fut, -1}).permute({0, 2, 1, 3});
        auto other_valid_sa = (all_valid_f & ~ego_mask_ba.unsqueeze(-1)).permute({0, 2, 1});

        auto ego_pos_sp = ego_pos_p.permute({0, 2, 1, 3});
        auto ego_heading_sp = ego_heading_p.permute({0, 2, 1});
        auto ego_size_sp = ego_size_p.permute({0, 2, 1, 3});
        auto ego_dir_sp = ego_dir_p.permute({0, 2, 1, 3});
        auto ego_speed_sp = ego_speed_p.permute({0, 2, 1});

        auto ego_rot_sp = heading_to_rot(ego_heading_sp);
        auto other_rot_sa = heading_to_rot(other_heading_sa);

        auto sat_dist = sat_signed_distance(
            ego_pos_sp.unsqueeze(3).expand({-1, -1, -1, A, -1}),
            ego_rot_sp.unsqueeze(3).expand({-1, -1, -1, A, -1, -1}),
            ego_size_sp.unsqueeze(3).expand({-1, -1, -1, A, -1}),
            other_pos_sa.unsqueeze(2).expand({-1, -1, P, -1, -1}),
            other_rot_sa.unsqueeze(2).expand({-1, -1, P, -1, -1, -1}),
            other_size_sa.unsqueeze(2).expand({-1, -1, P, -1, -1})
        );


        auto colliding = (sat_dist <= 0.0f) & other_valid_sa.unsqueeze(2);


        auto other_speed_sa = all_speed_now.unsqueeze(1).expand({B, S_fut, A});

        auto ego_speed_spa = ego_speed_sp.unsqueeze(3);
        auto to_other = other_pos_sa.unsqueeze(2) - ego_pos_sp.unsqueeze(3);
        auto dot_ahead = (to_other * ego_dir_sp.unsqueeze(3)).sum( -1);


        auto lat_dist = (
            ego_dir_sp.index({"...", 0}).unsqueeze(3) * to_other.index({"...", 1})
            - ego_dir_sp.index({"...", 1}).unsqueeze(3) * to_other.index({"...", 0})
        );


        auto ego_stopped = ego_speed_spa <= STOPPED_EGO_THRESHOLD;
        auto other_stopped_spa = (
            ~all_is_agent.unsqueeze(1) | (other_speed_sa <= STOPPED_EGO_THRESHOLD)
        ).unsqueeze(2);
        auto both_active = ~ego_stopped & ~other_stopped_spa;


        auto ego_rear_sp = ego_pos_sp - ego_dir_sp
                         * ego_size_sp.index({"...", Slice(0, 1)}) * NC_REAR_AXLE_TO_CENTER_RATIO;
        auto to_other_rear = other_pos_sa.unsqueeze(2) - ego_rear_sp.unsqueeze(3);
        auto cos_angle = (to_other_rear * ego_dir_sp.unsqueeze(3)).sum(-1)
                       / to_other_rear.norm(2, -1).clamp_min(EPS);

        auto is_behind = cos_angle < -0.866f;


        auto ego_corners_sp = build_ego_corners(ego_pos_sp, ego_dir_sp, ego_size_sp);
        auto fl = ego_corners_sp.index({"...", 0, Slice()}).unsqueeze(3).unsqueeze(-2);
        auto fr = ego_corners_sp.index({"...", 1, Slice()}).unsqueeze(3).unsqueeze(-2);
        auto other_pos_spa = other_pos_sa.unsqueeze(2).expand({-1, -1, P, -1, -1});
        auto other_rot_spa = other_rot_sa.unsqueeze(2).expand({-1, -1, P, -1, -1, -1});
        auto other_size_spa = other_size_sa.unsqueeze(2).expand({-1, -1, P, -1, -1});
        auto is_frontal = segment_obb_collide(
            other_pos_spa, other_rot_spa, other_size_spa,
            fl.expand({B, S_fut, P, A, 1, 2}), fr.expand({B, S_fut, P, A, 1, 2}),
            torch::ones({B, S_fut, P, A, 1}, torch::TensorOptions().dtype(torch::kBool).device(device))
        ).squeeze(-1);
        auto active_front = both_active & is_frontal;
        auto stopped_track = other_stopped_spa.expand_as(colliding);
        auto active_lateral = both_active & ~is_behind & ~is_frontal;


        auto at_fault_front_stop = (active_front | stopped_track) & colliding;
        auto offroad_spa = lateral_fault_area.permute({0, 2, 1}).unsqueeze(3);
        auto at_fault_lateral = active_lateral & colliding & offroad_spa;
        auto at_fault = (at_fault_front_stop | at_fault_lateral) & ~ego_stopped;

        auto all_type_sa = all_type_f.permute({0, 2, 1}).unsqueeze(2);
        auto is_agent_type = (
            (all_type_sa == AgentTypeConst::vehicle) |
            (all_type_sa == AgentTypeConst::pedestrian) |
            (all_type_sa == AgentTypeConst::cyclist)
        );


        auto colliding_bpa = colliding.permute({0, 2, 3, 1});
        has_any_collision_bpa = colliding_bpa.any( -1);
        auto first_t = colliding_bpa.to(torch::kFloat32).argmax( -1);

        auto at_fault_bpas = at_fault.permute({0, 2, 3, 1});
        auto first_t_idx = first_t.unsqueeze(-1);
        at_fault_first_bpa = torch::gather(at_fault_bpas, 3, first_t_idx).squeeze(-1) & has_any_collision_bpa;

        auto is_agent_bpa = is_agent_type.permute({0, 2, 3, 1}).index({"...", 0});

        auto collision_score_bpa = torch::where(
            at_fault_first_bpa,
            torch::where(is_agent_bpa, torch::zeros({}, device), torch::full({}, 0.5f, device)),
            torch::ones({}, device)
        );
        nc_score = collision_score_bpa.amin( 2);

        auto at_fault_step = first_t.masked_fill(~at_fault_first_bpa, S_fut);
        first_at_fault_step = at_fault_step.amin( 2);
    } else {
        nc_score = torch::ones({B, P}, device);
        first_at_fault_step = torch::full({B, P}, S_fut, torch::TensorOptions().device(device).dtype(torch::kLong));
        has_any_collision_bpa = torch::zeros({B, P, A}, torch::TensorOptions().dtype(torch::kBool).device(device));
        at_fault_first_bpa = torch::zeros({B, P, A}, torch::TensorOptions().dtype(torch::kBool).device(device));
    }

    if (first_at_fault_only) {
        return ScorerResult{
            torch::Tensor(), first_at_fault_step,
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()
        };
    }


    torch::Tensor ttc_score;

    if (A > 0 && S_fut > 0) {
        auto max_off = std::max(1, static_cast<int>(FUTURE_COLLISION_HORIZON / dt));
        auto offs_last = (max_off / 3) * 3;
        auto n_evaluable = std::max(static_cast<int64_t>(1), S_fut - offs_last + 1);
        auto offs = torch::arange(0, max_off + 1, 3, torch::TensorOptions().dtype(torch::kLong).device(device));
        auto ts = torch::arange(n_evaluable, torch::TensorOptions().dtype(torch::kLong).device(device));

        auto grids = torch::meshgrid({ts, offs}, "ij");
        auto t_grid = grids[0];
        auto off_grid = grids[1];
        auto target_grid = t_grid + off_grid;
        auto pair_valid = target_grid < S_fut;
        auto target_clamped = target_grid.clamp_max(S_fut - 1);
        auto t_flat = t_grid.reshape(-1);
        auto target_flat = target_clamped.reshape(-1);
        auto pair_valid_flat = pair_valid.reshape(-1);
        auto N = t_flat.size(0);

        auto ego_pos_n = ego_pos_p.index_select(2, t_flat);
        auto ego_dir_n = ego_dir_p.index_select(2, t_flat);
        auto ego_speed_n = ego_speed_p.index_select(2, t_flat);
        auto ego_heading_n = ego_heading_p.index_select(2, t_flat);
        auto ego_size_n = ego_size_p.index_select(2, t_flat);

        auto delta_t = off_grid.reshape(-1).to(torch::kFloat32) * dt;
        auto proj_pos = ego_pos_n + ego_dir_n * ego_speed_n.unsqueeze(-1) * delta_t.reshape({1, 1, N, 1});
        auto proj_rot = heading_to_rot(ego_heading_n);

        auto oth_pos_na = all_pos_f.index_select(2, target_flat).permute({0, 2, 1, 3});
        auto oth_heading_na = all_heading_f.index_select(2, target_flat).permute({0, 2, 1});
        auto oth_rot_na = heading_to_rot(oth_heading_na);
        auto oth_size_na = all_size_2d.unsqueeze(2).expand({-1, -1, N, -1}).permute({0, 2, 1, 3});
        auto oth_valid_na = (
            all_valid_f.index_select(2, target_flat) & ~ego_mask_ba.unsqueeze(-1)
        ).permute({0, 2, 1});

        auto proj_pos_np = proj_pos.permute({0, 2, 1, 3});
        auto proj_rot_np = proj_rot.permute({0, 2, 1, 3, 4});
        auto ego_size_np = ego_size_n.permute({0, 2, 1, 3});
        auto ego_dir_np = ego_dir_n.permute({0, 2, 1, 3});
        auto ego_speed_np = ego_speed_n.permute({0, 2, 1});

        auto sat_dist_ttc = sat_signed_distance(
            proj_pos_np.unsqueeze(3).expand({-1, -1, -1, A, -1}),
            proj_rot_np.unsqueeze(3).expand({-1, -1, -1, A, -1, -1}),
            ego_size_np.unsqueeze(3).expand({-1, -1, -1, A, -1}),
            oth_pos_na.unsqueeze(2).expand({-1, -1, P, -1, -1}),
            oth_rot_na.unsqueeze(2).expand({-1, -1, P, -1, -1, -1}),
            oth_size_na.unsqueeze(2).expand({-1, -1, P, -1, -1})
        );

        auto colliding_ttc = (sat_dist_ttc <= 0.0f) & oth_valid_na.unsqueeze(2)
            & pair_valid_flat.reshape({1, N, 1, 1}).to(device);
        auto stopped_ttc = ego_speed_np < STOPPED_SPEED_THRESHOLD;


        auto src_pos_np = ego_pos_n.permute({0, 2, 1, 3});
        auto src_rear_np = src_pos_np - ego_dir_np
                         * ego_size_np.index({"...", Slice(0, 1)}) * NC_REAR_AXLE_TO_CENTER_RATIO;
        auto to_other_ttc = oth_pos_na.unsqueeze(2) - src_rear_np.unsqueeze(3);
        auto dot_ahead_ttc = (to_other_ttc * ego_dir_np.unsqueeze(3)).sum( -1);

        auto dist_ttc = to_other_ttc.norm(2, -1).clamp_min(EPS);
        auto cos_angle_ttc = dot_ahead_ttc / dist_ttc;
        auto ahead_ttc = cos_angle_ttc > 0.866f;
        auto behind_ttc = cos_angle_ttc < -0.866f;


        auto ttc_area = lateral_fault_area | dac_in_intersection;
        auto offroad_S = ttc_area.size(2);
        torch::Tensor ego_offroad_n;
        if (offroad_S >= S_fut) {
            ego_offroad_n = ttc_area.index_select(2, t_flat);
        } else {
            namespace F = torch::nn::functional;
            auto padded = F::pad(ttc_area, F::PadFuncOptions({0, static_cast<int64_t>(S_fut - offroad_S)}));
            ego_offroad_n = padded.index_select(2, t_flat);
        }
        auto ego_offroad_np = ego_offroad_n.permute({0, 2, 1});


        auto nc_not_at_fault = has_any_collision_bpa & ~at_fault_first_bpa;
        auto skip_ttc = nc_not_at_fault.unsqueeze(1);

        auto violation = colliding_ttc & ~skip_ttc & (
            ahead_ttc | (ego_offroad_np.unsqueeze(-1) & ~behind_ttc)
        );
        violation = violation & ~stopped_ttc.unsqueeze(-1);
        ttc_score = torch::where(
            violation.any( 1).any( 2),
            torch::zeros({}, device), torch::ones({}, device)
        );
    } else {
        ttc_score = torch::ones({B, P}, device);
    }

    auto ego_pos_bp = proposals.index({"...", Slice(None, 2)});
    auto ego_heading_bp = proposals.index({"...", 2});
    auto ego_dir_bp = torch::stack({ego_heading_bp.cos(), ego_heading_bp.sin()},  -1);
    auto ego_valid_bp = torch::ones({B, P, S_fut}, torch::TensorOptions().dtype(torch::kBool).device(device));


    auto _exp = [&](const torch::Tensor& t) {
        auto shape = t.sizes().vec();
        std::vector<int64_t> expand_shape = {-1, P};
        for (size_t i = 1; i < shape.size(); ++i) expand_shape.push_back(shape[i]);
        auto expanded = t.unsqueeze(1).expand(expand_shape);
        std::vector<int64_t> reshape_shape = {BP};
        for (size_t i = 1; i < shape.size(); ++i) reshape_shape.push_back(shape[i]);
        return expanded.reshape(reshape_shape);
    };


    auto prop_flat = proposals.reshape({BP, S_fut, 4});
    auto past_heading_bp = ego_past_heading.unsqueeze(1).expand({-1, P, -1}).reshape({BP, -1});
    auto past_pos_bp = ego_past_pos.unsqueeze(1).expand({-1, P, -1, -1}).reshape({BP, -1, 2});
    auto hc_heading = torch::cat({past_heading_bp, prop_flat.index({Slice(), Slice(), 2})},  1);
    auto hc_pos = torch::cat({past_pos_bp, prop_flat.index({Slice(), Slice(), Slice(None, 2)})},  1);
    auto hc_score = compute_comfort(hc_pos, hc_heading, num_past_frames, dt).reshape({B, P});


    auto [pr_pos, pr_dir, pr_valid, pr_near, pr_has] = pick_ego_route(scene.paths);
    (void)pr_dir; (void)pr_near;


    auto cl_pos_bp = _exp(pr_pos);
    auto cl_valid_bp = _exp(pr_valid);
    auto ep_computed = compute_ego_progress(
        prop_flat.index({Slice(), Slice(), Slice(None, 2)}),
        cl_pos_bp, cl_valid_bp
    ).reshape({B, P});
    auto ep_raw = torch::where(pr_has.unsqueeze(-1), ep_computed, torch::zeros_like(ep_computed));


    auto ego_pos_t0 = scene.agents.pos_xy.index({bi, ego_idx, now});
    auto ego_valid_t0 = scene.agents.valid.index({bi, ego_idx, now});
    auto ddc_all_pos = torch::cat(
        {ego_pos_t0.unsqueeze(1).unsqueeze(1).expand({B, P, 1, 2}), ego_pos_bp},  2);
    auto ddc_all_valid = torch::cat(
        {ego_valid_t0.unsqueeze(1).unsqueeze(1).expand({B, P, 1}), ego_valid_bp},  2);
    auto ddc_score = compute_ddc(ddc_all_pos, ddc_all_valid, dac_ddc_on_route,
                                 dac_ddc_intersection, dac_has_ddc, dt);


    auto tlc_score = torch::where(dac_tlc_violation,
                                  torch::zeros({B, P}, device), torch::ones({B, P}, device));


    auto lk_score = compute_lk(
        ddc_all_pos, ddc_all_valid, dac_ddc_intersection, scene.paths, dt);


    auto multiplier = nc_score * dac_score * ddc_score * tlc_score;


    auto masked_progress = ep_raw * multiplier;
    auto norm_const = std::get<0>(masked_progress.max( 1,  true));
    auto ep_score = torch::where(
        norm_const > PROGRESS_DISTANCE_THRESHOLD,
        (ep_raw / norm_const.clamp_min(EPS)).clamp(0.0f, 1.0f),
        torch::ones_like(ep_raw)
    );

    auto weighted = (5.0f * ep_score + 5.0f * ttc_score + 2.0f * lk_score + 2.0f * hc_score) / 14.0f;
    auto scores = multiplier * weighted;

    return ScorerResult{
        scores, first_at_fault_step,
        nc_score, ttc_score, dac_score, ddc_score,
        tlc_score, ep_score, hc_score, lk_score
    };
}

}

ScorerResult score_proposals(
    const torch::Tensor& proposals,
    const Scene& scene,
    const PlannerConfig& configs
) {
    return score_proposals_impl(
        proposals, scene, configs,  nullptr,
         false);
}

torch::Tensor first_at_fault_step(
    const torch::Tensor& proposals,
    const Scene& scene,
    const PlannerConfig& configs
) {
    return score_proposals_impl(
        proposals, scene, configs,  nullptr,
         true).first_at_fault_step;
}

ScorerResult score_proposals_core(
    const torch::Tensor& proposals,
    const Scene& scene,
    const ScorerPrep& prep,
    const PlannerConfig& configs
) {
    return score_proposals_impl(
        proposals, scene, configs, &prep,
         false);
}

ScorerPrep score_proposals_prep(
    const torch::Tensor& proposals,
    const Scene& scene,
    const PlannerConfig& configs
) {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;

    auto B = proposals.size(0);
    auto P = proposals.size(1);
    auto S_fut = proposals.size(2);
    auto device = proposals.device();

    auto now = configs.time.now();


    auto ego_mask_ba = scene.agents.is_ego;
    auto ego_idx = ego_mask_ba.to(torch::kFloat32).argmax( 1);
    auto bi = torch::arange(B, device);
    auto ego_size_2d = scene.agents.size_xyz.index({bi, ego_idx}).index({Slice(), Slice(None, 2)});

    auto ego_pos_p = proposals.index({"...", Slice(None, 2)});
    auto ego_heading_p = proposals.index({"...", 2});
    auto ego_dir_p = torch::stack({ego_heading_p.cos(), ego_heading_p.sin()},  -1);
    auto ego_size_p = ego_size_2d.unsqueeze(1).unsqueeze(2).expand({-1, P, S_fut, -1});

    auto dac_r = compute_dac_block(B, P, S_fut, ego_pos_p, ego_dir_p, ego_size_p, scene, configs.drivable_area, now, device);

    return ScorerPrep{dac_r.score, dac_r.offroad, dac_r.multiple_lanes,
                      dac_r.in_intersection, dac_r.t0_in_intersection,
                      dac_r.ddc_on_route, dac_r.ddc_intersection, dac_r.has_ddc,
                      dac_r.tlc_violation};
}
