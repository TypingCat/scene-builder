#include "drivable_area.h"
#include <cmath>
#include <limits>

using namespace torch::indexing;

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();
constexpr int LINK_TYPE_VEHICLE = 1;


torch::Tensor gather_entities(const torch::Tensor& value, const torch::Tensor& index) {
    std::vector<int64_t> shape(index.sizes().begin(), index.sizes().end());
    for (int64_t i = 2; i < value.dim(); ++i) shape.push_back(value.size(i));

    auto gather_index = index;
    for (int64_t i = 2; i < value.dim(); ++i) gather_index = gather_index.unsqueeze(-1);
    return value.gather(1, gather_index.expand(shape));
}


std::pair<torch::Tensor, torch::Tensor> nearest_boundary_half_widths(
    const torch::Tensor& link_pos,
    const torch::Tensor& link_valid,
    const torch::Tensor& left_ids,
    const torch::Tensor& right_ids,
    const Lane& lanes,
    const torch::Tensor& center_xy,
    const DrivableAreaConfig& cfg
) {
    auto B = link_valid.size(0);
    auto K = link_valid.size(1);
    auto def = torch::full_like(link_pos.index({"...", 0}), cfg.default_half_width_m);
    if (!lanes.valid.defined() || lanes.valid.numel() == 0 || !lanes.valid.any().item<bool>()) {
        return {def, def.clone()};
    }

    auto ref_ids = torch::cat({left_ids, right_ids}, -1).reshape({B, -1});
    auto referenced = ((lanes.id.unsqueeze(-1) == ref_ids.unsqueeze(1))
                       & (ref_ids.unsqueeze(1) > 0)).any(-1);
    auto boundary_valid = lanes.valid & referenced.unsqueeze(-1);
    if (!boundary_valid.any().item<bool>()) {
        return {def, def.clone()};
    }

    auto num_lanes = std::min<int64_t>(cfg.max_lanes, lanes.valid.size(1));
    auto lane_dist = (lanes.pos_xy - center_xy.unsqueeze(1).unsqueeze(1)).norm(2, -1);
    lane_dist = lane_dist.masked_fill(~boundary_valid, INF).amin(-1);
    auto lane_index = std::get<1>(lane_dist.topk(num_lanes, 1,  false));
    auto lane_pos = gather_entities(lanes.pos_xy, lane_index);
    auto lane_valid = gather_entities(boundary_valid, lane_index);
    auto lane_id = lanes.id.gather(1, lane_index);


    auto left_of = ((lane_id.unsqueeze(1).unsqueeze(-1) == left_ids.unsqueeze(2))
                    & (left_ids.unsqueeze(2) > 0)).any(-1);
    auto right_of = ((lane_id.unsqueeze(1).unsqueeze(-1) == right_ids.unsqueeze(2))
                     & (right_ids.unsqueeze(2) > 0)).any(-1);

    auto seg_start = lane_pos.index({Slice(), Slice(), Slice(None, -1)});
    auto seg_vec = lane_pos.index({Slice(), Slice(), Slice(1, None)}) - seg_start;
    auto seg_sq = (seg_vec * seg_vec).sum(-1).clamp_min(1e-8);
    auto seg_valid = lane_valid.index({Slice(), Slice(), Slice(None, -1)})
                   & lane_valid.index({Slice(), Slice(), Slice(1, None)});

    auto left_width = torch::full_like(def, INF);
    auto right_width = torch::full_like(def, INF);
    for (int64_t start = 0; start < K; start += cfg.link_chunk) {
        auto end = std::min<int64_t>(start + cfg.link_chunk, K);

        auto station = link_pos.index({Slice(), Slice(start, end)}).unsqueeze(3).unsqueeze(3);
        auto relative = station - seg_start.unsqueeze(1).unsqueeze(1);
        auto fraction = ((relative * seg_vec.unsqueeze(1).unsqueeze(1)).sum(-1)
                         / seg_sq.unsqueeze(1).unsqueeze(1)).clamp(0.0, 1.0);
        auto offset = relative - fraction.unsqueeze(-1) * seg_vec.unsqueeze(1).unsqueeze(1);
        auto distance = offset.norm(2, -1);
        distance = distance.masked_fill(~seg_valid.unsqueeze(1).unsqueeze(1), INF);

        for (int side = 0; side < 2; ++side) {
            const auto& belongs = side == 0 ? left_of : right_of;
            auto& dest = side == 0 ? left_width : right_width;

            auto side_mask = belongs.index({Slice(), Slice(start, end)})
                                 .unsqueeze(2).unsqueeze(-1);
            auto measured = distance.masked_fill(~side_mask, INF);
            dest.index_put_({Slice(), Slice(start, end)}, measured.flatten(3).amin(-1));
        }
    }

    auto usable_of = [&](const torch::Tensor& width) {
        return torch::isfinite(width)
             & (width >= cfg.min_half_width_m)
             & (width <= cfg.max_half_width_m)
             & link_valid;
    };
    auto link_median = [&](const torch::Tensor& width, const torch::Tensor& usable) {
        auto unmeasured = torch::full_like(width, std::numeric_limits<float>::quiet_NaN());
        return std::get<0>(torch::where(usable, width, unmeasured).nanmedian(-1, true));
    };


    auto usable_left = usable_of(left_width), usable_right = usable_of(right_width);
    auto median_left = link_median(left_width, usable_left);
    auto median_right = link_median(right_width, usable_right);
    auto def_one = def.index({Slice(), Slice(), Slice(0, 1)});
    auto fill_left = torch::where(torch::isnan(median_left), median_right, median_left);
    auto fill_right = torch::where(torch::isnan(median_right), median_left, median_right);
    fill_left = torch::where(torch::isnan(fill_left), def_one, fill_left);
    fill_right = torch::where(torch::isnan(fill_right), def_one, fill_right);
    return {torch::where(usable_left, left_width, fill_left.expand_as(left_width)),
            torch::where(usable_right, right_width, fill_right.expand_as(right_width))};
}


std::pair<torch::Tensor, torch::Tensor> pack_ribbon(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& station_valid
) {
    auto B = left.size(0), K = left.size(1), L = left.size(2);
    auto V = 2 * L;
    auto vertices = torch::zeros({B, K, V, 2}, left.options());

    auto rank = station_valid.to(torch::kLong).cumsum(-1) - 1;
    auto count = station_valid.sum(-1);
    auto unused = torch::full_like(rank, V - 1);

    auto left_index = torch::where(station_valid, rank, unused);
    vertices.scatter_(2, left_index.unsqueeze(-1).expand({B, K, L, 2}),
                      torch::where(station_valid.unsqueeze(-1), left, torch::zeros_like(left)));

    auto right_index = 2 * count.unsqueeze(-1) - 1 - rank;
    right_index = torch::where(station_valid, right_index, unused);
    vertices.scatter_(2, right_index.unsqueeze(-1).expand({B, K, L, 2}),
                      torch::where(station_valid.unsqueeze(-1), right, torch::zeros_like(right)));

    auto valid = torch::arange(V, left.device()).unsqueeze(0).unsqueeze(0)
               < (2 * count).unsqueeze(-1);
    valid = valid & (count.unsqueeze(-1) >= 2);
    return {vertices, valid};
}

}


DrivablePolygons build_drivable_area_polygons(
    const Link& links,
    const Lane& lanes,
    const torch::Tensor& center_xy,
    const DrivableAreaConfig& cfg
) {
    DrivablePolygons out;
    if (!links.valid.defined() || links.valid.numel() == 0 || links.valid.size(-1) < 2) {
        return out;
    }
    auto drivable = (links.type == LINK_TYPE_VEHICLE) & (links.valid.sum(-1) >= 2);
    if (!drivable.any().item<bool>()) {
        return out;
    }

    auto num_links = links.valid.size(1);
    auto K = std::min<int64_t>(cfg.max_links, num_links);
    auto distance = (links.pos_xy - center_xy.unsqueeze(1).unsqueeze(1)).norm(2, -1);
    distance = distance.masked_fill(~links.valid, INF).amin(-1);
    distance = distance.masked_fill(~drivable, INF);
    auto selected = std::get<1>(distance.topk(K, 1,  false));

    auto link_pos = gather_entities(links.pos_xy, selected);
    auto link_dir = gather_entities(links.dir_xy, selected);
    auto link_valid = gather_entities(links.valid, selected);
    auto selected_drivable = drivable.gather(1, selected);
    link_valid = link_valid & selected_drivable.unsqueeze(-1);
    auto left_ids = gather_entities(links.left_lanes, selected);
    auto right_ids = gather_entities(links.right_lanes, selected);

    auto [left_width, right_width] = nearest_boundary_half_widths(
        link_pos, link_valid, left_ids, right_ids, lanes, center_xy, cfg);

    auto tangent = link_dir / link_dir.norm(2, -1,  true).clamp_min(1e-8);
    auto geometric = torch::zeros_like(link_pos);
    auto L = link_pos.size(2);
    geometric.index_put_(
        {Slice(), Slice(), Slice(None, -1)},
        link_pos.index({Slice(), Slice(), Slice(1, None)})
            - link_pos.index({Slice(), Slice(), Slice(None, -1)}));
    geometric.index_put_(
        {Slice(), Slice(), L - 1},
        link_pos.index({Slice(), Slice(), L - 1}) - link_pos.index({Slice(), Slice(), L - 2}));
    geometric = geometric / geometric.norm(2, -1,  true).clamp_min(1e-8);
    tangent = torch::where(
        link_dir.norm(2, -1,  true) > 1e-8, tangent, geometric);


    auto step = (link_pos.index({Slice(), Slice(), Slice(1, None)})
                 - link_pos.index({Slice(), Slice(), Slice(None, -1)})).norm(2, -1);
    auto step_valid = link_valid.index({Slice(), Slice(), Slice(1, None)})
                    & link_valid.index({Slice(), Slice(), Slice(None, -1)});
    auto spacing = (step * step_valid).sum(-1) / step_valid.sum(-1).clamp_min(1);
    auto rank = link_valid.to(torch::kLong).cumsum(-1) - 1;
    auto count = link_valid.sum(-1,  true);
    auto is_first = link_valid & (rank == 0);
    auto is_last = link_valid & (rank == count - 1);
    auto overhang = 0.5 * spacing.unsqueeze(-1).unsqueeze(-1) * tangent;
    auto extended = link_pos + overhang * (is_last.unsqueeze(-1).to(overhang.scalar_type())
                                           - is_first.unsqueeze(-1).to(overhang.scalar_type()));

    auto normal = torch::stack(
        {-tangent.index({"...", 1}), tangent.index({"...", 0})}, -1);
    auto left = extended + normal * left_width.unsqueeze(-1);
    auto right = extended - normal * right_width.unsqueeze(-1);

    auto [vertices, valid] = pack_ribbon(left, right, link_valid);
    auto zero_l = torch::zeros_like(selected_drivable.to(links.id.scalar_type()));
    out.vertices = vertices;
    out.valid = valid;
    out.link_id = torch::where(selected_drivable, links.id.gather(1, selected), zero_l);
    out.link_section = links.section.sizes() == links.id.sizes()
        ? torch::where(selected_drivable, links.section.gather(1, selected), zero_l)
        : zero_l;
    out.defined = true;
    return out;
}


torch::Tensor polygons_match_link_ids(
    const DrivablePolygons& polygons,
    const torch::Tensor& link_ids,
    const torch::Tensor& link_valid
) {
    auto B = polygons.link_id.size(0);
    auto flat_ids = link_ids.reshape({B, -1});
    auto flat_valid = link_valid.reshape({B, -1});
    return ((polygons.link_id.unsqueeze(-1) == flat_ids.unsqueeze(1))
            & flat_valid.unsqueeze(1)
            & (flat_ids.unsqueeze(1) > 0)
            & polygons.valid.any(-1).unsqueeze(-1)).any(-1);
}


torch::Tensor points_in_polygons(
    const torch::Tensor& points,
    const torch::Tensor& vertices,
    const torch::Tensor& valid,
    int polygon_chunk
) {
    constexpr float BOUNDARY_TOL = 1e-6f;
    auto B = vertices.size(0), K = vertices.size(1), V = vertices.size(2);
    auto Q = points.size(1);
    auto result = torch::zeros({B, Q, K},
                               torch::TensorOptions().dtype(torch::kBool).device(points.device()));
    auto vertex_index = torch::arange(V, points.device()).unsqueeze(0).unsqueeze(0);

    for (int64_t start = 0; start < K; start += polygon_chunk) {
        auto end = std::min<int64_t>(start + polygon_chunk, K);
        auto polygon = vertices.index({Slice(), Slice(start, end)});
        auto polygon_valid = valid.index({Slice(), Slice(start, end)});
        auto count = polygon_valid.sum(-1);
        auto next_index = torch::where(
            vertex_index + 1 < count.unsqueeze(-1),
            vertex_index + 1,
            torch::zeros_like(vertex_index));
        auto next_vertex = polygon.gather(
            2, next_index.unsqueeze(-1).expand({B, end - start, V, 2}));
        auto edge_valid = vertex_index < count.unsqueeze(-1);

        auto a = polygon.unsqueeze(1);
        auto b = next_vertex.unsqueeze(1);
        auto p = points.unsqueeze(2).unsqueeze(2);
        auto px = p.index({"...", 0}), py = p.index({"...", 1});
        auto ax = a.index({"...", 0}), ay = a.index({"...", 1});
        auto bx = b.index({"...", 0}), by = b.index({"...", 1});

        auto crosses_y = (ay > py) != (by > py);
        auto delta_y = by - ay;
        auto safe_delta_y = torch::where(delta_y.abs() > 1e-12, delta_y, torch::ones_like(delta_y));
        auto x_intersection = ax + (py - ay) * (bx - ax) / safe_delta_y;
        auto crossings = crosses_y & (px < x_intersection) & edge_valid.unsqueeze(1);
        auto inside = (crossings.sum(-1) % 2) == 1;

        auto edge = b - a;
        auto relative = p - a;
        auto cross = edge.index({"...", 0}) * relative.index({"...", 1})
                   - edge.index({"...", 1}) * relative.index({"...", 0});
        auto dot = (relative * edge).sum(-1);
        auto edge_sq = (edge * edge).sum(-1);
        auto on_boundary = ((cross.abs() <= BOUNDARY_TOL)
                            & (dot >= -BOUNDARY_TOL)
                            & (dot <= edge_sq + BOUNDARY_TOL)
                            & edge_valid.unsqueeze(1)).any(-1);

        result.index_put_({Slice(), Slice(), Slice(start, end)},
                          inside & ~on_boundary & (count.unsqueeze(1) >= 3));
    }
    return result;
}


torch::Tensor polygons_intersect_boxes(
    const torch::Tensor& box_corners,
    const torch::Tensor& vertices,
    const torch::Tensor& valid,
    int polygon_chunk
) {
    auto B = vertices.size(0), K = vertices.size(1), V = vertices.size(2);
    auto Q = box_corners.size(1);


    auto result = points_in_polygons(
        box_corners.reshape({B, Q * 4, 2}), vertices, valid, polygon_chunk)
        .reshape({B, Q, 4, K}).any(2);

    auto center = box_corners.mean(-2);
    auto longitudinal = box_corners.index({Slice(), Slice(), 0}) - box_corners.index({Slice(), Slice(), 3});
    auto lateral = box_corners.index({Slice(), Slice(), 0}) - box_corners.index({Slice(), Slice(), 1});
    auto half = torch::stack({longitudinal.norm(2, -1) * 0.5, lateral.norm(2, -1) * 0.5}, -1);
    longitudinal = longitudinal / longitudinal.norm(2, -1, true).clamp_min(1e-8);
    lateral = lateral / lateral.norm(2, -1, true).clamp_min(1e-8);

    auto vertex_index = torch::arange(V, vertices.device()).unsqueeze(0).unsqueeze(0);
    for (int64_t start = 0; start < K; start += polygon_chunk) {
        auto end = std::min<int64_t>(start + polygon_chunk, K);
        auto polygon = vertices.index({Slice(), Slice(start, end)});
        auto polygon_valid = valid.index({Slice(), Slice(start, end)});
        auto count = polygon_valid.sum(-1);
        auto next_index = torch::where(vertex_index + 1 < count.unsqueeze(-1),
                                       vertex_index + 1, torch::zeros_like(vertex_index));
        auto next_vertex = polygon.gather(
            2, next_index.unsqueeze(-1).expand({B, end - start, V, 2}));
        auto edge_valid = (vertex_index < count.unsqueeze(-1)) & (count.unsqueeze(-1) >= 3);

        auto rel_a = polygon.unsqueeze(1) - center.unsqueeze(2).unsqueeze(2);
        auto rel_b = next_vertex.unsqueeze(1) - center.unsqueeze(2).unsqueeze(2);
        auto axis_long = longitudinal.unsqueeze(2).unsqueeze(2);
        auto axis_lat = lateral.unsqueeze(2).unsqueeze(2);
        auto local_a = torch::stack({(rel_a * axis_long).sum(-1), (rel_a * axis_lat).sum(-1)}, -1);
        auto local_b = torch::stack({(rel_b * axis_long).sum(-1), (rel_b * axis_lat).sum(-1)}, -1);


        auto delta = local_b - local_a;
        auto box_half = half.unsqueeze(2).unsqueeze(2);
        auto parallel = delta.abs() <= 1e-12;
        auto safe_delta = torch::where(parallel, torch::ones_like(delta), delta);
        auto t1 = (-box_half - local_a) / safe_delta;
        auto t2 = (box_half - local_a) / safe_delta;
        auto near = torch::minimum(t1, t2);
        auto far = torch::maximum(t1, t2);
        auto outside_parallel = parallel & (local_a.abs() > box_half);
        near = torch::where(parallel, torch::full_like(near, -INFINITY), near);
        far = torch::where(parallel,
                           torch::where(outside_parallel,
                                        torch::full_like(far, -INFINITY),
                                        torch::full_like(far, INFINITY)),
                           far);
        auto t_min = near.amax(-1);
        auto t_max = far.amin(-1);
        auto edge_hits = ((t_min <= t_max) & (t_max >= 0.0) & (t_min <= 1.0)
                          & edge_valid.unsqueeze(1)).any(-1);
        result.index_put_({Slice(), Slice(), Slice(start, end)},
                          result.index({Slice(), Slice(), Slice(start, end)}) | edge_hits);
    }
    return result;
}
