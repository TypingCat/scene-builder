#pragma once
#include "types.h"
#include <torch/torch.h>


struct DrivablePolygons {
    torch::Tensor vertices;
    torch::Tensor valid;
    torch::Tensor link_id;
    torch::Tensor link_section;
    bool defined = false;
};


DrivablePolygons build_drivable_area_polygons(
    const Link& links,
    const Lane& lanes,
    const torch::Tensor& center_xy,
    const DrivableAreaConfig& cfg);


torch::Tensor polygons_match_link_ids(
    const DrivablePolygons& polygons,
    const torch::Tensor& link_ids,
    const torch::Tensor& link_valid);


torch::Tensor points_in_polygons(
    const torch::Tensor& points,
    const torch::Tensor& vertices,
    const torch::Tensor& valid,
    int polygon_chunk);


torch::Tensor polygons_intersect_boxes(
    const torch::Tensor& box_corners,
    const torch::Tensor& vertices,
    const torch::Tensor& valid,
    int polygon_chunk);
