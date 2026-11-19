#pragma once

#include <torch/torch.h>

namespace AgentTypeConst {
constexpr int64_t unknown = 0;
constexpr int64_t vehicle = 1;
constexpr int64_t pedestrian = 2;
constexpr int64_t cyclist = 3;
}

namespace RoadTypeConst {
constexpr int64_t unknown = 0;
constexpr int64_t lane_center_freeway = 1;
constexpr int64_t lane_center_surface_street = 2;
constexpr int64_t lane_center_bike_lane = 3;
constexpr int64_t road_line_broken_single_white = 4;
constexpr int64_t road_line_solid_single_white = 5;
constexpr int64_t road_line_solid_double_white = 6;
constexpr int64_t road_line_broken_single_yellow = 7;
constexpr int64_t road_line_broken_double_yellow = 8;
constexpr int64_t road_line_solid_single_yellow = 9;
constexpr int64_t road_line_solid_double_yellow = 10;
constexpr int64_t road_line_passing_double_yellow = 11;
constexpr int64_t road_edge_boundary = 12;
constexpr int64_t road_edge_median = 13;
constexpr int64_t stop_sign = 14;
constexpr int64_t crosswalk = 15;
constexpr int64_t speed_bump = 16;
constexpr int64_t driveway = 17;
}

namespace PathTypeConst {
constexpr int64_t unknown = 0;
constexpr int64_t vehicle = 1;
constexpr int64_t pedestrian = 2;
constexpr int64_t cyclist = 3;
}

namespace TrafficTypeConst {
constexpr int64_t unknown = 0;
constexpr int64_t vehicle = 1;
constexpr int64_t pedestrian = 2;
constexpr int64_t cyclist = 3;
}

namespace TrafficStateConst {
constexpr int64_t unknown = 0;
constexpr int64_t go = 1;
constexpr int64_t caution = 2;
constexpr int64_t stop = 3;
}

namespace LinkTypeConst {
constexpr int64_t unknown = 0;
constexpr int64_t vehicle = 1;
constexpr int64_t pedestrian = 2;
constexpr int64_t cyclist = 3;
}

namespace LinkSectionConst {
constexpr int64_t unknown = 0;
constexpr int64_t intersection = 1;
constexpr int64_t school_zone = 2;
constexpr int64_t tunnel = 3;
constexpr int64_t hi_pass = 4;
constexpr int64_t freeway = 5;
}

namespace LaneTypeConst {
constexpr int64_t unknown = 0;
constexpr int64_t center = 1;
constexpr int64_t solid = 2;
constexpr int64_t dashed = 3;
constexpr int64_t edge = 4;
constexpr int64_t boundary = 5;
constexpr int64_t median = 6;
}

namespace MarkerTypeConst {
constexpr int64_t unknown = 0;
constexpr int64_t stop_sign = 1;
constexpr int64_t stop_line = 2;
constexpr int64_t crosswalk = 3;
constexpr int64_t speed_bump = 4;
constexpr int64_t driveway = 5;
constexpr int64_t bus_stop = 6;
constexpr int64_t safe_point = 7;
constexpr int64_t arrow = 8;
}

struct Agent {
    torch::Tensor id;
    torch::Tensor type;
    torch::Tensor time_s;
    torch::Tensor pos_xy;
    torch::Tensor dir_xy;
    torch::Tensor vel_xy;
    torch::Tensor acc_xy;
    torch::Tensor size_xyz;
    torch::Tensor is_ego;
    torch::Tensor valid;
};

struct Traffic {
    torch::Tensor id;
    torch::Tensor type;
    torch::Tensor target_link;
    torch::Tensor dir_xy;
    torch::Tensor pos_xy;
    torch::Tensor state;
    torch::Tensor time_s;
    torch::Tensor valid;
};

struct Path {
    torch::Tensor near_link;
    torch::Tensor pos_xy;
    torch::Tensor dir_xy;
    torch::Tensor arc_length;
    torch::Tensor is_route;
    torch::Tensor valid;
};

struct Link {
    torch::Tensor id;
    torch::Tensor type;
    torch::Tensor section;
    torch::Tensor speed_limit;
    torch::Tensor pos_xy;
    torch::Tensor dir_xy;
    torch::Tensor valid;
    torch::Tensor entry_links;
    torch::Tensor exit_links;
    torch::Tensor left_links;
    torch::Tensor right_links;
    torch::Tensor left_lanes;
    torch::Tensor right_lanes;
};

struct Lane {
    torch::Tensor id;
    torch::Tensor type;
    torch::Tensor pos_xy;
    torch::Tensor dir_xy;
    torch::Tensor valid;
};

struct Marker {
    torch::Tensor id;
    torch::Tensor type;
    torch::Tensor pos_xy;
    torch::Tensor dir_xy;
    torch::Tensor valid;
};

struct Scene {
    Agent agents;
    Link links;
    Lane lanes;
    Marker markers;
    Traffic traffics;
    Path paths;
};

struct Output {
    torch::Tensor pos_xy;
    torch::Tensor dir_xy;
    torch::Tensor size_xyz;
    torch::Tensor valid;
    torch::Tensor is_ego;
    torch::Tensor type;
    torch::Tensor batch_valid;
};
