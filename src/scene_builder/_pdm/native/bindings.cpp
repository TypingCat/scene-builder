#include "planner.h"

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include "scene_ops.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

Agent to_cpp_agent(py::handle obj) {
    return Agent{
        obj.attr("id").cast<torch::Tensor>(),
        obj.attr("type").cast<torch::Tensor>(),
        obj.attr("time_s").cast<torch::Tensor>(),
        obj.attr("pos_xy").cast<torch::Tensor>(),
        obj.attr("dir_xy").cast<torch::Tensor>(),
        obj.attr("vel_xy").cast<torch::Tensor>(),
        obj.attr("acc_xy").cast<torch::Tensor>(),
        obj.attr("size_xyz").cast<torch::Tensor>(),
        obj.attr("is_ego").cast<torch::Tensor>(),
        obj.attr("valid").cast<torch::Tensor>(),
    };
}

Traffic to_cpp_traffic(py::handle obj) {
    return Traffic{
        obj.attr("id").cast<torch::Tensor>(),
        obj.attr("type").cast<torch::Tensor>(),
        obj.attr("target_link").cast<torch::Tensor>(),
        obj.attr("dir_xy").cast<torch::Tensor>(),
        obj.attr("pos_xy").cast<torch::Tensor>(),
        obj.attr("state").cast<torch::Tensor>(),
        obj.attr("time_s").cast<torch::Tensor>(),
        obj.attr("valid").cast<torch::Tensor>(),
    };
}

Path to_cpp_path(py::handle obj) {
    return Path{
        obj.attr("near_link").cast<torch::Tensor>(),
        obj.attr("pos_xy").cast<torch::Tensor>(),
        obj.attr("dir_xy").cast<torch::Tensor>(),
        obj.attr("arc_length").cast<torch::Tensor>(),
        obj.attr("is_route").cast<torch::Tensor>(),
        obj.attr("valid").cast<torch::Tensor>(),
    };
}

Link to_cpp_link(py::handle obj) {
    return Link{
        obj.attr("id").cast<torch::Tensor>(),
        obj.attr("type").cast<torch::Tensor>(),
        obj.attr("section").cast<torch::Tensor>(),
        obj.attr("speed_limit").cast<torch::Tensor>(),
        obj.attr("pos_xy").cast<torch::Tensor>(),
        obj.attr("dir_xy").cast<torch::Tensor>(),
        obj.attr("valid").cast<torch::Tensor>(),
        obj.attr("entry_links").cast<torch::Tensor>(),
        obj.attr("exit_links").cast<torch::Tensor>(),
        obj.attr("left_links").cast<torch::Tensor>(),
        obj.attr("right_links").cast<torch::Tensor>(),
        obj.attr("left_lanes").cast<torch::Tensor>(),
        obj.attr("right_lanes").cast<torch::Tensor>(),
    };
}

Lane to_cpp_lane(py::handle obj) {
    return Lane{
        obj.attr("id").cast<torch::Tensor>(),
        obj.attr("type").cast<torch::Tensor>(),
        obj.attr("pos_xy").cast<torch::Tensor>(),
        obj.attr("dir_xy").cast<torch::Tensor>(),
        obj.attr("valid").cast<torch::Tensor>(),
    };
}

Marker to_cpp_marker(py::handle obj) {
    return Marker{
        obj.attr("id").cast<torch::Tensor>(),
        obj.attr("type").cast<torch::Tensor>(),
        obj.attr("pos_xy").cast<torch::Tensor>(),
        obj.attr("dir_xy").cast<torch::Tensor>(),
        obj.attr("valid").cast<torch::Tensor>(),
    };
}

Scene to_cpp_scene(py::handle obj) {
    return Scene{
        to_cpp_agent(obj.attr("agents")),
        to_cpp_link(obj.attr("links")),
        to_cpp_lane(obj.attr("lanes")),
        to_cpp_marker(obj.attr("markers")),
        to_cpp_traffic(obj.attr("traffics")),
        to_cpp_path(obj.attr("paths")),
    };
}


py::dict to_py_output(const Output& output) {
    py::dict result;
    result["pos_xy"] = output.pos_xy;
    result["dir_xy"] = output.dir_xy;
    result["size_xyz"] = output.size_xyz;
    result["valid"] = output.valid;
    result["is_ego"] = output.is_ego;
    result["type"] = output.type;
    result["batch_valid"] = output.batch_valid;
    return result;
}

PlannerConfig planner_config(int past, int future, double frame_unit) {
    PlannerConfig config;
    config.time.num_past_frames = past;
    config.time.num_future_frames = future;
    config.frame_unit = static_cast<float>(frame_unit);

    config.model.speed_limit_fraction = {0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
    config.model.lateral_offsets = {-1.0f, 1.0f};
    config.model.lqr.q_lateral = {1.0f, 10.0f, 0.0f};
    config.model.lqr.r_lateral = {1.0f};
    return config;
}

py::dict plan(py::handle object, int past, int future, double frame_unit) {
    torch::InferenceMode guard;
    const auto config = planner_config(past, future, frame_unit);
    const auto scene = to_cpp_scene(object);
    auto [local, transform] = runtime::preprocess_scene(scene, past, config.preprocess);
    auto output = run_pdm_full(local, config);
    return to_py_output(runtime::transform_output_to_global(output, transform));
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("plan", &plan);
}
