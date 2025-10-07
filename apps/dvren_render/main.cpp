#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "dvren/core/context.hpp"
#include "dvren/core/plan.hpp"
#include "dvren/fields/dense_grid.hpp"
#include "dvren/render/renderer.hpp"
#include "nlohmann/json.hpp"

namespace dvren::app {

namespace {

struct RenderConfig {
    PlanDescriptor plan_desc;
    DenseGridConfig grid_config;
    std::filesystem::path output_path{"frame.ppm"};
    RenderOptions render_options{};
};

Status ParseSamplingMode(const std::string& value, SamplingMode& out_mode) {
    if (value == "fixed") {
        out_mode = SamplingMode::kFixed;
        return Status::Ok();
    }
    if (value == "stratified") {
        out_mode = SamplingMode::kStratified;
        return Status::Ok();
    }
    return Status(StatusCode::kInvalidArgument, "unsupported sampling mode: " + value);
}

Status ParseInterpMode(const std::string& value, hp_interp_mode& out_mode) {
    if (value == "linear") {
        out_mode = HP_INTERP_LINEAR;
        return Status::Ok();
    }
    if (value == "nearest") {
        out_mode = HP_INTERP_NEAREST;
        return Status::Ok();
    }
    return Status(StatusCode::kInvalidArgument, "unsupported interpolation mode: " + value);
}

Status ParseOobPolicy(const std::string& value, hp_oob_policy& out_policy) {
    if (value == "zero") {
        out_policy = HP_OOB_ZERO;
        return Status::Ok();
    }
    if (value == "clamp") {
        out_policy = HP_OOB_CLAMP;
        return Status::Ok();
    }
    return Status(StatusCode::kInvalidArgument, "unsupported oob policy: " + value);
}

template <typename T>
Status ParseVector(const nlohmann::json& node, size_t expected_count, std::vector<T>& out) {
    try {
        out = node.get<std::vector<T>>();
    } catch (const nlohmann::json::exception& ex) {
        return Status(StatusCode::kInvalidArgument, ex.what());
    }
    if (expected_count != 0 && out.size() != expected_count) {
        return Status(StatusCode::kInvalidArgument, "array length mismatch");
    }
    return Status::Ok();
}

Status ParsePlan(const nlohmann::json& render_node, PlanDescriptor& out_plan) {
    try {
        out_plan.width = render_node.at("width").get<uint32_t>();
        out_plan.height = render_node.at("height").get<uint32_t>();
        out_plan.t_near = render_node.value("t_near", 0.0f);
        out_plan.t_far = render_node.at("t_far").get<float>();
        out_plan.seed = render_node.value("seed", 0ULL);
        out_plan.sampling.dt = render_node.at("dt").get<float>();
        out_plan.sampling.max_steps = render_node.at("max_steps").get<uint32_t>();
        const std::string sampling_mode = render_node.value("sampling_mode", std::string("fixed"));
        Status sampling_status = ParseSamplingMode(sampling_mode, out_plan.sampling.mode);
        if (!sampling_status.ok()) {
            return sampling_status;
        }

        if (render_node.contains("roi")) {
            const auto& roi_node = render_node.at("roi");
            Roi roi{};
            roi.x = roi_node.value("x", 0U);
            roi.y = roi_node.value("y", 0U);
            roi.width = roi_node.value("width", out_plan.width);
            roi.height = roi_node.value("height", out_plan.height);
            out_plan.roi = roi;
        }

        if (render_node.contains("camera")) {
            const auto& cam_node = render_node.at("camera");
            const std::string model = cam_node.value("model", std::string("pinhole"));
            if (model == "orthographic") {
                out_plan.camera.model = CameraModel::kOrthographic;
            } else {
                out_plan.camera.model = CameraModel::kPinhole;
            }
            if (cam_node.contains("K")) {
                std::vector<float> K;
                Status status = ParseVector(cam_node.at("K"), 9, K);
                if (!status.ok()) {
                    return status;
                }
                std::copy(K.begin(), K.end(), out_plan.camera.K.begin());
            } else {
                out_plan.camera.K[0] = 1.0f;
                out_plan.camera.K[4] = 1.0f;
                out_plan.camera.K[8] = 1.0f;
                out_plan.camera.K[2] = static_cast<float>(out_plan.width) * 0.5f;
                out_plan.camera.K[5] = static_cast<float>(out_plan.height) * 0.5f;
            }
            if (cam_node.contains("c2w")) {
                std::vector<float> c2w;
                Status status = ParseVector(cam_node.at("c2w"), 12, c2w);
                if (!status.ok()) {
                    return status;
                }
                std::copy(c2w.begin(), c2w.end(), out_plan.camera.c2w.begin());
            }
            out_plan.camera.ortho_scale = cam_node.value("ortho_scale", 1.0f);
        } else {
            out_plan.camera.K[0] = 1.0f;
            out_plan.camera.K[4] = 1.0f;
            out_plan.camera.K[8] = 1.0f;
            out_plan.camera.K[2] = static_cast<float>(out_plan.width) * 0.5f;
            out_plan.camera.K[5] = static_cast<float>(out_plan.height) * 0.5f;
        }
    } catch (const nlohmann::json::exception& ex) {
        return Status(StatusCode::kInvalidArgument, ex.what());
    }
    return Status::Ok();
}

Status ParseVolume(const nlohmann::json& volume_node, DenseGridConfig& grid_config) {
    try {
        const std::vector<int32_t> dims = volume_node.at("size").get<std::vector<int32_t>>();
        if (dims.size() != 3) {
            return Status(StatusCode::kInvalidArgument, "volume.size must contain 3 integers");
        }
        grid_config.resolution = {dims[0], dims[1], dims[2]};
    } catch (const nlohmann::json::exception& ex) {
        return Status(StatusCode::kInvalidArgument, ex.what());
    }

    Status density_status = ParseVector(volume_node.at("density"), 0, grid_config.sigma);
    if (!density_status.ok()) {
        return density_status;
    }

    if (volume_node.contains("color")) {
        Status color_status = ParseVector(volume_node.at("color"), 0, grid_config.color);
        if (!color_status.ok()) {
            return color_status;
        }
    } else {
        const int64_t voxel_count = static_cast<int64_t>(grid_config.sigma.size());
        grid_config.color.resize(static_cast<size_t>(voxel_count) * 3ULL, 0.0f);
        for (int64_t i = 0; i < voxel_count; ++i) {
            const float value = grid_config.sigma[static_cast<size_t>(i)];
            const size_t base = static_cast<size_t>(i) * 3U;
            grid_config.color[base + 0] = value;
            grid_config.color[base + 1] = value;
            grid_config.color[base + 2] = value;
        }
    }

    if (volume_node.contains("bbox_min")) {
        std::vector<float> bbox;
        Status bbox_status = ParseVector(volume_node.at("bbox_min"), 3, bbox);
        if (!bbox_status.ok()) {
            return bbox_status;
        }
        std::copy(bbox.begin(), bbox.end(), grid_config.bbox_min.begin());
    }

    if (volume_node.contains("bbox_max")) {
        std::vector<float> bbox;
        Status bbox_status = ParseVector(volume_node.at("bbox_max"), 3, bbox);
        if (!bbox_status.ok()) {
            return bbox_status;
        }
        std::copy(bbox.begin(), bbox.end(), grid_config.bbox_max.begin());
    }

    const std::string interp = volume_node.value("interp", std::string("linear"));
    Status interp_status = ParseInterpMode(interp, grid_config.interp);
    if (!interp_status.ok()) {
        return interp_status;
    }

    const std::string oob = volume_node.value("oob", std::string("zero"));
    Status oob_status = ParseOobPolicy(oob, grid_config.oob);
    if (!oob_status.ok()) {
        return oob_status;
    }
    return Status::Ok();
}

Status ParseConfigInternal(const std::filesystem::path& path, RenderConfig& config) {
    if (!std::filesystem::exists(path)) {
        return Status(StatusCode::kInvalidArgument, "config file not found: " + path.string());
    }

    std::ifstream file(path);
    if (!file) {
        return Status(StatusCode::kInvalidArgument, "failed to open config file");
    }

    nlohmann::json root;
    try {
        file >> root;
    } catch (const nlohmann::json::exception& ex) {
        return Status(StatusCode::kInvalidArgument, ex.what());
    }

    const auto& render_node = root.at("render");

    Status render_status = ParsePlan(render_node, config.plan_desc);
    if (!render_status.ok()) {
        return render_status;
    }

    if (render_node.contains("options")) {
        const auto& opt_node = render_node.at("options");
        config.render_options.use_fused_path = opt_node.value("use_fused_path", true);
        config.render_options.enable_graph = opt_node.value("enable_graph", false);
        config.render_options.capture_stats = opt_node.value("capture_stats", true);
    }

    Status volume_status = ParseVolume(root.at("volume"), config.grid_config);
    if (!volume_status.ok()) {
        return volume_status;
    }

    if (root.contains("output")) {
        const auto& output_node = root.at("output");
        config.output_path = output_node.value("path", std::string("frame.ppm"));
    }

    return Status::Ok();
}

}  // namespace

Status ParseConfigFile(const std::filesystem::path& path, RenderConfig& config) {
    return ParseConfigInternal(path, config);
}

Status RenderToFile(const dvren::Context& ctx,
                    const dvren::Plan& plan,
                    const dvren::DenseGridField& field,
                    const RenderOptions& options,
                    const std::filesystem::path& output_path) {
    dvren::Renderer renderer(ctx, plan, options);
    dvren::ForwardResult forward;
    Status status = renderer.Forward(field, forward);
    if (!status.ok()) {
        return status;
    }

    const auto& desc = plan.descriptor();
    const size_t pixel_count = static_cast<size_t>(desc.width) * static_cast<size_t>(desc.height);
    if (forward.image.size() != pixel_count * 3ULL) {
        return Status(StatusCode::kInternalError, "forward output size mismatch");
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        return Status(StatusCode::kInvalidArgument, "failed to open output file: " + output_path.string());
    }

    out << "P6\n" << desc.width << " " << desc.height << "\n255\n";
    for (size_t idx = 0; idx < pixel_count; ++idx) {
        const size_t base = idx * 3ULL;
        const float r = std::clamp(forward.image[base + 0], 0.0f, 1.0f);
        const float g = std::clamp(forward.image[base + 1], 0.0f, 1.0f);
        const float b = std::clamp(forward.image[base + 2], 0.0f, 1.0f);
        const unsigned char r_byte = static_cast<unsigned char>(std::round(r * 255.0f));
        const unsigned char g_byte = static_cast<unsigned char>(std::round(g * 255.0f));
        const unsigned char b_byte = static_cast<unsigned char>(std::round(b * 255.0f));
        out.write(reinterpret_cast<const char*>(&r_byte), 1);
        out.write(reinterpret_cast<const char*>(&g_byte), 1);
        out.write(reinterpret_cast<const char*>(&b_byte), 1);
    }
    out.flush();

    std::cout << "Forward stats: rays=" << forward.ray_count
              << " samples=" << forward.sample_count
              << " total_ms=" << forward.stats.total_ms << std::endl;
    return Status::Ok();
}

}  // namespace dvren::app

int main(int argc, char** argv) {
    using dvren::Status;
    using dvren::Context;
    using dvren::ContextOptions;
    using dvren::Plan;
    using dvren::DenseGridField;
    using dvren::app::RenderConfig;

    if (argc < 2) {
        std::cerr << "Usage: dvren_render <config.json> [output.ppm]" << std::endl;
        return 1;
    }

    const std::filesystem::path config_path(argv[1]);
    const std::optional<std::filesystem::path> output_override =
        (argc >= 3) ? std::optional<std::filesystem::path>(std::filesystem::path(argv[2])) : std::nullopt;

    RenderConfig config{};
    Status config_status = dvren::app::ParseConfigFile(config_path, config);
    if (!config_status.ok()) {
        std::cerr << "Error parsing config: " << config_status.ToString() << std::endl;
        return 1;
    }
    if (output_override.has_value()) {
        config.output_path = *output_override;
    }

    Context ctx;
    Status ctx_status = Context::Create(ContextOptions{}, ctx);
    if (!ctx_status.ok()) {
        std::cerr << "Context creation failed: " << ctx_status.ToString() << std::endl;
        return 1;
    }

    Plan plan;
    Status plan_status = Plan::Create(ctx, config.plan_desc, plan);
    if (!plan_status.ok()) {
        std::cerr << "Plan creation failed: " << plan_status.ToString() << std::endl;
        return 1;
    }

    DenseGridField field;
    Status field_status = DenseGridField::Create(ctx, config.grid_config, field);
    if (!field_status.ok()) {
        std::cerr << "Field creation failed: " << field_status.ToString() << std::endl;
        return 1;
    }

    Status run_status = dvren::app::RenderToFile(ctx, plan, field, config.render_options, config.output_path);
    if (!run_status.ok()) {
        std::cerr << "Render failed: " << run_status.ToString() << std::endl;
        return 1;
    }

    std::cout << "Wrote " << std::filesystem::absolute(config.output_path) << std::endl;
    return 0;
}
