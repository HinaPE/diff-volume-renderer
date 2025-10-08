#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "dvren/core/context.hpp"
#include "dvren/core/plan.hpp"
#include "dvren/fields/dense_grid.hpp"
#include "dvren/render/renderer.hpp"
#include "hotpath/hp.h"
#include "smoke_test_utils.hpp"

namespace {

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

}  // namespace

int main() {
    dvren::Context ctx;
    dvren::Status status = dvren::Context::Create(dvren::ContextOptions{}, ctx);
    if (!status.ok()) {
        std::cerr << "Context::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::PlanDescriptor plan_desc{};
    plan_desc.width = 32;
    plan_desc.height = 32;
    plan_desc.t_near = 0.0f;
    plan_desc.t_far = 2.0f;
    plan_desc.sampling.dt = 0.02f;
    plan_desc.sampling.max_steps = 160;
    plan_desc.sampling.mode = dvren::SamplingMode::kFixed;
    plan_desc.max_rays = plan_desc.width * plan_desc.height;
    plan_desc.max_samples = plan_desc.max_rays * plan_desc.sampling.max_steps;
    plan_desc.seed = 1337;

    plan_desc.camera.model = dvren::CameraModel::kPinhole;
    plan_desc.camera.K = {48.0f, 0.0f, 16.0f,
                          0.0f, 48.0f, 16.0f,
                          0.0f, 0.0f, 1.0f};
    plan_desc.camera.c2w = {1.0f, 0.0f, 0.0f, 0.5f,
                            0.0f, 1.0f, 0.0f, 0.5f,
                            0.0f, 0.0f, 1.0f, -1.2f};

    dvren::Plan plan;
    status = dvren::Plan::Create(ctx, plan_desc, plan);
    if (!status.ok()) {
        std::cerr << "Plan::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::DenseGridConfig grid_config{};
    grid_config.resolution = {48, 48, 64};
    grid_config.bbox_min = {0.0f, 0.0f, 0.0f};
    grid_config.bbox_max = {1.0f, 1.0f, 1.0f};
    grid_config.interp = HP_INTERP_LINEAR;
    grid_config.oob = HP_OOB_ZERO;
    smoke_test::PopulateSmokeGrid(grid_config);

    dvren::DenseGridField field;
    status = dvren::DenseGridField::Create(ctx, grid_config, field);
    if (!status.ok()) {
        std::cerr << "DenseGridField::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::Renderer renderer(ctx, plan, dvren::RenderOptions{});
    dvren::ForwardResult forward{};
    status = renderer.Forward(field, forward);
    if (!status.ok()) {
        std::cerr << "Renderer::Forward failed: " << status.ToString() << std::endl;
        return 1;
    }

    const hp_plan_desc& actual_desc = plan.descriptor();

    if (forward.image.size() != static_cast<size_t>(actual_desc.width) * static_cast<size_t>(actual_desc.height) * 3ULL) {
        std::cerr << "Forward image size mismatch" << std::endl;
        return 1;
    }
    if (forward.transmittance.size() != static_cast<size_t>(actual_desc.width) * static_cast<size_t>(actual_desc.height)) {
        std::cerr << "Forward transmittance size mismatch" << std::endl;
        return 1;
    }
    if (forward.opacity.size() != forward.transmittance.size() ||
        forward.depth.size() != forward.transmittance.size() ||
        forward.hitmask.size() != forward.transmittance.size()) {
        std::cerr << "Forward auxiliary buffer size mismatch" << std::endl;
        return 1;
    }

    const size_t expected_ray_count = static_cast<size_t>(actual_desc.roi.width) * static_cast<size_t>(actual_desc.roi.height);
    if (forward.ray_count != expected_ray_count) {
        std::cerr << "Unexpected ray count: " << forward.ray_count << " expected " << expected_ray_count << std::endl;
        return 1;
    }
    if (forward.sample_count == 0) {
        std::cerr << "Forward produced zero samples" << std::endl;
        return 1;
    }

    const size_t pixel_count = static_cast<size_t>(actual_desc.width) * static_cast<size_t>(actual_desc.height);
    std::vector<float> ref_image(pixel_count * 3ULL, 0.0f);
    std::vector<float> ref_trans(pixel_count, 0.0f);
    std::vector<float> ref_opacity(pixel_count, 0.0f);
    std::vector<float> ref_depth(pixel_count, actual_desc.t_far);

    for (uint32_t py = 0; py < actual_desc.height; ++py) {
        for (uint32_t px = 0; px < actual_desc.width; ++px) {
            const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(actual_desc.width) + px;
            const size_t base = idx * 3ULL;

            smoke_test::PixelEvaluation eval = smoke_test::IntegratePixel(actual_desc, grid_config, px, py);
            ref_image[base + 0] = eval.radiance[0];
            ref_image[base + 1] = eval.radiance[1];
            ref_image[base + 2] = eval.radiance[2];
            ref_trans[idx] = eval.transmittance;
            ref_opacity[idx] = eval.opacity;
            ref_depth[idx] = eval.depth;
        }
    }

    const float image_diff = MaxAbsDiff(forward.image, ref_image);
    const float trans_diff = MaxAbsDiff(forward.transmittance, ref_trans);
    const float opacity_diff = MaxAbsDiff(forward.opacity, ref_opacity);
    const float depth_diff = MaxAbsDiff(forward.depth, ref_depth);

    if (image_diff > 2e-3f) {
        std::cerr << "Image mismatch, max abs diff = " << image_diff << std::endl;
        return 1;
    }
    if (trans_diff > 2e-3f) {
        std::cerr << "Transmittance mismatch, max abs diff = " << trans_diff << std::endl;
        return 1;
    }
    if (opacity_diff > 2e-3f) {
        std::cerr << "Opacity mismatch, max abs diff = " << opacity_diff << std::endl;
        return 1;
    }
    if (depth_diff > 1e-2f) {
        std::cerr << "Depth mismatch, max abs diff = " << depth_diff << std::endl;
        return 1;
    }

    const auto image_extrema = std::minmax_element(forward.image.begin(), forward.image.end());
    if (!(image_extrema.second != forward.image.end() &&
          *image_extrema.second > *image_extrema.first + 5e-3f)) {
        std::cerr << "Forward image lacks intensity variation" << std::endl;
        return 1;
    }

    const auto trans_extrema = std::minmax_element(forward.transmittance.begin(), forward.transmittance.end());
    if (!(trans_extrema.second != forward.transmittance.end() &&
          *trans_extrema.second > *trans_extrema.first + 5e-3f)) {
        std::cerr << "Transmittance lacks variation" << std::endl;
        return 1;
    }

    if (std::all_of(forward.hitmask.begin(), forward.hitmask.end(), [](uint32_t v) { return v == 0U; })) {
        std::cerr << "Hitmask contains no hits" << std::endl;
        return 1;
    }

    for (float value : forward.transmittance) {
        if (!(value >= 0.0f && value <= 1.0f)) {
            std::cerr << "Transmittance out of range" << std::endl;
            return 1;
        }
    }
    for (float value : forward.opacity) {
        if (!(value >= 0.0f && value <= 1.0f)) {
            std::cerr << "Opacity out of range" << std::endl;
            return 1;
        }
    }

    return 0;
}
