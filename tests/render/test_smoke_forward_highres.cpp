#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "dvren/core/context.hpp"
#include "dvren/core/plan.hpp"
#include "dvren/fields/dense_grid.hpp"
#include "dvren/render/renderer.hpp"
#include "smoke_test_utils.hpp"

namespace {

float MaxAbsDiff(float a, float b) {
    return std::fabs(a - b);
}

float MaxAbsDiff(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        max_diff = std::max(max_diff, MaxAbsDiff(a[i], b[i]));
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
    plan_desc.width = 960;
    plan_desc.height = 720;
    plan_desc.t_near = 0.0f;
    plan_desc.t_far = 1.6f;
    plan_desc.sampling.dt = 0.04f;
    plan_desc.sampling.max_steps = 64;
    plan_desc.sampling.mode = dvren::SamplingMode::kFixed;
    plan_desc.max_rays = plan_desc.width * plan_desc.height;
    plan_desc.max_samples = plan_desc.max_rays * plan_desc.sampling.max_steps;
    plan_desc.seed = 2024;

    plan_desc.camera.model = dvren::CameraModel::kPinhole;
    plan_desc.camera.K = {920.0f, 0.0f, 479.5f,
                          0.0f, 920.0f, 359.5f,
                          0.0f, 0.0f, 1.0f};
    plan_desc.camera.c2w = {1.0f, 0.0f, 0.0f, 0.45f,
                            0.0f, 1.0f, 0.0f, 0.55f,
                            0.0f, 0.0f, 1.0f, -1.0f};

    dvren::Plan plan;
    status = dvren::Plan::Create(ctx, plan_desc, plan);
    if (!status.ok()) {
        std::cerr << "Plan::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::DenseGridConfig grid_config{};
    grid_config.resolution = {64, 64, 80};
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
    const size_t pixel_count =
        static_cast<size_t>(actual_desc.width) * static_cast<size_t>(actual_desc.height);

    if (forward.image.size() != pixel_count * 3ULL ||
        forward.transmittance.size() != pixel_count ||
        forward.opacity.size() != pixel_count ||
        forward.depth.size() != pixel_count ||
        forward.hitmask.size() != pixel_count) {
        std::cerr << "Forward buffer size mismatch" << std::endl;
        return 1;
    }

    if (forward.ray_count != pixel_count) {
        std::cerr << "Unexpected ray count: " << forward.ray_count << " expected " << pixel_count << std::endl;
        return 1;
    }

    if (forward.sample_count == 0 || forward.sample_count > plan_desc.max_samples) {
        std::cerr << "Forward sample count invalid: " << forward.sample_count << std::endl;
        return 1;
    }

    const dvren::WorkspaceInfo ws = renderer.workspace_info();
    if (ws.total_bytes() == 0) {
        std::cerr << "Workspace info not populated" << std::endl;
        return 1;
    }

    const auto image_extrema = std::minmax_element(forward.image.begin(), forward.image.end());
    if (!(image_extrema.second != forward.image.end() &&
          *image_extrema.second > *image_extrema.first + 1e-3f)) {
        std::cerr << "Image lacks variation" << std::endl;
        return 1;
    }

    const auto trans_extrema = std::minmax_element(forward.transmittance.begin(), forward.transmittance.end());
    if (!(trans_extrema.second != forward.transmittance.end() &&
          *trans_extrema.second > *trans_extrema.first + 1e-3f)) {
        std::cerr << "Transmittance lacks variation" << std::endl;
        return 1;
    }

    if (std::all_of(forward.hitmask.begin(), forward.hitmask.end(), [](uint32_t v) { return v == 0U; })) {
        std::cerr << "Hitmask contains no active pixels" << std::endl;
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

    const smoke_test::FieldStats image_stats = smoke_test::ComputeStats(forward.image);
    const smoke_test::FieldStats trans_stats = smoke_test::ComputeStats(forward.transmittance);
    const smoke_test::FieldStats opacity_stats = smoke_test::ComputeStats(forward.opacity);
    const smoke_test::FieldStats depth_stats = smoke_test::ComputeStats(forward.depth);

    const size_t active_pixels = std::count_if(forward.hitmask.begin(),
                                               forward.hitmask.end(),
                                               [](uint32_t v) { return v != 0U; });
    const double active_ratio = pixel_count > 0
        ? static_cast<double>(active_pixels) / static_cast<double>(pixel_count)
        : 0.0;
    const double avg_samples_per_ray = forward.ray_count > 0
        ? static_cast<double>(forward.sample_count) / static_cast<double>(forward.ray_count)
        : 0.0;

    std::vector<std::pair<uint32_t, uint32_t>> sample_pixels;
    const uint32_t stride_x = 32;
    const uint32_t stride_y = 24;
    for (uint32_t py = 0; py < actual_desc.height; py += stride_y) {
        for (uint32_t px = 0; px < actual_desc.width; px += stride_x) {
            sample_pixels.emplace_back(px, py);
        }
    }
    if (sample_pixels.empty()) {
        sample_pixels.emplace_back(actual_desc.width / 2, actual_desc.height / 2);
    }
    sample_pixels.emplace_back(actual_desc.width - 1, actual_desc.height / 2);
    sample_pixels.emplace_back(actual_desc.width / 2, actual_desc.height - 1);
    sample_pixels.emplace_back(actual_desc.width - 1, actual_desc.height - 1);

    float max_image_diff = 0.0f;
    float max_trans_diff = 0.0f;
    float max_opacity_diff = 0.0f;
    float max_depth_diff = 0.0f;

    double sum_image_diff = 0.0;
    double sumsq_image_diff = 0.0;
    size_t image_diff_count = 0;

    double sum_trans_diff = 0.0;
    double sumsq_trans_diff = 0.0;
    size_t trans_diff_count = 0;

    double sum_opacity_diff = 0.0;
    double sumsq_opacity_diff = 0.0;
    size_t opacity_diff_count = 0;

    double sum_depth_diff = 0.0;
    double sumsq_depth_diff = 0.0;
    size_t depth_diff_count = 0;

    for (const auto& [px, py] : sample_pixels) {
        smoke_test::PixelEvaluation eval = smoke_test::IntegratePixel(actual_desc, grid_config, px, py);
        const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(actual_desc.width) + px;
        const size_t base = idx * 3ULL;

        std::array<float, 3> observed{
            forward.image[base + 0],
            forward.image[base + 1],
            forward.image[base + 2]
        };

        float pixel_max_diff = 0.0f;
        for (size_t c = 0; c < 3; ++c) {
            const float diff = std::fabs(observed[c] - eval.radiance[c]);
            pixel_max_diff = std::max(pixel_max_diff, diff);
            sum_image_diff += diff;
            sumsq_image_diff += static_cast<double>(diff) * static_cast<double>(diff);
            ++image_diff_count;
        }
        max_image_diff = std::max(max_image_diff, pixel_max_diff);

        const float trans_diff = std::fabs(forward.transmittance[idx] - eval.transmittance);
        max_trans_diff = std::max(max_trans_diff, trans_diff);
        sum_trans_diff += trans_diff;
        sumsq_trans_diff += static_cast<double>(trans_diff) * static_cast<double>(trans_diff);
        ++trans_diff_count;

        const float opacity_diff = std::fabs(forward.opacity[idx] - eval.opacity);
        max_opacity_diff = std::max(max_opacity_diff, opacity_diff);
        sum_opacity_diff += opacity_diff;
        sumsq_opacity_diff += static_cast<double>(opacity_diff) * static_cast<double>(opacity_diff);
        ++opacity_diff_count;

        const float depth_diff = std::fabs(forward.depth[idx] - eval.depth);
        max_depth_diff = std::max(max_depth_diff, depth_diff);
        sum_depth_diff += depth_diff;
        sumsq_depth_diff += static_cast<double>(depth_diff) * static_cast<double>(depth_diff);
        ++depth_diff_count;
    }

    if (max_image_diff > 3e-3f) {
        std::cerr << "Image mismatch (subset), max abs diff = " << max_image_diff << std::endl;
        return 1;
    }
    if (max_trans_diff > 3e-3f) {
        std::cerr << "Transmittance mismatch (subset), max abs diff = " << max_trans_diff << std::endl;
        return 1;
    }
    if (max_opacity_diff > 3e-3f) {
        std::cerr << "Opacity mismatch (subset), max abs diff = " << max_opacity_diff << std::endl;
        return 1;
    }
    if (max_depth_diff > 1e-2f) {
        std::cerr << "Depth mismatch (subset), max abs diff = " << max_depth_diff << std::endl;
        return 1;
    }

    const auto compute_mean_std = [](double sum, double sumsq, size_t count) -> std::pair<double, double> {
        if (count == 0) {
            return {0.0, 0.0};
        }
        const double mean = sum / static_cast<double>(count);
        const double variance = std::max(sumsq / static_cast<double>(count) - mean * mean, 0.0);
        return {mean, std::sqrt(variance)};
    };

    const auto [mean_image_diff, std_image_diff] = compute_mean_std(sum_image_diff, sumsq_image_diff, image_diff_count);
    const auto [mean_trans_diff, std_trans_diff] = compute_mean_std(sum_trans_diff, sumsq_trans_diff, trans_diff_count);
    const auto [mean_opacity_diff, std_opacity_diff] = compute_mean_std(sum_opacity_diff, sumsq_opacity_diff, opacity_diff_count);
    const auto [mean_depth_diff, std_depth_diff] = compute_mean_std(sum_depth_diff, sumsq_depth_diff, depth_diff_count);

    const std::string ppm_path = "dvren_smoke_forward_highres.ppm";
    std::ofstream ppm(ppm_path, std::ios::binary);
    if (!ppm) {
        std::cerr << "Failed to open PPM file for writing: " << ppm_path << std::endl;
        return 1;
    }

    const float scale = image_stats.max > 0.0f ? (255.0f / image_stats.max) : 0.0f;
    auto to_byte = [](float value) -> uint8_t {
        const float clamped = std::clamp(value, 0.0f, 255.0f);
        return static_cast<uint8_t>(clamped + 0.5f);
    };

    ppm << "P6\n" << actual_desc.width << ' ' << actual_desc.height << "\n255\n";
    for (size_t idx = 0; idx < pixel_count; ++idx) {
        const size_t base = idx * 3ULL;
        const float r = forward.image[base + 0] * scale;
        const float g = forward.image[base + 1] * scale;
        const float b = forward.image[base + 2] * scale;
        ppm.put(static_cast<char>(to_byte(r)));
        ppm.put(static_cast<char>(to_byte(g)));
        ppm.put(static_cast<char>(to_byte(b)));
    }
    ppm.flush();
    if (!ppm) {
        std::cerr << "Failed to write PPM image: " << ppm_path << std::endl;
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "ImageStats min=" << image_stats.min
              << " max=" << image_stats.max
              << " mean=" << image_stats.mean
              << " stddev=" << image_stats.stddev << '\n';
    std::cout << "TransmittanceStats min=" << trans_stats.min
              << " max=" << trans_stats.max
              << " mean=" << trans_stats.mean
              << " stddev=" << trans_stats.stddev << '\n';
    std::cout << "OpacityStats min=" << opacity_stats.min
              << " max=" << opacity_stats.max
              << " mean=" << opacity_stats.mean
              << " stddev=" << opacity_stats.stddev << '\n';
    std::cout << "DepthStats min=" << depth_stats.min
              << " max=" << depth_stats.max
              << " mean=" << depth_stats.mean
              << " stddev=" << depth_stats.stddev << '\n';
    std::cout << "Hitmask active_pixels=" << active_pixels
              << " ratio=" << active_ratio * 100.0 << "%\n";
    std::cout << "Samples total=" << forward.sample_count
              << " per_ray_avg=" << avg_samples_per_ray << '\n';
    std::cout << "WorkspaceBytes rays=" << ws.ray_buffer_bytes
              << " samples=" << ws.sample_buffer_bytes
              << " integration=" << ws.integration_buffer_bytes
              << " image=" << ws.image_buffer_bytes
              << " gradients=" << ws.gradient_buffer_bytes
              << " workspace=" << ws.workspace_buffer_bytes
              << " total=" << ws.total_bytes() << '\n';
    std::cout << "DifferenceStats(subset) image_max=" << max_image_diff
              << " image_mean=" << mean_image_diff
              << " image_std=" << std_image_diff
              << " trans_max=" << max_trans_diff
              << " trans_mean=" << mean_trans_diff
              << " trans_std=" << std_trans_diff
              << " opacity_max=" << max_opacity_diff
              << " opacity_mean=" << mean_opacity_diff
              << " opacity_std=" << std_opacity_diff
              << " depth_max=" << max_depth_diff
              << " depth_mean=" << mean_depth_diff
              << " depth_std=" << std_depth_diff << '\n';
    std::cout << "SubsetSamples count=" << sample_pixels.size()
              << " image_diff_samples=" << image_diff_count
              << " scalar_diff_samples=" << trans_diff_count << '\n';
    std::cout << "OutputImage path=" << ppm_path
              << " scale=" << scale << '\n';

    return 0;
}
