#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <utility>
#include <vector>

#include "dvren/core/context.hpp"
#include "dvren/core/plan.hpp"
#include "dvren/fields/dense_grid.hpp"
#include "dvren/render/renderer.hpp"
#include "smoke_test_utils.hpp"

namespace {

void PopulateSdfSphereGrid(dvren::DenseGridConfig& config) {
    const int nx = config.resolution[0];
    const int ny = config.resolution[1];
    const int nz = config.resolution[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        return;
    }

    const size_t voxel_count = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    config.sigma.assign(voxel_count, 0.0f);
    config.color.assign(voxel_count * 3ULL, 0.0f);

    const float extent_x = config.bbox_max[0] - config.bbox_min[0];
    const float extent_y = config.bbox_max[1] - config.bbox_min[1];
    const float extent_z = config.bbox_max[2] - config.bbox_min[2];
    const float min_extent = std::min({extent_x, extent_y, extent_z});
    const std::array<float, 3> center{
        config.bbox_min[0] + 0.5f * extent_x,
        config.bbox_min[1] + 0.5f * extent_y,
        config.bbox_min[2] + 0.5f * extent_z
    };

    const float radius = 0.35f * min_extent;
    const float shell_thickness = 0.08f * min_extent;
    const float density_scale = 8.0f;

    auto voxel_position = [&](int ix, int iy, int iz) -> std::array<float, 3> {
        const float fx = (static_cast<float>(ix) + 0.5f) / static_cast<float>(nx);
        const float fy = (static_cast<float>(iy) + 0.5f) / static_cast<float>(ny);
        const float fz = (static_cast<float>(iz) + 0.5f) / static_cast<float>(nz);
        return {
            config.bbox_min[0] + fx * extent_x,
            config.bbox_min[1] + fy * extent_y,
            config.bbox_min[2] + fz * extent_z
        };
    };

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                const auto p = voxel_position(ix, iy, iz);
                const float dx = p[0] - center[0];
                const float dy = p[1] - center[1];
                const float dz = p[2] - center[2];
                const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                const float sdf = distance - radius;

                const float shell_profile = std::exp(-(sdf * sdf) / (shell_thickness * shell_thickness));
                float sigma = density_scale * shell_profile;
                if (sigma < 1e-7f) {
                    sigma = 0.0f;
                }

                const float shell_mix = std::clamp(1.0f - std::fabs(sdf) / (shell_thickness * 1.5f), 0.0f, 1.0f);
                const float warm = 0.5f + 0.5f * shell_mix;
                const float cool = 0.3f + 0.7f * (1.0f - shell_mix);

                const size_t index = (static_cast<size_t>(iz) * static_cast<size_t>(ny) + static_cast<size_t>(iy)) * static_cast<size_t>(nx) + static_cast<size_t>(ix);
                config.sigma[index] = sigma;

                const size_t base = index * 3ULL;
                config.color[base + 0] = sigma * warm;
                config.color[base + 1] = sigma * 0.45f * (0.6f + shell_mix);
                config.color[base + 2] = sigma * cool;
            }
        }
    }
}

std::vector<std::pair<uint32_t, uint32_t>> BuildSamplePixels(uint32_t width, uint32_t height) {
    std::vector<std::pair<uint32_t, uint32_t>> samples;
    const float cx = 0.5f * static_cast<float>(width - 1);
    const float cy = 0.5f * static_cast<float>(height - 1);
    const float radius = 0.32f * std::min(width, height);

    samples.emplace_back(static_cast<uint32_t>(std::lround(cx)), static_cast<uint32_t>(std::lround(cy)));

    for (int step = 0; step < 12; ++step) {
        const float angle = static_cast<float>(step) * (std::numbers::pi_v<float> / 6.0f);
        const float px = cx + radius * std::cos(angle);
        const float py = cy + radius * std::sin(angle);
        const uint32_t ix = static_cast<uint32_t>(std::clamp<std::int64_t>(std::llround(px), 0, static_cast<std::int64_t>(width) - 1));
        const uint32_t iy = static_cast<uint32_t>(std::clamp<std::int64_t>(std::llround(py), 0, static_cast<std::int64_t>(height) - 1));
        samples.emplace_back(ix, iy);
    }

    samples.emplace_back(static_cast<uint32_t>(std::lround(cx)), 0U);
    samples.emplace_back(static_cast<uint32_t>(std::lround(cx)), height - 1U);
    samples.emplace_back(0U, static_cast<uint32_t>(std::lround(cy)));
    samples.emplace_back(width - 1U, static_cast<uint32_t>(std::lround(cy)));

    std::sort(samples.begin(), samples.end());
    samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
    return samples;
}

float PixelLuminance(const std::vector<float>& image, size_t base_index) {
    if (base_index + 2 >= image.size()) {
        return 0.0f;
    }
    const float r = image[base_index + 0];
    const float g = image[base_index + 1];
    const float b = image[base_index + 2];
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
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
    plan_desc.width = 800;
    plan_desc.height = 800;
    plan_desc.t_near = 0.2f;
    plan_desc.t_far = 3.8f;
    plan_desc.sampling.dt = 0.06f;
    plan_desc.sampling.max_steps = 64;
    plan_desc.sampling.mode = dvren::SamplingMode::kFixed;
    plan_desc.max_rays = plan_desc.width * plan_desc.height;
    plan_desc.max_samples = plan_desc.max_rays * plan_desc.sampling.max_steps;
    plan_desc.seed = 424242;
    plan_desc.camera.model = dvren::CameraModel::kPinhole;
    plan_desc.camera.K = {740.0f, 0.0f, 399.5f,
                          0.0f, 740.0f, 399.5f,
                          0.0f, 0.0f, 1.0f};
    plan_desc.camera.c2w = {1.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, -2.0f};

    dvren::Plan plan;
    status = dvren::Plan::Create(ctx, plan_desc, plan);
    if (!status.ok()) {
        std::cerr << "Plan::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::DenseGridConfig grid_config{};
    grid_config.resolution = {160, 160, 160};
    grid_config.bbox_min = {0.0f, 0.0f, 0.0f};
    grid_config.bbox_max = {1.0f, 1.0f, 1.0f};
    grid_config.interp = HP_INTERP_LINEAR;
    grid_config.oob = HP_OOB_ZERO;
    PopulateSdfSphereGrid(grid_config);

    const auto field_stats = smoke_test::ComputeStats(grid_config.sigma);
    if (!(field_stats.max > field_stats.min && field_stats.max > 1e-3f)) {
        std::cerr << "Populated field lacks variation" << std::endl;
        return 1;
    }

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
    const size_t pixel_count = static_cast<size_t>(actual_desc.width) * static_cast<size_t>(actual_desc.height);
    if (forward.image.size() != pixel_count * 3ULL ||
        forward.transmittance.size() != pixel_count ||
        forward.opacity.size() != pixel_count ||
        forward.depth.size() != pixel_count ||
        forward.hitmask.size() != pixel_count) {
        std::cerr << "Forward buffers size mismatch" << std::endl;
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

    const auto image_stats = smoke_test::ComputeStats(forward.image);
    const auto trans_stats = smoke_test::ComputeStats(forward.transmittance);
    const auto opacity_stats = smoke_test::ComputeStats(forward.opacity);
    const auto depth_stats = smoke_test::ComputeStats(forward.depth);

    if (!(image_stats.max > image_stats.min + 5e-4f)) {
        std::cerr << "Rendered image lacks variation" << std::endl;
        return 1;
    }
    if (!(opacity_stats.max > opacity_stats.min + 5e-4f)) {
        std::cerr << "Opacity lacks variation" << std::endl;
        return 1;
    }

    const size_t center_idx = static_cast<size_t>(actual_desc.height / 2) * static_cast<size_t>(actual_desc.width) + static_cast<size_t>(actual_desc.width / 2);
    const size_t center_base = center_idx * 3ULL;
    const float center_luma = PixelLuminance(forward.image, center_base);

    const size_t ring_px = static_cast<size_t>(actual_desc.width / 2 + actual_desc.width / 5);
    const size_t ring_idx = static_cast<size_t>(actual_desc.height / 2) * static_cast<size_t>(actual_desc.width) + std::min(ring_px, static_cast<size_t>(actual_desc.width - 1));
    const float ring_luma = PixelLuminance(forward.image, ring_idx * 3ULL);

    if (!(ring_luma > center_luma + 1e-4f)) {
        std::cerr << "Ring luminance not greater than center" << std::endl;
        return 1;
    }

    size_t active_pixels = 0;
    for (uint32_t hit : forward.hitmask) {
        if (hit != 0U) {
            ++active_pixels;
        }
    }
    if (active_pixels < pixel_count / 10) {
        std::cerr << "Too few active pixels: " << active_pixels << std::endl;
        return 1;
    }

    const auto sample_pixels = BuildSamplePixels(actual_desc.width, actual_desc.height);
    float max_image_diff = 0.0f;
    float max_trans_diff = 0.0f;
    float max_opacity_diff = 0.0f;
    float max_depth_diff = 0.0f;

    for (const auto& [px, py] : sample_pixels) {
        const smoke_test::PixelEvaluation eval = smoke_test::IntegratePixel(actual_desc, grid_config, px, py);
        const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(actual_desc.width) + static_cast<size_t>(px);
        const size_t base = idx * 3ULL;
        max_image_diff = std::max(max_image_diff, std::fabs(forward.image[base + 0] - eval.radiance[0]));
        max_image_diff = std::max(max_image_diff, std::fabs(forward.image[base + 1] - eval.radiance[1]));
        max_image_diff = std::max(max_image_diff, std::fabs(forward.image[base + 2] - eval.radiance[2]));
        max_trans_diff = std::max(max_trans_diff, std::fabs(forward.transmittance[idx] - eval.transmittance));
        max_opacity_diff = std::max(max_opacity_diff, std::fabs(forward.opacity[idx] - eval.opacity));
        max_depth_diff = std::max(max_depth_diff, std::fabs(forward.depth[idx] - eval.depth));
    }

    if (max_image_diff > 3e-3f) {
        std::cerr << "Image mismatch on sampled pixels, max abs diff = " << max_image_diff << std::endl;
        return 1;
    }
    if (max_trans_diff > 3e-3f) {
        std::cerr << "Transmittance mismatch on sampled pixels, max abs diff = " << max_trans_diff << std::endl;
        return 1;
    }
    if (max_opacity_diff > 3e-3f) {
        std::cerr << "Opacity mismatch on sampled pixels, max abs diff = " << max_opacity_diff << std::endl;
        return 1;
    }
    if (max_depth_diff > 1e-2f) {
        std::cerr << "Depth mismatch on sampled pixels, max abs diff = " << max_depth_diff << std::endl;
        return 1;
    }

    const std::string ppm_path = "dvren_sdf_sphere.ppm";
    std::ofstream ppm(ppm_path, std::ios::binary);
    if (!ppm) {
        std::cerr << "Failed to open PPM file for writing: " << ppm_path << std::endl;
        return 1;
    }

    const float scale = image_stats.max > 0.0f ? 255.0f / image_stats.max : 0.0f;
    auto to_byte = [](float value) -> uint8_t {
        const float clamped = std::clamp(value, 0.0f, 255.0f);
        return static_cast<uint8_t>(clamped + 0.5f);
    };

    ppm << "P6\n" << actual_desc.width << ' ' << actual_desc.height << "\n255\n";
    for (size_t idx = 0; idx < pixel_count; ++idx) {
        const size_t base = idx * 3ULL;
        ppm.put(static_cast<char>(to_byte(forward.image[base + 0] * scale)));
        ppm.put(static_cast<char>(to_byte(forward.image[base + 1] * scale)));
        ppm.put(static_cast<char>(to_byte(forward.image[base + 2] * scale)));
    }
    ppm.flush();
    if (!ppm) {
        std::cerr << "Failed to write PPM image: " << ppm_path << std::endl;
        return 1;
    }

    const dvren::WorkspaceInfo ws = renderer.workspace_info();

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
    std::cout << "ActivePixels " << active_pixels
              << " / " << pixel_count << '\n';
    std::cout << "SampleDiff image=" << max_image_diff
              << " trans=" << max_trans_diff
              << " opacity=" << max_opacity_diff
              << " depth=" << max_depth_diff << '\n';
    std::cout << "WorkspaceBytes total=" << ws.total_bytes()
              << " rays=" << ws.ray_buffer_bytes
              << " samples=" << ws.sample_buffer_bytes
              << " intl=" << ws.integration_buffer_bytes
              << " image=" << ws.image_buffer_bytes
              << " grads=" << ws.gradient_buffer_bytes << '\n';
    std::cout << "OutputImage path=" << ppm_path
              << " scale=" << scale << '\n';

    return 0;
}
