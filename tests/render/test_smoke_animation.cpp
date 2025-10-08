#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dvren/core/context.hpp"
#include "dvren/core/plan.hpp"
#include "dvren/fields/dense_grid.hpp"
#include "dvren/render/renderer.hpp"
#include "smoke_test_utils.hpp"

namespace {

struct GridPrecompute {
    std::vector<float> x_norm;
    std::vector<float> y_norm;
    std::vector<float> z_norm;

    explicit GridPrecompute(const std::array<int32_t, 3>& resolution) {
        x_norm.resize(static_cast<size_t>(resolution[0]));
        y_norm.resize(static_cast<size_t>(resolution[1]));
        z_norm.resize(static_cast<size_t>(resolution[2]));

        const float nx_inv = 1.0f / static_cast<float>(resolution[0]);
        const float ny_inv = 1.0f / static_cast<float>(resolution[1]);
        const float nz_inv = 1.0f / static_cast<float>(resolution[2]);

        for (int32_t ix = 0; ix < resolution[0]; ++ix) {
            x_norm[static_cast<size_t>(ix)] = (static_cast<float>(ix) + 0.5f) * nx_inv;
        }
        for (int32_t iy = 0; iy < resolution[1]; ++iy) {
            y_norm[static_cast<size_t>(iy)] = (static_cast<float>(iy) + 0.5f) * ny_inv;
        }
        for (int32_t iz = 0; iz < resolution[2]; ++iz) {
            z_norm[static_cast<size_t>(iz)] = (static_cast<float>(iz) + 0.5f) * nz_inv;
        }
    }
};

struct FrameDensityStats {
    double sum{0.0};
    float min{std::numeric_limits<float>::max()};
    float max{0.0f};
};

FrameDensityStats PopulateSmokeFrame(dvren::DenseGridConfig& config,
                                     const GridPrecompute& precompute,
                                     float time_norm) {
    const auto& resolution = config.resolution;
    const int32_t nx = resolution[0];
    const int32_t ny = resolution[1];
    const int32_t nz = resolution[2];
    const size_t voxel_count =
        static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    config.sigma.assign(voxel_count, 0.0f);
    config.color.assign(voxel_count * 3ULL, 0.0f);

    const float two_pi = 6.28318530717958647692f;
    const float phase = two_pi * time_norm;
    constexpr float extent_x = 768.0f;
    constexpr float extent_y = 1024.0f;
    constexpr float extent_z = 768.0f;
    const float center_x = 0.5f * extent_x;
    const float center_z = 0.5f * extent_z;

    FrameDensityStats stats{};

    for (int32_t iz = 0; iz < nz; ++iz) {
        const float z = precompute.z_norm[static_cast<size_t>(iz)];
        const float z_phys = z * extent_z;
        const float spiral_z = smoke_test::RandomizedSine(z * 4.0f - 0.7f * time_norm,
                                                          z * 2.3f + 0.4f * std::sin(phase),
                                                          0.5f * time_norm);
        for (int32_t iy = 0; iy < ny; ++iy) {
            const float y = precompute.y_norm[static_cast<size_t>(iy)];
            const float y_phys = y * extent_y;
            const float rise_center = (0.10f + 0.6f * time_norm) * extent_y;
            const float buoyancy = std::exp(-(y_phys - rise_center) * (y_phys - rise_center) /
                                            (1800.0f + 2600.0f * time_norm));
            const float taper = std::exp(-y * 2.0f) * (0.9f - 0.3f * time_norm) + 0.1f;
            for (int32_t ix = 0; ix < nx; ++ix) {
                const float x = precompute.x_norm[static_cast<size_t>(ix)];
                const float x_phys = x * extent_x;

                const float swirl = smoke_test::RandomizedSine(
                    x * 5.0f + 0.6f * time_norm,
                    y * 6.0f - 0.4f * std::cos(phase),
                    z * 5.0f + 0.9f * time_norm);
                const float vortex = smoke_test::RandomizedSine(
                    z * 6.0f - 0.5f * time_norm,
                    x * 6.0f + 0.3f * std::sin(phase),
                    y * 4.0f + 0.7f * std::cos(phase));

                const float dx = x_phys - (center_x + 110.0f * std::sin(phase + z * 1.8f));
                const float dz = z_phys - (center_z + 110.0f * std::cos(phase + x * 1.8f));
                const float radial = dx * dx + dz * dz;
                const float column = std::exp(-radial / (2800.0f + 3600.0f * std::abs(swirl)));

                float sigma_val = 14.0f * column * buoyancy * taper *
                                  (0.5f + 0.5f * 0.5f * (swirl + vortex + spiral_z));
                sigma_val = std::max(sigma_val, 0.0f);

                const size_t idx = (static_cast<size_t>(iz) * static_cast<size_t>(ny) +
                                    static_cast<size_t>(iy)) * static_cast<size_t>(nx) +
                                   static_cast<size_t>(ix);
                config.sigma[idx] = sigma_val;
                stats.sum += sigma_val;
                stats.min = std::min(stats.min, sigma_val);
                stats.max = std::max(stats.max, sigma_val);

                const float temperature = std::clamp(
                    0.55f + 0.35f * (1.0f - y) + 0.1f * vortex - 0.05f * swirl,
                    0.0f,
                    1.0f);
                const float brightness = std::clamp(sigma_val * 0.055f, 0.0f, 1.0f);
                const size_t base = idx * 3ULL;
                config.color[base + 0] = brightness * (0.55f + 0.35f * temperature);
                config.color[base + 1] = brightness * (0.5f + 0.3f * temperature);
                config.color[base + 2] = brightness * (0.4f + 0.4f * (1.0f - temperature));
            }
        }
    }

    if (!std::isfinite(stats.min)) {
        stats.min = 0.0f;
    }
    return stats;
}

std::vector<std::pair<uint32_t, uint32_t>> BuildSampleGrid(uint32_t width,
                                                           uint32_t height) {
    std::vector<std::pair<uint32_t, uint32_t>> samples;
    const uint32_t stride_x = 64;
    const uint32_t stride_y = 48;
    for (uint32_t py = 0; py < height; py += stride_y) {
        for (uint32_t px = 0; px < width; px += stride_x) {
            samples.emplace_back(px, py);
        }
    }
    samples.emplace_back(width / 2, height / 2);
    samples.emplace_back(width - 1, height / 2);
    samples.emplace_back(width / 2, height - 1);
    samples.emplace_back(width - 1, height - 1);
    return samples;
}

bool WritePPM(const std::filesystem::path& path,
              const std::vector<float>& image,
              uint32_t width,
              uint32_t height,
              float scale) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    auto to_byte = [](float value) -> uint8_t {
        const float clamped = std::clamp(value, 0.0f, 255.0f);
        return static_cast<uint8_t>(clamped + 0.5f);
    };
    out << "P6\n" << width << ' ' << height << "\n255\n";
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t idx = 0; idx < pixel_count; ++idx) {
        const size_t base = idx * 3ULL;
        out.put(static_cast<char>(to_byte(image[base + 0] * scale)));
        out.put(static_cast<char>(to_byte(image[base + 1] * scale)));
        out.put(static_cast<char>(to_byte(image[base + 2] * scale)));
    }
    out.flush();
    return static_cast<bool>(out);
}

struct DiffAccumulator {
    double sum{0.0};
    double sumsq{0.0};
    size_t count{0};
    float max{0.0f};

    void Add(float value) {
        const double v = static_cast<double>(value);
        sum += v;
        sumsq += v * v;
        ++count;
        if (value > max) {
            max = value;
        }
    }

    std::pair<double, double> MeanStd() const {
        if (count == 0) {
            return {0.0, 0.0};
        }
        const double mean = sum / static_cast<double>(count);
        const double variance = std::max(sumsq / static_cast<double>(count) - mean * mean, 0.0);
        return {mean, std::sqrt(variance)};
    }
};

}  // namespace

int main() {
    constexpr uint32_t kFrameCount = 120;
    const std::filesystem::path output_dir{"dvren_smoke_animation_frames"};

    dvren::Context ctx;
    dvren::Status status = dvren::Context::Create(dvren::ContextOptions{}, ctx);
    if (!status.ok()) {
        std::cerr << "Context::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::PlanDescriptor plan_desc{};
    plan_desc.width = 1024;
    plan_desc.height = 768;
    plan_desc.t_near = 0.0f;
    plan_desc.t_far = 2.4f;
    plan_desc.sampling.dt = 0.02f;
    plan_desc.sampling.max_steps = 160;
    plan_desc.sampling.mode = dvren::SamplingMode::kFixed;
    plan_desc.max_rays = plan_desc.width * plan_desc.height;
    plan_desc.max_samples = plan_desc.max_rays * plan_desc.sampling.max_steps;
    plan_desc.seed = 47;

    plan_desc.camera.model = dvren::CameraModel::kPinhole;
    plan_desc.camera.K = {950.0f, 0.0f, 512.0f,
                          0.0f, 950.0f, 384.0f,
                          0.0f, 0.0f, 1.0f};
    plan_desc.camera.c2w = {1.0f, 0.0f, 0.0f, 0.5f,
                            0.0f, 1.0f, 0.0f, 0.35f,
                            0.0f, 0.0f, 1.0f, -1.6f};

    dvren::Plan plan;
    status = dvren::Plan::Create(ctx, plan_desc, plan);
    if (!status.ok()) {
        std::cerr << "Plan::Create failed: " << status.ToString() << std::endl;
        return 1;
    }

    dvren::RenderOptions render_options{};
    render_options.use_fused_path = true;

    dvren::Renderer renderer(ctx, plan, render_options);

    dvren::DenseGridConfig grid_config{};
    grid_config.resolution = {160, 216, 160};
    grid_config.bbox_min = {0.0f, 0.0f, 0.0f};
    grid_config.bbox_max = {1.0f, 1.0f, 1.0f};
    grid_config.interp = HP_INTERP_LINEAR;
    grid_config.oob = HP_OOB_ZERO;

    GridPrecompute precompute(grid_config.resolution);

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "Failed to create output directory: " << output_dir << " (" << ec.message() << ")\n";
        return 1;
    }

    const auto sample_pixels = BuildSampleGrid(plan_desc.width, plan_desc.height);
    const hp_plan_desc& actual_desc = plan.descriptor();
    const size_t pixel_count =
        static_cast<size_t>(actual_desc.width) * static_cast<size_t>(actual_desc.height);

    double global_image_sum = 0.0;
    double global_image_sumsq = 0.0;
    size_t global_image_count = 0;
    float global_image_min = std::numeric_limits<float>::max();
    float global_image_max = 0.0f;

    double global_trans_sum = 0.0;
    double global_trans_sumsq = 0.0;
    size_t global_trans_count = 0;
    float global_trans_min = std::numeric_limits<float>::max();
    float global_trans_max = 0.0f;

    double global_opacity_sum = 0.0;
    double global_opacity_sumsq = 0.0;
    size_t global_opacity_count = 0;
    float global_opacity_min = std::numeric_limits<float>::max();
    float global_opacity_max = 0.0f;

    double global_depth_sum = 0.0;
    double global_depth_sumsq = 0.0;
    size_t global_depth_count = 0;
    float global_depth_min = std::numeric_limits<float>::max();
    float global_depth_max = 0.0f;

    double total_sigma_sum = 0.0;

    DiffAccumulator image_diff_acc;
    DiffAccumulator trans_diff_acc;
    DiffAccumulator opacity_diff_acc;
    DiffAccumulator depth_diff_acc;

    for (uint32_t frame = 0; frame < kFrameCount; ++frame) {
        const float time_norm = kFrameCount > 1
            ? static_cast<float>(frame) / static_cast<float>(kFrameCount - 1)
            : 0.0f;

        const FrameDensityStats density = PopulateSmokeFrame(grid_config, precompute, time_norm);
        if (!(density.sum > 0.0) || density.max <= 1e-6f) {
            std::cerr << "Frame " << frame << " produced insufficient density (sum=" << density.sum
                      << ", max=" << density.max << ")" << std::endl;
            return 1;
        }
        total_sigma_sum += density.sum;

        dvren::DenseGridField field;
        status = dvren::DenseGridField::Create(ctx, grid_config, field);
        if (!status.ok()) {
            std::cerr << "DenseGridField::Create failed at frame " << frame << ": "
                      << status.ToString() << std::endl;
            return 1;
        }

        dvren::ForwardResult forward{};
        status = renderer.Forward(field, forward);
        if (!status.ok()) {
            std::cerr << "Renderer::Forward failed at frame " << frame << ": "
                      << status.ToString() << std::endl;
            return 1;
        }

        if (forward.image.size() != pixel_count * 3ULL ||
            forward.transmittance.size() != pixel_count ||
            forward.opacity.size() != pixel_count ||
            forward.depth.size() != pixel_count ||
            forward.hitmask.size() != pixel_count) {
            std::cerr << "Frame " << frame << " buffer size mismatch" << std::endl;
            return 1;
        }

        if (forward.sample_count == 0 || forward.ray_count != pixel_count) {
            std::cerr << "Frame " << frame << " invalid sample/ray count (samples="
                      << forward.sample_count << ", rays=" << forward.ray_count << ")" << std::endl;
            return 1;
        }

        if (frame == 0) {
            std::cerr << "Frame 0 diagnostic: samples=" << forward.sample_count
                      << " image0=" << (forward.image.empty() ? 0.0f : forward.image[0])
                      << " trans0=" << (forward.transmittance.empty() ? 0.0f : forward.transmittance[0])
                      << " opacity0=" << (forward.opacity.empty() ? 0.0f : forward.opacity[0])
                      << std::endl;
        }

        const smoke_test::FieldStats image_stats = smoke_test::ComputeStats(forward.image);
        const smoke_test::FieldStats trans_stats = smoke_test::ComputeStats(forward.transmittance);
        const smoke_test::FieldStats opacity_stats = smoke_test::ComputeStats(forward.opacity);
        const smoke_test::FieldStats depth_stats = smoke_test::ComputeStats(forward.depth);

        if (image_stats.max <= image_stats.min + 1e-6f) {
            std::cerr << "Frame " << frame << " image lacks variation (min=" << image_stats.min
                      << ", max=" << image_stats.max << ", density_max=" << density.max << ")"
                      << std::endl;
            return 1;
        }
        if (trans_stats.max <= trans_stats.min + 1e-5f) {
            std::cerr << "Frame " << frame << " transmittance lacks variation" << std::endl;
            return 1;
        }

        const size_t active_pixels = std::count_if(
            forward.hitmask.begin(),
            forward.hitmask.end(),
            [](uint32_t v) { return v != 0U; });
        if (active_pixels == 0) {
            std::cerr << "Frame " << frame << " hitmask empty" << std::endl;
            return 1;
        }

        if (!std::all_of(forward.transmittance.begin(), forward.transmittance.end(),
                         [](float v) { return v >= 0.0f && v <= 1.0f; })) {
            std::cerr << "Frame " << frame << " transmittance out of range" << std::endl;
            return 1;
        }
        if (!std::all_of(forward.opacity.begin(), forward.opacity.end(),
                         [](float v) { return v >= 0.0f && v <= 1.0f; })) {
            std::cerr << "Frame " << frame << " opacity out of range" << std::endl;
            return 1;
        }

        global_image_sum += std::accumulate(forward.image.begin(), forward.image.end(), 0.0);
        global_image_sumsq += std::accumulate(
            forward.image.begin(),
            forward.image.end(),
            0.0,
            [](double acc, float v) { return acc + static_cast<double>(v) * static_cast<double>(v); });
        global_image_count += forward.image.size();
        global_image_min = std::min(global_image_min, image_stats.min);
        global_image_max = std::max(global_image_max, image_stats.max);

        global_trans_sum += std::accumulate(forward.transmittance.begin(), forward.transmittance.end(), 0.0);
        global_trans_sumsq += std::accumulate(
            forward.transmittance.begin(),
            forward.transmittance.end(),
            0.0,
            [](double acc, float v) { return acc + static_cast<double>(v) * static_cast<double>(v); });
        global_trans_count += forward.transmittance.size();
        global_trans_min = std::min(global_trans_min, trans_stats.min);
        global_trans_max = std::max(global_trans_max, trans_stats.max);

        global_opacity_sum += std::accumulate(forward.opacity.begin(), forward.opacity.end(), 0.0);
        global_opacity_sumsq += std::accumulate(
            forward.opacity.begin(),
            forward.opacity.end(),
            0.0,
            [](double acc, float v) { return acc + static_cast<double>(v) * static_cast<double>(v); });
        global_opacity_count += forward.opacity.size();
        global_opacity_min = std::min(global_opacity_min, opacity_stats.min);
        global_opacity_max = std::max(global_opacity_max, opacity_stats.max);

        global_depth_sum += std::accumulate(forward.depth.begin(), forward.depth.end(), 0.0);
        global_depth_sumsq += std::accumulate(
            forward.depth.begin(),
            forward.depth.end(),
            0.0,
            [](double acc, float v) { return acc + static_cast<double>(v) * static_cast<double>(v); });
        global_depth_count += forward.depth.size();
        global_depth_min = std::min(global_depth_min, depth_stats.min);
        global_depth_max = std::max(global_depth_max, depth_stats.max);

        for (const auto& [px, py] : sample_pixels) {
            smoke_test::PixelEvaluation eval =
                smoke_test::IntegratePixel(actual_desc, grid_config, px, py);
            const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(actual_desc.width) + px;
            const size_t base = idx * 3ULL;

            for (int c = 0; c < 3; ++c) {
                image_diff_acc.Add(std::fabs(forward.image[base + static_cast<size_t>(c)] - eval.radiance[c]));
            }
            trans_diff_acc.Add(std::fabs(forward.transmittance[idx] - eval.transmittance));
            opacity_diff_acc.Add(std::fabs(forward.opacity[idx] - eval.opacity));
            depth_diff_acc.Add(std::fabs(forward.depth[idx] - eval.depth));
        }

        const float scale = image_stats.max > 0.0f ? 255.0f / image_stats.max : 0.0f;
        std::ostringstream filename;
        filename << "frame_" << std::setw(3) << std::setfill('0') << frame << ".ppm";
        if (!WritePPM(output_dir / filename.str(), forward.image, actual_desc.width, actual_desc.height, scale)) {
            std::cerr << "Failed to write frame " << frame << " to disk" << std::endl;
            return 1;
        }
    }

    auto finalize_stats = [](double sum, double sumsq, size_t count) -> std::pair<double, double> {
        if (count == 0) {
            return {0.0, 0.0};
        }
        const double mean = sum / static_cast<double>(count);
        const double variance = std::max(sumsq / static_cast<double>(count) - mean * mean, 0.0);
        return {mean, std::sqrt(variance)};
    };

    const auto [image_mean, image_std] = finalize_stats(global_image_sum, global_image_sumsq, global_image_count);
    const auto [trans_mean, trans_std] = finalize_stats(global_trans_sum, global_trans_sumsq, global_trans_count);
    const auto [opacity_mean, opacity_std] = finalize_stats(global_opacity_sum, global_opacity_sumsq, global_opacity_count);
    const auto [depth_mean, depth_std] = finalize_stats(global_depth_sum, global_depth_sumsq, global_depth_count);

    const auto [image_diff_mean, image_diff_std] = image_diff_acc.MeanStd();
    const auto [trans_diff_mean, trans_diff_std] = trans_diff_acc.MeanStd();
    const auto [opacity_diff_mean, opacity_diff_std] = opacity_diff_acc.MeanStd();
    const auto [depth_diff_mean, depth_diff_std] = depth_diff_acc.MeanStd();

    if (image_diff_acc.max > 4e-3f ||
        trans_diff_acc.max > 4e-3f ||
        opacity_diff_acc.max > 4e-3f ||
        depth_diff_acc.max > 1e-2f) {
        std::cerr << "Animation subset drift too large: "
                  << "image=" << image_diff_acc.max
                  << " trans=" << trans_diff_acc.max
                  << " opacity=" << opacity_diff_acc.max
                  << " depth=" << depth_diff_acc.max << std::endl;
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "AnimationFrames=" << kFrameCount << '\n';
    std::cout << "SigmaTotal=" << total_sigma_sum << '\n';
    std::cout << "ImageStats range=[" << global_image_min << ", " << global_image_max
              << "] mean=" << image_mean << " std=" << image_std << '\n';
    std::cout << "TransmittanceStats range=[" << global_trans_min << ", " << global_trans_max
              << "] mean=" << trans_mean << " std=" << trans_std << '\n';
    std::cout << "OpacityStats range=[" << global_opacity_min << ", " << global_opacity_max
              << "] mean=" << opacity_mean << " std=" << opacity_std << '\n';
    std::cout << "DepthStats range=[" << global_depth_min << ", " << global_depth_max
              << "] mean=" << depth_mean << " std=" << depth_std << '\n';
    std::cout << "DifferenceStats image_max=" << image_diff_acc.max
              << " image_mean=" << image_diff_mean
              << " image_std=" << image_diff_std
              << " trans_max=" << trans_diff_acc.max
              << " trans_mean=" << trans_diff_mean
              << " trans_std=" << trans_diff_std
              << " opacity_max=" << opacity_diff_acc.max
              << " opacity_mean=" << opacity_diff_mean
              << " opacity_std=" << opacity_diff_std
              << " depth_max=" << depth_diff_acc.max
              << " depth_mean=" << depth_diff_mean
              << " depth_std=" << depth_diff_std << '\n';
    std::cout << "OutputDirectory=" << output_dir.generic_string() << '\n';
    return 0;
}
