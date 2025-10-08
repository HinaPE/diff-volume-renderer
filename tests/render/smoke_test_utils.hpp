#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "dvren/fields/dense_grid.hpp"
#include "hotpath/hp.h"

namespace smoke_test {

constexpr float kStopThreshold = 1e-4f;

inline float ComputeAlpha(float sigma, float dt) {
    const float optical_depth = sigma * dt;
    if (optical_depth <= 0.0f) {
        return 0.0f;
    }
    if (optical_depth < 1e-4f) {
        const float half = 0.5f * optical_depth;
        return optical_depth * (1.0f - half);
    }
    const double alpha = -std::expm1(-static_cast<double>(optical_depth));
    return static_cast<float>(std::clamp(alpha, 0.0, 1.0));
}

class VolumeSampler {
public:
    explicit VolumeSampler(const dvren::DenseGridConfig& config)
        : res_(config.resolution),
          sigma_(config.sigma),
          color_(config.color),
          bbox_min_(config.bbox_min),
          bbox_max_(config.bbox_max),
          interp_(config.interp),
          oob_(config.oob) {}

    void Sample(const std::array<float, 3>& pos, float& sigma, std::array<float, 3>& rgb) const {
        sigma = 0.0f;
        rgb = {0.0f, 0.0f, 0.0f};
        if (sigma_.empty() || color_.empty()) {
            return;
        }

        std::array<float, 3> local{};
        bool outside = false;
        for (int i = 0; i < 3; ++i) {
            const float extent = bbox_max_[i] - bbox_min_[i];
            float coord = 0.0f;
            if (extent != 0.0f) {
                coord = (pos[i] - bbox_min_[i]) / extent;
            }
            local[i] = coord;
            if (coord < 0.0f || coord > 1.0f) {
                outside = true;
            }
        }

        if (outside) {
            if (oob_ == HP_OOB_ZERO) {
                return;
            }
            for (float& v : local) {
                v = std::clamp(v, 0.0f, 1.0f);
            }
        }

        const int nx = res_[0];
        const int ny = res_[1];
        const int nz = res_[2];
        const float gx = local[0] * static_cast<float>(std::max(nx - 1, 0));
        const float gy = local[1] * static_cast<float>(std::max(ny - 1, 0));
        const float gz = local[2] * static_cast<float>(std::max(nz - 1, 0));

        if (interp_ == HP_INTERP_NEAREST || nx == 1 || ny == 1 || nz == 1) {
            const int ix = static_cast<int>(std::lround(gx));
            const int iy = static_cast<int>(std::lround(gy));
            const int iz = static_cast<int>(std::lround(gz));
            sigma = FetchSigma(ix, iy, iz);
            rgb[0] = FetchColor(ix, iy, iz, 0);
            rgb[1] = FetchColor(ix, iy, iz, 1);
            rgb[2] = FetchColor(ix, iy, iz, 2);
            return;
        }

        sigma = Trilinear([&](int ix, int iy, int iz) { return FetchSigma(ix, iy, iz); }, gx, gy, gz);
        rgb[0] = Trilinear([&](int ix, int iy, int iz) { return FetchColor(ix, iy, iz, 0); }, gx, gy, gz);
        rgb[1] = Trilinear([&](int ix, int iy, int iz) { return FetchColor(ix, iy, iz, 1); }, gx, gy, gz);
        rgb[2] = Trilinear([&](int ix, int iy, int iz) { return FetchColor(ix, iy, iz, 2); }, gx, gy, gz);
    }

private:
    template <typename FetchFn>
    float Trilinear(const FetchFn& fetch, float fx, float fy, float fz) const {
        const int ix0 = static_cast<int>(std::floor(fx));
        const int iy0 = static_cast<int>(std::floor(fy));
        const int iz0 = static_cast<int>(std::floor(fz));
        const int ix1 = std::min(ix0 + 1, res_[0] - 1);
        const int iy1 = std::min(iy0 + 1, res_[1] - 1);
        const int iz1 = std::min(iz0 + 1, res_[2] - 1);
        const float tx = fx - static_cast<float>(ix0);
        const float ty = fy - static_cast<float>(iy0);
        const float tz = fz - static_cast<float>(iz0);

        const float c000 = fetch(ix0, iy0, iz0);
        const float c100 = fetch(ix1, iy0, iz0);
        const float c010 = fetch(ix0, iy1, iz0);
        const float c110 = fetch(ix1, iy1, iz0);
        const float c001 = fetch(ix0, iy0, iz1);
        const float c101 = fetch(ix1, iy0, iz1);
        const float c011 = fetch(ix0, iy1, iz1);
        const float c111 = fetch(ix1, iy1, iz1);

        const float c00 = std::lerp(c000, c100, tx);
        const float c10 = std::lerp(c010, c110, tx);
        const float c01 = std::lerp(c001, c101, tx);
        const float c11 = std::lerp(c011, c111, tx);

        const float c0 = std::lerp(c00, c10, ty);
        const float c1 = std::lerp(c01, c11, ty);
        return std::lerp(c0, c1, tz);
    }

    float FetchSigma(int ix, int iy, int iz) const {
        if (ix < 0 || ix >= res_[0] || iy < 0 || iy >= res_[1] || iz < 0 || iz >= res_[2]) {
            return 0.0f;
        }
        const size_t idx = static_cast<size_t>((iz * res_[1] + iy) * res_[0] + ix);
        return sigma_[idx];
    }

    float FetchColor(int ix, int iy, int iz, int channel) const {
        if (ix < 0 || ix >= res_[0] || iy < 0 || iy >= res_[1] || iz < 0 || iz >= res_[2]) {
            return 0.0f;
        }
        const size_t base = static_cast<size_t>((iz * res_[1] + iy) * res_[0] + ix) * 3ULL;
        return color_[base + static_cast<size_t>(channel)];
    }

    std::array<int32_t, 3> res_{};
    const std::vector<float>& sigma_;
    const std::vector<float>& color_;
    std::array<float, 3> bbox_min_{};
    std::array<float, 3> bbox_max_{};
    hp_interp_mode interp_{HP_INTERP_LINEAR};
    hp_oob_policy oob_{HP_OOB_ZERO};
};

struct PixelEvaluation {
    std::array<float, 3> radiance{0.0f, 0.0f, 0.0f};
    float transmittance{1.0f};
    float opacity{0.0f};
    float depth{0.0f};
    uint32_t hitmask{0U};
};

inline PixelEvaluation IntegratePixel(const hp_plan_desc& desc,
                                      const dvren::DenseGridConfig& grid,
                                      uint32_t px,
                                      uint32_t py) {
    PixelEvaluation eval{};
    const uint32_t width = desc.width;
    const uint32_t height = desc.height;
    if (px >= width || py >= height) {
        return eval;
    }

    const float fx = desc.camera.K[0];
    const float fy = desc.camera.K[4];
    const float cx = desc.camera.K[2];
    const float cy = desc.camera.K[5];

    const float r00 = desc.camera.c2w[0];
    const float r01 = desc.camera.c2w[1];
    const float r02 = desc.camera.c2w[2];
    const float r10 = desc.camera.c2w[4];
    const float r11 = desc.camera.c2w[5];
    const float r12 = desc.camera.c2w[6];
    const float r20 = desc.camera.c2w[8];
    const float r21 = desc.camera.c2w[9];
    const float r22 = desc.camera.c2w[10];

    const std::array<float, 3> origin{
        desc.camera.c2w[3],
        desc.camera.c2w[7],
        desc.camera.c2w[11]
    };

    const float u = static_cast<float>(px) + 0.5f;
    const float v = static_cast<float>(py) + 0.5f;

    float dir_cam_x = (u - cx) / fx;
    float dir_cam_y = (v - cy) / fy;
    float dir_cam_z = 1.0f;
    if (desc.camera.model == HP_CAMERA_ORTHOGRAPHIC) {
        dir_cam_x = 0.0f;
        dir_cam_y = 0.0f;
        dir_cam_z = 1.0f;
    }

    std::array<float, 3> dir_world{
        r00 * dir_cam_x + r01 * dir_cam_y + r02 * dir_cam_z,
        r10 * dir_cam_x + r11 * dir_cam_y + r12 * dir_cam_z,
        r20 * dir_cam_x + r21 * dir_cam_y + r22 * dir_cam_z
    };
    const float len_sq = dir_world[0] * dir_world[0] +
                         dir_world[1] * dir_world[1] +
                         dir_world[2] * dir_world[2];
    const float inv_len = 1.0f / std::sqrt(std::max(len_sq, std::numeric_limits<float>::min()));
    for (float& d : dir_world) {
        d *= inv_len;
    }

    const float t_near = desc.t_near;
    const float t_far = desc.t_far;
    const float dt_step = desc.sampling.dt;
    const uint32_t max_steps = desc.sampling.max_steps;

    VolumeSampler sampler(grid);

    float T = 1.0f;
    std::array<float, 3> radiance{0.0f, 0.0f, 0.0f};
    float depth_weighted = 0.0f;
    float t_cursor = t_near;

    for (uint32_t step = 0; step < max_steps; ++step) {
        const float base_t = t_near + static_cast<float>(step) * dt_step;
        if (base_t >= t_far) {
            break;
        }
        const float segment_end = std::min(base_t + dt_step, t_far);
        const float dt_actual = segment_end - base_t;
        if (!(dt_actual > 0.0f)) {
            continue;
        }

        float sample_t = base_t + 0.5f * dt_step;
        if (sample_t >= t_far) {
            sample_t = std::nextafter(t_far, t_near);
        }

        std::array<float, 3> position{
            origin[0] + dir_world[0] * sample_t,
            origin[1] + dir_world[1] * sample_t,
            origin[2] + dir_world[2] * sample_t
        };

        float sigma_val = 0.0f;
        std::array<float, 3> color_val{};
        sampler.Sample(position, sigma_val, color_val);

        const float alpha = std::clamp(ComputeAlpha(sigma_val, dt_actual), 0.0f, 1.0f);
        const float weight = T * alpha;

        radiance[0] += weight * color_val[0];
        radiance[1] += weight * color_val[1];
        radiance[2] += weight * color_val[2];

        const float segment_mid = t_cursor + 0.5f * dt_actual;
        depth_weighted += weight * segment_mid;

        T *= std::max(1.0f - alpha, 0.0f);
        t_cursor += dt_actual;

        if (T <= kStopThreshold) {
            break;
        }
    }

    eval.radiance = radiance;
    eval.transmittance = T;
    eval.opacity = 1.0f - T;
    eval.depth = eval.opacity > 1e-6f ? depth_weighted / std::max(eval.opacity, 1e-6f) : t_far;
    eval.hitmask = eval.opacity > 0.0f ? 1U : 0U;
    return eval;
}

inline float RandomizedSine(float x, float y, float z) {
    return 0.5f + 0.5f * std::sin(9.0f * x + 7.0f * y + 5.0f * z + 1.3f);
}

inline void PopulateSmokeGrid(dvren::DenseGridConfig& config) {
    const int nx = config.resolution[0];
    const int ny = config.resolution[1];
    const int nz = config.resolution[2];
    const size_t voxel_count = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    config.sigma.assign(voxel_count, 0.0f);
    config.color.assign(voxel_count * 3ULL, 0.0f);

    for (int iz = 0; iz < nz; ++iz) {
        const float z = (static_cast<float>(iz) + 0.5f) / static_cast<float>(nz);
        for (int iy = 0; iy < ny; ++iy) {
            const float y = (static_cast<float>(iy) + 0.5f) / static_cast<float>(ny);
            for (int ix = 0; ix < nx; ++ix) {
                const float x = (static_cast<float>(ix) + 0.5f) / static_cast<float>(nx);
                const float radius = std::sqrt(std::max(0.0f, (x - 0.5f) * (x - 0.5f) + (y - 0.5f) * (y - 0.5f)));

                const float swirl = RandomizedSine(x, y, z);
                const float plume_profile =
                    std::exp(-radius * radius / 0.035f) *
                    (0.55f + 0.45f * std::exp(-(z - 0.35f) * (z - 0.35f) / 0.01f));
                const float vertical_layers =
                    0.35f * std::exp(-(z - 0.2f) * (z - 0.2f) / 0.02f) +
                    0.45f * std::exp(-(z - 0.55f) * (z - 0.55f) / 0.04f) +
                    0.25f * std::exp(-(z - 0.85f) * (z - 0.85f) / 0.03f);

                float sigma_val = 6.5f * plume_profile * vertical_layers * (0.6f + 0.4f * swirl);
                sigma_val = std::max(sigma_val, 0.0f);

                const float warm_tint = 0.4f + 0.6f * std::exp(-radius * 6.0f);
                const float cool_tint = 0.2f + 0.8f * (1.0f - z);
                const float accent = 0.1f + 0.9f * swirl;

                const float red = sigma_val * (0.25f * warm_tint + 0.05f * accent);
                const float green = sigma_val * (0.18f * warm_tint + 0.12f * accent);
                const float blue = sigma_val * (0.35f * cool_tint + 0.08f * accent);

                const size_t voxel_index = static_cast<size_t>((iz * ny + iy) * nx + ix);
                config.sigma[voxel_index] = sigma_val;
                const size_t color_base = voxel_index * 3ULL;
                config.color[color_base + 0] = red;
                config.color[color_base + 1] = green;
                config.color[color_base + 2] = blue;
            }
        }
    }
}

}  // namespace smoke_test

