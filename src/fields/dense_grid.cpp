#include "dvren/fields/dense_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

#include "dvren/core/tensor_utils.hpp"

namespace dvren {

namespace {

bool HasPositiveResolution(const std::array<int32_t, 3>& resolution) {
    return resolution[0] > 0 && resolution[1] > 0 && resolution[2] > 0;
}

inline float Clamp(float value, float min_value, float max_value) {
    return std::max(min_value, std::min(max_value, value));
}

inline size_t VoxelIndex(const std::array<int32_t, 3>& resolution, int32_t ix, int32_t iy, int32_t iz) {
    return static_cast<size_t>((iz * resolution[1] + iy) * resolution[0] + ix);
}

}  // namespace

DenseGridField::~DenseGridField() {
    Release();
}

DenseGridField::DenseGridField(DenseGridField&& other) noexcept {
    sigma_field_ = other.sigma_field_;
    color_field_ = other.color_field_;
    resolution_ = other.resolution_;
    bbox_min_ = other.bbox_min_;
    bbox_max_ = other.bbox_max_;
    interp_ = other.interp_;
    oob_ = other.oob_;
    sigma_data_ = std::move(other.sigma_data_);
    color_data_ = std::move(other.color_data_);
    sigma_grad_ = std::move(other.sigma_grad_);
    color_grad_ = std::move(other.color_grad_);
    other.sigma_field_ = nullptr;
    other.color_field_ = nullptr;
}

DenseGridField& DenseGridField::operator=(DenseGridField&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    Release();
    sigma_field_ = other.sigma_field_;
    color_field_ = other.color_field_;
    resolution_ = other.resolution_;
    bbox_min_ = other.bbox_min_;
    bbox_max_ = other.bbox_max_;
    interp_ = other.interp_;
    oob_ = other.oob_;
    sigma_data_ = std::move(other.sigma_data_);
    color_data_ = std::move(other.color_data_);
    sigma_grad_ = std::move(other.sigma_grad_);
    color_grad_ = std::move(other.color_grad_);
    other.sigma_field_ = nullptr;
    other.color_field_ = nullptr;
    return *this;
}

Status DenseGridField::Create(const Context& ctx, const DenseGridConfig& config, DenseGridField& out) {
    if (!ctx.valid()) {
        return Status(StatusCode::kInvalidArgument, "context is invalid");
    }
    if (!HasPositiveResolution(config.resolution)) {
        return Status(StatusCode::kInvalidArgument, "resolution must be positive");
    }
    const int64_t nx = static_cast<int64_t>(config.resolution[0]);
    const int64_t ny = static_cast<int64_t>(config.resolution[1]);
    const int64_t nz = static_cast<int64_t>(config.resolution[2]);
    const int64_t voxel_count = nx * ny * nz;
    if (voxel_count <= 0) {
        return Status(StatusCode::kInvalidArgument, "voxel count must be positive");
    }
    if (static_cast<int64_t>(config.sigma.size()) != voxel_count) {
        return Status(StatusCode::kInvalidArgument, "sigma data size mismatch");
    }
    if (static_cast<int64_t>(config.color.size()) != voxel_count * 3) {
        return Status(StatusCode::kInvalidArgument, "color data size mismatch");
    }

    DenseGridField temp;
    temp.sigma_data_ = config.sigma;
    temp.color_data_ = config.color;

    hp_tensor sigma_tensor = MakeHostTensor(
        temp.sigma_data_.data(),
        HP_DTYPE_F32,
        {nz, ny, nx});

    hp_tensor color_tensor = MakeHostTensor(
        temp.color_data_.data(),
        HP_DTYPE_F32,
        {nz, ny, nx, 3});

    hp_field* sigma_field = nullptr;
    const hp_status sigma_status = hp_field_create_grid_sigma(
        ctx.handle(),
        &sigma_tensor,
        static_cast<uint32_t>(config.interp),
        static_cast<uint32_t>(config.oob),
        &sigma_field);
    if (sigma_status != HP_STATUS_SUCCESS || sigma_field == nullptr) {
        return Status::FromHotpath(sigma_status, "hp_field_create_grid_sigma failed");
    }

    hp_field* color_field = nullptr;
    const hp_status color_status = hp_field_create_grid_color(
        ctx.handle(),
        &color_tensor,
        static_cast<uint32_t>(config.interp),
        static_cast<uint32_t>(config.oob),
        &color_field);
    if (color_status != HP_STATUS_SUCCESS || color_field == nullptr) {
        hp_field_release(sigma_field);
        return Status::FromHotpath(color_status, "hp_field_create_grid_color failed");
    }

    temp.Reset(sigma_field, color_field, config.resolution, config);
    out = std::move(temp);
    return Status::Ok();
}

void DenseGridField::Reset(hp_field* sigma,
                           hp_field* color,
                           std::array<int32_t, 3> resolution,
                           const DenseGridConfig& config) {
    Release();
    sigma_field_ = sigma;
    color_field_ = color;
    resolution_ = resolution;
    bbox_min_ = config.bbox_min;
    bbox_max_ = config.bbox_max;
    interp_ = config.interp;
    oob_ = config.oob;

    const size_t voxel_count = this->voxel_count();
    sigma_grad_.assign(voxel_count, 0.0f);
    color_grad_.assign(voxel_count * 3ULL, 0.0f);

}

void DenseGridField::Release() {
    if (sigma_field_ != nullptr) {
        hp_field_release(sigma_field_);
        sigma_field_ = nullptr;
    }
    if (color_field_ != nullptr) {
        hp_field_release(color_field_);
        color_field_ = nullptr;
    }
    sigma_data_.clear();
    color_data_.clear();
    sigma_grad_.clear();
    color_grad_.clear();
}

void DenseGridField::ZeroGradients() {
    std::fill(sigma_grad_.begin(), sigma_grad_.end(), 0.0f);
    std::fill(color_grad_.begin(), color_grad_.end(), 0.0f);
}

Status DenseGridField::AccumulateSampleGradients(const hp_samp_t& samples,
                                                 std::span<const float> grad_sigma,
                                                 std::span<const float> grad_color) {
    if (samples.positions.memspace != HP_MEMSPACE_HOST ||
        samples.positions.dtype != HP_DTYPE_F32 ||
        samples.dt.memspace != HP_MEMSPACE_HOST ||
        samples.dt.dtype != HP_DTYPE_F32) {
        return Status(StatusCode::kInvalidArgument, "samples must reside on host");
    }

    const int64_t position_rank = static_cast<int64_t>(samples.positions.rank);
    if (position_rank != 2 || samples.positions.shape[1] != 3) {
        return Status(StatusCode::kInvalidArgument, "sample positions must have shape (M,3)");
    }

    const size_t sample_count = static_cast<size_t>(samples.positions.shape[0]);
    if (grad_sigma.size() != sample_count) {
        return Status(StatusCode::kInvalidArgument, "grad_sigma size mismatch");
    }
    if (grad_color.size() != sample_count * 3ULL) {
        return Status(StatusCode::kInvalidArgument, "grad_color size mismatch");
    }

    if (voxel_count() == 0) {
        return Status(StatusCode::kInvalidArgument, "voxel count is zero");
    }

    const auto nx = resolution_[0];
    const auto ny = resolution_[1];
    const auto nz = resolution_[2];
    const float extent_x = bbox_max_[0] - bbox_min_[0];
    const float extent_y = bbox_max_[1] - bbox_min_[1];
    const float extent_z = bbox_max_[2] - bbox_min_[2];

    const float* positions = static_cast<const float*>(samples.positions.data);
    for (size_t idx = 0; idx < sample_count; ++idx) {
        const float px = positions[idx * 3U + 0];
        const float py = positions[idx * 3U + 1];
        const float pz = positions[idx * 3U + 2];

        float local_x = extent_x != 0.0f ? (px - bbox_min_[0]) / extent_x : 0.0f;
        float local_y = extent_y != 0.0f ? (py - bbox_min_[1]) / extent_y : 0.0f;
        float local_z = extent_z != 0.0f ? (pz - bbox_min_[2]) / extent_z : 0.0f;

        bool outside = local_x < 0.0f || local_x > 1.0f ||
                       local_y < 0.0f || local_y > 1.0f ||
                       local_z < 0.0f || local_z > 1.0f;

        if (outside) {
            if (oob_ == HP_OOB_ZERO) {
                continue;
            }
            local_x = Clamp(local_x, 0.0f, 1.0f);
            local_y = Clamp(local_y, 0.0f, 1.0f);
            local_z = Clamp(local_z, 0.0f, 1.0f);
        }

        const float grid_x = local_x * static_cast<float>(std::max(nx - 1, 1));
        const float grid_y = local_y * static_cast<float>(std::max(ny - 1, 1));
        const float grid_z = local_z * static_cast<float>(std::max(nz - 1, 1));

        if (interp_ == HP_INTERP_NEAREST || nx == 1 || ny == 1 || nz == 1) {
            const int32_t ix = static_cast<int32_t>(std::round(grid_x));
            const int32_t iy = static_cast<int32_t>(std::round(grid_y));
            const int32_t iz = static_cast<int32_t>(std::round(grid_z));
            if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
                continue;
            }
            const size_t voxel = VoxelIndex(resolution_, ix, iy, iz);
            sigma_grad_[voxel] += grad_sigma[idx];
            const size_t color_base = voxel * 3ULL;
            color_grad_[color_base + 0] += grad_color[idx * 3U + 0];
            color_grad_[color_base + 1] += grad_color[idx * 3U + 1];
            color_grad_[color_base + 2] += grad_color[idx * 3U + 2];
            continue;
        }

        const int32_t ix0 = static_cast<int32_t>(std::floor(grid_x));
        const int32_t iy0 = static_cast<int32_t>(std::floor(grid_y));
        const int32_t iz0 = static_cast<int32_t>(std::floor(grid_z));
        const int32_t ix1 = std::min(ix0 + 1, nx - 1);
        const int32_t iy1 = std::min(iy0 + 1, ny - 1);
        const int32_t iz1 = std::min(iz0 + 1, nz - 1);

        const float tx = grid_x - static_cast<float>(ix0);
        const float ty = grid_y - static_cast<float>(iy0);
        const float tz = grid_z - static_cast<float>(iz0);

        const float w000 = (1.0f - tx) * (1.0f - ty) * (1.0f - tz);
        const float w100 = tx * (1.0f - ty) * (1.0f - tz);
        const float w010 = (1.0f - tx) * ty * (1.0f - tz);
        const float w110 = tx * ty * (1.0f - tz);
        const float w001 = (1.0f - tx) * (1.0f - ty) * tz;
        const float w101 = tx * (1.0f - ty) * tz;
        const float w011 = (1.0f - tx) * ty * tz;
        const float w111 = tx * ty * tz;

        const float grad_sigma_sample = grad_sigma[idx];
        const float grad_color_r = grad_color[idx * 3U + 0];
        const float grad_color_g = grad_color[idx * 3U + 1];
        const float grad_color_b = grad_color[idx * 3U + 2];

        const std::array<int32_t, 2> xs{ix0, ix1};
        const std::array<int32_t, 2> ys{iy0, iy1};
        const std::array<int32_t, 2> zs{iz0, iz1};
        const float weights[2][2][2] = {
            {
                {w000, w001},
                {w010, w011}
            },
            {
                {w100, w101},
                {w110, w111}
            }
        };

        for (int dx = 0; dx < 2; ++dx) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dz = 0; dz < 2; ++dz) {
                    const int32_t ix = xs[dx];
                    const int32_t iy = ys[dy];
                    const int32_t iz = zs[dz];
                    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
                        continue;
                    }
                    const size_t voxel = VoxelIndex(resolution_, ix, iy, iz);
                    const float w = weights[dx][dy][dz];
                    sigma_grad_[voxel] += grad_sigma_sample * w;
                    const size_t base = voxel * 3ULL;
                    color_grad_[base + 0] += grad_color_r * w;
                    color_grad_[base + 1] += grad_color_g * w;
                    color_grad_[base + 2] += grad_color_b * w;
                }
            }
        }
    }

    return Status::Ok();
}

}  // namespace dvren

