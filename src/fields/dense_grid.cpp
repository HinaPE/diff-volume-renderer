#include "dvren/fields/dense_grid.hpp"

#include <numeric>

#include "dvren/core/tensor_utils.hpp"

namespace dvren {

namespace {

bool HasPositiveResolution(const std::array<int32_t, 3>& resolution) {
    return resolution[0] > 0 && resolution[1] > 0 && resolution[2] > 0;
}

}  // namespace

DenseGridField::~DenseGridField() {
    Release();
}

DenseGridField::DenseGridField(DenseGridField&& other) noexcept {
    sigma_field_ = other.sigma_field_;
    color_field_ = other.color_field_;
    resolution_ = other.resolution_;
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

    hp_tensor sigma_tensor = MakeHostTensor(
        const_cast<float*>(config.sigma.data()),
        HP_DTYPE_F32,
        {nz, ny, nx});

    hp_tensor color_tensor = MakeHostTensor(
        const_cast<float*>(config.color.data()),
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

    out.Reset(sigma_field, color_field, config.resolution);
    return Status::Ok();
}

void DenseGridField::Reset(hp_field* sigma, hp_field* color, std::array<int32_t, 3> resolution) {
    Release();
    sigma_field_ = sigma;
    color_field_ = color;
    resolution_ = resolution;
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
}

}  // namespace dvren

