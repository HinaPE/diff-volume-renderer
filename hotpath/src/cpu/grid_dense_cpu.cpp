#include "hp_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace {

struct GridShape {
    int32_t nx{};
    int32_t ny{};
    int32_t nz{};
    int32_t channels{};
};

GridShape infer_sigma_shape(const hp_tensor& tensor) {
    GridShape shape{};
    if (tensor.rank >= 3) {
        shape.nz = static_cast<int32_t>(tensor.shape[0]);
        shape.ny = static_cast<int32_t>(tensor.shape[1]);
        shape.nx = static_cast<int32_t>(tensor.shape[2]);
    }
    return shape;
}

GridShape infer_color_shape(const hp_tensor& tensor) {
    GridShape shape{};
    if (tensor.rank >= 4) {
        shape.nz = static_cast<int32_t>(tensor.shape[0]);
        shape.ny = static_cast<int32_t>(tensor.shape[1]);
        shape.nx = static_cast<int32_t>(tensor.shape[2]);
        shape.channels = static_cast<int32_t>(tensor.shape[3]);
    }
    return shape;
}

inline bool within_bounds(const GridShape& shape, int32_t ix, int32_t iy, int32_t iz) {
    return ix >= 0 && ix < shape.nx &&
           iy >= 0 && iy < shape.ny &&
           iz >= 0 && iz < shape.nz;
}

inline size_t sigma_index(const GridShape& shape, int32_t ix, int32_t iy, int32_t iz) {
    return static_cast<size_t>((iz * shape.ny + iy) * shape.nx + ix);
}

inline size_t color_index(const GridShape& shape, int32_t ix, int32_t iy, int32_t iz, int32_t channel) {
    return static_cast<size_t>(((iz * shape.ny + iy) * shape.nx + ix) * shape.channels + channel);
}

inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

template <typename FetchFn>
float trilinear_sample(const FetchFn& fetch, float fx, float fy, float fz, const GridShape& shape) {
    const int32_t ix0 = static_cast<int32_t>(std::floor(fx));
    const int32_t iy0 = static_cast<int32_t>(std::floor(fy));
    const int32_t iz0 = static_cast<int32_t>(std::floor(fz));
    const int32_t ix1 = std::min(ix0 + 1, shape.nx - 1);
    const int32_t iy1 = std::min(iy0 + 1, shape.ny - 1);
    const int32_t iz1 = std::min(iz0 + 1, shape.nz - 1);
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

    const float c00 = lerp(c000, c100, tx);
    const float c10 = lerp(c010, c110, tx);
    const float c01 = lerp(c001, c101, tx);
    const float c11 = lerp(c011, c111, tx);

    const float c0 = lerp(c00, c10, ty);
    const float c1 = lerp(c01, c11, ty);
    return lerp(c0, c1, tz);
}

struct GridCoordinates {
    float fx;
    float fy;
    float fz;
    bool outside;
};

GridCoordinates compute_grid_coordinates(const hp_field* field, const float pos[3]) {
    GridCoordinates coords{};
    coords.outside = false;

    float local[3];
    for (int i = 0; i < 3; ++i) {
        const float extent = field->world_max[i] - field->world_min[i];
        const float coord = extent != 0.0f ? (pos[i] - field->world_min[i]) / extent : 0.0f;
        local[i] = coord;
        if (coord < 0.0f || coord > 1.0f) {
            coords.outside = true;
        }
    }

    if (field->oob == HP_OOB_CLAMP) {
        for (float& v : local) {
            v = std::clamp(v, 0.0f, 1.0f);
        }
        coords.outside = false;
    }

    coords.fx = local[0];
    coords.fy = local[1];
    coords.fz = local[2];
    return coords;
}

}  // namespace

namespace hp_internal {

float sample_grid_sigma_cpu(const hp_field* field, const float pos[3], hp_status* status) {
    if (status != nullptr) {
        *status = HP_STATUS_SUCCESS;
    }
    if (field == nullptr || field->kind != hp_field_kind::dense_sigma) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return 0.0f;
    }
    const hp_tensor& tensor = field->source;
    if (tensor.data == nullptr) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return 0.0f;
    }

    const GridShape shape = infer_sigma_shape(tensor);
    if (shape.nx <= 0 || shape.ny <= 0 || shape.nz <= 0) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return 0.0f;
    }

    const float* data = static_cast<const float*>(tensor.data);
    const GridCoordinates coords = compute_grid_coordinates(field, pos);
    if (coords.outside && field->oob == HP_OOB_ZERO) {
        return 0.0f;
    }

    const float fx = coords.fx * static_cast<float>(shape.nx - 1);
    const float fy = coords.fy * static_cast<float>(shape.ny - 1);
    const float fz = coords.fz * static_cast<float>(shape.nz - 1);

    const auto fetch = [&](int32_t ix, int32_t iy, int32_t iz) -> float {
        if (!within_bounds(shape, ix, iy, iz)) {
            return 0.0f;
        }
        return data[sigma_index(shape, ix, iy, iz)];
    };

    if (field->interp == HP_INTERP_NEAREST) {
        const int32_t ix = static_cast<int32_t>(std::round(fx));
        const int32_t iy = static_cast<int32_t>(std::round(fy));
        const int32_t iz = static_cast<int32_t>(std::round(fz));
        return fetch(ix, iy, iz);
    }

    return trilinear_sample(fetch, fx, fy, fz, shape);
}

void sample_grid_color_cpu(const hp_field* field, const float pos[3], float out_rgb[3], hp_status* status) {
    if (status != nullptr) {
        *status = HP_STATUS_SUCCESS;
    }
    if (out_rgb == nullptr) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return;
    }
    out_rgb[0] = 0.0f;
    out_rgb[1] = 0.0f;
    out_rgb[2] = 0.0f;

    if (field == nullptr || field->kind != hp_field_kind::dense_color) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return;
    }
    const hp_tensor& tensor = field->source;
    if (tensor.data == nullptr) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return;
    }

    const GridShape shape = infer_color_shape(tensor);
    if (shape.nx <= 0 || shape.ny <= 0 || shape.nz <= 0 || shape.channels < 3) {
        if (status != nullptr) {
            *status = HP_STATUS_INVALID_ARGUMENT;
        }
        return;
    }

    const float* data = static_cast<const float*>(tensor.data);
    const GridCoordinates coords = compute_grid_coordinates(field, pos);
    if (coords.outside && field->oob == HP_OOB_ZERO) {
        return;
    }

    const float fx = coords.fx * static_cast<float>(shape.nx - 1);
    const float fy = coords.fy * static_cast<float>(shape.ny - 1);
    const float fz = coords.fz * static_cast<float>(shape.nz - 1);

    const auto fetch_channel = [&](int channel) {
        const auto fetch = [&](int32_t ix, int32_t iy, int32_t iz) -> float {
            if (!within_bounds(shape, ix, iy, iz)) {
                return 0.0f;
            }
            return data[color_index(shape, ix, iy, iz, channel)];
        };

        if (field->interp == HP_INTERP_NEAREST) {
            const int32_t ix = static_cast<int32_t>(std::round(fx));
            const int32_t iy = static_cast<int32_t>(std::round(fy));
            const int32_t iz = static_cast<int32_t>(std::round(fz));
            return fetch(ix, iy, iz);
        }

        return trilinear_sample(fetch, fx, fy, fz, shape);
    };

    out_rgb[0] = fetch_channel(0);
    out_rgb[1] = fetch_channel(1);
    out_rgb[2] = fetch_channel(2);
}

}  // namespace hp_internal
