#include "hp_internal.hpp"

#include "workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

struct IntlBuffers {
    float* radiance{nullptr};
    float* transmittance{nullptr};
    float* opacity{nullptr};
    float* depth{nullptr};
    float* aux{nullptr};
};

hp_status ensure_intl_buffers(hp_intl_t* intl,
                              size_t ray_count,
                              size_t sample_count,
                              hp_workspace_allocator& allocator,
                              IntlBuffers& buffers) {
    const int64_t ray_count_i64 = static_cast<int64_t>(ray_count);
    const int64_t sample_count_i64 = static_cast<int64_t>(sample_count);

    intl->radiance.dtype = HP_DTYPE_F32;
    intl->radiance.memspace = HP_MEMSPACE_HOST;
    intl->radiance.rank = 2;
    intl->radiance.shape[0] = ray_count_i64;
    intl->radiance.shape[1] = 3;
    intl->radiance.stride[1] = 1;
    intl->radiance.stride[0] = 3;

    intl->transmittance.dtype = HP_DTYPE_F32;
    intl->transmittance.memspace = HP_MEMSPACE_HOST;
    intl->transmittance.rank = 1;
    intl->transmittance.shape[0] = ray_count_i64;
    intl->transmittance.stride[0] = 1;

    intl->opacity.dtype = HP_DTYPE_F32;
    intl->opacity.memspace = HP_MEMSPACE_HOST;
    intl->opacity.rank = 1;
    intl->opacity.shape[0] = ray_count_i64;
    intl->opacity.stride[0] = 1;

    intl->depth.dtype = HP_DTYPE_F32;
    intl->depth.memspace = HP_MEMSPACE_HOST;
    intl->depth.rank = 1;
    intl->depth.shape[0] = ray_count_i64;
    intl->depth.stride[0] = 1;

    intl->aux.dtype = HP_DTYPE_F32;
    intl->aux.memspace = HP_MEMSPACE_HOST;
    intl->aux.rank = 2;
    intl->aux.shape[0] = sample_count_i64;
    intl->aux.shape[1] = 4;
    intl->aux.stride[1] = 1;
    intl->aux.stride[0] = 4;

    const size_t ray_vec_bytes = ray_count * 3U * sizeof(float);
    const size_t ray_scalar_bytes = ray_count * sizeof(float);
    const size_t aux_bytes = sample_count * 4U * sizeof(float);

    if (intl->radiance.data == nullptr && ray_vec_bytes > 0) {
        intl->radiance.data = allocator.allocate(ray_vec_bytes, alignof(float));
    }
    if (intl->transmittance.data == nullptr && ray_scalar_bytes > 0) {
        intl->transmittance.data = allocator.allocate(ray_scalar_bytes, alignof(float));
    }
    if (intl->opacity.data == nullptr && ray_scalar_bytes > 0) {
        intl->opacity.data = allocator.allocate(ray_scalar_bytes, alignof(float));
    }
    if (intl->depth.data == nullptr && ray_scalar_bytes > 0) {
        intl->depth.data = allocator.allocate(ray_scalar_bytes, alignof(float));
    }
    if (intl->aux.data == nullptr && aux_bytes > 0) {
        intl->aux.data = allocator.allocate(aux_bytes, alignof(float));
    }

    if ((ray_vec_bytes > 0 && intl->radiance.data == nullptr) ||
        (ray_scalar_bytes > 0 && (intl->transmittance.data == nullptr || intl->opacity.data == nullptr || intl->depth.data == nullptr)) ||
        (aux_bytes > 0 && intl->aux.data == nullptr)) {
        return HP_STATUS_OUT_OF_MEMORY;
    }

    buffers.radiance = static_cast<float*>(intl->radiance.data);
    buffers.transmittance = static_cast<float*>(intl->transmittance.data);
    buffers.opacity = static_cast<float*>(intl->opacity.data);
    buffers.depth = static_cast<float*>(intl->depth.data);
    buffers.aux = sample_count > 0 ? static_cast<float*>(intl->aux.data) : nullptr;
    return HP_STATUS_SUCCESS;
}

inline float compute_alpha(float sigma, float dt) {
    const float optical_depth = sigma * dt;
    if (optical_depth <= 0.0f) {
        return 0.0f;
    }
    if (optical_depth < 1e-4f) {
        const float half = 0.5f * optical_depth;
        return optical_depth * (1.0f - half);
    }
    const double alpha = -std::expm1(-static_cast<double>(optical_depth));
    return static_cast<float>(std::min(1.0, std::max(alpha, 0.0)));  // clamp to [0,1]
}

}  // namespace

namespace hp_internal {

hp_status int_generate_cpu(const hp_plan* plan,
                           const hp_samp_t* samp,
                           hp_intl_t* intl,
                           void* ws,
                           size_t ws_bytes) {
    if (plan == nullptr || samp == nullptr || intl == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (samp->dt.memspace != HP_MEMSPACE_HOST ||
        samp->sigma.memspace != HP_MEMSPACE_HOST ||
        samp->ray_offset.memspace != HP_MEMSPACE_HOST ||
        samp->color.memspace != HP_MEMSPACE_HOST) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    if (samp->dt.dtype != HP_DTYPE_F32 || samp->sigma.dtype != HP_DTYPE_F32 ||
        samp->color.dtype != HP_DTYPE_F32 || samp->ray_offset.dtype != HP_DTYPE_U32) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const size_t sample_count = (samp->dt.rank >= 1) ? static_cast<size_t>(samp->dt.shape[0]) : 0;
    const size_t ray_count = (samp->ray_offset.rank >= 1) ? static_cast<size_t>(samp->ray_offset.shape[0] > 0 ? samp->ray_offset.shape[0] - 1 : 0) : 0;

    if (sample_count > static_cast<size_t>(plan->desc.max_samples) ||
        ray_count > static_cast<size_t>(plan->desc.max_rays)) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const float* dt = static_cast<const float*>(samp->dt.data);
    const float* sigma = static_cast<const float*>(samp->sigma.data);
    const float* color = static_cast<const float*>(samp->color.data);
    const uint32_t* offset = static_cast<const uint32_t*>(samp->ray_offset.data);

    if ((sample_count > 0 && (dt == nullptr || sigma == nullptr || color == nullptr)) ||
        (ray_count > 0 && offset == nullptr)) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    hp_workspace_allocator allocator(ws, ws_bytes);
    IntlBuffers buffers{};
    const hp_status buffer_status = ensure_intl_buffers(intl, ray_count, sample_count, allocator, buffers);
    if (buffer_status != HP_STATUS_SUCCESS) {
        return buffer_status;
    }

    if (ray_count > 0) {
        std::fill(buffers.radiance, buffers.radiance + ray_count * 3U, 0.0f);
        std::fill(buffers.transmittance, buffers.transmittance + ray_count, 1.0f);
        std::fill(buffers.opacity, buffers.opacity + ray_count, 0.0f);
        std::fill(buffers.depth, buffers.depth + ray_count, plan->desc.t_far);
    }
    if (sample_count > 0 && buffers.aux != nullptr) {
        std::fill(buffers.aux, buffers.aux + sample_count * 4U, 0.0f);
    }

    const float t_near = plan->desc.t_near;
    const float stop_threshold = 1e-4f;

    for (size_t ray = 0; ray < ray_count; ++ray) {
        const uint32_t begin = offset[ray];
        const uint32_t end = offset[ray + 1];
        if (end < begin || end > sample_count) {
            return HP_STATUS_INVALID_ARGUMENT;
        }

        float T = 1.0f;
        float depth_weighted = 0.0f;
        float color_acc[3]{0.0f, 0.0f, 0.0f};
        float t_cursor = t_near;

        for (uint32_t idx = begin; idx < end; ++idx) {
            const float dt_val = dt[idx];
            const float sigma_val = sigma[idx];
            const float alpha = std::clamp(compute_alpha(sigma_val, dt_val), 0.0f, 1.0f);
            const float T_before = T;
            const float logT_before = std::log(std::max(T_before, 1e-30f));
            const float weight = T_before * alpha;

            const float* color_sample = color + idx * 3U;
            color_acc[0] += weight * color_sample[0];
            color_acc[1] += weight * color_sample[1];
            color_acc[2] += weight * color_sample[2];

            const float segment_mid = t_cursor + 0.5f * dt_val;
            depth_weighted += weight * segment_mid;

            if (buffers.aux != nullptr) {
                float* aux_row = buffers.aux + idx * 4U;
                aux_row[0] = alpha;
                aux_row[1] = weight;
                aux_row[2] = T_before;
                aux_row[3] = logT_before;
            }

            const float remaining = std::max(1.0f - alpha, 0.0f);
            T *= remaining;
            t_cursor += dt_val;

            if (T <= stop_threshold) {
                break;
            }
        }

        const float opacity = 1.0f - T;
        buffers.radiance[ray * 3U + 0] = color_acc[0];
        buffers.radiance[ray * 3U + 1] = color_acc[1];
        buffers.radiance[ray * 3U + 2] = color_acc[2];
        buffers.transmittance[ray] = T;
        buffers.opacity[ray] = opacity;
        const float depth = (opacity > 1e-6f) ? (depth_weighted / opacity) : plan->desc.t_far;
        buffers.depth[ray] = depth;
    }
    intl->aux.shape[0] = static_cast<int64_t>(sample_count);

    return HP_STATUS_SUCCESS;
}

}  // namespace hp_internal
