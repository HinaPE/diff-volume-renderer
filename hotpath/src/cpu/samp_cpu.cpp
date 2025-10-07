#include "hp_internal.hpp"

#include "workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

uint64_t mix_seed(uint64_t state) {
    state = (state ^ (state >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    state = (state ^ (state >> 27U)) * 0x94d049bb133111ebULL;
    state = state ^ (state >> 31U);
    return state;
}

float stratified_jitter(uint64_t base_seed, size_t ray_index, uint32_t step) {
    uint64_t seed = base_seed;
    seed ^= static_cast<uint64_t>(ray_index) << 32U;
    seed ^= static_cast<uint64_t>(step);
    seed = mix_seed(seed);
    const double norm = static_cast<double>(seed & 0x000fffffffffffffULL) / static_cast<double>(0x0010000000000000ULL);
    return static_cast<float>(norm);
}

bool ensure_host_tensor(const hp_tensor& tensor, size_t expected_rank1) {
    if (tensor.memspace != HP_MEMSPACE_HOST) {
        return false;
    }
    if (tensor.rank == 0 && expected_rank1 > 0) {
        return false;
    }
    return true;
}

size_t infer_ray_count(const hp_rays_t* rays, const hp_plan_desc& desc) {
    if (rays == nullptr) {
        return static_cast<size_t>(desc.roi.width) * static_cast<size_t>(desc.roi.height);
    }
    if (rays->t_near.rank >= 1 && rays->t_near.shape[0] > 0) {
        return static_cast<size_t>(rays->t_near.shape[0]);
    }
    if (rays->origins.rank >= 2 && rays->origins.shape[0] > 0) {
        return static_cast<size_t>(rays->origins.shape[0]);
    }
    return static_cast<size_t>(desc.roi.width) * static_cast<size_t>(desc.roi.height);
}

struct SampBuffers {
    float* positions{nullptr};
    float* dt{nullptr};
    uint32_t* offsets{nullptr};
    float* sigma{nullptr};
    float* color{nullptr};
};

hp_status ensure_sample_buffers(hp_samp_t* samp,
                                size_t ray_count,
                                size_t capacity,
                                hp_workspace_allocator& allocator,
                                SampBuffers& buffers) {
    const int64_t sample_capacity_i64 = static_cast<int64_t>(capacity);
    const int64_t ray_offset_len = static_cast<int64_t>(ray_count + 1);

    samp->positions.dtype = HP_DTYPE_F32;
    samp->positions.memspace = HP_MEMSPACE_HOST;
    samp->positions.rank = 2;
    samp->positions.shape[0] = sample_capacity_i64;
    samp->positions.shape[1] = 3;
    samp->positions.stride[1] = 1;
    samp->positions.stride[0] = 3;

    samp->dt.dtype = HP_DTYPE_F32;
    samp->dt.memspace = HP_MEMSPACE_HOST;
    samp->dt.rank = 1;
    samp->dt.shape[0] = sample_capacity_i64;
    samp->dt.stride[0] = 1;

    samp->sigma.dtype = HP_DTYPE_F32;
    samp->sigma.memspace = HP_MEMSPACE_HOST;
    samp->sigma.rank = 1;
    samp->sigma.shape[0] = sample_capacity_i64;
    samp->sigma.stride[0] = 1;

    samp->color.dtype = HP_DTYPE_F32;
    samp->color.memspace = HP_MEMSPACE_HOST;
    samp->color.rank = 2;
    samp->color.shape[0] = sample_capacity_i64;
    samp->color.shape[1] = 3;
    samp->color.stride[1] = 1;
    samp->color.stride[0] = 3;

    samp->ray_offset.dtype = HP_DTYPE_U32;
    samp->ray_offset.memspace = HP_MEMSPACE_HOST;
    samp->ray_offset.rank = 1;
    samp->ray_offset.shape[0] = ray_offset_len;
    samp->ray_offset.stride[0] = 1;

    const size_t vec3_bytes = capacity * 3U * sizeof(float);
    const size_t scalar_bytes = capacity * sizeof(float);
    const size_t color_bytes = capacity * 3U * sizeof(float);
    const size_t offsets_bytes = (ray_count + 1) * sizeof(uint32_t);

    if (samp->positions.data == nullptr) {
        samp->positions.data = allocator.allocate(vec3_bytes, alignof(float));
    }
    if (samp->dt.data == nullptr) {
        samp->dt.data = allocator.allocate(scalar_bytes, alignof(float));
    }
    if (samp->sigma.data == nullptr) {
        samp->sigma.data = allocator.allocate(scalar_bytes, alignof(float));
    }
    if (samp->color.data == nullptr) {
        samp->color.data = allocator.allocate(color_bytes, alignof(float));
    }
    if (samp->ray_offset.data == nullptr) {
        samp->ray_offset.data = allocator.allocate(offsets_bytes, alignof(uint32_t));
    }

    if (samp->positions.data == nullptr ||
        samp->dt.data == nullptr ||
        samp->sigma.data == nullptr ||
        samp->color.data == nullptr ||
        samp->ray_offset.data == nullptr) {
        return HP_STATUS_OUT_OF_MEMORY;
    }

    buffers.positions = static_cast<float*>(samp->positions.data);
    buffers.dt = static_cast<float*>(samp->dt.data);
    buffers.sigma = static_cast<float*>(samp->sigma.data);
    buffers.color = static_cast<float*>(samp->color.data);
    buffers.offsets = static_cast<uint32_t*>(samp->ray_offset.data);
    return HP_STATUS_SUCCESS;
}

}  // namespace

namespace hp_internal {

hp_status samp_generate_cpu(const hp_plan* plan,
                            const hp_field* fs,
                            const hp_field* fc,
                            const hp_rays_t* rays,
                            hp_samp_t* samp,
                            void* ws,
                            size_t ws_bytes) {
    if (plan == nullptr || samp == nullptr || rays == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    if (fs == nullptr && fc == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    if (!ensure_host_tensor(rays->origins, 1) ||
        !ensure_host_tensor(rays->directions, 1) ||
        !ensure_host_tensor(rays->t_near, 1) ||
        !ensure_host_tensor(rays->t_far, 1)) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const size_t ray_count = infer_ray_count(rays, plan->desc);
    if (ray_count > plan->desc.max_rays) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const size_t capacity = static_cast<size_t>(plan->desc.max_samples);
    if (capacity == 0U && ray_count > 0) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    hp_workspace_allocator allocator(ws, ws_bytes);
    SampBuffers buffers{};
    const hp_status buffer_status = ensure_sample_buffers(samp, ray_count, capacity, allocator, buffers);
    if (buffer_status != HP_STATUS_SUCCESS) {
        return buffer_status;
    }

    std::memset(buffers.offsets, 0, (ray_count + 1) * sizeof(uint32_t));

    if (rays->origins.data == nullptr || rays->directions.data == nullptr ||
        rays->t_near.data == nullptr || rays->t_far.data == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const float* origins = static_cast<const float*>(rays->origins.data);
    const float* directions = static_cast<const float*>(rays->directions.data);
    const float* t_near = static_cast<const float*>(rays->t_near.data);
    const float* t_far = static_cast<const float*>(rays->t_far.data);

    const hp_sampling_desc sampling = plan->desc.sampling;
    const float dt_step = sampling.dt;
    const uint64_t base_seed = plan->desc.seed;

    size_t total_samples = 0;
    hp_status field_status = HP_STATUS_SUCCESS;

    for (size_t ray_idx = 0; ray_idx < ray_count; ++ray_idx) {
        buffers.offsets[ray_idx] = static_cast<uint32_t>(total_samples);

        float origin[3];
        float dir[3];
        const size_t vec_idx = ray_idx * 3U;
        origin[0] = origins[vec_idx + 0];
        origin[1] = origins[vec_idx + 1];
        origin[2] = origins[vec_idx + 2];
        dir[0] = directions[vec_idx + 0];
        dir[1] = directions[vec_idx + 1];
        dir[2] = directions[vec_idx + 2];

        const float ray_t_near = t_near[ray_idx];
        const float ray_t_far = t_far[ray_idx];
        if (!(ray_t_far > ray_t_near)) {
            continue;
        }

        for (uint32_t step = 0; step < sampling.max_steps; ++step) {
            const float base_t = ray_t_near + static_cast<float>(step) * dt_step;
            if (base_t >= ray_t_far) {
                break;
            }
            float jitter = 0.5f;
            if (sampling.mode == HP_SAMPLING_STRATIFIED) {
                jitter = stratified_jitter(base_seed, ray_idx, step);
            }
            jitter = std::clamp(jitter, 0.0f, 1.0f);
            float sample_t = base_t + jitter * dt_step;
            if (sample_t >= ray_t_far) {
                sample_t = std::nextafter(ray_t_far, ray_t_near);
            }
            const float segment_end = std::min(base_t + dt_step, ray_t_far);
            const float dt_actual = segment_end - base_t;
            if (!(dt_actual > 0.0f)) {
                continue;
            }
            if (total_samples >= capacity) {
                return HP_STATUS_INVALID_ARGUMENT;
            }

            const size_t sample_vec_idx = total_samples * 3U;
            buffers.positions[sample_vec_idx + 0] = origin[0] + dir[0] * sample_t;
            buffers.positions[sample_vec_idx + 1] = origin[1] + dir[1] * sample_t;
            buffers.positions[sample_vec_idx + 2] = origin[2] + dir[2] * sample_t;
            buffers.dt[total_samples] = dt_actual;

            if (fs != nullptr) {
                const float sigma_val = sample_grid_sigma_cpu(fs, &buffers.positions[sample_vec_idx], &field_status);
                if (field_status != HP_STATUS_SUCCESS) {
                    return field_status;
                }
                buffers.sigma[total_samples] = sigma_val;
            } else {
                buffers.sigma[total_samples] = 0.0f;
            }

            if (fc != nullptr) {
                float rgb[3]{0.0f, 0.0f, 0.0f};
                sample_grid_color_cpu(fc, &buffers.positions[sample_vec_idx], rgb, &field_status);
                if (field_status != HP_STATUS_SUCCESS) {
                    return field_status;
                }
                buffers.color[sample_vec_idx + 0] = rgb[0];
                buffers.color[sample_vec_idx + 1] = rgb[1];
                buffers.color[sample_vec_idx + 2] = rgb[2];
            } else {
                buffers.color[sample_vec_idx + 0] = 0.0f;
                buffers.color[sample_vec_idx + 1] = 0.0f;
                buffers.color[sample_vec_idx + 2] = 0.0f;
            }

            ++total_samples;
        }
    }

    buffers.offsets[ray_count] = static_cast<uint32_t>(total_samples);

    const int64_t total_samples_i64 = static_cast<int64_t>(total_samples);
    samp->positions.shape[0] = total_samples_i64;
    samp->dt.shape[0] = total_samples_i64;
    samp->sigma.shape[0] = total_samples_i64;
    samp->color.shape[0] = total_samples_i64;

    if (samp->positions.rank >= 1 && total_samples == 0) {
        samp->positions.shape[0] = 0;
        samp->dt.shape[0] = 0;
        samp->sigma.shape[0] = 0;
        samp->color.shape[0] = 0;
    }

    samp->ray_offset.shape[0] = static_cast<int64_t>(ray_count + 1);

    return HP_STATUS_SUCCESS;
}

#if defined(HP_WITH_CUDA)
hp_status samp_generate_cuda(const hp_plan*,
                             const hp_field*,
                             const hp_field*,
                             const hp_rays_t*,
                             hp_samp_t*,
                             void*,
                             size_t) {
    return HP_STATUS_NOT_IMPLEMENTED;
}
#endif

}  // namespace hp_internal
