#include "hp_internal.hpp"

#if defined(HP_WITH_CUDA)

#include <cuda_runtime.h>
#include <cstdint>

namespace hp_internal {

hp_status samp_int_fused_cuda(const hp_plan* plan,
                              const hp_field* fs,
                              const hp_field* fc,
                              const hp_rays_t* rays,
                              hp_samp_t* samp,
                              hp_intl_t* intl,
                              void* ws,
                              size_t ws_bytes) {
    if (plan == nullptr || rays == nullptr || samp == nullptr || intl == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    if (ws == nullptr || ws_bytes == 0) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    hp_status status = samp_generate_cuda(plan, fs, fc, rays, samp, ws, ws_bytes);
    if (status != HP_STATUS_SUCCESS) {
        return status;
    }

    const size_t capacity = static_cast<size_t>(plan->desc.max_samples);
    const size_t ray_offset_len = (samp->ray_offset.rank >= 1) ? static_cast<size_t>(samp->ray_offset.shape[0]) : 0;
    const size_t ray_count = (ray_offset_len > 0) ? (ray_offset_len - 1) : 0;

    const uintptr_t base = reinterpret_cast<uintptr_t>(ws);
    uintptr_t max_addr = base;
    auto update_max = [&](void* ptr, size_t bytes) {
        if (ptr != nullptr && bytes > 0) {
            const uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
            const uintptr_t end = start + bytes;
            if (end > max_addr) {
                max_addr = end;
            }
        }
    };

    update_max(samp->positions.data, capacity * 3U * sizeof(float));
    update_max(samp->dt.data, capacity * sizeof(float));
    update_max(samp->sigma.data, capacity * sizeof(float));
    update_max(samp->color.data, capacity * 3U * sizeof(float));
    update_max(samp->ray_offset.data, (ray_count + 1) * sizeof(uint32_t));

    const size_t sample_bytes = (max_addr > base) ? static_cast<size_t>(max_addr - base) : 0;
    if (sample_bytes > ws_bytes) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    void* intl_ws = reinterpret_cast<void*>(base + sample_bytes);
    const size_t intl_ws_bytes = (ws_bytes > sample_bytes) ? (ws_bytes - sample_bytes) : 0;

    status = int_generate_cuda(plan, samp, intl, intl_ws, intl_ws_bytes);
    return status;
}

}  // namespace hp_internal

#endif  // HP_WITH_CUDA
