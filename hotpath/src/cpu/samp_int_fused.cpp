#include "hp_internal.hpp"


#include "workspace.hpp"

#include <cstring>
#include <vector>

namespace hp_internal {

hp_status samp_int_fused_cpu(const hp_plan* plan,
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

    hp_status status = samp_generate_cpu(plan, fs, fc, rays, samp, ws, ws_bytes);
    if (status != HP_STATUS_SUCCESS) {
        return status;
    }

    const size_t sample_count = (samp->dt.rank >= 1) ? static_cast<size_t>(samp->dt.shape[0]) : 0;
    const size_t ray_count = (samp->ray_offset.rank >= 1) ? static_cast<size_t>(samp->ray_offset.shape[0]) : 0;

    std::byte* base = static_cast<std::byte*>(ws);
    std::byte* max_ptr = base;

    if (samp->positions.data != nullptr) {
        auto* ptr = static_cast<std::byte*>(samp->positions.data) + sample_count * 3U * sizeof(float);
        if (ptr > max_ptr) max_ptr = ptr;
    }
    if (samp->dt.data != nullptr) {
        auto* ptr = static_cast<std::byte*>(samp->dt.data) + sample_count * sizeof(float);
        if (ptr > max_ptr) max_ptr = ptr;
    }
    if (samp->sigma.data != nullptr) {
        auto* ptr = static_cast<std::byte*>(samp->sigma.data) + sample_count * sizeof(float);
        if (ptr > max_ptr) max_ptr = ptr;
    }
    if (samp->color.data != nullptr) {
        auto* ptr = static_cast<std::byte*>(samp->color.data) + sample_count * 3U * sizeof(float);
        if (ptr > max_ptr) max_ptr = ptr;
    }
    if (samp->ray_offset.data != nullptr) {
        auto* ptr = static_cast<std::byte*>(samp->ray_offset.data) + ray_count * sizeof(uint32_t);
        if (ptr > max_ptr) max_ptr = ptr;
    }

    const size_t sample_bytes = static_cast<size_t>(max_ptr - base);
    if (sample_bytes > ws_bytes) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    std::byte* intl_base = base + sample_bytes;
    const size_t intl_bytes = (ws_bytes > sample_bytes) ? (ws_bytes - sample_bytes) : 0;
    if (intl_bytes == 0) {
        return HP_STATUS_OUT_OF_MEMORY;
    }

    status = int_generate_cpu(plan, samp, intl, intl_base, intl_bytes);
    if (status != HP_STATUS_SUCCESS) {
        return status;
    }

    return HP_STATUS_SUCCESS;
}

}  // namespace hp_internal
