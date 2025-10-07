#include "hp_internal.hpp"

#include "workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

struct GradBuffers {
    float* sigma{nullptr};
    float* color{nullptr};
    float* camera{nullptr};
};

hp_status ensure_grad_buffers(const hp_plan* plan,
                              const hp_samp_t* samp,
                              hp_grads_t* grads,
                              hp_workspace_allocator& allocator,
                              GradBuffers& buffers) {
    const size_t sample_count = (samp->dt.rank >= 1) ? static_cast<size_t>(samp->dt.shape[0]) : 0;
    const size_t camera_count = 12;

    grads->sigma.dtype = HP_DTYPE_F32;
    grads->sigma.memspace = HP_MEMSPACE_HOST;
    grads->sigma.rank = 1;
    grads->sigma.shape[0] = static_cast<int64_t>(sample_count);
    grads->sigma.stride[0] = 1;

    grads->color.dtype = HP_DTYPE_F32;
    grads->color.memspace = HP_MEMSPACE_HOST;
    grads->color.rank = 2;
    grads->color.shape[0] = static_cast<int64_t>(sample_count);
    grads->color.shape[1] = 3;
    grads->color.stride[1] = 1;
    grads->color.stride[0] = 3;

    grads->camera.dtype = HP_DTYPE_F32;
    grads->camera.memspace = HP_MEMSPACE_HOST;
    grads->camera.rank = 2;
    grads->camera.shape[0] = 3;
    grads->camera.shape[1] = 4;
    grads->camera.stride[1] = 1;
    grads->camera.stride[0] = 4;

    const size_t sigma_bytes = sample_count * sizeof(float);
    const size_t color_bytes = sample_count * 3U * sizeof(float);
    const size_t camera_bytes = camera_count * sizeof(float);

    if (grads->sigma.data == nullptr && sigma_bytes > 0) {
        grads->sigma.data = allocator.allocate(sigma_bytes, alignof(float));
    }
    if (grads->color.data == nullptr && color_bytes > 0) {
        grads->color.data = allocator.allocate(color_bytes, alignof(float));
    }
    if (grads->camera.data == nullptr && camera_bytes > 0) {
        grads->camera.data = allocator.allocate(camera_bytes, alignof(float));
    }

    if ((sigma_bytes > 0 && grads->sigma.data == nullptr) ||
        (color_bytes > 0 && grads->color.data == nullptr) ||
        (camera_bytes > 0 && grads->camera.data == nullptr)) {
        return HP_STATUS_OUT_OF_MEMORY;
    }

    buffers.sigma = static_cast<float*>(grads->sigma.data);
    buffers.color = static_cast<float*>(grads->color.data);
    buffers.camera = static_cast<float*>(grads->camera.data);

    const size_t camera_values = camera_count;
    std::fill(buffers.camera, buffers.camera + camera_values, 0.0f);

    if (sample_count > 0) {
        std::fill(buffers.sigma, buffers.sigma + sample_count, 0.0f);
        std::fill(buffers.color, buffers.color + sample_count * 3U, 0.0f);
    }

    (void)plan;
    return HP_STATUS_SUCCESS;
}

}  // namespace

namespace hp_internal {

hp_status diff_generate_cpu(const hp_plan* plan,
                            const hp_tensor* dL_dI,
                            const hp_samp_t* samp,
                            const hp_intl_t* intl,
                            const hp_rays_t* /*rays*/,
                            hp_grads_t* grads,
                            void* ws,
                            size_t ws_bytes) {
    if (plan == nullptr || dL_dI == nullptr || samp == nullptr || intl == nullptr || grads == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (dL_dI->memspace != HP_MEMSPACE_HOST ||
        samp->dt.memspace != HP_MEMSPACE_HOST ||
        samp->sigma.memspace != HP_MEMSPACE_HOST ||
        samp->color.memspace != HP_MEMSPACE_HOST ||
        samp->ray_offset.memspace != HP_MEMSPACE_HOST ||
        intl->radiance.memspace != HP_MEMSPACE_HOST ||
        intl->transmittance.memspace != HP_MEMSPACE_HOST ||
        intl->opacity.memspace != HP_MEMSPACE_HOST ||
        intl->depth.memspace != HP_MEMSPACE_HOST) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (dL_dI->dtype != HP_DTYPE_F32 ||
        samp->dt.dtype != HP_DTYPE_F32 ||
        samp->sigma.dtype != HP_DTYPE_F32 ||
        samp->color.dtype != HP_DTYPE_F32 ||
        samp->ray_offset.dtype != HP_DTYPE_U32 ||
        intl->aux.dtype != HP_DTYPE_F32) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const size_t sample_count = (samp->dt.rank >= 1) ? static_cast<size_t>(samp->dt.shape[0]) : 0;
    const size_t ray_count = (samp->ray_offset.rank >= 1) ? static_cast<size_t>(samp->ray_offset.shape[0] > 0 ? samp->ray_offset.shape[0] - 1 : 0) : 0;
    if (sample_count > static_cast<size_t>(plan->desc.max_samples) ||
        ray_count > static_cast<size_t>(plan->desc.max_rays)) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    hp_workspace_allocator allocator(ws, ws_bytes);
    GradBuffers buffers{};
    const hp_status buffer_status = ensure_grad_buffers(plan, samp, grads, allocator, buffers);
    if (buffer_status != HP_STATUS_SUCCESS) {
        return buffer_status;
    }

    if (sample_count == 0 || ray_count == 0) {
        return HP_STATUS_SUCCESS;
    }

    const float* dt = static_cast<const float*>(samp->dt.data);
    const float* sigma = static_cast<const float*>(samp->sigma.data);
    const float* color = static_cast<const float*>(samp->color.data);
    const uint32_t* offsets = static_cast<const uint32_t*>(samp->ray_offset.data);
    const float* aux = static_cast<const float*>(intl->aux.data);
    if (aux == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (dL_dI->rank < 2 || dL_dI->shape[0] != static_cast<int64_t>(ray_count) || dL_dI->shape[1] < 3) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    const int64_t grad_stride_ray = dL_dI->stride[0];
    const int64_t grad_stride_c = (dL_dI->rank >= 2) ? dL_dI->stride[1] : 1;
    const float* grad_base = static_cast<const float*>(dL_dI->data);

    for (size_t ray = 0; ray < ray_count; ++ray) {
        const uint32_t begin = offsets[ray];
        const uint32_t end = offsets[ray + 1];
        if (end < begin || end > sample_count) {
            return HP_STATUS_INVALID_ARGUMENT;
        }

        const float* grad_ptr = grad_base + ray * grad_stride_ray;
        const float grad_color_vec[3]{
            grad_ptr[0 * grad_stride_c],
            grad_ptr[1 * grad_stride_c],
            grad_ptr[2 * grad_stride_c]
        };

        float adj_T_next = 0.0f;
        for (uint32_t idx = end; idx-- > begin;) {
            const float dt_val = dt[idx];
            const float alpha = aux[idx * 4U + 0];
            const float weight = aux[idx * 4U + 1];
            const float T_prev = aux[idx * 4U + 2];
            (void)sigma;  // sigma accessed if needed for validation only

            const float* color_sample = color + idx * 3U;
            const float dot_grad = grad_color_vec[0] * color_sample[0] +
                                   grad_color_vec[1] * color_sample[1] +
                                   grad_color_vec[2] * color_sample[2];

            buffers.color[idx * 3U + 0] += grad_color_vec[0] * weight;
            buffers.color[idx * 3U + 1] += grad_color_vec[1] * weight;
            buffers.color[idx * 3U + 2] += grad_color_vec[2] * weight;

            float adj_alpha = dot_grad * T_prev - adj_T_next * T_prev;
            const float adj_T_prev = dot_grad * alpha + adj_T_next * (1.0f - alpha);

            const float d_alpha_d_sigma = dt_val * (1.0f - alpha);
            buffers.sigma[idx] += adj_alpha * d_alpha_d_sigma;

            adj_T_next = adj_T_prev;
        }
    }

    return HP_STATUS_SUCCESS;
}

#if defined(HP_WITH_CUDA)
hp_status diff_generate_cuda(const hp_plan*,
                             const hp_tensor*,
                             const hp_samp_t*,
                             const hp_intl_t*,
                             const hp_rays_t*,
                             hp_grads_t*,
                             void*,
                             size_t) {
    return HP_STATUS_NOT_IMPLEMENTED;
}
#endif

}  // namespace hp_internal
