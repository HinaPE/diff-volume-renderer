#include "hp_internal.hpp"

#if defined(HP_WITH_CUDA)

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

__global__ void backward_kernel(
    const float* __restrict__ dL_dI,
    const int64_t grad_stride_ray,
    const int64_t grad_stride_c,
    const float* __restrict__ dt,
    const float* __restrict__ sigma,
    const float* __restrict__ color,
    const uint32_t* __restrict__ offsets,
    const float* __restrict__ aux,
    float* __restrict__ grad_sigma,
    float* __restrict__ grad_color,
    const uint32_t ray_count,
    const uint32_t sample_count
) {
    const uint32_t ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= ray_count) return;

    const uint32_t begin = offsets[ray];
    const uint32_t end = offsets[ray + 1];
    if (end < begin || end > sample_count) return;

    const float* grad_ptr = dL_dI + ray * grad_stride_ray;
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

        const float* color_sample = color + idx * 3U;
        const float dot_grad = grad_color_vec[0] * color_sample[0] +
                               grad_color_vec[1] * color_sample[1] +
                               grad_color_vec[2] * color_sample[2];

        atomicAdd(&grad_color[idx * 3U + 0], grad_color_vec[0] * weight);
        atomicAdd(&grad_color[idx * 3U + 1], grad_color_vec[1] * weight);
        atomicAdd(&grad_color[idx * 3U + 2], grad_color_vec[2] * weight);

        float adj_alpha = dot_grad * T_prev - adj_T_next * T_prev;
        const float adj_T_prev = dot_grad * alpha + adj_T_next * (1.0f - alpha);

        const float d_alpha_d_sigma = dt_val * (1.0f - alpha);
        atomicAdd(&grad_sigma[idx], adj_alpha * d_alpha_d_sigma);

        adj_T_next = adj_T_prev;
    }
}

}  // namespace

namespace hp_internal {

hp_status diff_generate_cuda(const hp_plan* plan,
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

    if (dL_dI->memspace != HP_MEMSPACE_DEVICE ||
        samp->dt.memspace != HP_MEMSPACE_DEVICE ||
        samp->sigma.memspace != HP_MEMSPACE_DEVICE ||
        samp->color.memspace != HP_MEMSPACE_DEVICE ||
        samp->ray_offset.memspace != HP_MEMSPACE_DEVICE ||
        intl->aux.memspace != HP_MEMSPACE_DEVICE) {
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

    if (sample_count == 0 || ray_count == 0) {
        return HP_STATUS_SUCCESS;
    }

    // Allocate gradients on device
    const size_t sigma_bytes = sample_count * sizeof(float);
    const size_t color_bytes = sample_count * 3U * sizeof(float);
    const size_t camera_bytes = 12U * sizeof(float);

    float* d_grad_sigma = nullptr;
    float* d_grad_color = nullptr;
    float* d_grad_camera = nullptr;

    cudaError_t err = cudaMalloc(&d_grad_sigma, sigma_bytes);
    if (err != cudaSuccess) {
        return HP_STATUS_OUT_OF_MEMORY;
    }

    err = cudaMalloc(&d_grad_color, color_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_grad_sigma);
        return HP_STATUS_OUT_OF_MEMORY;
    }

    err = cudaMalloc(&d_grad_camera, camera_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_grad_sigma);
        cudaFree(d_grad_color);
        return HP_STATUS_OUT_OF_MEMORY;
    }

    // Initialize gradients to zero
    cudaMemset(d_grad_sigma, 0, sigma_bytes);
    cudaMemset(d_grad_color, 0, color_bytes);
    cudaMemset(d_grad_camera, 0, camera_bytes);

    // Set up gradient tensor descriptors
    grads->sigma.data = d_grad_sigma;
    grads->sigma.dtype = HP_DTYPE_F32;
    grads->sigma.memspace = HP_MEMSPACE_DEVICE;
    grads->sigma.rank = 1;
    grads->sigma.shape[0] = static_cast<int64_t>(sample_count);
    grads->sigma.stride[0] = 1;

    grads->color.data = d_grad_color;
    grads->color.dtype = HP_DTYPE_F32;
    grads->color.memspace = HP_MEMSPACE_DEVICE;
    grads->color.rank = 2;
    grads->color.shape[0] = static_cast<int64_t>(sample_count);
    grads->color.shape[1] = 3;
    grads->color.stride[1] = 1;
    grads->color.stride[0] = 3;

    grads->camera.data = d_grad_camera;
    grads->camera.dtype = HP_DTYPE_F32;
    grads->camera.memspace = HP_MEMSPACE_DEVICE;
    grads->camera.rank = 2;
    grads->camera.shape[0] = 3;
    grads->camera.shape[1] = 4;
    grads->camera.stride[1] = 1;
    grads->camera.stride[0] = 4;

    // Extract pointers
    const float* dt = static_cast<const float*>(samp->dt.data);
    const float* sigma = static_cast<const float*>(samp->sigma.data);
    const float* color = static_cast<const float*>(samp->color.data);
    const uint32_t* offsets = static_cast<const uint32_t*>(samp->ray_offset.data);
    const float* aux = static_cast<const float*>(intl->aux.data);

    if (dL_dI->rank < 2 || dL_dI->shape[0] != static_cast<int64_t>(ray_count) || dL_dI->shape[1] < 3) {
        cudaFree(d_grad_sigma);
        cudaFree(d_grad_color);
        cudaFree(d_grad_camera);
        return HP_STATUS_INVALID_ARGUMENT;
    }
    const int64_t grad_stride_ray = dL_dI->stride[0];
    const int64_t grad_stride_c = (dL_dI->rank >= 2) ? dL_dI->stride[1] : 1;
    const float* grad_base = static_cast<const float*>(dL_dI->data);

    // Launch kernel
    const int threads = 256;
    const int blocks = (ray_count + threads - 1) / threads;

    backward_kernel<<<blocks, threads>>>(
        grad_base,
        grad_stride_ray,
        grad_stride_c,
        dt,
        sigma,
        color,
        offsets,
        aux,
        d_grad_sigma,
        d_grad_color,
        static_cast<uint32_t>(ray_count),
        static_cast<uint32_t>(sample_count)
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_grad_sigma);
        cudaFree(d_grad_color);
        cudaFree(d_grad_camera);
        return HP_STATUS_INTERNAL_ERROR;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_grad_sigma);
        cudaFree(d_grad_color);
        cudaFree(d_grad_camera);
        return HP_STATUS_INTERNAL_ERROR;
    }

    (void)ws;
    (void)ws_bytes;
    return HP_STATUS_SUCCESS;
}

}  // namespace hp_internal

#endif  // HP_WITH_CUDA
