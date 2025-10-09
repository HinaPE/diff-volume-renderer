#include "hp_internal.hpp"

#if defined(HP_WITH_CUDA)

#include <cuda_runtime.h>
#include "../runtime/workspace.hpp"
#include <vector>
#include <algorithm>

namespace {

struct DeviceIntlBuffers {
    float* radiance{nullptr};
    float* transmittance{nullptr};
    float* opacity{nullptr};
    float* depth{nullptr};
    float* aux{nullptr};
};

hp_status ensure_intl_buffers_device(hp_intl_t* intl,
                                     size_t ray_count,
                                     size_t sample_count,
                                     void* ws,
                                     size_t ws_bytes,
                                     DeviceIntlBuffers& buffers) {
    const int64_t ray_count_i64 = static_cast<int64_t>(ray_count);
    const int64_t sample_count_i64 = static_cast<int64_t>(sample_count);

    intl->radiance.dtype = HP_DTYPE_F32;
    intl->radiance.memspace = HP_MEMSPACE_DEVICE;
    intl->radiance.rank = 2;
    intl->radiance.shape[0] = ray_count_i64;
    intl->radiance.shape[1] = 3;
    intl->radiance.stride[1] = 1;
    intl->radiance.stride[0] = 3;

    intl->transmittance.dtype = HP_DTYPE_F32;
    intl->transmittance.memspace = HP_MEMSPACE_DEVICE;
    intl->transmittance.rank = 1;
    intl->transmittance.shape[0] = ray_count_i64;
    intl->transmittance.stride[0] = 1;

    intl->opacity.dtype = HP_DTYPE_F32;
    intl->opacity.memspace = HP_MEMSPACE_DEVICE;
    intl->opacity.rank = 1;
    intl->opacity.shape[0] = ray_count_i64;
    intl->opacity.stride[0] = 1;

    intl->depth.dtype = HP_DTYPE_F32;
    intl->depth.memspace = HP_MEMSPACE_DEVICE;
    intl->depth.rank = 1;
    intl->depth.shape[0] = ray_count_i64;
    intl->depth.stride[0] = 1;

    intl->aux.dtype = HP_DTYPE_F32;
    intl->aux.memspace = HP_MEMSPACE_DEVICE;
    intl->aux.rank = 2;
    intl->aux.shape[0] = sample_count_i64;
    intl->aux.shape[1] = 4;
    intl->aux.stride[1] = 1;
    intl->aux.stride[0] = 4;

    hp_workspace_allocator allocator(ws, ws_bytes);
    const size_t ray_vec_bytes = ray_count * 3U * sizeof(float);
    const size_t ray_scalar_bytes = ray_count * sizeof(float);
    const size_t aux_bytes = sample_count * 4U * sizeof(float);

    if (ray_vec_bytes > 0 && intl->radiance.data == nullptr) {
        intl->radiance.data = allocator.allocate(ray_vec_bytes, alignof(float));
    }
    if (ray_scalar_bytes > 0) {
        if (intl->transmittance.data == nullptr) {
            intl->transmittance.data = allocator.allocate(ray_scalar_bytes, alignof(float));
        }
        if (intl->opacity.data == nullptr) {
            intl->opacity.data = allocator.allocate(ray_scalar_bytes, alignof(float));
        }
        if (intl->depth.data == nullptr) {
            intl->depth.data = allocator.allocate(ray_scalar_bytes, alignof(float));
        }
    }
    if (aux_bytes > 0 && intl->aux.data == nullptr) {
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
    buffers.aux = static_cast<float*>(intl->aux.data);
    return HP_STATUS_SUCCESS;
}

bool tensor_is_device_f32(const hp_tensor& tensor) {
    return tensor.memspace == HP_MEMSPACE_DEVICE &&
           tensor.dtype == HP_DTYPE_F32 &&
           tensor.rank >= 1;
}

bool tensor_is_device_u32(const hp_tensor& tensor) {
    return tensor.memspace == HP_MEMSPACE_DEVICE &&
           tensor.dtype == HP_DTYPE_U32 &&
           tensor.rank >= 1;
}

}  // namespace

namespace hp_internal {

hp_status int_generate_cuda(const hp_plan* plan,
                            const hp_samp_t* samp,
                            hp_intl_t* intl,
                            void* ws,
                            size_t ws_bytes) {
    if (plan == nullptr || samp == nullptr || intl == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (!tensor_is_device_f32(samp->dt) ||
        !tensor_is_device_f32(samp->sigma) ||
        !tensor_is_device_f32(samp->color) ||
        !tensor_is_device_u32(samp->ray_offset)) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const size_t sample_count = (samp->dt.rank >= 1) ? static_cast<size_t>(samp->dt.shape[0]) : 0;
    const size_t ray_offset_len = (samp->ray_offset.rank >= 1) ? static_cast<size_t>(samp->ray_offset.shape[0]) : 0;
    const size_t ray_count = (ray_offset_len > 0) ? (ray_offset_len - 1) : 0;

    if (sample_count > static_cast<size_t>(plan->desc.max_samples) ||
        ray_count > static_cast<size_t>(plan->desc.max_rays)) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    DeviceIntlBuffers device_buffers{};
    const hp_status device_status = ensure_intl_buffers_device(intl, ray_count, sample_count, ws, ws_bytes, device_buffers);
    if (device_status != HP_STATUS_SUCCESS) {
        return device_status;
    }

    std::vector<float> h_dt(sample_count, 0.0f);
    std::vector<float> h_sigma(sample_count, 0.0f);
    std::vector<float> h_color(sample_count * 3U, 0.0f);
    std::vector<uint32_t> h_offsets(ray_count + 1, 0U);

    if (sample_count > 0) {
        if (cudaMemcpy(h_dt.data(), samp->dt.data, sample_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_sigma.data(), samp->sigma.data, sample_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(h_color.data(), samp->color.data, sample_count * 3U * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return HP_STATUS_INTERNAL_ERROR;
        }
    }
    if (ray_count + 1 > 0) {
        if (cudaMemcpy(h_offsets.data(), samp->ray_offset.data, (ray_count + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return HP_STATUS_INTERNAL_ERROR;
        }
    }

    hp_samp_t host_samp{};
    host_samp.dt.data = h_dt.data();
    host_samp.dt.dtype = HP_DTYPE_F32;
    host_samp.dt.memspace = HP_MEMSPACE_HOST;
    host_samp.dt.rank = 1;
    host_samp.dt.shape[0] = static_cast<int64_t>(sample_count);
    host_samp.dt.stride[0] = 1;

    host_samp.sigma = host_samp.dt;
    host_samp.sigma.data = h_sigma.data();

    host_samp.color.data = h_color.data();
    host_samp.color.dtype = HP_DTYPE_F32;
    host_samp.color.memspace = HP_MEMSPACE_HOST;
    host_samp.color.rank = 2;
    host_samp.color.shape[0] = static_cast<int64_t>(sample_count);
    host_samp.color.shape[1] = 3;
    host_samp.color.stride[1] = 1;
    host_samp.color.stride[0] = 3;

    host_samp.ray_offset.data = h_offsets.data();
    host_samp.ray_offset.dtype = HP_DTYPE_U32;
    host_samp.ray_offset.memspace = HP_MEMSPACE_HOST;
    host_samp.ray_offset.rank = 1;
    host_samp.ray_offset.shape[0] = static_cast<int64_t>(ray_count + 1);
    host_samp.ray_offset.stride[0] = 1;

    std::vector<std::byte> host_ws;
    const size_t ray_vec_bytes = ray_count * 3U * sizeof(float);
    const size_t ray_scalar_bytes = ray_count * sizeof(float);
    const size_t aux_bytes = sample_count * 4U * sizeof(float);
    const size_t host_ws_bytes = ray_vec_bytes + ray_scalar_bytes * 3 + aux_bytes + 128;
    host_ws.resize(host_ws_bytes);

    hp_intl_t host_intl{};
    const hp_status cpu_status = int_generate_cpu(plan, &host_samp, &host_intl, host_ws.data(), host_ws.size());
    if (cpu_status != HP_STATUS_SUCCESS) {
        return cpu_status;
    }

    const size_t copy_ray_vec_bytes = ray_count * 3U * sizeof(float);
    const size_t copy_ray_scalar_bytes = ray_count * sizeof(float);
    const size_t copy_aux_bytes = sample_count * 4U * sizeof(float);

    if (copy_ray_vec_bytes > 0 &&
        cudaMemcpy(device_buffers.radiance, host_intl.radiance.data, copy_ray_vec_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }
    if (copy_ray_scalar_bytes > 0) {
        if (cudaMemcpy(device_buffers.transmittance, host_intl.transmittance.data, copy_ray_scalar_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_buffers.opacity, host_intl.opacity.data, copy_ray_scalar_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(device_buffers.depth, host_intl.depth.data, copy_ray_scalar_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            return HP_STATUS_INTERNAL_ERROR;
        }
    }
    if (copy_aux_bytes > 0 &&
        cudaMemcpy(device_buffers.aux, host_intl.aux.data, copy_aux_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }

    intl->radiance.shape[0] = static_cast<int64_t>(ray_count);
    intl->transmittance.shape[0] = static_cast<int64_t>(ray_count);
    intl->opacity.shape[0] = static_cast<int64_t>(ray_count);
    intl->depth.shape[0] = static_cast<int64_t>(ray_count);
    intl->aux.shape[0] = static_cast<int64_t>(sample_count);

    return HP_STATUS_SUCCESS;
}

}  // namespace hp_internal

#endif  // HP_WITH_CUDA
