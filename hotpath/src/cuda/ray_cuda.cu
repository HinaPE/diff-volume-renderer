#include "hp_internal.hpp"

#if defined(HP_WITH_CUDA)

#include <cuda_runtime.h>

#include <cstdint>
namespace {

struct RayGenParams {
    float origin[3];
    float rot[9];
    float fx;
    float fy;
    float cx;
    float cy;
    float ortho_dir[3];
    float ortho_scale;
    uint32_t image_width;
    uint32_t roi_x;
    uint32_t roi_y;
    uint32_t roi_width;
    uint32_t roi_height;
    float t_near;
    float t_far;
    hp_camera_model model;
};

__global__ void generate_rays_kernel(const RayGenParams params,
                                     float* origins,
                                     float* directions,
                                     float* t_near,
                                     float* t_far,
                                     uint32_t* pixel_ids,
                                     size_t ray_count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= ray_count) {
        return;
    }

    const uint32_t local_x = static_cast<uint32_t>(idx % params.roi_width);
    const uint32_t local_y = static_cast<uint32_t>(idx / params.roi_width);
    const uint32_t px = params.roi_x + local_x;
    const uint32_t py = params.roi_y + local_y;

    float dir_x = 0.0f;
    float dir_y = 0.0f;
    float dir_z = 1.0f;

    if (params.model == HP_CAMERA_PINHOLE) {
        const float u = static_cast<float>(px) + 0.5f;
        const float v = static_cast<float>(py) + 0.5f;
        const float cam_x = (u - params.cx) / params.fx;
        const float cam_y = (v - params.cy) / params.fy;
        dir_x = params.rot[0] * cam_x + params.rot[1] * cam_y + params.rot[2] * dir_z;
        dir_y = params.rot[3] * cam_x + params.rot[4] * cam_y + params.rot[5] * dir_z;
        dir_z = params.rot[6] * cam_x + params.rot[7] * cam_y + params.rot[8] * 1.0f;
    } else {
        dir_x = params.ortho_dir[0];
        dir_y = params.ortho_dir[1];
        dir_z = params.ortho_dir[2];
    }

    const float len_sq = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z;
    float inv_len = rsqrtf(len_sq + 1.0e-20f);
    dir_x *= inv_len;
    dir_y *= inv_len;
    dir_z *= inv_len;

    const size_t vec_offset = idx * 3U;
    origins[vec_offset + 0] = params.origin[0];
    origins[vec_offset + 1] = params.origin[1];
    origins[vec_offset + 2] = params.origin[2];

    if (params.model == HP_CAMERA_ORTHOGRAPHIC) {
        const float u = (static_cast<float>(px) - params.cx) / params.fx;
        const float v = (static_cast<float>(py) - params.cy) / params.fy;
        origins[vec_offset + 0] += params.rot[0] * (u * params.ortho_scale) +
                                   params.rot[1] * (v * params.ortho_scale);
        origins[vec_offset + 1] += params.rot[3] * (u * params.ortho_scale) +
                                   params.rot[4] * (v * params.ortho_scale);
        origins[vec_offset + 2] += params.rot[6] * (u * params.ortho_scale) +
                                   params.rot[7] * (v * params.ortho_scale);
    }

    directions[vec_offset + 0] = dir_x;
    directions[vec_offset + 1] = dir_y;
    directions[vec_offset + 2] = dir_z;

    t_near[idx] = params.t_near;
    t_far[idx] = params.t_far;
    pixel_ids[idx] = py * params.image_width + px;
}

cudaError_t launch_generate_kernel(const RayGenParams& params,
                                   float* origins,
                                   float* directions,
                                   float* t_near,
                                   float* t_far,
                                   uint32_t* pixel_ids,
                                   size_t ray_count) {
    if (ray_count == 0) {
        return cudaSuccess;
    }
    constexpr int kBlockSize = 256;
    const int blocks = static_cast<int>((ray_count + kBlockSize - 1) / kBlockSize);
    generate_rays_kernel<<<blocks, kBlockSize>>>(params,
                                                 origins,
                                                 directions,
                                                 t_near,
                                                 t_far,
                                                 pixel_ids,
                                                 ray_count);
    return cudaGetLastError();
}

hp_status copy_override_cuda(const hp_rays_t* override_rays,
                             hp_rays_t* rays,
                             size_t ray_count) {
    if (override_rays == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    if (override_rays->origins.data == nullptr || override_rays->directions.data == nullptr ||
        override_rays->t_near.data == nullptr || override_rays->t_far.data == nullptr ||
        override_rays->pixel_ids.data == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const size_t vec3_bytes = ray_count * 3U * sizeof(float);
    const size_t scalar_bytes = ray_count * sizeof(float);
    const size_t ids_bytes = ray_count * sizeof(uint32_t);

    cudaError_t err = cudaMemcpy(rays->origins.data, override_rays->origins.data, vec3_bytes, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }
    err = cudaMemcpy(rays->directions.data, override_rays->directions.data, vec3_bytes, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }
    err = cudaMemcpy(rays->t_near.data, override_rays->t_near.data, scalar_bytes, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }
    err = cudaMemcpy(rays->t_far.data, override_rays->t_far.data, scalar_bytes, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }
    err = cudaMemcpy(rays->pixel_ids.data, override_rays->pixel_ids.data, ids_bytes, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }
    return HP_STATUS_SUCCESS;
}

void configure_tensor_device(hp_tensor& tensor, size_t elements, hp_dtype dtype) {
    tensor.dtype = dtype;
    tensor.memspace = HP_MEMSPACE_DEVICE;
    tensor.rank = 2;
    tensor.shape[0] = static_cast<int64_t>(elements);
    tensor.shape[1] = 3;
    tensor.stride[1] = 1;
    tensor.stride[0] = 3;
}

void configure_scalar_tensor_device(hp_tensor& tensor, size_t elements, hp_dtype dtype) {
    tensor.dtype = dtype;
    tensor.memspace = HP_MEMSPACE_DEVICE;
    tensor.rank = 1;
    tensor.shape[0] = static_cast<int64_t>(elements);
    tensor.stride[0] = 1;
}

}  // namespace

namespace hp_internal {

hp_status ray_generate_cuda(const hp_plan* plan,
                            const hp_rays_t* override_or_null,
                            hp_rays_t* rays,
                            void*,
                            size_t) {
    if (plan == nullptr || rays == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const hp_plan_desc& desc = plan->desc;
    const hp_roi_desc& roi = desc.roi;
    const uint64_t ray_count_u64 = static_cast<uint64_t>(roi.width) * static_cast<uint64_t>(roi.height);
    if (ray_count_u64 > static_cast<uint64_t>(desc.max_rays)) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    const size_t ray_count = static_cast<size_t>(ray_count_u64);

    configure_tensor_device(rays->origins, ray_count, HP_DTYPE_F32);
    configure_tensor_device(rays->directions, ray_count, HP_DTYPE_F32);
    configure_scalar_tensor_device(rays->t_near, ray_count, HP_DTYPE_F32);
    configure_scalar_tensor_device(rays->t_far, ray_count, HP_DTYPE_F32);
    configure_scalar_tensor_device(rays->pixel_ids, ray_count, HP_DTYPE_U32);

    if (rays->origins.memspace != HP_MEMSPACE_DEVICE ||
        rays->directions.memspace != HP_MEMSPACE_DEVICE ||
        rays->t_near.memspace != HP_MEMSPACE_DEVICE ||
        rays->t_far.memspace != HP_MEMSPACE_DEVICE ||
        rays->pixel_ids.memspace != HP_MEMSPACE_DEVICE) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (ray_count == 0U) {
        return HP_STATUS_SUCCESS;
    }

    if (rays->origins.data == nullptr || rays->directions.data == nullptr ||
        rays->t_near.data == nullptr || rays->t_far.data == nullptr ||
        rays->pixel_ids.data == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (override_or_null != nullptr) {
        return copy_override_cuda(override_or_null, rays, ray_count);
    }

    RayGenParams params{};
    const hp_camera_desc& cam = desc.camera;

    params.origin[0] = cam.c2w[3];
    params.origin[1] = cam.c2w[7];
    params.origin[2] = cam.c2w[11];
    params.rot[0] = cam.c2w[0];
    params.rot[1] = cam.c2w[1];
    params.rot[2] = cam.c2w[2];
    params.rot[3] = cam.c2w[4];
    params.rot[4] = cam.c2w[5];
    params.rot[5] = cam.c2w[6];
    params.rot[6] = cam.c2w[8];
    params.rot[7] = cam.c2w[9];
    params.rot[8] = cam.c2w[10];
    params.fx = cam.K[0];
    params.fy = cam.K[4];
    params.cx = cam.K[2];
    params.cy = cam.K[5];
    params.ortho_scale = cam.ortho_scale <= 0.0f ? 1.0f : cam.ortho_scale;
    params.ortho_dir[0] = params.rot[2];
    params.ortho_dir[1] = params.rot[5];
    params.ortho_dir[2] = params.rot[8];
    params.image_width = desc.width;
    params.roi_x = roi.x;
    params.roi_y = roi.y;
    params.roi_width = roi.width;
    params.roi_height = roi.height;
    params.t_near = desc.t_near;
    params.t_far = desc.t_far;
    params.model = cam.model;

    const cudaError_t launch_status = launch_generate_kernel(
        params,
        static_cast<float*>(rays->origins.data),
        static_cast<float*>(rays->directions.data),
        static_cast<float*>(rays->t_near.data),
        static_cast<float*>(rays->t_far.data),
        static_cast<uint32_t*>(rays->pixel_ids.data),
        ray_count);

    if (launch_status != cudaSuccess) {
        return HP_STATUS_INTERNAL_ERROR;
    }
    return HP_STATUS_SUCCESS;
}

}  // namespace hp_internal

#endif  // HP_WITH_CUDA
