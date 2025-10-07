#include "hp_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "workspace.hpp"

namespace {

hp_status copy_override_cpu(const hp_rays_t* override_rays,
                            hp_rays_t* rays,
                            size_t ray_count) {
    if (override_rays == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }
    if (override_rays->origins.memspace != HP_MEMSPACE_HOST ||
        override_rays->directions.memspace != HP_MEMSPACE_HOST ||
        override_rays->t_near.memspace != HP_MEMSPACE_HOST ||
        override_rays->t_far.memspace != HP_MEMSPACE_HOST ||
        override_rays->pixel_ids.memspace != HP_MEMSPACE_HOST) {
        return HP_STATUS_UNSUPPORTED;
    }
    if (override_rays->origins.data == nullptr || override_rays->directions.data == nullptr ||
        override_rays->t_near.data == nullptr || override_rays->t_far.data == nullptr ||
        override_rays->pixel_ids.data == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const size_t vec3_bytes = ray_count * 3U * sizeof(float);
    const size_t scalar_bytes = ray_count * sizeof(float);
    const size_t ids_bytes = ray_count * sizeof(uint32_t);

    std::memcpy(rays->origins.data, override_rays->origins.data, vec3_bytes);
    std::memcpy(rays->directions.data, override_rays->directions.data, vec3_bytes);
    std::memcpy(rays->t_near.data, override_rays->t_near.data, scalar_bytes);
    std::memcpy(rays->t_far.data, override_rays->t_far.data, scalar_bytes);
    std::memcpy(rays->pixel_ids.data, override_rays->pixel_ids.data, ids_bytes);
    return HP_STATUS_SUCCESS;
}

void configure_output_tensors(hp_rays_t* rays, size_t ray_count) {
    const int64_t count_i64 = static_cast<int64_t>(ray_count);
    const int64_t vec_dim = 3;

    rays->origins.dtype = HP_DTYPE_F32;
    rays->origins.memspace = HP_MEMSPACE_HOST;
    rays->origins.rank = 2;
    rays->origins.shape[0] = count_i64;
    rays->origins.shape[1] = vec_dim;
    rays->origins.stride[1] = 1;
    rays->origins.stride[0] = vec_dim;

    rays->directions.dtype = HP_DTYPE_F32;
    rays->directions.memspace = HP_MEMSPACE_HOST;
    rays->directions.rank = 2;
    rays->directions.shape[0] = count_i64;
    rays->directions.shape[1] = vec_dim;
    rays->directions.stride[1] = 1;
    rays->directions.stride[0] = vec_dim;

    rays->t_near.dtype = HP_DTYPE_F32;
    rays->t_near.memspace = HP_MEMSPACE_HOST;
    rays->t_near.rank = 1;
    rays->t_near.shape[0] = count_i64;
    rays->t_near.stride[0] = 1;

    rays->t_far.dtype = HP_DTYPE_F32;
    rays->t_far.memspace = HP_MEMSPACE_HOST;
    rays->t_far.rank = 1;
    rays->t_far.shape[0] = count_i64;
    rays->t_far.stride[0] = 1;

    rays->pixel_ids.dtype = HP_DTYPE_U32;
    rays->pixel_ids.memspace = HP_MEMSPACE_HOST;
    rays->pixel_ids.rank = 1;
    rays->pixel_ids.shape[0] = count_i64;
    rays->pixel_ids.stride[0] = 1;
}

hp_status ensure_output_buffers(hp_rays_t* rays, size_t ray_count, hp_workspace_allocator& allocator) {
    configure_output_tensors(rays, ray_count);
    if (ray_count == 0U) {
        return HP_STATUS_SUCCESS;
    }
    const size_t vec3_bytes = ray_count * 3U * sizeof(float);
    const size_t scalar_bytes = ray_count * sizeof(float);
    const size_t ids_bytes = ray_count * sizeof(uint32_t);

    if (rays->origins.data == nullptr) {
        rays->origins.data = allocator.allocate(vec3_bytes, alignof(float));
    }
    if (rays->directions.data == nullptr) {
        rays->directions.data = allocator.allocate(vec3_bytes, alignof(float));
    }
    if (rays->t_near.data == nullptr) {
        rays->t_near.data = allocator.allocate(scalar_bytes, alignof(float));
    }
    if (rays->t_far.data == nullptr) {
        rays->t_far.data = allocator.allocate(scalar_bytes, alignof(float));
    }
    if (rays->pixel_ids.data == nullptr) {
        rays->pixel_ids.data = allocator.allocate(ids_bytes, alignof(uint32_t));
    }

    if (rays->origins.data == nullptr || rays->directions.data == nullptr ||
        rays->t_near.data == nullptr || rays->t_far.data == nullptr ||
        rays->pixel_ids.data == nullptr) {
        return HP_STATUS_OUT_OF_MEMORY;
    }

    return HP_STATUS_SUCCESS;
}

}  // namespace

namespace hp_internal {

hp_status ray_generate_cpu(const hp_plan* plan,
                           const hp_rays_t* override_or_null,
                           hp_rays_t* rays,
                           void* ws,
                           size_t ws_bytes) {
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

    hp_workspace_allocator allocator(ws, ws_bytes);
    const hp_status buffer_status = ensure_output_buffers(rays, ray_count, allocator);
    if (buffer_status != HP_STATUS_SUCCESS) {
        return buffer_status;
    }
    if (ray_count == 0U) {
        return HP_STATUS_SUCCESS;
    }

    if (override_or_null != nullptr) {
        return copy_override_cpu(override_or_null, rays, ray_count);
    }

    auto* origins = static_cast<float*>(rays->origins.data);
    auto* directions = static_cast<float*>(rays->directions.data);
    auto* t_near = static_cast<float*>(rays->t_near.data);
    auto* t_far = static_cast<float*>(rays->t_far.data);
    auto* pixel_ids = static_cast<uint32_t*>(rays->pixel_ids.data);

    const hp_camera_desc& cam = desc.camera;
    const float fx = cam.K[0];
    const float fy = cam.K[4];
    const float cx = cam.K[2];
    const float cy = cam.K[5];

    const float r00 = cam.c2w[0];
    const float r01 = cam.c2w[1];
    const float r02 = cam.c2w[2];
    const float r10 = cam.c2w[4];
    const float r11 = cam.c2w[5];
    const float r12 = cam.c2w[6];
    const float r20 = cam.c2w[8];
    const float r21 = cam.c2w[9];
    const float r22 = cam.c2w[10];

    const float cam_tx = cam.c2w[3];
    const float cam_ty = cam.c2w[7];
    const float cam_tz = cam.c2w[11];

    const float near_val = desc.t_near;
    const float far_val = desc.t_far;
    const uint32_t image_width = desc.width;

    for (uint32_t local_y = 0; local_y < roi.height; ++local_y) {
        const uint32_t py = roi.y + local_y;
        for (uint32_t local_x = 0; local_x < roi.width; ++local_x) {
            const uint32_t px = roi.x + local_x;
            const size_t idx = static_cast<size_t>(local_y) * static_cast<size_t>(roi.width) +
                               static_cast<size_t>(local_x);

            const float u = static_cast<float>(px) + 0.5f;
            const float v = static_cast<float>(py) + 0.5f;
            float dir_cam_x = (u - cx) / fx;
            float dir_cam_y = (v - cy) / fy;
            float dir_cam_z = 1.0f;

            if (cam.model == HP_CAMERA_ORTHOGRAPHIC) {
                dir_cam_x = 0.0f;
                dir_cam_y = 0.0f;
                dir_cam_z = 1.0f;
            }

            float dir_world_x = r00 * dir_cam_x + r01 * dir_cam_y + r02 * dir_cam_z;
            float dir_world_y = r10 * dir_cam_x + r11 * dir_cam_y + r12 * dir_cam_z;
            float dir_world_z = r20 * dir_cam_x + r21 * dir_cam_y + r22 * dir_cam_z;

            const float len_sq = dir_world_x * dir_world_x +
                                 dir_world_y * dir_world_y +
                                 dir_world_z * dir_world_z;
            const float inv_len = 1.0f / std::sqrt(std::max(len_sq, std::numeric_limits<float>::min()));
            dir_world_x *= inv_len;
            dir_world_y *= inv_len;
            dir_world_z *= inv_len;

            const size_t vec_offset = idx * 3U;
            origins[vec_offset + 0] = cam_tx;
            origins[vec_offset + 1] = cam_ty;
            origins[vec_offset + 2] = cam_tz;

            directions[vec_offset + 0] = dir_world_x;
            directions[vec_offset + 1] = dir_world_y;
            directions[vec_offset + 2] = dir_world_z;

            t_near[idx] = near_val;
            t_far[idx] = far_val;
            pixel_ids[idx] = py * image_width + px;
        }
    }

    return HP_STATUS_SUCCESS;
}

}  // namespace hp_internal
