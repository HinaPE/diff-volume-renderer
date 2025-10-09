#include "hp_internal.hpp"

#include "workspace.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

struct ImgBuffers {
    float* image{nullptr};
    float* transmittance{nullptr};
    float* opacity{nullptr};
    float* depth{nullptr};
    uint32_t* hitmask{nullptr};
};

hp_status ensure_image_buffers(const hp_plan* plan,
                               hp_img_t* img,
                               hp_workspace_allocator& allocator,
                               ImgBuffers& buffers) {
    const size_t width = plan->desc.width;
    const size_t height = plan->desc.height;
    const size_t pixel_count = width * height;
    const int64_t width_i64 = static_cast<int64_t>(width);
    const int64_t height_i64 = static_cast<int64_t>(height);

    img->image.dtype = HP_DTYPE_F32;
    img->image.memspace = HP_MEMSPACE_HOST;
    img->image.rank = 3;
    img->image.shape[0] = height_i64;
    img->image.shape[1] = width_i64;
    img->image.shape[2] = 3;
    img->image.stride[2] = 1;
    img->image.stride[1] = 3;
    img->image.stride[0] = width_i64 * 3;

    img->trans.dtype = HP_DTYPE_F32;
    img->trans.memspace = HP_MEMSPACE_HOST;
    img->trans.rank = 2;
    img->trans.shape[0] = height_i64;
    img->trans.shape[1] = width_i64;
    img->trans.stride[1] = 1;
    img->trans.stride[0] = width_i64;

    img->opacity.dtype = HP_DTYPE_F32;
    img->opacity.memspace = HP_MEMSPACE_HOST;
    img->opacity.rank = 2;
    img->opacity.shape[0] = height_i64;
    img->opacity.shape[1] = width_i64;
    img->opacity.stride[1] = 1;
    img->opacity.stride[0] = width_i64;

    img->depth.dtype = HP_DTYPE_F32;
    img->depth.memspace = HP_MEMSPACE_HOST;
    img->depth.rank = 2;
    img->depth.shape[0] = height_i64;
    img->depth.shape[1] = width_i64;
    img->depth.stride[1] = 1;
    img->depth.stride[0] = width_i64;

    img->hitmask.dtype = HP_DTYPE_U32;
    img->hitmask.memspace = HP_MEMSPACE_HOST;
    img->hitmask.rank = 2;
    img->hitmask.shape[0] = height_i64;
    img->hitmask.shape[1] = width_i64;
    img->hitmask.stride[1] = 1;
    img->hitmask.stride[0] = width_i64;

    const size_t image_bytes = pixel_count * 3U * sizeof(float);
    const size_t scalar_bytes = pixel_count * sizeof(float);
    const size_t mask_bytes = pixel_count * sizeof(uint32_t);

    if (img->image.data == nullptr && image_bytes > 0) {
        img->image.data = allocator.allocate(image_bytes, alignof(float));
    }
    if (img->trans.data == nullptr && scalar_bytes > 0) {
        img->trans.data = allocator.allocate(scalar_bytes, alignof(float));
    }
    if (img->opacity.data == nullptr && scalar_bytes > 0) {
        img->opacity.data = allocator.allocate(scalar_bytes, alignof(float));
    }
    if (img->depth.data == nullptr && scalar_bytes > 0) {
        img->depth.data = allocator.allocate(scalar_bytes, alignof(float));
    }
    if (img->hitmask.data == nullptr && mask_bytes > 0) {
        img->hitmask.data = allocator.allocate(mask_bytes, alignof(uint32_t));
    }

    if ((image_bytes > 0 && img->image.data == nullptr) ||
        (scalar_bytes > 0 && (img->trans.data == nullptr || img->opacity.data == nullptr || img->depth.data == nullptr)) ||
        (mask_bytes > 0 && img->hitmask.data == nullptr)) {
        return HP_STATUS_OUT_OF_MEMORY;
    }

    buffers.image = static_cast<float*>(img->image.data);
    buffers.transmittance = static_cast<float*>(img->trans.data);
    buffers.opacity = static_cast<float*>(img->opacity.data);
    buffers.depth = static_cast<float*>(img->depth.data);
    buffers.hitmask = static_cast<uint32_t*>(img->hitmask.data);
    return HP_STATUS_SUCCESS;
}

}  // namespace

namespace hp_internal {

hp_status img_generate_cpu(const hp_plan* plan,
                           const hp_intl_t* intl,
                           const hp_rays_t* rays,
                           hp_img_t* img,
                           void* ws,
                           size_t ws_bytes) {
    if (plan == nullptr || intl == nullptr || rays == nullptr || img == nullptr) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (rays->pixel_ids.memspace != HP_MEMSPACE_HOST ||
        intl->radiance.memspace != HP_MEMSPACE_HOST ||
        intl->transmittance.memspace != HP_MEMSPACE_HOST ||
        intl->opacity.memspace != HP_MEMSPACE_HOST ||
        intl->depth.memspace != HP_MEMSPACE_HOST) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    if (rays->pixel_ids.dtype != HP_DTYPE_U32 ||
        intl->radiance.dtype != HP_DTYPE_F32 ||
        intl->transmittance.dtype != HP_DTYPE_F32 ||
        intl->opacity.dtype != HP_DTYPE_F32 ||
        intl->depth.dtype != HP_DTYPE_F32) {
        return HP_STATUS_INVALID_ARGUMENT;
    }

    const size_t ray_count = (intl->transmittance.rank >= 1) ? static_cast<size_t>(intl->transmittance.shape[0]) : 0;
    const size_t width = plan->desc.width;
    const size_t height = plan->desc.height;
    const size_t pixel_count = width * height;

    const uint32_t* pixel_ids = static_cast<const uint32_t*>(rays->pixel_ids.data);
    const float* radiance = static_cast<const float*>(intl->radiance.data);
    const float* transmittance = static_cast<const float*>(intl->transmittance.data);
    const float* opacity = static_cast<const float*>(intl->opacity.data);
    const float* depth = static_cast<const float*>(intl->depth.data);

    hp_workspace_allocator allocator(ws, ws_bytes);
    ImgBuffers buffers{};
    const hp_status buffer_status = ensure_image_buffers(plan, img, allocator, buffers);
    if (buffer_status != HP_STATUS_SUCCESS) {
        return buffer_status;
    }

    const size_t image_len = pixel_count * 3U;
    std::fill(buffers.image, buffers.image + image_len, 0.0f);
    std::fill(buffers.transmittance, buffers.transmittance + pixel_count, 1.0f);
    std::fill(buffers.opacity, buffers.opacity + pixel_count, 0.0f);
    std::fill(buffers.depth, buffers.depth + pixel_count, plan->desc.t_far);
    std::fill(buffers.hitmask, buffers.hitmask + pixel_count, 0U);

    for (size_t ray = 0; ray < ray_count; ++ray) {
        const uint32_t pixel = pixel_ids != nullptr ? pixel_ids[ray] : 0;
        if (pixel >= pixel_count) {
            return HP_STATUS_INVALID_ARGUMENT;
        }
        const size_t base = static_cast<size_t>(pixel) * 3U;
        const size_t ray_base = ray * 3U;

        if (buffers.hitmask[pixel] == 0U) {
            buffers.image[base + 0] = radiance[ray_base + 0];
            buffers.image[base + 1] = radiance[ray_base + 1];
            buffers.image[base + 2] = radiance[ray_base + 2];
            buffers.transmittance[pixel] = transmittance[ray];
            buffers.opacity[pixel] = opacity[ray];
            buffers.depth[pixel] = depth[ray];
            buffers.hitmask[pixel] = 1U;
        } else {
            buffers.image[base + 0] += radiance[ray_base + 0];
            buffers.image[base + 1] += radiance[ray_base + 1];
            buffers.image[base + 2] += radiance[ray_base + 2];
            buffers.transmittance[pixel] *= transmittance[ray];
            buffers.opacity[pixel] = 1.0f - buffers.transmittance[pixel];
            buffers.depth[pixel] = std::min(buffers.depth[pixel], depth[ray]);
        }
    }

    return HP_STATUS_SUCCESS;
}

}  // namespace hp_internal
